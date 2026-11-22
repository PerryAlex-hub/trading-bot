"""Environment-driven settings.

Phase 0 read configuration relative to the repository checkout. That breaks the
moment this runs anywhere real: in a container the package is installed, not
checked out, and there is no writable project directory. Everything that
differs between a laptop, a GitHub Actions runner and a Render dyno is resolved
here and nowhere else.

Precedence: real environment variables win over `.env`, so a platform's
injected secrets always beat a file that got committed by accident.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class Environment(str, Enum):
    LOCAL = "local"
    CI = "ci"
    PRODUCTION = "production"


class TradingMode(str, Enum):
    """What the bot does with a signal it has approved.

    SIGNAL_ONLY - journal it and send it to Telegram. Nothing is tracked.
    PAPER       - also open a simulated position so the journal can measure
                  whether the signals actually worked, in R.

    There is deliberately no LIVE mode. This bot does not place orders, and
    that is enforced by the absence of any code that could: there are no
    exchange credentials in the settings and no authenticated endpoints in the
    exchanges package. Removing the capability is a stronger guarantee than
    gating it behind a flag.
    """

    SIGNAL_ONLY = "signal_only"
    PAPER = "paper"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Environment = Environment.LOCAL

    # -- database ---------------------------------------------------------
    # Neon hands out two connection strings. Use the POOLED one (the host with
    # `-pooler` in it) for the API and for short cron jobs: Neon's free compute
    # suspends after 5 idle minutes, and the pooler absorbs the reconnect.
    # The direct URL is only needed for migrations, since PgBouncer in
    # transaction mode cannot run them.
    database_url: SecretStr | None = None
    database_direct_url: SecretStr | None = Field(default=None)
    db_statement_timeout_ms: int = 30_000
    db_pool_size: int = 3
    db_max_overflow: int = 2

    # No venue credentials. WEEX market data (quotes, funding, contract specs)
    # is public, and nothing here places orders.

    # -- market data source -----------------------------------------------
    # Binance's main API hosts are DNS-blocked by some ISPs and geo-blocked
    # (HTTP 451) from some cloud IP ranges, which includes CI runners in
    # certain regions. `data-api.binance.vision` is Binance's public
    # market-data mirror and is usually reachable where the others are not.
    # It serves SPOT candles only, hence the futures flag below.
    binance_base_url: str | None = None
    # Futures candles match a perpetual contract more closely, but the mirror
    # has no futures route. Set false to use spot.
    binance_use_futures: bool = True

    # -- Telegram: both the delivery channel AND the user interface --------
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    # Shared secret Telegram echoes back in X-Telegram-Bot-Api-Secret-Token.
    # The webhook URL is effectively public, so this is what stops anyone who
    # guesses it from injecting fake commands.
    telegram_webhook_secret: SecretStr | None = None
    # Chat ids allowed to issue commands. Defaults to telegram_chat_id alone.
    # Without this the bot would answer anyone who found it in search.
    telegram_allowed_chat_ids: str | None = None

    # -- reasoning layer --------------------------------------------------
    anthropic_api_key: SecretStr | None = None
    # Judgment layer. Deliberately pinned: an analyst whose model silently
    # changes underneath a live journal makes the journal meaningless.
    anthropic_model: str = "claude-opus-5"
    anthropic_max_tokens: int = 4096
    reasoning_enabled: bool = False

    # -- behaviour --------------------------------------------------------
    trading_mode: TradingMode = TradingMode.SIGNAL_ONLY
    log_level: str = "INFO"
    # Public base URL of the deployed API, used to register the Telegram
    # webhook. Telegram requires HTTPS and will not accept localhost.
    public_base_url: str | None = None

    # Writable scratch space. On a read-only container filesystem this must
    # point somewhere real, hence /tmp rather than the project directory.
    cache_dir: Path = Path("/tmp/tradingbot-cache")

    # Config files ship inside the package by default so a container does not
    # need the repository mounted; override to point at a checkout.
    config_dir: Path | None = None

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @field_validator("telegram_chat_id", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    # -- derived ----------------------------------------------------------

    @property
    def can_place_real_orders(self) -> bool:
        """Always False, and kept as an explicit, testable statement of fact.

        Retained rather than deleted so that any caller still asking the
        question gets a hard "no" instead of an AttributeError that someone
        might "fix" by reintroducing the capability.
        """
        return False

    @property
    def tracks_paper_positions(self) -> bool:
        return self.trading_mode is TradingMode.PAPER

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    def require_database_url(self) -> str:
        if self.database_url is None:
            raise RuntimeError(
                "DATABASE_URL is not set. Copy the POOLED connection string from "
                "the Neon dashboard (the host containing '-pooler')."
            )
        return self.database_url.get_secret_value()

    def migration_database_url(self) -> str:
        """Alembic needs the unpooled URL; PgBouncer cannot run DDL reliably."""
        if self.database_direct_url is not None:
            return self.database_direct_url.get_secret_value()
        return self.require_database_url().replace("-pooler", "")

    def allowed_chat_ids(self) -> frozenset[str]:
        """Chat ids permitted to command the bot.

        Falls back to the delivery chat id so a single-user setup needs no
        extra configuration. An empty set means the bot answers nobody, which
        is the correct failure mode for an unconfigured deployment.
        """
        if self.telegram_allowed_chat_ids:
            return frozenset(
                part.strip()
                for part in self.telegram_allowed_chat_ids.replace(";", ",").split(",")
                if part.strip()
            )
        return frozenset({self.telegram_chat_id}) if self.telegram_chat_id else frozenset()

    def resolved_config_dir(self) -> Path:
        if self.config_dir is not None:
            return self.config_dir
        packaged = Path(__file__).resolve().parent / "resources" / "config"
        if packaged.is_dir():
            return packaged
        # Editable install / repo checkout.
        return Path(__file__).resolve().parents[2] / "config"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_logging(settings: Settings | None = None) -> None:
    """Structured-ish logging that reads well in both Actions and Render logs."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
