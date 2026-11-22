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

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class Environment(str, Enum):
    LOCAL = "local"
    CI = "ci"
    PRODUCTION = "production"


class TradingMode(str, Enum):
    """What the bot is permitted to do with a signal it has approved.

    SIGNAL_ONLY  - journal it and notify. No broker is touched at all.
    PAPER        - journal it and simulate fills against the paper broker.
    LIVE         - place real orders. Requires LIVE_TRADING_ENABLED as well.
    """

    SIGNAL_ONLY = "signal_only"
    PAPER = "paper"
    LIVE = "live"


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

    # -- venue credentials (execution only; market data is public) --------
    weex_api_key: SecretStr | None = None
    weex_api_secret: SecretStr | None = None
    weex_api_passphrase: SecretStr | None = None

    # -- delivery ---------------------------------------------------------
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    # -- reasoning layer --------------------------------------------------
    anthropic_api_key: SecretStr | None = None
    # Judgment layer. Deliberately pinned: an analyst whose model silently
    # changes underneath a live journal makes the journal meaningless.
    anthropic_model: str = "claude-opus-5"
    anthropic_max_tokens: int = 4096
    reasoning_enabled: bool = False

    # -- behaviour --------------------------------------------------------
    trading_mode: TradingMode = TradingMode.SIGNAL_ONLY
    live_trading_enabled: bool = False
    log_level: str = "INFO"

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

    @model_validator(mode="after")
    def _guard_live_trading(self) -> Settings:
        """Two independent switches must agree before real money is at risk.

        `trading_mode=live` alone is not enough, and neither is the
        `live_trading_enabled` flag. Requiring both means a single stray
        environment variable in a deploy config cannot arm the bot.
        """
        if self.trading_mode is TradingMode.LIVE:
            if not self.live_trading_enabled:
                raise ValueError(
                    "trading_mode=live requires LIVE_TRADING_ENABLED=true as well. "
                    "Two switches must agree before real orders can be placed."
                )
            missing = [
                name
                for name, value in (
                    ("WEEX_API_KEY", self.weex_api_key),
                    ("WEEX_API_SECRET", self.weex_api_secret),
                    ("WEEX_API_PASSPHRASE", self.weex_api_passphrase),
                )
                if value is None or not value.get_secret_value().strip()
            ]
            if missing:
                raise ValueError(
                    "trading_mode=live but these credentials are missing: "
                    + ", ".join(missing)
                )
        return self

    # -- derived ----------------------------------------------------------

    @property
    def can_place_real_orders(self) -> bool:
        return self.trading_mode is TradingMode.LIVE and self.live_trading_enabled

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
