"""Binance public market data, used for history and backtesting.

Why a second venue at all: WEEX caps klines at a few hundred rows per call and
its history is shallow, which is useless for a backtest. Binance serves years
of clean candles with no account and no key. On a 4h timeframe the cross-venue
price difference is noise; a live signal is still validated against the WEEX
mark price before it is acted on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import InvalidOperation
from typing import Any

import httpx

from ..models import Candle

log = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
# Binance's public market-data mirror. Spot routes only, but reachable from
# networks and cloud regions where the main hosts are DNS- or geo-blocked.
DATA_MIRROR_URL = "https://data-api.binance.vision"

MAX_LIMIT = 1000

# Binance interval strings happen to match the ones used in config/universe.yaml.
SUPPORTED_INTERVALS = (
    "1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w",
)


@dataclass
class BinanceHistory:
    """Paged OHLCV reader.

    `futures=True` reads USD-M perpetual candles, which match what WEEX
    actually trades more closely than spot does.
    """

    futures: bool = True
    timeout: float = 20.0
    # Explicit host override. Needed because Binance's main API is DNS-blocked
    # by some ISPs and returns HTTP 451 from some cloud IP ranges; the public
    # data mirror is reachable where those are not.
    base_url_override: str | None = None
    _client: httpx.Client | None = None
    # Set once a geo-block has pushed this client onto the mirror, so the
    # fallback is attempted at most once instead of on every page.
    _on_mirror: bool = False

    def __post_init__(self) -> None:
        if self.base_url_override and DATA_MIRROR_URL in self.base_url_override:
            self._on_mirror = True

    @classmethod
    def from_settings(cls, settings=None, *, timeout: float = 20.0) -> BinanceHistory:
        """Build from configuration.

        Keeps the host and spot/futures choice in one place, so a blocked
        network is a config change rather than a code change.
        """
        if settings is None:
            from ..settings import get_settings

            settings = get_settings()
        return cls(
            futures=settings.binance_use_futures,
            base_url_override=settings.binance_base_url,
            timeout=timeout,
        )

    @property
    def base_url(self) -> str:
        if self.base_url_override:
            return self.base_url_override.rstrip("/")
        return FUTURES_BASE_URL if self.futures else BASE_URL

    @property
    def _path(self) -> str:
        return "/fapi/v1/klines" if self.futures else "/api/v3/klines"

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> BinanceHistory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # Status codes Binance uses to refuse a caller by location rather than by
    # request. 451 is the documented one; 403 appears from some ranges too.
    _GEO_BLOCKED = frozenset({403, 451})

    def _fall_back_to_mirror(self) -> bool:
        """Switch to the public data mirror after a geo-block. Returns success.

        Worth doing automatically rather than leaving to configuration. The
        workflow already carried a comment telling the operator to set
        BINANCE_BASE_URL by hand when this happened, and the first deployment
        shows why that is not enough: every ingest returned 451, the newly
        added symbols had no stored candles to fall back on, and so they were
        skipped silently while the run reported success. A failure mode whose
        fix is "read a comment and set two repository variables" is a failure
        mode that stays broken.

        The mirror serves SPOT routes only, so this also turns futures off.
        That is a real difference - spot candles carry no perpetual basis - but
        on a 4h decision timeframe it is far smaller than the difference
        between having candles and having none.
        """
        if self._on_mirror:
            return False
        log.warning(
            "Binance returned a geo-block; falling back to the public data "
            "mirror at %s (spot routes)",
            DATA_MIRROR_URL,
        )
        self.base_url_override = DATA_MIRROR_URL
        self.futures = False
        self._on_mirror = True
        self.close()  # force a new client bound to the new base URL
        return True

    def _page(
        self, symbol: str, interval: str, start_ms: int | None, limit: int
    ) -> list[list[Any]]:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, MAX_LIMIT),
        }
        if start_ms is not None:
            params["startTime"] = start_ms

        try:
            response = self._http().get(self._path, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in self._GEO_BLOCKED and self._fall_back_to_mirror():
                response = self._http().get(self._path, params=params)
                response.raise_for_status()
            else:
                raise

        payload = response.json()
        return payload if isinstance(payload, list) else []

    def candles(
        self,
        symbol: str,
        interval: str = "4h",
        start: datetime | None = None,
        max_candles: int = 5000,
    ) -> list[Candle]:
        """Fetch candles forward from `start`, paging until caught up.

        Closed bars only: the in-progress final bar is dropped, because feeding
        a partial candle into an indicator is a quiet source of lookahead bias.
        """
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(
                f"Unsupported interval {interval!r}. "
                f"Expected one of: {', '.join(SUPPORTED_INTERVALS)}"
            )

        start_ms = int(start.timestamp() * 1000) if start is not None else None
        collected: list[Candle] = []
        seen: set[datetime] = set()

        while len(collected) < max_candles:
            rows = self._page(symbol, interval, start_ms, MAX_LIMIT)
            if not rows:
                break

            added = 0
            for row in rows:
                try:
                    candle = Candle.from_row(row)
                except (ValueError, TypeError, IndexError, InvalidOperation) as exc:
                    log.debug("Skipping unparseable Binance row %r: %s", row, exc)
                    continue
                if candle.open_time in seen:
                    continue
                seen.add(candle.open_time)
                collected.append(candle)
                added += 1

            if added == 0:
                break  # no forward progress; avoid looping on a repeated page

            start_ms = int(rows[-1][0]) + 1
            if len(rows) < MAX_LIMIT:
                break

        collected.sort(key=lambda candle: candle.open_time)
        collected = self._drop_unclosed(collected, interval)
        return collected[:max_candles]

    @staticmethod
    def _drop_unclosed(candles: list[Candle], interval: str) -> list[Candle]:
        if not candles:
            return candles
        seconds = _interval_seconds(interval)
        now = datetime.now(tz=UTC)
        last = candles[-1]
        if (now - last.open_time).total_seconds() < seconds:
            return candles[:-1]
        return candles


def _interval_seconds(interval: str) -> int:
    unit = interval[-1]
    amount = int(interval[:-1])
    multipliers = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    if unit not in multipliers:
        raise ValueError(f"Cannot interpret interval {interval!r}")
    return amount * multipliers[unit]
