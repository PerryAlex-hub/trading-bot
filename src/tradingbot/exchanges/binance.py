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
        response = self._http().get(self._path, params=params)
        response.raise_for_status()
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
