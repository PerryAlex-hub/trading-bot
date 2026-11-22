"""WEEX contract (perpetual futures) client.

Market-data endpoints are public and need no authentication.

Two things this module refuses to do:

1. Hardcode contract minimums. WEEX changes them (BTC went 0.001 -> 0.0001,
   ETH 0.01 -> 0.001), and a stale constant means either rejected orders or a
   position size you did not intend.
2. Assume a single API path or field naming. The documented contract-info route
   has moved across /capi/v1, /v2 and /v3, and published field names disagree
   between the official docs, the ccxt implicit API and third-party SDKs. So
   the client probes known candidates and normalises whatever it gets, keeping
   the raw payload so a mismatch is visible instead of silent.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

from ..models import Candle, ContractSpec

log = logging.getLogger(__name__)

BASE_URL = "https://api-contract.weex.com"

# Tried in order until one returns a usable contract list.
CONTRACT_INFO_PATHS: tuple[str, ...] = (
    "/capi/v3/market/contracts",
    "/capi/v2/market/contracts",
    "/capi/v3/market/exchangeInfo",
    "/capi/v1/market/contracts",
)

KLINE_PATHS: tuple[str, ...] = (
    "/capi/v3/market/klines",
    "/capi/v2/market/candles",
)

# Field-name candidates, most specific first. WEEX, ccxt and the community
# SDKs all spell these differently.
_SYMBOL_KEYS = ("symbol", "symbolName", "contract_code", "contractCode", "instrument_id")
_MIN_SIZE_KEYS = ("minOrderSize", "min_order_size", "minTradeNum", "minOrderQty", "minQty", "sizeMin")
_MIN_NOTIONAL_KEYS = ("minOrderValue", "min_order_value", "minNotional", "minTradeValue", "minAmount")
_SIZE_STEP_KEYS = ("sizeMultiplier", "sizeStep", "stepSize", "volumePlace", "sizeIncrement", "qtyStep")
_PRICE_STEP_KEYS = ("priceEndStep", "tickSize", "priceStep", "pricePlace", "priceIncrement")
_MAX_LEV_KEYS = ("maxLeverage", "max_leverage", "leverageMax", "initLeverage")


class WeexEndpointError(RuntimeError):
    """Raised when no known WEEX endpoint path returned a usable payload."""


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _first(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _unwrap(payload: Any) -> list[dict[str, Any]]:
    """Pull the list of contract dicts out of whatever envelope WEEX used."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "result", "symbols", "contracts", "list"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
            if isinstance(inner, dict):
                nested = _unwrap(inner)
                if nested:
                    return nested
    return []


def _decimals_to_step(value: Any) -> Decimal | None:
    """Some venues report a precision (decimal places) rather than a step size.

    A small non-negative integer is almost certainly a place count, so convert
    it; anything else is treated as an actual step.
    """
    dec = _to_decimal(value)
    if dec is None:
        return None
    if dec == dec.to_integral_value() and 0 <= dec <= 12:
        return Decimal(1).scaleb(-int(dec))
    return dec


@dataclass
class WeexClient:
    base_url: str = BASE_URL
    timeout: float = 20.0
    cache_dir: Path | None = None
    cache_ttl_seconds: int = 24 * 60 * 60
    _client: httpx.Client | None = None

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- plumbing ---------------------------------------------------------

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

    def __enter__(self) -> WeexClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _try_paths(
        self, paths: Sequence[str], params: dict[str, Any] | None = None
    ) -> tuple[str, Any]:
        """Return (path, payload) for the first path that yields JSON."""
        errors: list[str] = []
        for path in paths:
            try:
                response = self._http().get(path, params=params)
            except httpx.HTTPError as exc:
                errors.append(f"{path}: transport error: {exc}")
                continue
            if response.status_code != 200:
                errors.append(f"{path}: HTTP {response.status_code}")
                continue
            try:
                payload = response.json()
            except json.JSONDecodeError:
                errors.append(f"{path}: response was not JSON")
                continue
            log.debug("WEEX endpoint %s responded", path)
            return path, payload
        raise WeexEndpointError(
            "No known WEEX endpoint responded usefully.\n  " + "\n  ".join(errors)
        )

    # -- contract specifications -----------------------------------------

    def _cache_path(self) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / "weex_contracts.json"

    def _read_cache(self) -> Any | None:
        path = self._cache_path()
        if path is None or not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.cache_ttl_seconds:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, payload: Any) -> None:
        path = self._cache_path()
        if path is None:
            return
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk issues
            log.warning("Could not cache WEEX contracts: %s", exc)

    def fetch_contracts_raw(self, *, use_cache: bool = True) -> Any:
        if use_cache:
            cached = self._read_cache()
            if cached is not None:
                log.debug("Using cached WEEX contract info")
                return cached
        _path, payload = self._try_paths(CONTRACT_INFO_PATHS)
        self._write_cache(payload)
        return payload

    def contract_specs(self, *, use_cache: bool = True) -> dict[str, ContractSpec]:
        """Every contract WEEX reports, keyed by normalised symbol."""
        entries = _unwrap(self.fetch_contracts_raw(use_cache=use_cache))
        if not entries:
            raise WeexEndpointError(
                "WEEX returned a payload with no recognisable contract list. "
                "Inspect the cached JSON and extend _unwrap()."
            )

        specs: dict[str, ContractSpec] = {}
        for entry in entries:
            symbol_raw = _first(entry, _SYMBOL_KEYS)
            if symbol_raw is None:
                continue
            symbol = normalise_symbol(str(symbol_raw))
            max_lev = _to_decimal(_first(entry, _MAX_LEV_KEYS))
            specs[symbol] = ContractSpec(
                symbol=symbol,
                min_order_size=_to_decimal(_first(entry, _MIN_SIZE_KEYS)),
                min_notional=_to_decimal(_first(entry, _MIN_NOTIONAL_KEYS)),
                size_step=_decimals_to_step(_first(entry, _SIZE_STEP_KEYS)),
                price_step=_decimals_to_step(_first(entry, _PRICE_STEP_KEYS)),
                max_leverage=int(max_lev) if max_lev else None,
                raw=entry,
            )
        return specs

    def contract_spec(self, symbol: str, *, use_cache: bool = True) -> ContractSpec:
        specs = self.contract_specs(use_cache=use_cache)
        key = normalise_symbol(symbol)
        if key not in specs:
            raise KeyError(
                f"{symbol} is not in the WEEX contract list "
                f"({len(specs)} contracts seen). It may be delisted or spot-only."
            )
        return specs[key]

    # -- candles ----------------------------------------------------------

    def klines(self, symbol: str, interval: str = "4h", limit: int = 200) -> list[Candle]:
        """Recent candles.

        WEEX caps `limit` low (300 on the historical route), which is why
        backtest history comes from Binance instead.
        """
        _path, payload = self._try_paths(
            KLINE_PATHS,
            params={
                "symbol": symbol,
                "interval": interval,
                "granularity": interval,
                "limit": limit,
            },
        )
        rows = payload if isinstance(payload, list) else _unwrap(payload)
        candles: list[Candle] = []
        for row in rows:
            try:
                if isinstance(row, dict):
                    candles.append(
                        Candle.from_row(
                            [
                                row.get("t") or row.get("time") or row.get("openTime"),
                                row.get("o") or row.get("open"),
                                row.get("h") or row.get("high"),
                                row.get("l") or row.get("low"),
                                row.get("c") or row.get("close"),
                                row.get("v") or row.get("volume") or 0,
                            ]
                        )
                    )
                else:
                    candles.append(Candle.from_row(row))
            except (ValueError, TypeError, IndexError, InvalidOperation) as exc:
                log.debug("Skipping unparseable WEEX kline row %r: %s", row, exc)
        candles.sort(key=lambda candle: candle.open_time)
        return candles


def normalise_symbol(symbol: str) -> str:
    return symbol.upper().replace("-", "").replace("_", "").replace("/", "")
