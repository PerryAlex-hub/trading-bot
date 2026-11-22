"""Configuration loading.

Universe and risk rules live in YAML so the trading logic never needs editing to
change what is traded or how much is risked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

from .models import SymbolConfig, Tier  # noqa: E402  (after PROJECT_ROOT for clarity)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass(frozen=True)
class Universe:
    quote: str
    symbols: tuple[SymbolConfig, ...]
    context: tuple[SymbolConfig, ...]
    primary_timeframe: str
    additional_timeframes: tuple[str, ...]

    @property
    def all_timeframes(self) -> tuple[str, ...]:
        return (self.primary_timeframe, *self.additional_timeframes)

    def tradeable(self) -> tuple[SymbolConfig, ...]:
        return tuple(s for s in self.symbols if s.tier.tradeable)

    def by_symbol(self, symbol: str) -> SymbolConfig | None:
        for candidate in (*self.symbols, *self.context):
            if candidate.symbol == symbol:
                return candidate
        return None

    def symbols_requiring_data(self) -> tuple[str, ...]:
        """Every symbol we need candles for, including paper and context names."""
        seen: dict[str, None] = {}
        for entry in (*self.symbols, *self.context):
            if entry.tier is Tier.CONTEXT and entry.symbol.endswith(".D"):
                continue  # derived series, not an exchange symbol
            seen.setdefault(entry.symbol, None)
        return tuple(seen)


def _parse_symbol(raw: dict[str, Any], default_tier: Tier) -> SymbolConfig:
    tier_name = str(raw.get("tier", default_tier.value)).lower()
    try:
        tier = Tier(tier_name)
    except ValueError as exc:
        raise ValueError(
            f"Unknown tier {tier_name!r} for {raw.get('symbol')!r}. "
            f"Expected one of: {', '.join(t.value for t in Tier)}"
        ) from exc
    return SymbolConfig(
        symbol=str(raw["symbol"]),
        tier=tier,
        notes=str(raw.get("notes", "") or "").strip(),
        min_history_days=raw.get("min_history_days"),
    )


@lru_cache(maxsize=1)
def load_universe(path: Path | None = None) -> Universe:
    data = _load_yaml(path or CONFIG_DIR / "universe.yaml")
    timeframes = data.get("timeframes", {}) or {}
    return Universe(
        quote=str(data.get("quote", "USDT")),
        symbols=tuple(
            _parse_symbol(entry, Tier.CORE) for entry in data.get("symbols", [])
        ),
        context=tuple(
            _parse_symbol(entry, Tier.CONTEXT) for entry in data.get("context", [])
        ),
        primary_timeframe=str(timeframes.get("primary", "4h")),
        additional_timeframes=tuple(
            str(tf) for tf in (timeframes.get("additional") or [])
        ),
    )


@dataclass(frozen=True)
class RiskConfig:
    """Thin typed wrapper over risk.yaml.

    Deliberately keeps the raw mapping around: the risk engine reads nested
    rules directly, and silently dropping an unrecognised rule would be worse
    than carrying it.
    """

    raw: dict[str, Any]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def decimal(self, *keys: str, default: str = "0") -> Decimal:
        value = self.get(*keys, default=default)
        return Decimal(str(value))

    @property
    def assumed_equity(self) -> Decimal:
        return self.decimal("account", "assumed_equity_usdt", default="0")

    @property
    def risk_pct(self) -> Decimal:
        return self.decimal("per_trade", "risk_pct", default="0.5")

    @property
    def min_risk_reward(self) -> Decimal:
        return self.decimal("per_trade", "min_risk_reward", default="2.0")

    def max_leverage(self, tier: Tier) -> int:
        key = "satellite_max_effective" if tier is Tier.SATELLITE else "max_effective"
        return int(self.get("leverage", key, default=self.get("leverage", "max_effective", default=10)))

    def beta_to_btc(self, symbol: str) -> Decimal:
        table = self.get("portfolio", "beta_to_btc", default={}) or {}
        if symbol in table:
            return Decimal(str(table[symbol]))
        return Decimal(str(table.get("default", "2.0")))


@lru_cache(maxsize=1)
def load_risk(path: Path | None = None) -> RiskConfig:
    return RiskConfig(raw=_load_yaml(path or CONFIG_DIR / "risk.yaml"))


def live_trading_enabled() -> bool:
    """Live execution requires an explicit opt-in, checked at the call site.

    Phase 0 has no execution path at all. This exists so that when one is added
    it is gated from the first commit rather than bolted on afterwards.
    """
    return os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() == "true"
