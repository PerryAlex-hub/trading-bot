"""Core value types.

Decimal is used for anything that touches order sizing or price. Floats are fine
for indicators, not for deciding how much of an asset to buy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any


class Tier(str, Enum):
    CORE = "core"
    SATELLITE = "satellite"
    PAPER = "paper"
    CONTEXT = "context"

    @property
    def tradeable(self) -> bool:
        """PAPER and CONTEXT never produce a live order."""
        return self in (Tier.CORE, Tier.SATELLITE)


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class Candle:
    """A single OHLCV bar. `open_time` is always timezone-aware UTC."""

    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self) -> None:
        if self.open_time.tzinfo is None:
            raise ValueError("Candle.open_time must be timezone-aware (UTC)")

    @classmethod
    def from_row(cls, row: list[Any] | tuple[Any, ...]) -> Candle:
        """Build from the common [ts_ms, o, h, l, c, v] ordering."""
        ts_ms, o, h, low, c, v = row[:6]
        return cls(
            open_time=datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc),
            open=Decimal(str(o)),
            high=Decimal(str(h)),
            low=Decimal(str(low)),
            close=Decimal(str(c)),
            volume=Decimal(str(v)),
        )


@dataclass(frozen=True)
class ContractSpec:
    """Exchange-reported trading constraints for one contract.

    These values are ALWAYS fetched at runtime and cached. They are never
    hardcoded: WEEX adjusts minimums regularly, and a stale constant here means
    silently rejected orders or an unintended position size.
    """

    symbol: str
    min_order_size: Decimal | None = None       # in base asset
    size_step: Decimal | None = None
    price_step: Decimal | None = None
    min_notional: Decimal | None = None         # in quote asset (USDT)
    max_leverage: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def min_cost(self, price: Decimal, leverage: Decimal | int) -> Decimal | None:
        """Margin required to open the smallest permitted position.

        This is the number that actually matters for a small account. An
        exchange minimum is expressed as notional; margin is notional/leverage.
        Returns None when the exchange did not report enough information.
        """
        notional = self.effective_min_notional(price)
        if notional is None:
            return None
        lev = Decimal(str(leverage))
        if lev <= 0:
            raise ValueError("leverage must be positive")
        return notional / lev

    def effective_min_notional(self, price: Decimal) -> Decimal | None:
        """The binding minimum position value in quote terms.

        Exchanges often enforce both a base-asset minimum and a quote-value
        minimum. Whichever is larger is the one you actually have to clear.
        """
        candidates: list[Decimal] = []
        if self.min_order_size is not None:
            candidates.append(self.min_order_size * price)
        if self.min_notional is not None:
            candidates.append(self.min_notional)
        return max(candidates) if candidates else None


@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    tier: Tier
    notes: str = ""
    min_history_days: int | None = None

    @property
    def base(self) -> str:
        for quote in ("USDT", "USDC", "USD"):
            if self.symbol.endswith(quote):
                return self.symbol[: -len(quote)]
        return self.symbol
