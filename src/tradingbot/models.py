"""Core value types.

Decimal is used for anything that touches order sizing or price. Floats are fine
for indicators, not for deciding how much of an asset to buy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
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

    @property
    def opposite(self) -> Side:
        if self is Side.LONG:
            return Side.SHORT
        if self is Side.SHORT:
            return Side.LONG
        return Side.FLAT

    @property
    def sign(self) -> int:
        return {"long": 1, "short": -1, "flat": 0}[self.value]


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
            open_time=datetime.fromtimestamp(int(ts_ms) / 1000, tz=UTC),
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

    def round_quantity(self, quantity: Decimal) -> Decimal:
        """Snap a quantity DOWN to the venue step.

        Always down: rounding up can push required margin past what the risk
        budget approved, which would mean the sized trade risks more than the
        risk gate agreed to.
        """
        step = self.size_step
        if step is None or step <= 0:
            return quantity
        return (quantity // step) * step

    def round_quantity_up(self, quantity: Decimal) -> Decimal:
        """Snap a quantity UP to the venue step.

        The deliberate exception to `round_quantity`. Used only when clearing a
        venue minimum, where rounding down would land back under the floor and
        the order would simply be rejected. The extra risk this creates is
        disclosed on the ticket rather than absorbed silently.
        """
        step = self.size_step
        if step is None or step <= 0:
            return quantity
        steps = quantity / step
        whole = steps.quantize(Decimal("1"), rounding=ROUND_DOWN)
        if whole < steps:
            whole += 1
        return whole * step

    def round_price(self, price: Decimal) -> Decimal:
        step = self.price_step
        if step is None or step <= 0:
            return price
        return (price / step).quantize(Decimal("1")) * step


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


# ---------------------------------------------------------------------------
# Trading domain
#
# These are the types the journal, the risk gate and the execution seam all
# speak. They are deliberately frozen: a TradeSetup that can be mutated after
# the risk gate approved it is a setup whose journal entry is a lie.
# ---------------------------------------------------------------------------


class SignalStatus(str, Enum):
    """Lifecycle of one analysis result.

    Every terminal state is journaled, including rejections. A bot that only
    records its approved signals cannot tell you whether its risk rules are too
    tight, and that is usually the most valuable thing in the log.
    """

    CANDIDATE = "candidate"                     # screener found it, not yet gated
    REJECTED_RISK = "rejected_risk"             # deterministic gate vetoed it
    REJECTED_REASONING = "rejected_reasoning"   # analyst declined it
    APPROVED = "approved"                       # passed everything, not yet acted on
    EXECUTED = "executed"                       # an order went out (paper or live)
    EXPIRED = "expired"                         # entry never triggered in validity
    CLOSED = "closed"                           # position opened and finished


class Timeframe(str, Enum):
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def seconds(self) -> int:
        return {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[self.value]


class Trend(str, Enum):
    STRONG_UP = "strong_up"
    UP = "up"
    RANGE = "range"
    DOWN = "down"
    STRONG_DOWN = "strong_down"

    @property
    def direction(self) -> int:
        """Signed strength, so trend agreement across timeframes is arithmetic."""
        return {
            "strong_up": 2,
            "up": 1,
            "range": 0,
            "down": -1,
            "strong_down": -2,
        }[self.value]

    @property
    def is_up(self) -> bool:
        return self.direction > 0

    @property
    def is_down(self) -> bool:
        return self.direction < 0


@dataclass(frozen=True)
class TradeSetup:
    """A concrete, fully-specified trade proposal.

    Prices are Decimal because they become order parameters. Nothing here is
    optional for convenience: a setup without a stop is not a setup.
    """

    symbol: str
    side: Side
    entry: Decimal
    stop: Decimal
    targets: tuple[Decimal, ...]
    timeframe: Timeframe = Timeframe.H4

    def __post_init__(self) -> None:
        if self.side not in (Side.LONG, Side.SHORT):
            raise ValueError(f"A setup needs a direction, got {self.side}")
        if not self.targets:
            raise ValueError("A setup needs at least one target")
        if self.side is Side.LONG and self.stop >= self.entry:
            raise ValueError(f"Long stop {self.stop} must sit below entry {self.entry}")
        if self.side is Side.SHORT and self.stop <= self.entry:
            raise ValueError(f"Short stop {self.stop} must sit above entry {self.entry}")

    @property
    def risk_per_unit(self) -> Decimal:
        """Distance to stop. Always positive, direction-independent."""
        return abs(self.entry - self.stop)

    @property
    def primary_target(self) -> Decimal:
        return self.targets[0]

    def reward_per_unit(self, target: Decimal | None = None) -> Decimal:
        chosen = target if target is not None else self.primary_target
        return abs(chosen - self.entry)

    def risk_reward(self, target: Decimal | None = None) -> Decimal:
        risk = self.risk_per_unit
        if risk == 0:
            raise ValueError("Setup has a zero stop distance")
        return self.reward_per_unit(target) / risk

    def is_favourable(self, price: Decimal) -> bool:
        """True when `price` has not already run past the entry.

        Guards the gap between analysis and order placement: a cron that fires
        a few minutes late must not chase a move it already missed.
        """
        if self.side is Side.LONG:
            return price <= self.entry
        return price >= self.entry


@dataclass(frozen=True)
class PositionSize:
    """Output of sizing. Carries its own refusal reason rather than raising.

    Sizing legitimately fails on a small account: the exchange minimum can
    exceed what the risk budget permits. That is a normal, journalable outcome,
    not an exception.
    """

    symbol: str
    quantity: Decimal
    notional: Decimal
    margin: Decimal
    leverage: Decimal
    risk_amount: Decimal
    viable: bool = True
    reason: str = ""
    # True when the risk budget could not buy a tradeable position and the size
    # was raised to the venue minimum instead. The ticket must say so: the
    # trade then risks more than the configured per-trade budget.
    at_venue_minimum: bool = False
    # What the position actually risks, as a % of equity. Equal to the
    # configured risk_pct in the normal case, higher when at_venue_minimum.
    effective_risk_pct: Decimal | None = None

    @classmethod
    def unviable(cls, symbol: str, reason: str) -> PositionSize:
        zero = Decimal("0")
        return cls(
            symbol=symbol,
            quantity=zero,
            notional=zero,
            margin=zero,
            leverage=zero,
            risk_amount=zero,
            viable=False,
            reason=reason,
        )


@dataclass(frozen=True)
class RiskVeto:
    """One failed rule. Carries the rule name so the journal is diagnosable."""

    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    vetoes: tuple[RiskVeto, ...] = ()
    size: PositionSize | None = None
    warnings: tuple[str, ...] = ()

    @property
    def veto_summary(self) -> str:
        return "; ".join(str(veto) for veto in self.vetoes) or "none"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT_MARKET = "take_profit_market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.FAILED,
        )


@dataclass(frozen=True)
class OrderRequest:
    """An intent to place one order.

    `client_order_id` is mandatory and deterministic. The bot runs on a cron
    that can be retried or double-fired by the scheduler, so every order must
    be idempotent at the venue: resubmitting the same intent has to come back
    as a duplicate rather than opening a second position.
    """

    symbol: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    client_order_id: str
    price: Decimal | None = None
    trigger_price: Decimal | None = None
    leverage: Decimal | None = None
    reduce_only: bool = False

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if self.order_type is OrderType.LIMIT and self.price is None:
            raise ValueError("A limit order needs a price")
        if self.order_type in (OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET):
            if self.trigger_price is None:
                raise ValueError(f"{self.order_type.value} needs a trigger_price")
            if not self.reduce_only:
                raise ValueError(
                    "Protective orders must be reduce_only, otherwise a trigger "
                    "opens a new position instead of closing the existing one."
                )
        if not self.client_order_id:
            raise ValueError("client_order_id is required for idempotency")


@dataclass(frozen=True)
class OrderResult:
    client_order_id: str
    status: OrderStatus
    venue_order_id: str | None = None
    filled_quantity: Decimal = Decimal("0")
    average_price: Decimal | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return self.status not in (OrderStatus.REJECTED, OrderStatus.FAILED)


@dataclass(frozen=True)
class Quote:
    """Live top-of-book, used for spread and slippage checks."""

    symbol: str
    bid: Decimal
    ask: Decimal
    mark: Decimal
    at: datetime

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> Decimal:
        mid = self.mid
        if mid <= 0:
            raise ValueError(f"Nonsensical quote for {self.symbol}: mid={mid}")
        return (self.ask - self.bid) / mid * Decimal("100")


@dataclass(frozen=True)
class FundingInfo:
    symbol: str
    rate_8h_pct: Decimal
    next_funding_time: datetime | None = None

    def carry_pct(self, intervals: int) -> Decimal:
        """Total funding cost over an expected hold, as a % of notional."""
        return self.rate_8h_pct * Decimal(intervals)
