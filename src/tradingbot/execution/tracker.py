"""Paper position tracking.

The bot does not place orders. This module exists for one reason: **to find out
whether the signals were any good.** A journal of predictions nobody scores is
just a diary, so every approved signal in `paper` mode opens a simulated
position, and that position is later resolved against real candle data to a
realised R-multiple.

There is no venue here and no credentials. Two consequences shape the design:

**Resolution happens from candles, not from a broker.** For each open paper
position the resolver walks the bars printed since entry and asks whether the
stop or the target was touched. That is honest, free, and reproducible.

**The same-bar ambiguity is resolved pessimistically.** If a single 4h bar's
range contains both the stop and the target, OHLC data cannot tell you which
came first. The resolver assumes **the stop filled.** That is not pedantry: the
optimistic assumption is precisely how a backtest manufactures an edge that
does not survive contact with a real market, and on a 4h bar the ambiguity is
common rather than rare.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from ..db.repo import OrderRepo, PositionRepo
from ..db.schema import Position, Signal
from ..models import (
    Candle,
    ContractSpec,
    OrderRequest,
    OrderStatus,
    OrderType,
    PositionSize,
    Quote,
    Side,
    TradeSetup,
)
from .base_helpers import apply_slippage, client_order_id

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackResult:
    """Outcome of trying to open a simulated position."""

    ok: bool
    detail: str
    position: Position | None = None
    created: bool = False


@dataclass(frozen=True)
class Resolution:
    """One paper position resolved to an outcome."""

    symbol: str
    reason: str
    exit_price: Decimal
    r_multiple: Decimal | None
    bar_time: datetime
    ambiguous: bool = False

    def describe(self) -> str:
        r_text = f"{self.r_multiple:+.2f}R" if self.r_multiple is not None else "n/a"
        suffix = " (same-bar: assumed stop)" if self.ambiguous else ""
        return f"{self.symbol} {self.reason} @ {self.exit_price} -> {r_text}{suffix}"


@dataclass
class PaperTracker:
    """Opens and resolves simulated positions.

    Idempotent on `client_order_id`, exactly as the live path had to be. The
    scheduler can fire twice, and a duplicated run must not produce two
    simulated positions - otherwise the measured performance is wrong in a way
    that quietly flatters or punishes the strategy.
    """

    session: Session
    taker_fee_pct: Decimal = Decimal("0.06")
    slippage_pct: Decimal = Decimal("0.02")
    satellite_slippage_pct: Decimal = Decimal("0.15")
    satellite_symbols: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        self.positions = PositionRepo(self.session)
        self.orders = OrderRepo(self.session)

    # -- opening -----------------------------------------------------------

    def open(
        self,
        *,
        signal: Signal,
        setup: TradeSetup,
        size: PositionSize,
        quote: Quote | None = None,
        spec: ContractSpec | None = None,
    ) -> TrackResult:
        """Record a simulated entry for an approved signal."""
        if not size.viable:
            return TrackResult(False, f"not sized: {size.reason}")

        existing = self.positions.open_for_symbol(setup.symbol)
        if existing is not None:
            return TrackResult(
                False,
                f"already tracking {existing.side} {existing.symbol}",
                position=existing,
            )

        quantity = size.quantity
        if spec is not None:
            quantity = spec.round_quantity(quantity)
            if quantity <= 0:
                return TrackResult(False, "quantity rounds to zero at the venue step")

        token = client_order_id(
            symbol=setup.symbol,
            side=setup.side,
            bar_time=signal.bar_time,
            purpose="paper_entry",
            signal_id=signal.id,
        )
        request = OrderRequest(
            symbol=setup.symbol,
            side=setup.side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            client_order_id=token,
            price=setup.entry,
            leverage=size.leverage,
        )
        order, created = self.orders.reserve(
            request, signal_id=signal.id, venue="paper", is_paper=True
        )
        if not created:
            log.info("Paper entry %s already recorded; not duplicating", token)
            return TrackResult(
                False, "this bar's paper entry was already recorded", created=False
            )

        # Fill at the entry with adverse slippage. The reference is the planned
        # entry rather than the live mark, because the setup specifies a limit
        # price and that is what a real order would have rested at.
        slippage = (
            self.satellite_slippage_pct
            if setup.symbol in self.satellite_symbols
            else self.slippage_pct
        )
        fill = apply_slippage(setup.entry, side=setup.side, slippage_pct=slippage)

        order.status = OrderStatus.FILLED.value
        order.filled_quantity = quantity
        order.average_price = fill
        self.session.flush()

        position = self.positions.create(
            symbol=setup.symbol,
            side=setup.side,
            quantity=quantity,
            entry_price=fill,
            stop_price=setup.stop,
            target_price=setup.primary_target,
            leverage=size.leverage,
            signal_id=signal.id,
            is_paper=True,
        )
        return TrackResult(
            True,
            f"tracking paper {setup.side.value} {quantity} {setup.symbol} @ {fill} "
            f"(stop {setup.stop}, target {setup.primary_target})",
            position=position,
            created=True,
        )

    # -- resolution --------------------------------------------------------

    def resolve(
        self, position: Position, candles: Sequence[Candle]
    ) -> Resolution | None:
        """Close a paper position if price reached its stop or target.

        Walks bars in order and returns on the first one that touches either
        level. Only bars at or after entry are considered, so a level touched
        before the signal existed cannot resolve it.
        """
        if not position.is_open or not position.is_paper:
            return None

        # Normalised because the comparison below is against timezone-aware
        # candle timestamps. Postgres returns an aware value, but SQLite drops
        # tzinfo on round-trip, and comparing naive to aware raises TypeError.
        # Depending on the driver to get this right is not worth the risk.
        opened = _as_utc(position.opened_at)
        side = Side(position.side)
        stop = position.stop_price
        target = position.target_price
        if stop is None:
            return None

        for candle in candles:
            if candle.open_time < opened:
                continue

            hit_stop = (
                candle.low <= stop if side is Side.LONG else candle.high >= stop
            )
            hit_target = target is not None and (
                candle.high >= target if side is Side.LONG else candle.low <= target
            )

            if not hit_stop and not hit_target:
                continue

            # The pessimistic convention. See the module docstring: OHLC cannot
            # order two touches inside one bar, and assuming the favourable one
            # is how paper results stop predicting reality.
            ambiguous = hit_stop and hit_target
            if hit_stop:
                exit_price, reason = stop, "stop"
            else:
                exit_price, reason = target, "target"  # type: ignore[assignment]

            fees = self._round_trip_fees(position, exit_price)
            self.positions.close(
                position,
                exit_price=exit_price,
                reason=f"{reason}_same_bar" if ambiguous else reason,
                fees=fees,
            )
            return Resolution(
                symbol=position.symbol,
                reason=reason,
                exit_price=exit_price,
                r_multiple=position.r_multiple,
                bar_time=candle.open_time,
                ambiguous=ambiguous,
            )
        return None

    def _round_trip_fees(self, position: Position, exit_price: Decimal) -> Decimal:
        """Taker fee on both legs.

        Included so a reported R-multiple is net. A paper journal that ignores
        fees overstates every small-account result, where friction is a
        meaningful fraction of the move.
        """
        entry_notional = position.entry_price * position.quantity
        exit_notional = exit_price * position.quantity
        rate = self.taker_fee_pct / Decimal("100")
        return (entry_notional + exit_notional) * rate

    def expire_stale(
        self, position: Position, *, max_age_hours: float, last_price: Decimal
    ) -> Resolution | None:
        """Close a position that has gone nowhere for too long.

        Without this, a position whose stop and target are both never touched
        stays open forever, permanently consuming one of the two portfolio
        slots `risk.yaml` allows and blocking every later signal on that
        symbol.
        """
        if not position.is_open or not position.is_paper:
            return None
        opened = _as_utc(position.opened_at)
        age_hours = (
            datetime.now(tz=UTC) - opened
        ).total_seconds() / 3600
        if age_hours < max_age_hours:
            return None

        fees = self._round_trip_fees(position, last_price)
        self.positions.close(
            position, exit_price=last_price, reason="expired", fees=fees
        )
        return Resolution(
            symbol=position.symbol,
            reason="expired",
            exit_price=last_price,
            r_multiple=position.r_multiple,
            bar_time=datetime.now(tz=UTC),
        )


def _as_utc(value: datetime) -> datetime:
    """Treat a naive timestamp as UTC.

    Everything this project stores is UTC by convention, so a missing tzinfo is
    a storage artefact rather than genuine local time.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
