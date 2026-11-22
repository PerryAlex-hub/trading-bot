"""The execution engine.

This module is the reason cron-driven live trading is safe rather than
reckless, and the whole design follows from one fact about the deployment:

    **The bot is asleep most of the time.**

It wakes every few hours on a free scheduler that can be delayed. It therefore
cannot watch price, cannot react to a move, and must not be the thing standing
between an open position and a loss. So:

**The stop is placed at the exchange, in the same operation as the entry.**
WEEX enforces it 24/7 whether or not this process ever runs again. An open
position with no resting stop order is treated as an incident, not a warning.

**Order of operations is not negotiable.** Entry first, then the protective
stop, then the target. If the stop cannot be placed after the entry filled,
the engine *closes the position immediately* rather than leaving it naked. An
unprotected position on a thin satellite book is a worse outcome than a small
realised loss on the round trip.

**Every write happens before the network call.** The order row is persisted as
PENDING, then sent. If the process dies mid-request, the next run finds a
PENDING row and reconciles it against the venue. The opposite order - send
first, record after - can produce a live position this bot does not know about,
which is the single worst failure mode available to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from sqlalchemy.orm import Session

from ..db.repo import OrderRepo, PositionRepo
from ..db.schema import Position, Signal
from ..models import (
    ContractSpec,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    PositionSize,
    Quote,
    Side,
    TradeSetup,
)
from .base_helpers import client_order_id

log = logging.getLogger(__name__)


class Venue(Protocol):
    """Minimal execution surface, satisfied by both the live and paper clients."""

    @property
    def is_paper(self) -> bool: ...
    def place_order(self, request: OrderRequest) -> OrderResult: ...
    def cancel_order(self, symbol: str, client_order_id: str) -> bool: ...


@dataclass(frozen=True)
class ExecutionReport:
    """What actually happened when a signal was acted on."""

    ok: bool
    detail: str
    position: Position | None = None
    entry: OrderResult | None = None
    stop: OrderResult | None = None
    target: OrderResult | None = None
    unprotected: bool = False

    @property
    def needs_attention(self) -> bool:
        """Whether a human should look at this now.

        True when a position exists without a stop at the venue. That state is
        an alarm: on free infrastructure nothing will be watching it.
        """
        return self.unprotected


@dataclass
class ExecutionEngine:
    """Places entries with their protective orders, atomically in intent."""

    session: Session
    venue: Venue
    spec: ContractSpec | None = None
    # Entry as a resting limit order. Market entries are available but
    # discouraged: see the module docstring on cron delay.
    use_limit_entry: bool = True

    def __post_init__(self) -> None:
        self.orders = OrderRepo(self.session)
        self.positions = PositionRepo(self.session)

    def execute(
        self,
        *,
        signal: Signal,
        setup: TradeSetup,
        size: PositionSize,
        quote: Quote | None = None,
    ) -> ExecutionReport:
        """Open a position and protect it, or explain why it did not happen."""
        if not size.viable:
            return ExecutionReport(False, f"not sized: {size.reason}")

        self._publish_quote(quote)

        existing = self.positions.open_for_symbol(setup.symbol)
        if existing is not None:
            return ExecutionReport(
                False,
                f"already holding {existing.side} {existing.symbol}; "
                "the risk gate should have caught this",
                position=existing,
            )

        bar_time = signal.bar_time
        quantity = size.quantity
        if self.spec is not None:
            quantity = self.spec.round_quantity(quantity)
            if quantity <= 0:
                return ExecutionReport(False, "quantity rounds to zero at the venue step")

        # --- 1. Entry -----------------------------------------------------
        entry_price = (
            self.spec.round_price(setup.entry) if self.spec is not None else setup.entry
        )
        entry_request = OrderRequest(
            symbol=setup.symbol,
            side=setup.side,
            order_type=OrderType.LIMIT if self.use_limit_entry else OrderType.MARKET,
            quantity=quantity,
            client_order_id=client_order_id(
                symbol=setup.symbol,
                side=setup.side,
                bar_time=bar_time,
                purpose="entry",
                signal_id=signal.id,
            ),
            price=entry_price if self.use_limit_entry else None,
            leverage=size.leverage,
        )

        entry_result = self._submit(entry_request, signal_id=signal.id)
        if not entry_result.ok:
            return ExecutionReport(
                False, f"entry rejected: {entry_result.error}", entry=entry_result
            )
        if entry_result.status is OrderStatus.FAILED:
            # Ambiguous: the venue may or may not have received it. Do NOT
            # retry here; the reconciler checks the venue before anything else
            # is attempted on this symbol.
            return ExecutionReport(
                False,
                "entry outcome is unknown (transport failure). The reconcile job "
                "must confirm against the venue before retrying.",
                entry=entry_result,
            )

        filled_price = entry_result.average_price or entry_price
        position = self.positions.create(
            symbol=setup.symbol,
            side=setup.side,
            quantity=entry_result.filled_quantity or quantity,
            entry_price=filled_price,
            stop_price=setup.stop,
            target_price=setup.primary_target,
            leverage=size.leverage,
            signal_id=signal.id,
            is_paper=self.venue.is_paper,
        )

        # --- 2. Protective stop (mandatory) -------------------------------
        stop_price = (
            self.spec.round_price(setup.stop) if self.spec is not None else setup.stop
        )
        stop_request = OrderRequest(
            symbol=setup.symbol,
            # The POSITION's side, with reduce_only, which the venue maps to
            # "close long" / "close short". Passing the opposite side here
            # would open a hedge instead of protecting the position.
            side=setup.side,
            order_type=OrderType.STOP_MARKET,
            quantity=position.quantity,
            client_order_id=client_order_id(
                symbol=setup.symbol,
                side=setup.side,
                bar_time=bar_time,
                purpose="stop",
                signal_id=signal.id,
            ),
            trigger_price=stop_price,
            reduce_only=True,
        )

        stop_result = self._submit(
            stop_request, signal_id=signal.id, position_id=position.id
        )
        if not stop_result.ok or stop_result.status is OrderStatus.FAILED:
            # The position is live and unprotected. Flatten it now.
            log.error(
                "Stop placement failed for %s; closing the position immediately",
                setup.symbol,
            )
            closed = self._emergency_close(position, bar_time, signal.id)
            return ExecutionReport(
                ok=False,
                detail=(
                    f"stop could not be placed ({stop_result.error}); position was "
                    + ("closed immediately" if closed else "LEFT OPEN AND UNPROTECTED")
                ),
                position=position,
                entry=entry_result,
                stop=stop_result,
                unprotected=not closed,
            )

        self.positions.attach_protective_orders(
            position, stop_order_id=stop_result.venue_order_id
        )

        # --- 3. Target (optional) -----------------------------------------
        # A failed target is not fatal: the position is protected, and the exit
        # can be managed on a later run. Only the stop is mandatory.
        target_result: OrderResult | None = None
        target_price = (
            self.spec.round_price(setup.primary_target)
            if self.spec is not None
            else setup.primary_target
        )
        target_request = OrderRequest(
            symbol=setup.symbol,
            side=setup.side,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=position.quantity,
            client_order_id=client_order_id(
                symbol=setup.symbol,
                side=setup.side,
                bar_time=bar_time,
                purpose="target",
                signal_id=signal.id,
            ),
            trigger_price=target_price,
            reduce_only=True,
        )
        target_result = self._submit(
            target_request, signal_id=signal.id, position_id=position.id
        )
        if target_result.ok:
            self.positions.attach_protective_orders(
                position, target_order_id=target_result.venue_order_id
            )
        else:
            log.warning(
                "Target order failed for %s (%s); position is protected but has "
                "no resting take-profit",
                setup.symbol,
                target_result.error,
            )

        return ExecutionReport(
            ok=True,
            detail=(
                f"{'paper' if self.venue.is_paper else 'LIVE'} {setup.side.value} "
                f"{position.quantity} {setup.symbol} @ {filled_price}, "
                f"stop {stop_price} resting at the venue"
                + ("" if target_result and target_result.ok else ", no target order")
            ),
            position=position,
            entry=entry_result,
            stop=stop_result,
            target=target_result,
        )

    def _publish_quote(self, quote: Quote | None) -> None:
        """Hand the current price to a venue that needs telling.

        `place_order` takes only a request, so there is nowhere to carry a
        quote. The live venue knows the market price by definition; the paper
        broker does not, and without this it priced fills from the position's
        own entry, making every simulated P&L approximately zero. Duck-typed so
        the protocol stays narrow.
        """
        if quote is None:
            return
        update = getattr(self.venue, "update_mark", None)
        if callable(update):
            update(quote.symbol, quote.mark)

    def _submit(
        self,
        request: OrderRequest,
        *,
        signal_id: int | None,
        position_id: int | None = None,
    ) -> OrderResult:
        """Persist the intent, then send it.

        The ordering is the whole point. `reserve()` returning `created=False`
        means this exact intent was already recorded, so the venue call is
        skipped entirely - that is what makes a duplicated cron run harmless.
        """
        order, created = self.orders.reserve(
            request,
            signal_id=signal_id,
            position_id=position_id,
            venue="paper" if self.venue.is_paper else "weex",
            is_paper=self.venue.is_paper,
        )
        if not created:
            log.info(
                "Order %s was already submitted (status %s); not resending",
                request.client_order_id,
                order.status,
            )
            return OrderResult(
                client_order_id=request.client_order_id,
                status=OrderStatus(order.status),
                venue_order_id=order.venue_order_id,
                filled_quantity=order.filled_quantity or Decimal("0"),
                average_price=order.average_price,
            )

        # Flush so the PENDING row survives even if the process dies during the
        # request below.
        self.session.flush()

        try:
            result = self.venue.place_order(request)
        except Exception as exc:  # pragma: no cover - venue-specific
            log.exception("Venue raised while placing %s", request.client_order_id)
            result = OrderResult(
                client_order_id=request.client_order_id,
                status=OrderStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )

        self.orders.apply_result(order, result)
        return result

    def _emergency_close(
        self, position: Position, bar_time: datetime, signal_id: int | None
    ) -> bool:
        """Flatten a position that could not be protected."""
        request = OrderRequest(
            symbol=position.symbol,
            side=Side(position.side),
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            client_order_id=client_order_id(
                symbol=position.symbol,
                side=Side(position.side),
                bar_time=bar_time,
                purpose="emergency_close",
                signal_id=signal_id,
            ),
            reduce_only=True,
        )
        result = self._submit(request, signal_id=signal_id, position_id=position.id)
        if result.ok and result.status is not OrderStatus.FAILED:
            self.positions.close(
                position,
                exit_price=result.average_price or position.entry_price,
                reason="emergency_close_no_stop",
            )
            return True
        return False

    def close_position(
        self, position: Position, *, reason: str, quote: Quote | None = None
    ) -> ExecutionReport:
        """Close a position and cancel its resting protective orders.

        Cancel order matters: the protective orders are cancelled *after* the
        close fills. Cancelling first would leave a window - however brief - in
        which the position is open and unprotected.
        """
        self._publish_quote(quote)
        request = OrderRequest(
            symbol=position.symbol,
            side=Side(position.side),
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            client_order_id=client_order_id(
                symbol=position.symbol,
                side=Side(position.side),
                bar_time=position.opened_at,
                purpose=f"close_{reason}",
                signal_id=position.signal_id,
            ),
            reduce_only=True,
        )
        result = self._submit(request, signal_id=position.signal_id, position_id=position.id)
        if not result.ok or result.status is OrderStatus.FAILED:
            return ExecutionReport(
                False, f"close failed: {result.error}", position=position
            )

        exit_price = result.average_price or (quote.mark if quote else position.entry_price)
        self.positions.close(position, exit_price=exit_price, reason=reason)

        for order_id in (position.stop_order_id, position.target_order_id):
            if order_id:
                try:
                    self.venue.cancel_order(position.symbol, order_id)
                except Exception:  # pragma: no cover
                    log.warning("Could not cancel resting order %s", order_id)

        return ExecutionReport(
            True,
            f"closed {position.symbol} at {exit_price} ({reason}), "
            f"R={position.r_multiple if position.r_multiple is not None else 'n/a'}",
            position=position,
        )
