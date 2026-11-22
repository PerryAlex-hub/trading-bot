"""Paper broker.

Implements the same `ExecutionVenue` interface as the live client so paper and
live share one execution code path. That shared path is the point: a paper
broker that works differently from the live one produces results that do not
predict live behaviour, which is worse than no paper mode at all.

**Modelled honestly.** Fills include fee and slippage rather than filling at
the requested price. Satellite symbols get wider assumed slippage because thin
books are where the real cost hides. A paper broker that fills perfectly is an
optimism generator, and the README already warns that this is exactly why paper
results look brilliant and live trading does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

from ..models import (
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Quote,
    Side,
)
from .base_helpers import apply_slippage

log = logging.getLogger(__name__)


@dataclass
class PaperBroker:
    """Simulated venue. Keeps no state beyond the positions it has opened."""

    equity: Decimal = Decimal("200")
    taker_fee_pct: Decimal = Decimal("0.06")
    # Assumed adverse slippage on a market fill, in percent.
    slippage_pct: Decimal = Decimal("0.02")
    satellite_slippage_pct: Decimal = Decimal("0.15")
    satellite_symbols: frozenset[str] = frozenset()
    _positions: dict[str, "PaperPosition"] = field(default_factory=dict)
    _filled: dict[str, OrderResult] = field(default_factory=dict)
    _marks: dict[str, Decimal] = field(default_factory=dict)

    @property
    def is_paper(self) -> bool:
        return True

    def update_mark(self, symbol: str, price: Decimal) -> None:
        """Tell the broker the current market price.

        The execution engine calls this before submitting, because the `Venue`
        protocol's `place_order` takes only a request - it has nowhere to carry
        a quote. Without it the broker priced every fill from the position's
        own entry price, so a closed paper trade showed a P&L of roughly zero
        no matter what the market had done, which made paper mode useless for
        measuring anything. Caught by test [5] in the execution suite.
        """
        if price > 0:
            self._marks[symbol] = price

    def place_order(
        self, request: OrderRequest, *, quote: Quote | None = None
    ) -> OrderResult:
        """Simulate a fill.

        Idempotent on `client_order_id`, matching the live venue's behaviour:
        a replayed cron run must not open a second simulated position either,
        or paper and live would diverge in exactly the case that matters.
        """
        if request.client_order_id in self._filled:
            log.info("Paper order %s already filled; returning the original result",
                     request.client_order_id)
            return self._filled[request.client_order_id]

        reference = self._reference_price(request, quote)
        if reference is None:
            result = OrderResult(
                client_order_id=request.client_order_id,
                status=OrderStatus.REJECTED,
                error="no reference price available to simulate a fill",
            )
            self._filled[request.client_order_id] = result
            return result

        # Protective orders rest at the venue rather than filling now. Marking
        # them SUBMITTED mirrors reality: the stop exists but has not triggered.
        if request.order_type in (OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET):
            result = OrderResult(
                client_order_id=request.client_order_id,
                status=OrderStatus.SUBMITTED,
                venue_order_id=f"paper-{request.client_order_id}",
                raw={"simulated": True, "resting": True},
            )
            self._filled[request.client_order_id] = result
            return result

        slippage = (
            self.satellite_slippage_pct
            if request.symbol in self.satellite_symbols
            else self.slippage_pct
        )
        fill_price = apply_slippage(
            reference,
            side=request.side,
            slippage_pct=slippage,
            is_exit=request.reduce_only,
        )

        if request.reduce_only:
            self._positions.pop(request.symbol, None)
        else:
            self._positions[request.symbol] = PaperPosition(
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                entry_price=fill_price,
                leverage=request.leverage or Decimal("1"),
            )

        fee = fill_price * request.quantity * self.taker_fee_pct / Decimal("100")
        self.equity -= fee

        result = OrderResult(
            client_order_id=request.client_order_id,
            status=OrderStatus.FILLED,
            venue_order_id=f"paper-{request.client_order_id}",
            filled_quantity=request.quantity,
            average_price=fill_price,
            raw={
                "simulated": True,
                "requested_price": str(reference),
                "slippage_pct": str(slippage),
                "fee": str(fee),
            },
        )
        self._filled[request.client_order_id] = result
        return result

    def _reference_price(
        self, request: OrderRequest, quote: Quote | None
    ) -> Decimal | None:
        """Find a price to simulate a fill at.

        The position fallback is load-bearing, not a nicety. A market
        `reduce_only` close carries neither `price` nor `trigger_price`, so
        without it an emergency close would be rejected here for want of a
        reference price - while the same close would succeed live, because the
        venue prices it. That divergence would make paper mode report an
        unprotected position in the one scenario the execution engine exists to
        handle. Caught by test [4] in the execution suite.
        """
        if quote is not None:
            return quote.mark
        mark = self._marks.get(request.symbol)
        if mark is not None:
            return mark
        explicit = request.price or request.trigger_price
        if explicit is not None:
            return explicit
        held = self._positions.get(request.symbol)
        if held is not None:
            return held.entry_price
        return None

    def cancel_order(self, symbol: str, client_order_id: str) -> bool:
        return self._filled.pop(client_order_id, None) is not None

    def open_positions(self) -> list["PaperPosition"]:
        return list(self._positions.values())

    def account_equity(self) -> Decimal | None:
        return self.equity


@dataclass
class PaperPosition:
    """A simulated position, shaped like `VenuePosition`."""

    symbol: str
    side: Side
    quantity: Decimal
    entry_price: Decimal
    leverage: Decimal = Decimal("1")
    unrealised_pnl: Decimal | None = None
    raw: dict = field(default_factory=dict)
