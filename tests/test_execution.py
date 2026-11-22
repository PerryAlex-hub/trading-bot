"""Execution safety.

These tests encode the guarantees that make cron-driven trading on free
infrastructure defensible. Each one corresponds to a way real money is lost:

* a replayed scheduler run opening a second position;
* a position left open with no stop resting at the venue;
* paper and live behaving differently, so paper results mislead.

Two of these tests found real bugs during development. Both were paper/live
divergences, and both are noted where they are asserted.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from tradingbot.db.schema import Order, Position, Signal
from tradingbot.execution import ExecutionEngine, PaperBroker, client_order_id
from tradingbot.execution.base_helpers import apply_slippage
from tradingbot.models import (
    ContractSpec,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    PositionSize,
    Quote,
    Side,
    SignalStatus,
    TradeSetup,
)

BAR = dt.datetime(2026, 11, 22, 16, 0, tzinfo=dt.UTC)


@pytest.fixture
def setup() -> TradeSetup:
    return TradeSetup(
        "BTCUSDT", Side.LONG, Decimal("92000"), Decimal("89200"), (Decimal("97600"),)
    )


@pytest.fixture
def size() -> PositionSize:
    return PositionSize(
        "BTCUSDT", Decimal("0.003"), Decimal("276"), Decimal("27.6"),
        Decimal("10"), Decimal("1.0"),
    )


@pytest.fixture
def spec() -> ContractSpec:
    return ContractSpec(
        "BTCUSDT", min_order_size=Decimal("0.0001"), size_step=Decimal("0.0001"),
        price_step=Decimal("0.1"), min_notional=Decimal("5"), max_leverage=125,
    )


def make_signal(session, symbol: str = "BTCUSDT") -> Signal:
    signal = Signal(
        symbol=symbol, tier="core", timeframe="4h", bar_time=BAR,
        side=Side.LONG.value, status=SignalStatus.APPROVED.value,
    )
    session.add(signal)
    session.flush()
    return signal


def btc_quote(mark: str = "92000") -> Quote:
    return Quote(
        "BTCUSDT", Decimal(mark) - 5, Decimal(mark) + 5, Decimal(mark),
        dt.datetime.now(dt.UTC),
    )


class TestIdempotency:
    def test_client_order_id_is_deterministic(self) -> None:
        """The keystone of safe cron execution."""
        first = client_order_id(
            symbol="BTCUSDT", side=Side.LONG, bar_time=BAR, purpose="entry", signal_id=1
        )
        second = client_order_id(
            symbol="BTCUSDT", side=Side.LONG, bar_time=BAR, purpose="entry", signal_id=1
        )
        assert first == second

    def test_different_purposes_get_different_ids(self) -> None:
        """Entry and stop are placed on the same bar; they must not collide."""
        entry = client_order_id(
            symbol="BTCUSDT", side=Side.LONG, bar_time=BAR, purpose="entry", signal_id=1
        )
        stop = client_order_id(
            symbol="BTCUSDT", side=Side.LONG, bar_time=BAR, purpose="stop", signal_id=1
        )
        assert entry != stop

    def test_different_bars_get_different_ids(self) -> None:
        later = BAR + dt.timedelta(hours=4)
        assert client_order_id(
            symbol="BTCUSDT", side=Side.LONG, bar_time=BAR, purpose="entry"
        ) != client_order_id(
            symbol="BTCUSDT", side=Side.LONG, bar_time=later, purpose="entry"
        )

    def test_id_fits_venue_length_limits(self) -> None:
        generated = client_order_id(
            symbol="BTCUSDT", side=Side.LONG, bar_time=BAR, purpose="entry", signal_id=1
        )
        assert len(generated) <= 32

    def test_a_replayed_run_does_not_open_a_second_position(
        self, session, setup, size, spec
    ) -> None:
        """GitHub Actions can fire the same schedule twice."""
        signal = make_signal(session)
        broker = PaperBroker(equity=Decimal("200"))
        engine = ExecutionEngine(session=session, venue=broker, spec=spec)

        first = engine.execute(signal=signal, setup=setup, size=size, quote=btc_quote())
        assert first.ok

        again = ExecutionEngine(session=session, venue=broker, spec=spec).execute(
            signal=signal, setup=setup, size=size, quote=btc_quote()
        )
        assert not again.ok
        assert session.query(Position).count() == 1
        assert session.query(Order).count() == 3  # entry, stop, target

    def test_paper_broker_is_itself_idempotent(self) -> None:
        """Paper must match live here, or the two diverge on retry."""
        broker = PaperBroker(equity=Decimal("200"))
        request = OrderRequest(
            symbol="BTCUSDT", side=Side.LONG, order_type=OrderType.MARKET,
            quantity=Decimal("0.01"), client_order_id="fixed-id",
        )
        first = broker.place_order(request, quote=btc_quote())
        second = broker.place_order(request, quote=btc_quote())
        assert first.average_price == second.average_price


class TestStopIsMandatory:
    def test_entry_places_a_resting_stop(self, session, setup, size, spec) -> None:
        signal = make_signal(session)
        engine = ExecutionEngine(
            session=session, venue=PaperBroker(equity=Decimal("200")), spec=spec
        )
        report = engine.execute(signal=signal, setup=setup, size=size, quote=btc_quote())

        assert report.ok
        assert report.position is not None
        assert report.position.stop_order_id, "no stop resting at the venue"
        stops = [
            order for order in session.query(Order).all()
            if order.order_type == OrderType.STOP_MARKET.value
        ]
        assert len(stops) == 1
        assert stops[0].reduce_only is True

    def test_position_is_flattened_when_the_stop_cannot_be_placed(
        self, session, setup, size, spec
    ) -> None:
        """The core guarantee. Unprotected capital is worse than a small loss."""

        class RefusesStops(PaperBroker):
            def place_order(self, request, *, quote=None):
                if request.order_type is OrderType.STOP_MARKET:
                    return OrderResult(
                        request.client_order_id, OrderStatus.REJECTED,
                        error="simulated venue refusal",
                    )
                return super().place_order(request, quote=quote)

        signal = make_signal(session)
        engine = ExecutionEngine(
            session=session, venue=RefusesStops(equity=Decimal("200")), spec=spec
        )
        report = engine.execute(signal=signal, setup=setup, size=size, quote=btc_quote())

        assert not report.ok
        assert not report.unprotected, "position was left open without a stop"
        position = session.query(Position).filter(Position.symbol == "BTCUSDT").one()
        assert position.is_open is False
        assert position.close_reason == "emergency_close_no_stop"

    def test_protective_orders_must_be_reduce_only(self) -> None:
        """Otherwise a trigger opens a new position instead of closing one."""
        with pytest.raises(ValueError, match="reduce_only"):
            OrderRequest(
                symbol="BTCUSDT", side=Side.LONG, order_type=OrderType.STOP_MARKET,
                quantity=Decimal("1"), client_order_id="x",
                trigger_price=Decimal("89000"), reduce_only=False,
            )

    def test_a_stop_order_requires_a_trigger_price(self) -> None:
        with pytest.raises(ValueError, match="trigger_price"):
            OrderRequest(
                symbol="BTCUSDT", side=Side.LONG, order_type=OrderType.STOP_MARKET,
                quantity=Decimal("1"), client_order_id="x", reduce_only=True,
            )


class TestPaperLiveParity:
    """Both bugs found during development were failures of parity."""

    def test_market_reduce_only_close_is_priced(self, session, setup, size, spec) -> None:
        """REGRESSION: a market reduce_only order carries no price.

        The paper broker rejected it for want of a reference price, so an
        emergency close that works live failed in paper - in exactly the
        scenario the execution engine exists to handle.
        """
        broker = PaperBroker(equity=Decimal("200"))
        signal = make_signal(session)
        ExecutionEngine(session=session, venue=broker, spec=spec).execute(
            signal=signal, setup=setup, size=size, quote=btc_quote()
        )
        close = OrderRequest(
            symbol="BTCUSDT", side=Side.LONG, order_type=OrderType.MARKET,
            quantity=Decimal("0.003"), client_order_id="close-1", reduce_only=True,
        )
        result = broker.place_order(close)
        assert result.status is OrderStatus.FILLED
        assert result.average_price is not None

    def test_paper_pnl_reflects_the_market_not_the_entry(
        self, session, setup, size, spec
    ) -> None:
        """REGRESSION: the quote never reached the venue.

        Fills were priced from the position's own entry, so every closed paper
        trade showed a P&L of roughly zero regardless of what the market did,
        which made paper mode useless for measuring anything.
        """
        broker = PaperBroker(equity=Decimal("200"))
        signal = make_signal(session)
        engine = ExecutionEngine(session=session, venue=broker, spec=spec)
        report = engine.execute(signal=signal, setup=setup, size=size, quote=btc_quote())
        position = report.position
        assert position is not None

        # Price moves up 2%; closing a long must realise a profit.
        closed = engine.close_position(
            position, reason="target", quote=btc_quote("94000")
        )
        assert closed.ok
        assert position.realised_pnl is not None
        assert position.realised_pnl > 0, "paper P&L ignored the market price"
        assert position.r_multiple is not None and position.r_multiple > 0


class TestSlippage:
    def test_slippage_is_always_adverse(self) -> None:
        """Favourable slippage manufactures an edge that does not exist."""
        price = Decimal("100")
        assert apply_slippage(price, side=Side.LONG, slippage_pct=Decimal("1")) > price
        assert apply_slippage(price, side=Side.SHORT, slippage_pct=Decimal("1")) < price

    def test_exit_slippage_reverses_direction(self) -> None:
        """Closing a long means selling into the bid."""
        price = Decimal("100")
        assert apply_slippage(
            price, side=Side.LONG, slippage_pct=Decimal("1"), is_exit=True
        ) < price

    def test_satellites_assume_wider_slippage(self) -> None:
        """Thin books are where the real cost hides."""
        broker = PaperBroker(
            equity=Decimal("200"),
            slippage_pct=Decimal("0.02"),
            satellite_slippage_pct=Decimal("0.5"),
            satellite_symbols=frozenset({"GOATUSDT"}),
        )
        core = broker.place_order(
            OrderRequest(
                symbol="BTCUSDT", side=Side.LONG, order_type=OrderType.MARKET,
                quantity=Decimal("1"), client_order_id="a", price=Decimal("100"),
            )
        )
        thin = broker.place_order(
            OrderRequest(
                symbol="GOATUSDT", side=Side.LONG, order_type=OrderType.MARKET,
                quantity=Decimal("1"), client_order_id="b", price=Decimal("100"),
            )
        )
        assert thin.average_price > core.average_price


class TestPositionAccounting:
    def test_r_multiple_is_computed_on_close(self, session, setup, size, spec) -> None:
        """R is the only honest way to compare a BTC trade with a GOAT trade."""
        broker = PaperBroker(equity=Decimal("200"))
        signal = make_signal(session)
        engine = ExecutionEngine(session=session, venue=broker, spec=spec)
        report = engine.execute(signal=signal, setup=setup, size=size, quote=btc_quote())
        position = report.position
        assert position is not None

        engine.close_position(position, reason="target", quote=btc_quote("97600"))
        assert position.r_multiple is not None
        # Entry 92000, stop 89200, exit ~97600 is roughly +2R before costs.
        assert position.r_multiple > Decimal("1.5")

    def test_a_losing_trade_has_a_negative_r(self, session, setup, size, spec) -> None:
        broker = PaperBroker(equity=Decimal("200"))
        signal = make_signal(session)
        engine = ExecutionEngine(session=session, venue=broker, spec=spec)
        report = engine.execute(signal=signal, setup=setup, size=size, quote=btc_quote())
        position = report.position
        assert position is not None

        engine.close_position(position, reason="stop", quote=btc_quote("89200"))
        assert position.r_multiple is not None and position.r_multiple < 0
