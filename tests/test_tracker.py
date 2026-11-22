"""Paper position tracking and resolution.

This replaces the old execution tests. The bot places no orders, so there is no
venue to guarantee anything about; what matters now is whether the *scoring* of
its own predictions is honest and reproducible.

The same-bar tests are the important ones. If one 4h bar's range contains both
the stop and the target, OHLC cannot tell you which was touched first, and
assuming the favourable one is precisely how a paper record manufactures an
edge that does not survive a real market.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from tradingbot.db.schema import Order, Position, Signal
from tradingbot.execution import PaperTracker, apply_slippage, client_order_id
from tradingbot.models import (
    Candle,
    ContractSpec,
    PositionSize,
    Side,
    SignalStatus,
    TradeSetup,
)

BAR = dt.datetime(2026, 11, 22, 16, 0, tzinfo=dt.UTC)


def candle(
    offset_hours: int, *, high: str, low: str, close: str, base: dt.datetime | None = None
) -> Candle:
    """One bar at a fixed offset from the entry."""
    return Candle(
        open_time=(base or BAR) + dt.timedelta(hours=offset_hours),
        open=Decimal(close),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


@pytest.fixture
def setup() -> TradeSetup:
    return TradeSetup(
        "BTCUSDT", Side.LONG, Decimal("92000"), Decimal("89200"), (Decimal("97600"),)
    )


@pytest.fixture
def short_setup() -> TradeSetup:
    return TradeSetup(
        "BTCUSDT", Side.SHORT, Decimal("92000"), Decimal("94800"), (Decimal("86400"),)
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


def make_signal(session, side: Side = Side.LONG) -> Signal:
    signal = Signal(
        symbol="BTCUSDT", tier="core", timeframe="4h", bar_time=BAR,
        side=side.value, status=SignalStatus.APPROVED.value,
    )
    session.add(signal)
    session.flush()
    return signal


class TestNoExecutionCapability:
    """The absence of execution is a structural property worth asserting."""

    def test_there_is_no_execution_engine(self) -> None:
        import tradingbot.execution as execution

        assert not hasattr(execution, "ExecutionEngine")

    def test_there_is_no_authenticated_exchange_client(self) -> None:
        import tradingbot.exchanges as exchanges

        assert not hasattr(exchanges, "WeexTradingClient")

    def test_settings_have_no_exchange_credentials(self) -> None:
        from tradingbot.settings import Settings

        fields = set(Settings.model_fields)
        assert not {
            "weex_api_key", "weex_api_secret", "weex_api_passphrase",
            "live_trading_enabled",
        } & fields

    def test_live_mode_does_not_exist(self) -> None:
        from tradingbot.settings import TradingMode

        assert {mode.value for mode in TradingMode} == {"signal_only", "paper"}

    def test_can_place_real_orders_is_always_false(self) -> None:
        from tradingbot.settings import Settings, TradingMode

        for mode in TradingMode:
            assert not Settings(trading_mode=mode).can_place_real_orders


class TestOpening:
    def test_a_position_is_opened_with_stop_and_target_recorded(
        self, session, setup, size, spec
    ) -> None:
        signal = make_signal(session)
        result = PaperTracker(session=session).open(
            signal=signal, setup=setup, size=size, spec=spec
        )

        assert result.ok and result.created
        position = result.position
        assert position is not None
        assert position.is_paper is True
        assert position.stop_price == setup.stop
        assert position.target_price == setup.primary_target

    def test_the_fill_includes_adverse_slippage(self, session, setup, size, spec) -> None:
        """A paper fill at the exact requested price flatters every result."""
        signal = make_signal(session)
        result = PaperTracker(session=session).open(
            signal=signal, setup=setup, size=size, spec=spec
        )
        assert result.position is not None
        assert result.position.entry_price > setup.entry

    def test_satellites_get_wider_slippage(self, session, size) -> None:
        setup = TradeSetup(
            "GOATUSDT", Side.LONG, Decimal("0.52"), Decimal("0.44"), (Decimal("0.70"),)
        )
        signal = Signal(
            symbol="GOATUSDT", tier="satellite", timeframe="4h", bar_time=BAR,
            side=Side.LONG.value, status=SignalStatus.APPROVED.value,
        )
        session.add(signal)
        session.flush()

        narrow = PaperTracker(session=session).open(
            signal=signal,
            setup=setup,
            size=PositionSize("GOATUSDT", Decimal("10"), Decimal("5.2"),
                              Decimal("5.2"), Decimal("1"), Decimal("0.5")),
        )
        assert narrow.position is not None
        narrow_fill = narrow.position.entry_price

        # Same setup, but now flagged as a thin book.
        session.query(Position).delete()
        session.query(Order).delete()
        session.flush()
        wide = PaperTracker(
            session=session, satellite_symbols=frozenset({"GOATUSDT"})
        ).open(
            signal=signal,
            setup=setup,
            size=PositionSize("GOATUSDT", Decimal("10"), Decimal("5.2"),
                              Decimal("5.2"), Decimal("1"), Decimal("0.5")),
        )
        assert wide.position is not None
        assert wide.position.entry_price > narrow_fill

    def test_a_replayed_run_does_not_open_a_second_position(
        self, session, setup, size, spec
    ) -> None:
        """A duplicated schedule would otherwise corrupt the measured results."""
        signal = make_signal(session)
        tracker = PaperTracker(session=session)

        first = tracker.open(signal=signal, setup=setup, size=size, spec=spec)
        assert first.ok

        # Close it so the "already tracking" guard is not what blocks the
        # replay; the idempotency token must be what does.
        first.position.is_open = False  # type: ignore[union-attr]
        session.flush()

        again = tracker.open(signal=signal, setup=setup, size=size, spec=spec)
        assert not again.ok
        assert session.query(Position).count() == 1

    def test_an_unviable_size_opens_nothing(self, session, setup, spec) -> None:
        signal = make_signal(session)
        result = PaperTracker(session=session).open(
            signal=signal,
            setup=setup,
            size=PositionSize.unviable("BTCUSDT", "below the venue minimum"),
            spec=spec,
        )
        assert not result.ok
        assert session.query(Position).count() == 0


class TestResolution:
    """Bars are anchored on `position.opened_at`, not on the signal's bar time.

    Resolution deliberately ignores bars that printed before the position was
    opened, because a level touched before the signal existed is not a result.
    `opened_at` is the wall-clock moment of entry, so the tests build their
    bars relative to it.
    """

    def _open(self, session, setup, size, spec) -> Position:
        signal = make_signal(session, Side(setup.side))
        result = PaperTracker(session=session).open(
            signal=signal, setup=setup, size=size, spec=spec
        )
        assert result.position is not None
        return result.position

    @staticmethod
    def _after(position: Position, hours: int, *, high: str, low: str, close: str) -> Candle:
        return candle(hours, high=high, low=low, close=close, base=position.opened_at)

    def test_target_hit_resolves_positive(self, session, setup, size, spec) -> None:
        position = self._open(session, setup, size, spec)
        bars = [
            self._after(position, 4, high="93000", low="91500", close="92500"),
            self._after(position, 8, high="98000", low="96000", close="97800"),
        ]
        resolution = PaperTracker(session=session).resolve(position, bars)

        assert resolution is not None
        assert resolution.reason == "target"
        assert not resolution.ambiguous
        assert position.is_open is False
        assert position.r_multiple is not None and position.r_multiple > 0

    def test_stop_hit_resolves_negative(self, session, setup, size, spec) -> None:
        position = self._open(session, setup, size, spec)
        bars = [self._after(position, 4, high="92500", low="89000", close="89100")]
        resolution = PaperTracker(session=session).resolve(position, bars)

        assert resolution is not None
        assert resolution.reason == "stop"
        assert position.r_multiple is not None and position.r_multiple < 0

    def test_same_bar_touch_assumes_the_stop(self, session, setup, size, spec) -> None:
        """The honesty rule. OHLC cannot order two touches inside one bar."""
        position = self._open(session, setup, size, spec)
        # One bar spanning BOTH the stop (89200) and the target (97600).
        bars = [self._after(position, 4, high="98000", low="89000", close="95000")]
        resolution = PaperTracker(session=session).resolve(position, bars)

        assert resolution is not None
        assert resolution.reason == "stop", "optimistic resolution would flatter results"
        assert resolution.ambiguous is True
        assert position.close_reason == "stop_same_bar"
        assert position.r_multiple is not None and position.r_multiple < 0

    def test_an_untouched_position_stays_open(self, session, setup, size, spec) -> None:
        position = self._open(session, setup, size, spec)
        bars = [self._after(position, 4, high="93000", low="91000", close="92000")]
        assert PaperTracker(session=session).resolve(position, bars) is None
        assert position.is_open is True

    def test_bars_before_entry_cannot_resolve_it(self, session, setup, size, spec) -> None:
        """A level touched before the signal existed is not a result."""
        position = self._open(session, setup, size, spec)
        bars = [self._after(position, -8, high="98000", low="89000", close="92000")]
        assert PaperTracker(session=session).resolve(position, bars) is None
        assert position.is_open is True

    def test_short_target_resolves_on_a_fall(self, session, short_setup, size, spec) -> None:
        position = self._open(session, short_setup, size, spec)
        bars = [self._after(position, 4, high="92000", low="86000", close="86200")]
        resolution = PaperTracker(session=session).resolve(position, bars)

        assert resolution is not None
        assert resolution.reason == "target"
        assert position.r_multiple is not None and position.r_multiple > 0

    def test_short_stop_resolves_on_a_rise(self, session, short_setup, size, spec) -> None:
        position = self._open(session, short_setup, size, spec)
        bars = [self._after(position, 4, high="95000", low="91800", close="94900")]
        resolution = PaperTracker(session=session).resolve(position, bars)

        assert resolution is not None
        assert resolution.reason == "stop"
        assert position.r_multiple is not None and position.r_multiple < 0

    def test_resolution_is_net_of_fees(self, session, setup, size, spec) -> None:
        """A gross-only R-multiple overstates every small-account result."""
        position = self._open(session, setup, size, spec)
        bars = [self._after(position, 4, high="98000", low="96000", close="97800")]
        PaperTracker(session=session, taker_fee_pct=Decimal("0.06")).resolve(
            position, bars
        )
        assert position.fees_paid is not None and position.fees_paid > 0
        gross = (position.exit_price - position.entry_price) * position.quantity
        assert position.realised_pnl is not None and position.realised_pnl < gross

    def test_the_first_touching_bar_wins(self, session, setup, size, spec) -> None:
        position = self._open(session, setup, size, spec)
        bars = [
            self._after(position, 4, high="92500", low="89000", close="89500"),
            self._after(position, 8, high="98000", low="97000", close="97800"),
        ]
        resolution = PaperTracker(session=session).resolve(position, bars)
        assert resolution is not None and resolution.reason == "stop"


class TestExpiry:
    def test_a_stale_position_is_closed(self, session, setup, size, spec) -> None:
        """Otherwise it holds a portfolio slot forever."""
        signal = make_signal(session)
        result = PaperTracker(session=session).open(
            signal=signal, setup=setup, size=size, spec=spec
        )
        position = result.position
        assert position is not None
        position.opened_at = dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=30)
        session.flush()

        resolution = PaperTracker(session=session).expire_stale(
            position, max_age_hours=24 * 14, last_price=Decimal("93000")
        )
        assert resolution is not None
        assert resolution.reason == "expired"
        assert position.is_open is False

    def test_a_young_position_is_not_expired(self, session, setup, size, spec) -> None:
        signal = make_signal(session)
        position = PaperTracker(session=session).open(
            signal=signal, setup=setup, size=size, spec=spec
        ).position
        assert position is not None
        assert (
            PaperTracker(session=session).expire_stale(
                position, max_age_hours=24 * 14, last_price=Decimal("93000")
            )
            is None
        )
        assert position.is_open is True


class TestSlippageHelper:
    def test_slippage_is_always_adverse(self) -> None:
        price = Decimal("100")
        assert apply_slippage(price, side=Side.LONG, slippage_pct=Decimal("1")) > price
        assert apply_slippage(price, side=Side.SHORT, slippage_pct=Decimal("1")) < price

    def test_exit_slippage_reverses_direction(self) -> None:
        price = Decimal("100")
        assert apply_slippage(
            price, side=Side.LONG, slippage_pct=Decimal("1"), is_exit=True
        ) < price


class TestIdempotencyToken:
    def test_it_is_deterministic(self) -> None:
        args = {"symbol": "BTCUSDT", "side": Side.LONG, "bar_time": BAR,
                "purpose": "paper_entry", "signal_id": 1}
        assert client_order_id(**args) == client_order_id(**args)

    def test_different_bars_differ(self) -> None:
        assert client_order_id(
            symbol="BTCUSDT", side=Side.LONG, bar_time=BAR, purpose="paper_entry"
        ) != client_order_id(
            symbol="BTCUSDT", side=Side.LONG,
            bar_time=BAR + dt.timedelta(hours=4), purpose="paper_entry",
        )

    def test_it_fits_venue_length_limits(self) -> None:
        assert len(
            client_order_id(
                symbol="BTCUSDT", side=Side.LONG, bar_time=BAR, purpose="paper_entry"
            )
        ) <= 32
