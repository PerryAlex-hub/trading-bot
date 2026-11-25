"""The fill model and the metrics.

These are the tests that decide whether the backtester can be trusted. A
replay loop that calls the real pipeline is easy to verify by inspection; a
fill model is not, and it is where a backtest invents an edge - by filling
orders that never traded, or by resolving an ambiguous bar in its own favour.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pandas as pd

from tradingbot.backtest.engine import (
    BacktestResult,
    TradeOutcome,
    _max_drawdown_r,
    _simulate,
    _slice,
    summarise,
)
from tradingbot.config import load_risk
from tradingbot.models import Side, Timeframe, TradeSetup

BAR = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def frame(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLC frame from (open, high, low, close) tuples."""
    index = pd.date_range(BAR, periods=len(bars), freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
            "volume": [1000.0] * len(bars),
        },
        index=index,
    )


def long_setup(entry="100", stop="98", target="104") -> TradeSetup:
    return TradeSetup(
        symbol="TESTUSDT",
        side=Side.LONG,
        entry=Decimal(entry),
        stop=Decimal(stop),
        targets=(Decimal(target),),
        timeframe=Timeframe.H4,
    )


class _Size:
    """Minimal stand-in for PositionSize: only two fields are read."""

    def __init__(self, quantity="1", notional="100") -> None:
        self.quantity = Decimal(quantity)
        self.notional = Decimal(notional)


def simulate(bars, setup=None, *, max_wait=3, max_hold=30):
    return _simulate(
        setup=setup or long_setup(),
        size=_Size(),
        frame=frame(bars),
        decision_position=0,
        bar_time=BAR,
        max_wait_bars=max_wait,
        max_hold_bars=max_hold,
        risk_config=load_risk(),
        reasons=(),
    )


class TestFillModel:
    def test_an_untouched_limit_never_fills(self) -> None:
        """Price must actually trade through the entry.

        The laziest backtest bug is assuming every signal fills at its stated
        price. A limit entry 0.15 ATR below price frequently never trades.
        """
        trade = simulate([(100, 101, 100.5, 101)] * 5)
        assert trade.outcome is TradeOutcome.NEVER_FILLED
        assert not trade.filled
        assert trade.r_multiple is None

    def test_an_unfilled_order_expires_rather_than_resting(self) -> None:
        """The live bot re-analyses each bar; it would not leave a stale order."""
        bars = [(100, 101, 100.5, 101)] * 3 + [(100, 101, 99.0, 100)] * 3
        trade = simulate(bars, max_wait=2)
        assert trade.outcome is TradeOutcome.NEVER_FILLED

    def test_target_reached_is_a_win(self) -> None:
        bars = [
            (100, 100.5, 99.5, 100),   # touches entry 100
            (100, 104.5, 100, 104),    # touches target 104
        ]
        trade = simulate(bars)
        assert trade.outcome is TradeOutcome.TARGET
        assert trade.outcome.is_win
        assert trade.exit_price == Decimal("104")

    def test_stop_reached_is_a_loss(self) -> None:
        bars = [
            (100, 100.5, 99.5, 100),
            (100, 100.2, 97.5, 98),    # through stop 98
        ]
        trade = simulate(bars)
        assert trade.outcome is TradeOutcome.STOP
        assert trade.pnl < 0

    def test_a_bar_containing_both_is_booked_as_a_loss(self) -> None:
        """The decision that keeps this honest.

        OHLC cannot say whether the high or the low came first. Resolving the
        ambiguity as a win turns every wide-range bar into free money and is
        the most common way a backtest manufactures an edge.
        """
        bars = [
            (100, 100.5, 99.5, 100),
            (100, 105.0, 97.0, 100),   # spans target AND stop
        ]
        trade = simulate(bars)
        assert trade.outcome is TradeOutcome.AMBIGUOUS_STOP
        assert trade.exit_price == Decimal("98")
        assert trade.pnl < 0

    def test_a_trade_going_nowhere_times_out_at_the_close(self) -> None:
        """Otherwise an open loser is deferred forever and flatters the result."""
        bars = [(100, 100.5, 99.5, 100)] + [(100, 100.4, 99.6, 100.1)] * 6
        trade = simulate(bars, max_hold=3)
        assert trade.outcome is TradeOutcome.TIMEOUT
        assert trade.exit_price == Decimal("100.1")

    def test_running_out_of_history_leaves_a_trade_unresolved(self) -> None:
        """An unfinished trade must not be counted as a result."""
        bars = [(100, 100.5, 99.5, 100), (100, 100.4, 99.6, 100)]
        trade = simulate(bars, max_hold=30)
        assert trade.exit_price is None
        assert trade.r_multiple is None

    def test_a_short_fills_on_the_high_and_stops_above(self) -> None:
        setup = TradeSetup(
            symbol="TESTUSDT",
            side=Side.SHORT,
            entry=Decimal("100"),
            stop=Decimal("102"),
            targets=(Decimal("96"),),
            timeframe=Timeframe.H4,
        )
        filled_then_stopped = simulate(
            [(99, 100.5, 98.5, 99), (99, 102.5, 99, 102)], setup=setup
        )
        assert filled_then_stopped.outcome is TradeOutcome.STOP
        assert filled_then_stopped.pnl < 0


class TestFrictionAndR:
    def test_friction_is_deducted_from_a_winner(self) -> None:
        """A win must not be reported gross."""
        bars = [(100, 100.5, 99.5, 100), (100, 104.5, 100, 104)]
        trade = simulate(bars)
        gross = (trade.exit_price - trade.entry) * trade.quantity
        assert trade.friction > 0
        assert trade.pnl == gross - trade.friction
        assert trade.pnl < gross

    def test_r_is_pnl_over_risk(self) -> None:
        bars = [(100, 100.5, 99.5, 100), (100, 104.5, 100, 104)]
        trade = simulate(bars)
        # entry 100, stop 98 -> 2.0 of risk per unit on 1 unit
        assert trade.risk_amount == Decimal("2")
        assert trade.r_multiple == trade.pnl / Decimal("2")
        # A 1:2 setup nets slightly under 2R once friction is charged.
        assert Decimal("1.5") < trade.r_multiple < Decimal("2")


class TestNoLookahead:
    def test_slicing_excludes_every_later_bar(self) -> None:
        """The guarantee the whole replay rests on."""
        data = frame([(100, 101, 99, 100)] * 10)
        cut = data.index[4]
        sliced = _slice(data, cut)
        assert sliced is not None
        assert len(sliced) == 5
        assert sliced.index[-1] == cut


class TestMetrics:
    def test_max_drawdown_tracks_peak_to_trough(self) -> None:
        # Curve: +2, +1, -2, -1 -> peak 3, trough 0, so 3R of drawdown.
        values = [Decimal("2"), Decimal("1"), Decimal("-2"), Decimal("-1")]
        assert _max_drawdown_r(values) == Decimal("3")

    def test_a_monotonic_winner_has_no_drawdown(self) -> None:
        assert _max_drawdown_r([Decimal("1"), Decimal("2")]) == Decimal("0")

    def test_empty_results_report_zero_trades_not_a_crash(self) -> None:
        stats = summarise(BacktestResult(bars_examined=500, signals_rejected=12))
        assert stats["trades"] == 0
        assert stats["bars_examined"] == 500

    def test_expectancy_is_reported_alongside_win_rate(self) -> None:
        """Win rate alone cannot say whether a strategy pays."""
        result = BacktestResult()
        result.trades = [
            simulate([(100, 100.5, 99.5, 100), (100, 104.5, 100, 104)]),
            simulate([(100, 100.5, 99.5, 100), (100, 100.2, 97.5, 98)]),
        ]
        stats = summarise(result)
        assert stats["trades"] == 2
        assert stats["win_rate"] == 50.0
        assert "expectancy_r" in stats
        # One +~1.9R win against one -~1.1R loss is net positive at a 50% rate.
        assert isinstance(stats["expectancy_r"], float)


class TestBootstrapInference:
    """Whether an expectancy figure is distinguishable from luck.

    A single expectancy number is the most dangerous output this project
    produces: "+0.14R per trade" reads identically over 15 trades and 1,500,
    and the first is almost certainly noise.
    """

    def _r(self, wins: int, losses: int, win_r="2.4", loss_r="-1.1") -> list[Decimal]:
        return [Decimal(win_r)] * wins + [Decimal(loss_r)] * losses

    def test_too_few_trades_returns_nothing_rather_than_a_number(self) -> None:
        """Refusing to quote an interval is more honest than quoting a useless one."""
        from tradingbot.backtest.engine import (
            MIN_TRADES_FOR_INFERENCE,
            bootstrap_expectancy,
        )

        assert bootstrap_expectancy(self._r(3, 5)) is None
        assert len(self._r(3, 5)) < MIN_TRADES_FOR_INFERENCE

    def test_a_marginal_edge_straddles_zero(self) -> None:
        """The case that matters: a positive mean that means nothing.

        Ten wins against twenty losses at a 1:2.2 payoff averages a little
        above zero, and an interval spanning zero is the correct reading.
        """
        from tradingbot.backtest.engine import bootstrap_expectancy

        stats = bootstrap_expectancy(self._r(10, 20))
        assert stats is not None
        assert stats["expectancy_r"] > 0          # looks like an edge
        assert stats["straddles_zero"] is True    # is not one
        assert 0.3 < stats["prob_positive"] < 0.8

    def test_a_strong_edge_does_not_straddle_zero(self) -> None:
        from tradingbot.backtest.engine import bootstrap_expectancy

        stats = bootstrap_expectancy(self._r(40, 20))
        assert stats is not None
        assert stats["straddles_zero"] is False
        assert stats["ci_low_r"] > 0
        assert stats["prob_positive"] > 0.95

    def test_a_clear_loser_is_reliably_negative(self) -> None:
        from tradingbot.backtest.engine import bootstrap_expectancy

        stats = bootstrap_expectancy(self._r(5, 45))
        assert stats is not None
        assert stats["ci_high_r"] < 0
        assert stats["prob_positive"] < 0.05

    def test_the_interval_is_reproducible(self) -> None:
        """A backtest reporting different confidence each run is worthless."""
        from tradingbot.backtest.engine import bootstrap_expectancy

        sample = self._r(18, 22)
        assert bootstrap_expectancy(sample) == bootstrap_expectancy(sample)

    def test_more_trades_narrow_the_interval(self) -> None:
        """The only honest way to turn a hint into evidence."""
        from tradingbot.backtest.engine import bootstrap_expectancy

        small = bootstrap_expectancy(self._r(12, 18))
        large = bootstrap_expectancy(self._r(120, 180))
        assert small is not None and large is not None
        small_width = small["ci_high_r"] - small["ci_low_r"]
        large_width = large["ci_high_r"] - large["ci_low_r"]
        assert large_width < small_width


class TestWalkForward:
    """Whether a result held across the period or came from one stretch."""

    def _trade(self, day: int, r: str):
        """A resolved trade whose r_multiple is exactly `r`."""
        from tradingbot.backtest.engine import SimulatedTrade

        risk = Decimal("1")
        pnl = Decimal(r) * risk
        return SimulatedTrade(
            symbol="TESTUSDT",
            side=Side.LONG,
            decision_bar=BAR + dt.timedelta(days=day),
            entry=Decimal("100"),
            stop=Decimal("99"),
            target=Decimal("103"),
            quantity=Decimal("1"),
            risk_amount=risk,
            outcome=TradeOutcome.TARGET,
            fill_bar=BAR + dt.timedelta(days=day),
            exit_bar=BAR + dt.timedelta(days=day),
            exit_price=Decimal("100") + pnl,
            friction=Decimal("0"),
        )

    def test_splits_by_time_not_by_trade_count(self) -> None:
        from tradingbot.backtest.engine import walk_forward

        # Eight trades over 80 days, all in the first and last quarters.
        trades = [self._trade(d, "1") for d in (0, 5, 10, 15)]
        trades += [self._trade(d, "1") for d in (65, 70, 75, 80)]
        windows = walk_forward(trades, windows=4)

        assert len(windows) == 4
        assert windows[0]["trades"] == 4
        assert windows[1]["trades"] == 0   # a genuinely empty period
        assert windows[3]["trades"] == 4

    def test_exposes_an_edge_concentrated_in_one_window(self) -> None:
        """The most common false positive: three bad windows and one great one.

        Aggregate expectancy here is positive, and the windows show why that
        is not a strategy.
        """
        from tradingbot.backtest.engine import walk_forward

        trades = [self._trade(d, "-0.5") for d in (0, 5, 10, 20, 25, 30, 40, 45, 50)]
        trades += [self._trade(d, "6") for d in (62, 68, 74)]
        windows = walk_forward(trades, windows=4)

        positive = [w for w in windows if (w["expectancy_r"] or 0) > 0]
        assert len(positive) == 1
        assert positive[0]["window"] == 4
        # The aggregate hides it.
        total = sum(float(w["total_r"]) for w in windows)
        assert total > 0

    def test_too_few_trades_returns_no_windows(self) -> None:
        from tradingbot.backtest.engine import walk_forward

        assert walk_forward([self._trade(0, "1"), self._trade(5, "1")]) == []
