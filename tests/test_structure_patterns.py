"""Market structure, level clustering and pattern detection.

The swing-confirmation tests are the most important in the suite. A swing high
cannot be known until N bars after it formed, and any code that forgets this
produces a backtest that reads the future and results that are fiction.
"""

from __future__ import annotations

from tradingbot.analysis import levels as L
from tradingbot.analysis import patterns as P
from tradingbot.analysis import structure as S
from tradingbot.analysis.frame import check_health
from tradingbot.models import Trend

from .conftest import make_frame, path_frame, ramp


class TestSwingCausality:
    """A swing must never be knowable before it is confirmed."""

    def test_confirmation_always_lags_formation(self) -> None:
        frame = make_frame(n=300, seed=11)
        swings = S.find_swings(frame, left=3, right=3)
        assert swings, "no swings found in 300 bars"
        assert all(swing.confirmed_index > swing.index for swing in swings)

    def test_confirmation_lag_equals_the_right_window(self) -> None:
        frame = make_frame(n=200, seed=12)
        for right in (2, 3, 5):
            swings = S.find_swings(frame, left=3, right=right)
            assert all(
                swing.confirmed_index == swing.index + right for swing in swings
            )

    def test_confirmed_swings_filters_the_future(self) -> None:
        frame = make_frame(n=300, seed=13)
        swings = S.find_swings(frame)
        as_of = 150
        visible = S.confirmed_swings(swings, as_of)
        assert all(swing.confirmed_index <= as_of for swing in visible)
        assert len(visible) < len(swings)

    def test_as_of_truncation_does_not_see_later_bars(self) -> None:
        """Structure computed at bar N must match structure on a frame cut at N.

        If these disagree, something downstream is reading beyond `as_of`.
        """
        frame = make_frame(n=300, seed=14)
        cut = 200
        truncated = S.classify_structure(frame.iloc[: cut + 1])
        in_place = S.classify_structure(frame, as_of=cut)
        assert truncated.trend == in_place.trend
        assert truncated.higher_highs == in_place.higher_highs
        assert truncated.lower_lows == in_place.lower_lows


class TestStructureClassification:
    def test_strong_uptrend_is_detected(self) -> None:
        frame = make_frame(n=300, seed=21, drift=0.012, vol=0.006)
        state = S.classify_structure(frame)
        assert state.trend.is_up
        assert state.structure_direction >= 0

    def test_strong_downtrend_is_detected(self) -> None:
        frame = make_frame(n=300, seed=22, drift=-0.012, vol=0.006)
        state = S.classify_structure(frame)
        assert state.trend.is_down

    def test_empty_frame_returns_a_neutral_state(self) -> None:
        frame = make_frame(n=300).iloc[0:0]
        state = S.classify_structure(frame)
        assert state.trend is Trend.RANGE
        assert state.swings == ()

    def test_a_parabolic_move_is_not_called_a_range(self) -> None:
        """REGRESSION: a near-vertical move produces NO fractal pivots.

        No bar in a monotonic rally has lower highs on both sides, so the
        swing sequence is empty exactly when the trend is strongest. That
        previously fell through to RANGE, which zeroed the trend contribution,
        disarmed the strong-trend blockers, and re-enabled the mean-reversion
        check that is meant to switch itself off in a trend - i.e. the bot
        would have been primed to fade a parabolic move.
        """
        frame = make_frame(n=300, seed=21, drift=0.012, vol=0.006)
        assert S.find_swings(frame) == [], "test premise broken: pivots exist here"
        state = S.classify_structure(frame)
        assert state.trend is Trend.STRONG_UP
        assert state.ema_stack == "bullish"

    def test_a_parabolic_decline_is_not_called_a_range(self) -> None:
        frame = make_frame(n=300, seed=22, drift=-0.012, vol=0.006)
        assert S.classify_structure(frame).trend is Trend.STRONG_DOWN

    def test_no_swings_and_a_mixed_stack_is_genuinely_a_range(self) -> None:
        """The fallback must not invent a direction when there is no evidence."""
        import numpy as np
        import pandas as pd

        close = np.full(100, 50.0)
        flat = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": np.ones(100)},
            index=pd.date_range("2025-01-01", periods=100, freq="4h", tz="UTC"),
        )
        assert S.classify_structure(flat).trend is Trend.RANGE

    def test_range_position_is_bounded(self) -> None:
        frame = make_frame(n=200, seed=23)
        position = S.classify_structure(frame).range_position
        assert position is None or 0.0 <= position <= 1.0


class TestLevels:
    def test_every_level_meets_the_touch_minimum(self) -> None:
        frame = make_frame(n=400, seed=31)
        level_map = L.build_levels(frame, min_touches=2)
        for level in level_map.supports + level_map.resistances:
            assert level.touches >= 2

    def test_supports_sit_below_price_and_resistances_above(self) -> None:
        frame = make_frame(n=400, seed=32)
        level_map = L.build_levels(frame)
        price = float(frame["close"].iloc[-1])
        assert all(level.price < price for level in level_map.supports)
        assert all(level.price > price for level in level_map.resistances)

    def test_nearest_level_accessors_pick_the_closest(self) -> None:
        frame = make_frame(n=400, seed=33)
        level_map = L.build_levels(frame)
        price = float(frame["close"].iloc[-1])
        support = level_map.nearest_support(price)
        if support is not None:
            assert all(
                level.price <= support.price
                for level in level_map.supports
                if level.price < price
            )

    def test_round_numbers_are_builtin_floats(self) -> None:
        """numpy scalars are not JSON-serialisable and these reach the journal."""
        frame = make_frame(n=200, seed=34)
        for value in L.build_levels(frame).round_numbers:
            assert type(value) is float

    def test_volume_profile_value_area_contains_the_poc(self) -> None:
        frame = make_frame(n=400, seed=35)
        profile = L.build_volume_profile(frame)
        assert profile is not None
        assert profile.value_area_low <= profile.poc <= profile.value_area_high


class TestChartPatterns:
    """Synthetic shapes, so a miss is a detector bug rather than noise."""

    def test_double_top(self) -> None:
        frame = path_frame(
            ramp(100, 120, 20) + ramp(120, 110, 10)
            + ramp(110, 120.2, 10) + ramp(120.2, 104, 14)
        )
        names = [p.name for p in P.detect_chart_patterns(frame)]
        assert "double_top" in names

    def test_double_bottom(self) -> None:
        frame = path_frame(
            ramp(120, 100, 20) + ramp(100, 110, 10)
            + ramp(110, 99.8, 10) + ramp(99.8, 116, 14)
        )
        patterns = P.detect_chart_patterns(frame)
        assert "double_bottom" in [p.name for p in patterns]
        found = next(p for p in patterns if p.name == "double_bottom")
        assert found.direction is P.PatternDirection.BULLISH
        assert found.target is not None and found.target > 110

    def test_head_and_shoulders(self) -> None:
        frame = path_frame(
            ramp(100, 112, 12) + ramp(112, 104, 8) + ramp(104, 124, 12)
            + ramp(124, 103, 10) + ramp(103, 113, 8) + ramp(113, 98, 12)
        )
        patterns = P.detect_chart_patterns(frame)
        assert "head_and_shoulders" in [p.name for p in patterns]
        found = next(p for p in patterns if p.name == "head_and_shoulders")
        assert found.direction is P.PatternDirection.BEARISH

    def test_inverse_head_and_shoulders(self) -> None:
        frame = path_frame(
            ramp(120, 108, 12) + ramp(108, 116, 8) + ramp(116, 96, 12)
            + ramp(96, 117, 10) + ramp(117, 107, 8) + ramp(107, 122, 12)
        )
        names = [p.name for p in P.detect_chart_patterns(frame)]
        assert "inverse_head_and_shoulders" in names

    def test_ascending_triangle(self) -> None:
        frame = path_frame(
            ramp(100, 120, 12) + ramp(120, 100, 10) + ramp(100, 119.8, 10)
            + ramp(119.8, 106, 9) + ramp(106, 120.1, 9) + ramp(120.1, 112, 8)
            + ramp(112, 134, 12)
        )
        patterns = P.detect_chart_patterns(frame)
        assert "ascending_triangle" in [p.name for p in patterns]

    def test_an_unbroken_double_top_is_not_reported(self) -> None:
        """Two peaks with no neckline break is a shape, not a signal."""
        frame = path_frame(
            ramp(100, 120, 20) + ramp(120, 112, 10) + ramp(112, 120.1, 10)
            + ramp(120.1, 118, 6)
        )
        assert "double_top" not in [p.name for p in P.detect_chart_patterns(frame)]

    def test_too_few_bars_returns_nothing_rather_than_raising(self) -> None:
        assert P.detect_chart_patterns(path_frame(ramp(100, 110, 10))) == []


class TestCandlestickPatterns:
    def test_hammer_is_detected(self) -> None:
        frame = make_frame(n=60, seed=41, vol=0.01)
        # A long lower wick with the body at the top.
        frame.iloc[-1, frame.columns.get_loc("open")] = 100.0
        frame.iloc[-1, frame.columns.get_loc("close")] = 100.5
        frame.iloc[-1, frame.columns.get_loc("high")] = 100.7
        frame.iloc[-1, frame.columns.get_loc("low")] = 96.0
        names = [p.name for p in P.detect_candlestick_patterns(frame, lookback=1)]
        assert "hammer" in names

    def test_shooting_star_is_detected(self) -> None:
        frame = make_frame(n=60, seed=42, vol=0.01)
        frame.iloc[-1, frame.columns.get_loc("open")] = 100.0
        frame.iloc[-1, frame.columns.get_loc("close")] = 99.5
        frame.iloc[-1, frame.columns.get_loc("low")] = 99.3
        frame.iloc[-1, frame.columns.get_loc("high")] = 104.0
        names = [p.name for p in P.detect_candlestick_patterns(frame, lookback=1)]
        assert "shooting_star" in names

    def test_confidence_is_always_a_probability(self) -> None:
        frame = make_frame(n=200, seed=43)
        for pattern in P.detect_candlestick_patterns(frame, lookback=50):
            assert 0.0 <= pattern.confidence <= 1.0


class TestDivergence:
    def test_strength_is_bounded_and_serialisable(self) -> None:
        frame = make_frame(n=300, seed=51)
        for divergence in P.detect_divergence(frame):
            assert 0.0 <= divergence.strength <= 1.0
            assert type(divergence.strength) is float

    def test_short_series_returns_nothing(self) -> None:
        assert P.detect_divergence(make_frame(n=20, seed=52)) == []


class TestSeriesHealth:
    def test_a_hole_in_the_series_is_detected(self) -> None:
        """A gap makes every rolling window silently wrong."""
        frame = make_frame(n=300, seed=61)
        gapped = frame.drop(frame.index[100:112])
        health = check_health(gapped, "TEST", "4h")
        assert health.gaps >= 1
        assert health.largest_gap_bars >= 12

    def test_a_clean_series_reports_no_gaps(self) -> None:
        health = check_health(make_frame(n=300, seed=62), "TEST", "4h")
        assert health.gaps == 0

    def test_usability_requires_enough_bars(self) -> None:
        health = check_health(make_frame(n=50, seed=63), "TEST", "4h")
        assert not health.is_usable(min_bars=200)
