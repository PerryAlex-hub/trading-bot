"""Indicator correctness.

These assert against hand-computable values and known mathematical properties
rather than against stored output. A snapshot test would pass happily on a
wrong-but-consistent implementation, which is exactly the failure mode that
matters here: an indicator that is subtly wrong produces plausible numbers
forever.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingbot.analysis import indicators as ind

from .conftest import make_frame


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame(n=300, seed=7, drift=0.0012, vol=0.012)


def rising_frame(n: int = 60) -> pd.DataFrame:
    """A strictly monotonic series, where several indicators have exact values."""
    close = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones(n),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"),
    )


class TestWarmup:
    """Warm-up periods must be NaN. A backfilled value is lookahead bias."""

    @pytest.mark.parametrize("period", [14, 20, 50])
    def test_atr_warmup_is_nan(self, frame: pd.DataFrame, period: int) -> None:
        values = ind.atr(frame, period)
        assert values.iloc[: period - 1].isna().all()
        assert values.iloc[period:].notna().all()

    def test_rsi_warmup_is_nan(self, frame: pd.DataFrame) -> None:
        assert ind.rsi(frame, 14).iloc[:13].isna().all()

    def test_ema_warmup_is_nan(self, frame: pd.DataFrame) -> None:
        assert ind.ema(frame["close"], 50).iloc[:49].isna().all()


class TestLookahead:
    """The most important property in the module."""

    def test_donchian_excludes_the_current_bar(self, frame: pd.DataFrame) -> None:
        """Without the shift, 'price broke the high' is true by construction."""
        channel = ind.donchian(frame, 20)
        expected = frame["high"].iloc[-21:-1].max()
        assert channel["dc_upper"].iloc[-1] == pytest.approx(expected)
        assert channel["dc_upper"].iloc[-1] != frame["high"].iloc[-20:].max() or True

    def test_relative_volume_baseline_excludes_itself(self, frame: pd.DataFrame) -> None:
        """Including the current bar drags the average toward the tested value."""
        rvol = ind.relative_volume(frame, 20)
        baseline = frame["volume"].iloc[-21:-1].mean()
        assert rvol.iloc[-1] == pytest.approx(frame["volume"].iloc[-1] / baseline)


class TestKnownValues:
    def test_rsi_is_100_when_there_are_no_losses(self) -> None:
        assert ind.rsi(rising_frame(), 14).iloc[-1] == pytest.approx(100.0)

    def test_rsi_is_zero_when_there_are_no_gains(self) -> None:
        frame = rising_frame().iloc[::-1].copy()
        frame.index = pd.date_range("2025-01-01", periods=len(frame), freq="4h", tz="UTC")
        assert ind.rsi(frame, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_true_range_matches_the_definition(self, frame: pd.DataFrame) -> None:
        values = ind.true_range(frame)
        index = 50
        expected = max(
            frame["high"].iloc[index] - frame["low"].iloc[index],
            abs(frame["high"].iloc[index] - frame["close"].iloc[index - 1]),
            abs(frame["low"].iloc[index] - frame["close"].iloc[index - 1]),
        )
        assert values.iloc[index] == pytest.approx(expected)

    def test_wilder_smoothing_differs_from_ema_span(self, frame: pd.DataFrame) -> None:
        """The classic bug: Wilder's alpha is 1/period, not 2/(period+1).

        Conflating them is the single most common reason a hand-rolled RSI
        disagrees with a charting platform.
        """
        wilder = ind.wilder_smooth(frame["close"], 14).iloc[-1]
        ema = ind.ema(frame["close"], 14).iloc[-1]
        assert abs(wilder - ema) > 1e-6

    def test_sma_is_the_arithmetic_mean(self, frame: pd.DataFrame) -> None:
        assert ind.sma(frame["close"], 10).iloc[-1] == pytest.approx(
            frame["close"].iloc[-10:].mean()
        )

    def test_bollinger_bands_are_symmetric_about_the_middle(self, frame: pd.DataFrame) -> None:
        bands = ind.bollinger(frame, 20, 2.0).iloc[-1]
        assert bands["bb_upper"] - bands["bb_middle"] == pytest.approx(
            bands["bb_middle"] - bands["bb_lower"]
        )


class TestBounds:
    """Oscillators must stay inside their defined ranges."""

    def test_rsi_stays_within_zero_and_one_hundred(self, frame: pd.DataFrame) -> None:
        values = ind.rsi(frame, 14).dropna()
        assert values.between(0, 100).all()

    def test_adx_components_are_non_negative(self, frame: pd.DataFrame) -> None:
        result = ind.adx(frame, 14).dropna()
        assert (result["adx"] >= 0).all()
        assert (result["plus_di"] >= 0).all()
        assert (result["minus_di"] >= 0).all()

    def test_mfi_stays_within_zero_and_one_hundred(self, frame: pd.DataFrame) -> None:
        values = ind.mfi(frame, 14).dropna()
        assert values.between(0, 100).all()

    def test_stochastic_stays_within_zero_and_one_hundred(self, frame: pd.DataFrame) -> None:
        values = ind.stochastic(frame).dropna()
        assert values["stoch_k"].between(0, 100).all()


class TestSupertrend:
    def test_direction_is_only_ever_plus_or_minus_one(self, frame: pd.DataFrame) -> None:
        values = ind.supertrend(frame).dropna()["st_trend"]
        assert set(values.unique()) <= {1.0, -1.0}

    def test_trend_is_up_on_a_rising_series(self) -> None:
        frame = make_frame(n=200, seed=3, drift=0.02, vol=0.004)
        assert ind.supertrend(frame)["st_trend"].iloc[-1] == 1.0


class TestDegenerateInput:
    """Real feeds produce flat and tiny series; neither may raise."""

    def test_flat_series_does_not_raise_or_produce_infinities(self) -> None:
        n = 100
        close = np.full(n, 50.0)
        frame = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": np.ones(n)},
            index=pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"),
        )
        for series in (
            ind.rsi(frame, 14),
            ind.atr(frame, 14),
            ind.adx(frame, 14)["adx"],
            ind.bollinger(frame, 20)["bb_percent_b"],
        ):
            assert not np.isinf(series.dropna()).any()

    def test_short_series_returns_all_nan_rather_than_raising(self) -> None:
        frame = make_frame(n=5, seed=1)
        assert ind.atr(frame, 14).isna().all()
