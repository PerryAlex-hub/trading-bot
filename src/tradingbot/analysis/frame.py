"""The Decimal/float boundary.

Candles arrive as `Decimal` because they may become order parameters, and
indicators want floats because they involve exponential smoothing and
regression. This module is the one place that conversion happens, so the rule
stays easy to state and easy to audit:

    Decimal in storage and sizing. float in analysis. Converted here, nowhere
    else.

Also the one place that verifies the series is actually usable. A gap in the
candle history makes every rolling-window indicator silently wrong - the window
still contains N bars, they just are not the N consecutive bars the indicator
assumes - so gaps are detected and reported rather than discovered later as an
inexplicable backtest result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Sequence

import pandas as pd

from ..models import Candle, Timeframe

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class SeriesHealth:
    """Whether a candle series can be trusted to compute indicators on."""

    symbol: str
    timeframe: str
    bars: int
    gaps: int
    largest_gap_bars: int
    duplicate_timestamps: int
    first_bar: pd.Timestamp | None
    last_bar: pd.Timestamp | None

    def is_usable(self, *, min_bars: int, max_gaps: int = 3) -> bool:
        return self.bars >= min_bars and self.gaps <= max_gaps

    def describe(self) -> str:
        return (
            f"{self.symbol} {self.timeframe}: {self.bars} bars, {self.gaps} gaps "
            f"(largest {self.largest_gap_bars}), "
            f"{self.duplicate_timestamps} duplicates"
        )


def to_frame(candles: Sequence[Candle]) -> pd.DataFrame:
    """Convert candles to a float OHLCV frame indexed by `open_time`.

    Decimal goes via `float()` directly rather than through pandas' own
    inference, which would produce an `object` dtype column that silently
    breaks every rolling operation downstream.
    """
    if not candles:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS), index=pd.DatetimeIndex([], tz="UTC"))

    frame = pd.DataFrame(
        {
            "open": [float(candle.open) for candle in candles],
            "high": [float(candle.high) for candle in candles],
            "low": [float(candle.low) for candle in candles],
            "close": [float(candle.close) for candle in candles],
            "volume": [float(candle.volume) for candle in candles],
        },
        index=pd.DatetimeIndex(
            [candle.open_time for candle in candles], name="open_time"
        ),
    )

    if not frame.index.is_monotonic_increasing:
        log.debug("Candle series arrived unsorted; sorting")
        frame = frame.sort_index()
    if frame.index.has_duplicates:
        # Keep the last occurrence: a restated bar is more current than the
        # version we first stored.
        log.warning("Duplicate candle timestamps found; keeping the latest of each")
        frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def check_health(
    frame: pd.DataFrame, symbol: str, timeframe: Timeframe | str
) -> SeriesHealth:
    """Detect missing bars.

    A rolling(20) over a series with a hole in it still returns a number; it is
    just not the number the indicator is defined to produce. Surfacing that
    here is much cheaper than tracking down why a backtest disagrees with the
    chart six weeks later.
    """
    tf = timeframe.value if isinstance(timeframe, Timeframe) else timeframe
    if frame.empty:
        return SeriesHealth(symbol, tf, 0, 0, 0, 0, None, None)

    expected = timedelta(seconds=Timeframe(tf).seconds)
    deltas = frame.index.to_series().diff().dropna()
    # Tolerate a little jitter; anything above 1.5 bars is a real gap.
    gap_mask = deltas > expected * 1.5
    gap_sizes = (deltas[gap_mask] / expected).round().astype(int) if gap_mask.any() else []

    return SeriesHealth(
        symbol=symbol,
        timeframe=tf,
        bars=len(frame),
        gaps=int(gap_mask.sum()),
        largest_gap_bars=int(max(gap_sizes)) if len(gap_sizes) else 0,
        duplicate_timestamps=int(frame.index.duplicated().sum()),
        first_bar=frame.index[0],
        last_bar=frame.index[-1],
    )


def to_decimal(value: float | int | Decimal | None) -> Decimal | None:
    """Back across the boundary, for values that become order parameters.

    Via `str()` deliberately. `Decimal(0.1)` captures the binary float error
    and yields 0.1000000000000000055511151231257827, whereas `Decimal("0.1")`
    is exactly 0.1. For a price that becomes an order parameter, that matters.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if pd.isna(value):
        return None
    return Decimal(str(value))
