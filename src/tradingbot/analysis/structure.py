"""Market structure: swing points, trend classification and structure breaks.

This is the backbone of the chart reading. Levels, trendlines, patterns and
divergence all build on the swing points identified here, so the definition of
a swing has to be strict and, above all, causal.

**The lookahead trap.** A swing high is "a bar whose high exceeds the N bars on
either side". Confirming it requires N bars of hindsight, which means at the
moment that bar printed, it was not yet a swing. Any code that treats the most
recent swing as tradeable information without accounting for that delay is
reading the future. Every swing here carries `confirmed_at`, and
`confirmed_swings()` is the only accessor the strategy layer is allowed to use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from ..models import Trend
from .indicators import adx, atr, ema

log = logging.getLogger(__name__)


class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True)
class Swing:
    """One confirmed pivot.

    `index` is the bar that formed the pivot; `confirmed_index` is the bar by
    which it became knowable. The gap between them is the lookahead the naive
    implementation steals.
    """

    type: SwingType
    index: int
    confirmed_index: int
    time: pd.Timestamp
    price: float
    # How many bars on the weaker side were cleared. A pivot that survived 20
    # bars of challenge is structurally more important than one that survived 3.
    strength: int

    @property
    def is_high(self) -> bool:
        return self.type is SwingType.HIGH


@dataclass(frozen=True)
class StructureState:
    """The structural read for one symbol on one timeframe."""

    trend: Trend
    swings: tuple[Swing, ...]
    last_high: Swing | None
    last_low: Swing | None
    # Break of structure: trend continuation (a new extreme in the trend's
    # direction). Change of character: the first failure, which is the earliest
    # warning that the trend is finished.
    last_bos: Literal["bullish", "bearish"] | None
    last_choch: Literal["bullish", "bearish"] | None
    higher_highs: int
    higher_lows: int
    lower_highs: int
    lower_lows: int
    adx_value: float | None
    ema_stack: Literal["bullish", "bearish", "mixed"]
    # Where price sits in the recent range: 0 = range low, 1 = range high.
    range_position: float | None

    @property
    def is_trending(self) -> bool:
        return self.adx_value is not None and self.adx_value >= 20.0

    @property
    def structure_direction(self) -> int:
        """Net structural bias from swing sequence alone, ignoring indicators."""
        bullish = self.higher_highs + self.higher_lows
        bearish = self.lower_highs + self.lower_lows
        if bullish > bearish:
            return 1
        if bearish > bullish:
            return -1
        return 0


def find_swings(
    frame: pd.DataFrame, left: int = 3, right: int = 3, *, min_strength: int = 0
) -> list[Swing]:
    """Fractal pivot detection.

    A swing high needs `left` lower highs before it and `right` lower highs
    after it. Smaller `right` confirms sooner but produces more noise; the
    default of 3 is a reasonable 4h compromise.

    Strict inequality on the left and non-strict on the right breaks ties on
    equal highs consistently, so a double top reports one pivot rather than two
    or none depending on floating-point luck.
    """
    if frame.empty or len(frame) < left + right + 1:
        return []

    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    times = frame.index

    swings: list[Swing] = []
    for i in range(left, len(frame) - right):
        window_left = slice(i - left, i)
        window_right = slice(i + 1, i + right + 1)

        if highs[i] > highs[window_left].max() and highs[i] >= highs[window_right].max():
            strength = _pivot_strength(highs, i, is_high=True)
            if strength >= min_strength:
                swings.append(
                    Swing(
                        type=SwingType.HIGH,
                        index=i,
                        confirmed_index=i + right,
                        time=times[i],
                        price=float(highs[i]),
                        strength=strength,
                    )
                )
        elif lows[i] < lows[window_left].min() and lows[i] <= lows[window_right].min():
            strength = _pivot_strength(lows, i, is_high=False)
            if strength >= min_strength:
                swings.append(
                    Swing(
                        type=SwingType.LOW,
                        index=i,
                        confirmed_index=i + right,
                        time=times[i],
                        price=float(lows[i]),
                        strength=strength,
                    )
                )

    return swings


def _pivot_strength(values: np.ndarray, index: int, *, is_high: bool, cap: int = 50) -> int:
    """How many bars each side the pivot dominates, taking the weaker side.

    The weaker side is the honest measure: a high that towers over 30 bars to
    its left but is exceeded by the very next bar to its right is not a
    significant pivot, and taking the max would claim it was.
    """
    pivot = values[index]
    left = 0
    for offset in range(1, min(index, cap) + 1):
        candidate = values[index - offset]
        if (candidate >= pivot) if is_high else (candidate <= pivot):
            break
        left += 1
    right = 0
    for offset in range(1, min(len(values) - index - 1, cap) + 1):
        candidate = values[index + offset]
        if (candidate >= pivot) if is_high else (candidate <= pivot):
            break
        right += 1
    return min(left, right)


def confirmed_swings(swings: Sequence[Swing], as_of_index: int) -> list[Swing]:
    """Only the swings knowable at `as_of_index`.

    The single most important function in this module. Every consumer that
    makes a trading decision must filter through it, or the backtest is
    reading the future and its results are fiction.
    """
    return [swing for swing in swings if swing.confirmed_index <= as_of_index]


def classify_structure(
    frame: pd.DataFrame,
    *,
    left: int = 3,
    right: int = 3,
    swing_lookback: int = 6,
    as_of: int | None = None,
) -> StructureState:
    """Build the full structural read.

    Combines three independent sources of evidence: the swing sequence (pure
    price structure), the EMA stack (smoothed direction) and ADX (whether any
    of it is organised enough to act on). They are kept separate on purpose so
    the confluence layer can see when they disagree, which is itself a signal.
    """
    if frame.empty:
        return _empty_structure()

    end = len(frame) - 1 if as_of is None else as_of
    all_swings = find_swings(frame, left=left, right=right)
    swings = confirmed_swings(all_swings, end)

    highs = [swing for swing in swings if swing.is_high][-swing_lookback:]
    lows = [swing for swing in swings if not swing.is_high][-swing_lookback:]

    higher_highs, lower_highs = _count_sequence(highs)
    higher_lows, lower_lows = _count_sequence(lows)

    adx_value = _latest(adx(frame, 14)["adx"], end)
    stack = _ema_stack(frame, end)
    trend = _classify_trend(
        higher_highs, higher_lows, lower_highs, lower_lows, adx_value, stack
    )
    bos, choch = _detect_structure_break(frame, swings, end)

    return StructureState(
        trend=trend,
        swings=tuple(swings),
        last_high=highs[-1] if highs else None,
        last_low=lows[-1] if lows else None,
        last_bos=bos,
        last_choch=choch,
        higher_highs=higher_highs,
        higher_lows=higher_lows,
        lower_highs=lower_highs,
        lower_lows=lower_lows,
        adx_value=adx_value,
        ema_stack=stack,
        range_position=_range_position(frame, end),
    )


def _count_sequence(swings: Sequence[Swing]) -> tuple[int, int]:
    """Count rising vs falling steps in a run of same-type swings."""
    rising = falling = 0
    for previous, current in zip(swings, swings[1:]):
        if current.price > previous.price:
            rising += 1
        elif current.price < previous.price:
            falling += 1
    return rising, falling


def _ema_stack(
    frame: pd.DataFrame, end: int
) -> Literal["bullish", "bearish", "mixed"]:
    """Whether the 20/50/200 EMAs are stacked in order.

    Falls back to shorter periods on a young series rather than returning
    nothing: AXISUSDT has about a month of history, so demanding a 200 EMA
    would make it unanalysable rather than merely less certain.
    """
    close = frame["close"]
    periods = (20, 50, 200) if len(frame) >= 200 else (10, 20, 50)
    if len(frame) < periods[-1]:
        return "mixed"

    values = [_latest(ema(close, period), end) for period in periods]
    if any(value is None for value in values):
        return "mixed"
    fast, mid, slow = values  # type: ignore[misc]
    if fast > mid > slow:
        return "bullish"
    if fast < mid < slow:
        return "bearish"
    return "mixed"


def _classify_trend(
    higher_highs: int,
    higher_lows: int,
    lower_highs: int,
    lower_lows: int,
    adx_value: float | None,
    stack: Literal["bullish", "bearish", "mixed"],
) -> Trend:
    """Fold the evidence into one label.

    `STRONG_*` requires structure, EMA stack and ADX all to agree. Reserving
    the strong label for genuine unanimity is what makes it worth anything to
    the sizing and confluence layers.
    """
    bullish = higher_highs + higher_lows
    bearish = lower_highs + lower_lows
    organised = adx_value is not None and adx_value >= 25.0

    if bullish > bearish + 1 and stack == "bullish" and organised:
        return Trend.STRONG_UP
    if bearish > bullish + 1 and stack == "bearish" and organised:
        return Trend.STRONG_DOWN
    if bullish > bearish:
        return Trend.UP
    if bearish > bullish:
        return Trend.DOWN
    return Trend.RANGE


def _detect_structure_break(
    frame: pd.DataFrame, swings: Sequence[Swing], end: int
) -> tuple[Literal["bullish", "bearish"] | None, Literal["bullish", "bearish"] | None]:
    """Find the most recent break of structure and change of character.

    BOS   - price closes beyond the prior swing in the trend's direction.
    CHoCH - price closes beyond the prior swing *against* the trend, the first
            structural evidence the trend has ended.

    Closes are used rather than wicks throughout. A wick through a level is a
    liquidity grab and reverses constantly; a close beyond it is a decision by
    the market. On the thin satellite books this distinction is the difference
    between a signal and a stop-out.
    """
    if len(swings) < 3:
        return None, None

    highs = [swing for swing in swings if swing.is_high]
    lows = [swing for swing in swings if not swing.is_high]
    if not highs or not lows:
        return None, None

    closes = frame["close"].to_numpy(dtype=float)
    bos: Literal["bullish", "bearish"] | None = None
    choch: Literal["bullish", "bearish"] | None = None

    last_high, last_low = highs[-1], lows[-1]
    # Only consider bars after the pivot was confirmed, not after it formed.
    after_high = closes[min(last_high.confirmed_index + 1, end + 1) : end + 1]
    after_low = closes[min(last_low.confirmed_index + 1, end + 1) : end + 1]

    broke_up = bool(len(after_high)) and bool((after_high > last_high.price).any())
    broke_down = bool(len(after_low)) and bool((after_low < last_low.price).any())

    was_rising = len(lows) >= 2 and lows[-1].price > lows[-2].price
    was_falling = len(highs) >= 2 and highs[-1].price < highs[-2].price

    if broke_up:
        bos, choch = ("bullish", choch) if was_rising else (bos, "bullish")
    if broke_down:
        bos, choch = ("bearish", choch) if was_falling else (bos, "bearish")

    return bos, choch


def _range_position(frame: pd.DataFrame, end: int, lookback: int = 50) -> float | None:
    """Where price sits in its recent range, 0 (low) to 1 (high).

    Guards against the classic error of buying a breakout at the top of a
    range that has already made its move.
    """
    start = max(0, end - lookback + 1)
    window = frame.iloc[start : end + 1]
    if window.empty:
        return None
    high = float(window["high"].max())
    low = float(window["low"].min())
    if high <= low:
        return None
    return (float(window["close"].iloc[-1]) - low) / (high - low)


def _latest(series: pd.Series, index: int) -> float | None:
    """Read one value, treating NaN and out-of-range as 'unknown'."""
    if index < 0 or index >= len(series):
        return None
    value = series.iloc[index]
    return None if pd.isna(value) else float(value)


def _empty_structure() -> StructureState:
    return StructureState(
        trend=Trend.RANGE,
        swings=(),
        last_high=None,
        last_low=None,
        last_bos=None,
        last_choch=None,
        higher_highs=0,
        higher_lows=0,
        lower_highs=0,
        lower_lows=0,
        adx_value=None,
        ema_stack="mixed",
        range_position=None,
    )


def swing_atr_distance(
    frame: pd.DataFrame, swing: Swing, *, atr_period: int = 14
) -> float | None:
    """Distance from the latest close to a swing, measured in ATR.

    ATR units rather than percent so the same thresholds work for BTC and for
    a satellite token whose daily range is twenty times wider.
    """
    atr_value = _latest(atr(frame, atr_period), len(frame) - 1)
    if not atr_value:
        return None
    return abs(float(frame["close"].iloc[-1]) - swing.price) / atr_value
