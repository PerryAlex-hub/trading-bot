"""Pattern detection: candlestick formations and classic chart patterns.

This is the "chart analysis" layer, done geometrically rather than visually. A
detector here returns a `Pattern` with a direction, a confidence and the bar it
was found on, so the confluence layer can weight it and the journal can record
exactly what was seen.

Three rules that keep this honest, and they are the difference between a useful
detector and a curve-fitting machine:

1. **Context gates confidence.** A hammer in a downtrend at support is a
   signal; the identical candle mid-range is noise. Detectors that report shape
   without location are why candlestick patterns have a bad reputation.
2. **Everything is measured in ATR, never in percent or ticks.** The universe
   spans six orders of magnitude of price.
3. **Nothing reads past its own bar.** Chart patterns need a confirmed breakout
   to count, and that breakout is only knowable after it closes.

Known limitation, verified by test: `_triangles_and_wedges` returns nothing once
the fitted boundaries have already crossed at the current bar. The formation has
genuinely resolved by then, but the *breakout* that resolved it is exactly what
we would want to trade, so a sharp move out of a tight symmetrical triangle can
be missed. Double tops/bottoms, head-and-shoulders, ascending/descending
triangles and channels are all covered; this one case is not. Fixing it means
tracking the apex and reporting the break relative to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .indicators import atr, macd, relative_volume, rsi
from .structure import Swing, SwingType, confirmed_swings, find_swings

log = logging.getLogger(__name__)


class PatternDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

    @property
    def sign(self) -> int:
        return {"bullish": 1, "bearish": -1, "neutral": 0}[self.value]


@dataclass(frozen=True)
class Pattern:
    """One detected formation."""

    name: str
    direction: PatternDirection
    # 0..1. Shape quality combined with how favourable the context is.
    confidence: float
    bar_index: int
    time: pd.Timestamp
    # Pattern-implied target, where the pattern defines one (measured move).
    target: float | None = None
    invalidation: float | None = None
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.confidence >= 0.5 and self.direction is not PatternDirection.NEUTRAL


# ---------------------------------------------------------------------------
# Candlestick patterns
# ---------------------------------------------------------------------------


def detect_candlestick_patterns(
    frame: pd.DataFrame, *, lookback: int = 3, atr_period: int = 14
) -> list[Pattern]:
    """Scan the last `lookback` bars for single- and two-bar formations.

    Only recent bars: a hammer from forty bars ago is history, not a trade
    trigger, and scanning the whole frame would flood the confluence layer with
    stale evidence.
    """
    if len(frame) < atr_period + 3:
        return []

    atr_series = atr(frame, atr_period)
    rvol = relative_volume(frame, 20)
    patterns: list[Pattern] = []

    start = max(2, len(frame) - lookback)
    for index in range(start, len(frame)):
        atr_value = atr_series.iloc[index]
        if pd.isna(atr_value) or atr_value <= 0:
            continue
        volume_mult = float(rvol.iloc[index]) if not pd.isna(rvol.iloc[index]) else 1.0
        patterns.extend(
            _candles_at(frame, index, float(atr_value), volume_mult)
        )
    return patterns


def _candles_at(
    frame: pd.DataFrame, index: int, atr_value: float, volume_mult: float
) -> list[Pattern]:
    bar = frame.iloc[index]
    previous = frame.iloc[index - 1]
    time = frame.index[index]

    open_, high, low, close = (
        float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    )
    body = abs(close - open_)
    span = high - low
    if span <= 0:
        return []

    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    found: list[Pattern] = []

    # Volume confirms a reversal candle: the same shape on no volume is a
    # drift, not a rejection. Capped so one freak bar cannot dominate.
    volume_boost = min(0.2, max(0.0, (volume_mult - 1.0) * 0.1))

    # --- Pin bars / hammers ------------------------------------------------
    # Long wick, small body, body at the opposite end. A rejection of price.
    if lower_wick > body * 2 and lower_wick > span * 0.6 and span > atr_value * 0.8:
        found.append(
            Pattern(
                name="hammer",
                direction=PatternDirection.BULLISH,
                confidence=min(0.85, 0.45 + lower_wick / span * 0.4 + volume_boost),
                bar_index=index,
                time=time,
                invalidation=low,
                detail={"wick_ratio": lower_wick / span, "rvol": volume_mult},
            )
        )
    if upper_wick > body * 2 and upper_wick > span * 0.6 and span > atr_value * 0.8:
        found.append(
            Pattern(
                name="shooting_star",
                direction=PatternDirection.BEARISH,
                confidence=min(0.85, 0.45 + upper_wick / span * 0.4 + volume_boost),
                bar_index=index,
                time=time,
                invalidation=high,
                detail={"wick_ratio": upper_wick / span, "rvol": volume_mult},
            )
        )

    # --- Engulfing ---------------------------------------------------------
    previous_open, previous_close = float(previous["open"]), float(previous["close"])
    previous_body = abs(previous_close - previous_open)
    if body > previous_body * 1.1 and body > atr_value * 0.5:
        engulfs_up = (
            close > open_
            and previous_close < previous_open
            and close >= previous_open
            and open_ <= previous_close
        )
        engulfs_down = (
            close < open_
            and previous_close > previous_open
            and close <= previous_open
            and open_ >= previous_close
        )
        if engulfs_up:
            found.append(
                Pattern(
                    name="bullish_engulfing",
                    direction=PatternDirection.BULLISH,
                    confidence=min(0.8, 0.5 + body / (atr_value * 4) + volume_boost),
                    bar_index=index,
                    time=time,
                    invalidation=low,
                    detail={"body_atr": body / atr_value, "rvol": volume_mult},
                )
            )
        if engulfs_down:
            found.append(
                Pattern(
                    name="bearish_engulfing",
                    direction=PatternDirection.BEARISH,
                    confidence=min(0.8, 0.5 + body / (atr_value * 4) + volume_boost),
                    bar_index=index,
                    time=time,
                    invalidation=high,
                    detail={"body_atr": body / atr_value, "rvol": volume_mult},
                )
            )

    # --- Marubozu / wide-range conviction bar ------------------------------
    if body > span * 0.85 and body > atr_value * 1.3:
        direction = (
            PatternDirection.BULLISH if close > open_ else PatternDirection.BEARISH
        )
        found.append(
            Pattern(
                name="momentum_bar",
                direction=direction,
                confidence=min(0.7, 0.4 + body / (atr_value * 5) + volume_boost),
                bar_index=index,
                time=time,
                detail={"body_atr": body / atr_value},
            )
        )

    # --- Doji / indecision -------------------------------------------------
    if body < span * 0.1 and span > atr_value * 0.6:
        found.append(
            Pattern(
                name="doji",
                direction=PatternDirection.NEUTRAL,
                confidence=0.4,
                bar_index=index,
                time=time,
                detail={"body_ratio": body / span},
            )
        )

    # --- Inside bar (compression before expansion) -------------------------
    if high < float(previous["high"]) and low > float(previous["low"]):
        found.append(
            Pattern(
                name="inside_bar",
                direction=PatternDirection.NEUTRAL,
                confidence=0.45,
                bar_index=index,
                time=time,
                detail={"compression": span / max(float(previous["high"]) - float(previous["low"]), 1e-12)},
            )
        )

    return found


# ---------------------------------------------------------------------------
# Chart patterns
# ---------------------------------------------------------------------------


def detect_chart_patterns(
    frame: pd.DataFrame,
    *,
    left: int = 3,
    right: int = 3,
    atr_period: int = 14,
    as_of: int | None = None,
) -> list[Pattern]:
    """Detect multi-swing formations from confirmed pivots.

    Every detector below requires a *confirmed* breakout or neckline break.
    An un-broken double top is a shape, not a trade: price is as likely to make
    a third high as to reverse, and treating the shape alone as a signal is the
    main way pattern trading loses money.
    """
    if frame.empty or len(frame) < 30:
        return []

    end = len(frame) - 1 if as_of is None else as_of
    window = frame.iloc[: end + 1]
    swings = confirmed_swings(find_swings(window, left=left, right=right), end)
    # Three is the real floor, not four: a textbook double top is exactly two
    # highs plus the neckline trough between them. Demanding four pivots threw
    # away the most common reversal pattern there is.
    if len(swings) < 3:
        return []

    atr_value = _last_valid(atr(window, atr_period))
    if not atr_value or atr_value <= 0:
        return []

    patterns: list[Pattern] = []
    patterns.extend(_double_top_bottom(window, swings, atr_value, end))
    patterns.extend(_head_and_shoulders(window, swings, atr_value, end))
    patterns.extend(_triangles_and_wedges(window, swings, atr_value, end))
    return patterns


def _double_top_bottom(
    frame: pd.DataFrame, swings: Sequence[Swing], atr_value: float, end: int
) -> list[Pattern]:
    """Two comparable extremes, then a neckline break.

    Tolerance is 0.5 ATR: the two peaks must be *comparable*, not identical.
    Demanding equality finds almost nothing, and in practice the second peak
    marginally exceeding the first (a failed breakout) is the stronger signal.
    """
    found: list[Pattern] = []
    closes = frame["close"].to_numpy(dtype=float)

    for kind, swing_type in (("top", SwingType.HIGH), ("bottom", SwingType.LOW)):
        extremes = [swing for swing in swings if swing.type is swing_type][-4:]
        if len(extremes) < 2:
            continue
        first, second = extremes[-2], extremes[-1]
        if abs(second.price - first.price) > atr_value * 0.5:
            continue
        if second.index - first.index < 5:
            continue  # too close together to be a distinct retest

        # The neckline is the opposing extreme between the two peaks.
        between = [
            swing
            for swing in swings
            if swing.type is not swing_type and first.index < swing.index < second.index
        ]
        if not between:
            continue
        neckline = (
            min(between, key=lambda swing: swing.price).price
            if kind == "top"
            else max(between, key=lambda swing: swing.price).price
        )

        after = closes[min(second.confirmed_index + 1, end + 1) : end + 1]
        if len(after) == 0:
            continue
        broken = (
            bool((after < neckline).any()) if kind == "top"
            else bool((after > neckline).any())
        )
        if not broken:
            continue

        height = abs(second.price - neckline)
        found.append(
            Pattern(
                name=f"double_{kind}",
                direction=(
                    PatternDirection.BEARISH if kind == "top"
                    else PatternDirection.BULLISH
                ),
                # Symmetry of the two peaks drives confidence.
                confidence=min(
                    0.8,
                    0.5 + (1.0 - abs(second.price - first.price) / (atr_value * 0.5)) * 0.25,
                ),
                bar_index=end,
                time=frame.index[end],
                # Measured move: the pattern's own height projected from the break.
                target=neckline - height if kind == "top" else neckline + height,
                invalidation=second.price,
                detail={
                    "neckline": neckline,
                    "height_atr": height / atr_value,
                    "peak_gap_atr": abs(second.price - first.price) / atr_value,
                },
            )
        )
    return found


def _head_and_shoulders(
    frame: pd.DataFrame, swings: Sequence[Swing], atr_value: float, end: int
) -> list[Pattern]:
    """Three extremes with a dominant middle, then a neckline break."""
    found: list[Pattern] = []
    closes = frame["close"].to_numpy(dtype=float)

    for inverse, swing_type in ((False, SwingType.HIGH), (True, SwingType.LOW)):
        extremes = [swing for swing in swings if swing.type is swing_type][-3:]
        if len(extremes) < 3:
            continue
        left_shoulder, head, right_shoulder = extremes

        if not inverse:
            if not (head.price > left_shoulder.price and head.price > right_shoulder.price):
                continue
        else:
            if not (head.price < left_shoulder.price and head.price < right_shoulder.price):
                continue

        # Shoulders should be roughly level; a badly skewed one is a different
        # formation that happens to have three pivots.
        if abs(right_shoulder.price - left_shoulder.price) > atr_value * 1.0:
            continue
        # The head must clear the shoulders decisively.
        if abs(head.price - left_shoulder.price) < atr_value * 0.6:
            continue

        troughs = [
            swing
            for swing in swings
            if swing.type is not swing_type
            and left_shoulder.index < swing.index < right_shoulder.index
        ]
        if len(troughs) < 2:
            continue
        neckline = float(np.mean([swing.price for swing in troughs[-2:]]))

        after = closes[min(right_shoulder.confirmed_index + 1, end + 1) : end + 1]
        if len(after) == 0:
            continue
        broken = (
            bool((after > neckline).any()) if inverse
            else bool((after < neckline).any())
        )
        if not broken:
            continue

        height = abs(head.price - neckline)
        found.append(
            Pattern(
                name="inverse_head_and_shoulders" if inverse else "head_and_shoulders",
                direction=(
                    PatternDirection.BULLISH if inverse else PatternDirection.BEARISH
                ),
                confidence=min(
                    0.82,
                    0.55
                    + (1.0 - abs(right_shoulder.price - left_shoulder.price) / atr_value) * 0.2,
                ),
                bar_index=end,
                time=frame.index[end],
                target=neckline + height if inverse else neckline - height,
                invalidation=head.price,
                detail={
                    "neckline": neckline,
                    "height_atr": height / atr_value,
                    "shoulder_symmetry_atr": abs(
                        right_shoulder.price - left_shoulder.price
                    ) / atr_value,
                },
            )
        )
    return found


def _triangles_and_wedges(
    frame: pd.DataFrame, swings: Sequence[Swing], atr_value: float, end: int
) -> list[Pattern]:
    """Classify converging or parallel swing boundaries.

    Fits a line through the last three highs and the last three lows and reads
    the pair of slopes. Converging slopes are a triangle or wedge; same-signed
    parallel slopes are a channel or flag. Direction comes from the breakout
    where there is one, and from the pattern's own bias where there is not.
    """
    highs = [swing for swing in swings if swing.type is SwingType.HIGH][-3:]
    lows = [swing for swing in swings if swing.type is SwingType.LOW][-3:]
    if len(highs) < 3 or len(lows) < 3:
        return []

    high_slope, high_intercept = _fit_line(
        [swing.index for swing in highs], [swing.price for swing in highs]
    )
    low_slope, low_intercept = _fit_line(
        [swing.index for swing in lows], [swing.price for swing in lows]
    )
    if high_slope is None or low_slope is None:
        return []

    # Normalise slopes to ATR per bar so thresholds are scale-free.
    high_norm = high_slope / atr_value
    low_norm = low_slope / atr_value
    flat = 0.05  # less than 0.05 ATR per bar counts as horizontal

    upper_now = high_slope * end + high_intercept
    lower_now = low_slope * end + low_intercept
    if upper_now <= lower_now:
        return []  # lines already crossed; the formation has resolved

    close = float(frame["close"].iloc[end])
    height = upper_now - lower_now
    broke_up = close > upper_now
    broke_down = close < lower_now

    name: str | None = None
    bias = PatternDirection.NEUTRAL

    if high_norm < -flat and low_norm > flat:
        name, bias = "symmetrical_triangle", PatternDirection.NEUTRAL
    elif abs(high_norm) <= flat and low_norm > flat:
        name, bias = "ascending_triangle", PatternDirection.BULLISH
    elif high_norm < -flat and abs(low_norm) <= flat:
        name, bias = "descending_triangle", PatternDirection.BEARISH
    elif high_norm > flat and low_norm > flat and high_norm < low_norm:
        # Both rising but converging: a rising wedge, which resolves down.
        name, bias = "rising_wedge", PatternDirection.BEARISH
    elif high_norm < -flat and low_norm < -flat and high_norm > low_norm:
        name, bias = "falling_wedge", PatternDirection.BULLISH
    elif high_norm > flat and low_norm > flat:
        name, bias = "ascending_channel", PatternDirection.BULLISH
    elif high_norm < -flat and low_norm < -flat:
        name, bias = "descending_channel", PatternDirection.BEARISH

    if name is None:
        return []

    if broke_up:
        direction = PatternDirection.BULLISH
    elif broke_down:
        direction = PatternDirection.BEARISH
    else:
        direction = bias

    # An unbroken formation is weaker evidence than a broken one, and this is
    # recorded in the confidence rather than by discarding it: a coiling
    # triangle is genuinely useful context for a breakout setup.
    confirmed = broke_up or broke_down
    return [
        Pattern(
            name=name,
            direction=direction,
            confidence=0.65 if confirmed else 0.4,
            bar_index=end,
            time=frame.index[end],
            target=(
                close + height if broke_up else close - height if broke_down else None
            ),
            invalidation=lower_now if broke_up else upper_now if broke_down else None,
            detail={
                "upper": upper_now,
                "lower": lower_now,
                "height_atr": height / atr_value,
                "upper_slope_atr": high_norm,
                "lower_slope_atr": low_norm,
                "confirmed_break": float(confirmed),
            },
        )
    ]


def _fit_line(
    xs: Sequence[int], ys: Sequence[float]
) -> tuple[float | None, float | None]:
    """Least-squares fit. Returns (slope, intercept)."""
    if len(xs) < 2:
        return None, None
    try:
        slope, intercept = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 1)
        return float(slope), float(intercept)
    except (np.linalg.LinAlgError, ValueError):
        return None, None


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Divergence:
    """Price and an oscillator disagreeing.

    Regular divergence warns of exhaustion (price extends, momentum does not).
    Hidden divergence suggests continuation. Both are recorded, because
    labelling every disagreement "reversal" is a classic way to fight a trend.
    """

    indicator: str
    kind: Literal["regular", "hidden"]
    direction: PatternDirection
    strength: float
    from_time: pd.Timestamp
    to_time: pd.Timestamp


def detect_divergence(
    frame: pd.DataFrame,
    *,
    left: int = 3,
    right: int = 3,
    max_swing_gap: int = 40,
    as_of: int | None = None,
) -> list[Divergence]:
    """Compare consecutive confirmed price pivots against RSI and MACD."""
    if len(frame) < 40:
        return []

    end = len(frame) - 1 if as_of is None else as_of
    window = frame.iloc[: end + 1]
    swings = confirmed_swings(find_swings(window, left=left, right=right), end)

    oscillators = {
        "rsi": rsi(window, 14),
        "macd": macd(window)["macd"],
    }

    found: list[Divergence] = []
    for name, series in oscillators.items():
        values = series.to_numpy(dtype=float)
        for swing_type in (SwingType.HIGH, SwingType.LOW):
            pivots = [swing for swing in swings if swing.type is swing_type][-3:]
            for previous, current in zip(pivots, pivots[1:]):
                if current.index - previous.index > max_swing_gap:
                    continue
                before, after = values[previous.index], values[current.index]
                if np.isnan(before) or np.isnan(after):
                    continue

                price_rose = current.price > previous.price
                osc_rose = after > before
                if price_rose == osc_rose:
                    continue  # they agree; no divergence

                # Scale by how far apart the oscillator readings are, so a
                # 2-point RSI difference is not treated like a 20-point one.
                strength = float(
                    min(
                        1.0,
                        abs(after - before)
                        / (20.0 if name == "rsi" else max(abs(before), 1e-9)),
                    )
                )

                if swing_type is SwingType.HIGH and price_rose and not osc_rose:
                    found.append(
                        Divergence(name, "regular", PatternDirection.BEARISH, strength, previous.time, current.time)
                    )
                elif swing_type is SwingType.LOW and not price_rose and osc_rose:
                    found.append(
                        Divergence(name, "regular", PatternDirection.BULLISH, strength, previous.time, current.time)
                    )
                elif swing_type is SwingType.LOW and price_rose and not osc_rose:
                    found.append(
                        Divergence(name, "hidden", PatternDirection.BEARISH, strength, previous.time, current.time)
                    )
                elif swing_type is SwingType.HIGH and not price_rose and osc_rose:
                    found.append(
                        Divergence(name, "hidden", PatternDirection.BULLISH, strength, previous.time, current.time)
                    )
    return found


def _last_valid(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    return float(cleaned.iloc[-1]) if not cleaned.empty else None
