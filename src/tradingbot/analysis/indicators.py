"""Technical indicators, implemented directly on pandas/numpy.

No TA-Lib. Two reasons, and the second is the one that matters:

1. TA-Lib needs a compiled C library, which free hosts cannot build.
2. An indicator whose implementation is a black box cannot be unit-tested
   against a hand-computed expected value. Since this bot's governing rule is
   "code does math, the LLM does judgment", the math has to be inspectable.

Conventions that hold throughout this module:

* Input is a DataFrame indexed by `open_time` with float `open/high/low/close/
  volume` columns, as built by `frame.py`. Floats are correct here: indicators
  feed judgment, not order sizing. Decimal re-enters at the sizing boundary.
* Every function returns a Series aligned to the input index, with NaN for the
  warm-up period. Nothing silently backfills, because a backfilled indicator
  value is lookahead bias wearing a disguise.
* Wilder's smoothing is used where the original indicator specified it (RSI,
  ATR, ADX). The EMA approximation is common in charting libraries and gives
  visibly different values on short series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "adx",
    "atr",
    "bollinger",
    "cci",
    "donchian",
    "ema",
    "keltner",
    "macd",
    "mfi",
    "obv",
    "relative_volume",
    "rsi",
    "sma",
    "stoch_rsi",
    "stochastic",
    "supertrend",
    "true_range",
    "vwap",
    "williams_r",
    "wilder_smooth",
]


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(
            f"Column {column!r} missing. Expected an OHLCV frame, got: "
            f"{list(frame.columns)}"
        )
    return frame[column].astype(float)


# ---------------------------------------------------------------------------
# Moving averages and smoothing
# ---------------------------------------------------------------------------


def sma(values: pd.Series, period: int) -> pd.Series:
    return values.rolling(window=period, min_periods=period).mean()


def ema(values: pd.Series, period: int) -> pd.Series:
    """Exponential moving average.

    `adjust=False` gives the recursive form every charting platform uses. With
    `adjust=True` pandas computes a different (and for our purposes wrong)
    weighting that will not match what you see on a chart.
    """
    return values.ewm(span=period, adjust=False, min_periods=period).mean()


def wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, as used by the original RSI/ATR/ADX definitions.

    Equivalent to an EMA with alpha = 1/period, which is a *different* curve
    from an EMA with span=period. Conflating the two is the single most common
    reason a hand-rolled RSI disagrees with TradingView.
    """
    return values.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def true_range(frame: pd.DataFrame) -> pd.Series:
    """True range: the greater of the bar's own span and its gap from the last close."""
    high = _series(frame, "high")
    low = _series(frame, "low")
    prev_close = _series(frame, "close").shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range. Drives every stop distance in `risk.yaml`."""
    return wilder_smooth(true_range(frame), period)


def atr_pct(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as a percentage of price.

    The comparable form. A 500-point ATR means nothing until you know whether
    the instrument trades at 600 or 60,000, and the risk rules apply across
    symbols whose prices differ by six orders of magnitude.
    """
    return atr(frame, period) / _series(frame, "close") * 100.0


def bollinger(
    frame: pd.DataFrame, period: int = 20, std_mult: float = 2.0
) -> pd.DataFrame:
    """Bollinger bands plus bandwidth and %B.

    Bandwidth is the useful column: a multi-month low in bandwidth ("the
    squeeze") precedes expansion, which is the regime in which breakout setups
    actually work.
    """
    close = _series(frame, "close")
    middle = sma(close, period)
    # ddof=0 matches the population standard deviation the indicator specifies.
    deviation = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + deviation * std_mult
    lower = middle - deviation * std_mult
    width = upper - lower
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": (width / middle.replace(0, np.nan)) * 100.0,
            # Where price sits in the band: 0 = lower, 1 = upper.
            "bb_percent_b": (close - lower) / width.replace(0, np.nan),
        },
        index=frame.index,
    )


def keltner(
    frame: pd.DataFrame, period: int = 20, atr_period: int = 10, mult: float = 2.0
) -> pd.DataFrame:
    """Keltner channels. ATR-based, so unlike Bollinger they ignore outliers."""
    middle = ema(_series(frame, "close"), period)
    band = atr(frame, atr_period) * mult
    return pd.DataFrame(
        {"kc_middle": middle, "kc_upper": middle + band, "kc_lower": middle - band},
        index=frame.index,
    )


def donchian(frame: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Rolling extremes.

    `shift(1)` is essential: the channel must describe the *prior* N bars. An
    unshifted Donchian high includes the current bar, so "price broke the high"
    becomes true by construction and every breakout backtest looks brilliant.
    """
    high = _series(frame, "high").rolling(period, min_periods=period).max().shift(1)
    low = _series(frame, "low").rolling(period, min_periods=period).min().shift(1)
    return pd.DataFrame(
        {"dc_upper": high, "dc_lower": low, "dc_middle": (high + low) / 2},
        index=frame.index,
    )


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def rsi(frame: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """Wilder's RSI."""
    delta = _series(frame, column).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    # A zero average loss is a genuine 100 reading, not a division error.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(
    frame: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    close = _series(frame, "close")
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line.dropna(), signal).reindex(frame.index)
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        },
        index=frame.index,
    )


def stochastic(
    frame: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> pd.DataFrame:
    high = _series(frame, "high").rolling(period, min_periods=period).max()
    low = _series(frame, "low").rolling(period, min_periods=period).min()
    span = (high - low).replace(0, np.nan)
    raw_k = (_series(frame, "close") - low) / span * 100.0
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    return pd.DataFrame(
        {"stoch_k": k, "stoch_d": k.rolling(smooth_d, min_periods=smooth_d).mean()},
        index=frame.index,
    )


def stoch_rsi(
    frame: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14, smooth: int = 3
) -> pd.DataFrame:
    """Stochastic RSI: where RSI sits inside its own recent range.

    Reaches its extremes far more often than RSI does, which makes it useful
    for timing an entry inside an already-established trend and useless as a
    standalone reversal signal.
    """
    base = rsi(frame, rsi_period)
    low = base.rolling(stoch_period, min_periods=stoch_period).min()
    high = base.rolling(stoch_period, min_periods=stoch_period).max()
    raw = (base - low) / (high - low).replace(0, np.nan) * 100.0
    k = raw.rolling(smooth, min_periods=smooth).mean()
    return pd.DataFrame(
        {"stochrsi_k": k, "stochrsi_d": k.rolling(smooth, min_periods=smooth).mean()},
        index=frame.index,
    )


def williams_r(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high = _series(frame, "high").rolling(period, min_periods=period).max()
    low = _series(frame, "low").rolling(period, min_periods=period).min()
    return (high - _series(frame, "close")) / (high - low).replace(0, np.nan) * -100.0


def cci(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (
        _series(frame, "high") + _series(frame, "low") + _series(frame, "close")
    ) / 3.0
    moving = typical.rolling(period, min_periods=period).mean()
    # Mean absolute deviation, which is what the original CCI specifies - not
    # standard deviation.
    deviation = typical.rolling(period, min_periods=period).apply(
        lambda window: np.abs(window - window.mean()).mean(), raw=True
    )
    return (typical - moving) / (0.015 * deviation.replace(0, np.nan))


# ---------------------------------------------------------------------------
# Trend strength and direction
# ---------------------------------------------------------------------------


def adx(frame: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX with its directional components.

    ADX measures how *organised* a move is, not which way it points. It is used
    here as a gate: trend-following setups require a rising ADX above a floor,
    and mean-reversion setups require the opposite, which stops the bot
    applying breakout logic to a chopping range.
    """
    high = _series(frame, "high")
    low = _series(frame, "low")

    up_move = high.diff()
    down_move = -low.diff()
    # Only the larger of the two counts, and only when positive. A bar that is
    # an inside bar contributes no directional movement at all.
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )

    atr_values = wilder_smooth(true_range(frame), period).replace(0, np.nan)
    plus_di = 100.0 * wilder_smooth(plus_dm, period) / atr_values
    minus_di = 100.0 * wilder_smooth(minus_dm, period) / atr_values

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return pd.DataFrame(
        {
            "adx": wilder_smooth(dx, period),
            "plus_di": plus_di,
            "minus_di": minus_di,
        },
        index=frame.index,
    )


def supertrend(
    frame: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """SuperTrend: an ATR band that only ever ratchets in the trend's favour.

    Implemented as an explicit loop because each bar's band depends on the
    previous bar's band and direction. That recurrence cannot be vectorised,
    and the common "vectorised SuperTrend" implementations get the flip
    condition wrong in exactly the cases that matter.
    """
    close = _series(frame, "close").to_numpy()
    hl2 = ((_series(frame, "high") + _series(frame, "low")) / 2.0).to_numpy()
    band = (atr(frame, period) * multiplier).to_numpy()

    size = len(frame)
    upper = np.full(size, np.nan)
    lower = np.full(size, np.nan)
    trend = np.full(size, np.nan)

    for i in range(size):
        if np.isnan(band[i]):
            continue
        basic_upper = hl2[i] + band[i]
        basic_lower = hl2[i] - band[i]

        if i == 0 or np.isnan(trend[i - 1]):
            upper[i], lower[i], trend[i] = basic_upper, basic_lower, 1.0
            continue

        # The band tightens toward price but never loosens away from it while
        # the trend holds. That ratchet is the whole indicator.
        upper[i] = (
            min(basic_upper, upper[i - 1])
            if close[i - 1] <= upper[i - 1]
            else basic_upper
        )
        lower[i] = (
            max(basic_lower, lower[i - 1])
            if close[i - 1] >= lower[i - 1]
            else basic_lower
        )

        if close[i] > upper[i - 1]:
            trend[i] = 1.0
        elif close[i] < lower[i - 1]:
            trend[i] = -1.0
        else:
            trend[i] = trend[i - 1]

    return pd.DataFrame(
        {
            "st_trend": trend,
            "st_upper": upper,
            "st_lower": lower,
            "st_line": np.where(trend > 0, lower, upper),
        },
        index=frame.index,
    )


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def obv(frame: pd.DataFrame) -> pd.Series:
    """On-balance volume: signed cumulative volume."""
    close = _series(frame, "close")
    volume = _series(frame, "volume")
    return (np.sign(close.diff()).fillna(0.0) * volume).cumsum()


def mfi(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money flow index: RSI weighted by volume.

    Divergence between MFI and price is more informative than RSI divergence
    on thin books, because it distinguishes a move that carried size from one
    that drifted on no volume - exactly the GOATUSDT failure mode.
    """
    typical = (
        _series(frame, "high") + _series(frame, "low") + _series(frame, "close")
    ) / 3.0
    flow = typical * _series(frame, "volume")
    direction = typical.diff()
    positive = flow.where(direction > 0, 0.0).rolling(period, min_periods=period).sum()
    negative = flow.where(direction < 0, 0.0).rolling(period, min_periods=period).sum()
    ratio = positive / negative.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + ratio))).where(negative != 0, 100.0)


def vwap(frame: pd.DataFrame, period: int | None = None) -> pd.Series:
    """Volume-weighted average price.

    With `period`, a rolling VWAP. Without, a cumulative one over the whole
    frame - only meaningful if the frame starts at a session boundary, which
    for 24/7 crypto perpetuals it generally does not. Prefer the rolling form.
    """
    typical = (
        _series(frame, "high") + _series(frame, "low") + _series(frame, "close")
    ) / 3.0
    volume = _series(frame, "volume")
    weighted = typical * volume
    if period is None:
        return weighted.cumsum() / volume.cumsum().replace(0, np.nan)
    return (
        weighted.rolling(period, min_periods=period).sum()
        / volume.rolling(period, min_periods=period).sum().replace(0, np.nan)
    )


def relative_volume(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume as a multiple of its recent average.

    `shift(1)` on the baseline so the current bar is compared against history
    that excludes itself. Including it drags the average toward the value being
    tested and systematically understates genuine volume spikes.
    """
    volume = _series(frame, "volume")
    baseline = volume.rolling(period, min_periods=period).mean().shift(1)
    return volume / baseline.replace(0, np.nan)
