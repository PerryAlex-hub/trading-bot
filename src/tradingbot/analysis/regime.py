"""Market regime: the context that decides whether a setup is worth taking.

A long on a satellite token during a BTC breakdown is a bad trade no matter how
clean its chart looks. `universe.yaml` already treats BTC and ETH as context
feeds for exactly this reason; this module turns those feeds into a verdict.

The core measurement is **realised beta to BTC**, computed from log returns
rather than read from the static table in `risk.yaml`. That table is a sensible
prior, but beta moves: a token can trade at 1.2x BTC for a month and 3x during
a liquidation cascade. The risk gate uses the static table as a floor and the
measured value when there is enough data, so the portfolio cap cannot be
defeated by a stale constant.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ..models import Trend
from .indicators import atr_pct, ema, rsi
from .structure import classify_structure

log = logging.getLogger(__name__)


class RegimeState(str, Enum):
    RISK_ON = "risk_on"          # BTC trending up, alts can follow
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"        # BTC breaking down; satellites are dangerous
    VOLATILE = "volatile"        # direction unclear, volatility elevated

    @property
    def allows_satellite_longs(self) -> bool:
        """Satellite longs need a benign backdrop.

        Gating this is the single highest-value regime rule for a small
        account: thin-book altcoins fall hardest when BTC turns, and that is
        where a 0.5%-risk trade becomes a 3% loss through slippage.
        """
        return self is RegimeState.RISK_ON

    @property
    def size_multiplier(self) -> float:
        """How much of normal size this regime justifies."""
        return {
            "risk_on": 1.0,
            "neutral": 0.75,
            "volatile": 0.5,
            "risk_off": 0.5,
        }[self.value]


@dataclass(frozen=True)
class BetaEstimate:
    """Realised beta of a symbol to BTC."""

    symbol: str
    beta: float
    correlation: float
    samples: int

    @property
    def is_reliable(self) -> bool:
        """Beta from a short or uncorrelated series is not meaningful.

        Below ~0.3 correlation the regression slope is mostly noise, and using
        it in a portfolio-exposure calculation would produce confident
        nonsense.
        """
        return self.samples >= 60 and abs(self.correlation) >= 0.3


@dataclass(frozen=True)
class MarketRegime:
    """The market backdrop every signal is judged against."""

    state: RegimeState
    btc_trend: Trend
    btc_rsi: float | None
    btc_atr_pct: float | None
    # ETH/BTC rising means capital is rotating out along the risk curve, which
    # is the classic precondition for altcoins working at all.
    eth_btc_trend: Trend | None
    btc_above_200ema: bool | None
    # Fraction of the universe's alts outperforming BTC over the lookback.
    # This is the project's answer to BTC dominance - see `alt_breadth`.
    alt_breadth: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def allows_satellite_longs(self) -> bool:
        """Whether a thin-book long is defensible right now.

        Supersedes `RegimeState.allows_satellite_longs`, which looked only at
        BTC's own trend. BTC can be trending up while capital hides in BTC, and
        that is the backdrop where an alt long looks reasonable on its own
        chart and still bleeds. Both measures here answer "is risk appetite
        actually travelling out along the curve", which is what `universe.yaml`
        wanted BTC dominance for.
        """
        if not self.state.allows_satellite_longs:
            return False
        # ETH/BTC strongly down: BTC is absorbing the flows, so alts lag even
        # in an up-trend. A merely weak ratio is not enough to block.
        if self.eth_btc_trend is not None and self.eth_btc_trend is Trend.STRONG_DOWN:
            return False
        return not (self.alt_breadth is not None and self.alt_breadth < 0.25)

    def describe(self) -> str:
        return (
            f"{self.state.value} (BTC {self.btc_trend.value}, "
            f"RSI {self.btc_rsi:.0f}, ATR {self.btc_atr_pct:.2f}%)"
            if self.btc_rsi is not None and self.btc_atr_pct is not None
            else f"{self.state.value} (BTC {self.btc_trend.value})"
        )


def alt_breadth(
    alt_frames: Mapping[str, pd.DataFrame],
    btc_frame: pd.DataFrame,
    *,
    lookback: int = 30,
) -> float | None:
    """Fraction of the universe's alts outperforming BTC over `lookback` bars.

    **This is deliberately not literal BTC dominance.** `universe.yaml` listed
    `BTC.D` as a context feed and nothing ever read it. Implementing the real
    thing means BTC market cap over total market cap, which needs circulating
    supplies from a market-data provider - and that route was rejected for a
    specific reason: such an API serves the CURRENT value, with no history we
    store. Every backfilled bar and every future backtest would then be scored
    against today's dominance, which is lookahead bias, and the whole `as_of`
    discipline elsewhere in this codebase exists to avoid exactly that.

    Breadth answers the same question - is risk appetite travelling out along
    the curve, or hiding in BTC - from candles already in the database, so it
    is computable for any past bar and survives a backtest.

    Returns None with fewer than two comparable alts, since a "fraction" over
    one symbol is just that symbol.
    """
    btc_return = _return_over(btc_frame, lookback)
    if btc_return is None:
        return None

    outcomes: list[float] = []
    for frame in alt_frames.values():
        alt_return = _return_over(frame, lookback)
        if alt_return is not None:
            outcomes.append(1.0 if alt_return > btc_return else 0.0)

    if len(outcomes) < 2:
        return None
    return sum(outcomes) / len(outcomes)


def _return_over(frame: pd.DataFrame, lookback: int) -> float | None:
    closes = frame["close"].dropna() if "close" in frame else pd.Series(dtype=float)
    if len(closes) < lookback + 1:
        return None
    start = float(closes.iloc[-1 - lookback])
    if start <= 0:
        return None
    return float(closes.iloc[-1]) / start - 1.0


def assess_regime(
    btc_frame: pd.DataFrame,
    *,
    eth_frame: pd.DataFrame | None = None,
    alt_frames: Mapping[str, pd.DataFrame] | None = None,
    high_volatility_atr_pct: float = 4.0,
) -> MarketRegime:
    """Classify the backdrop from BTC, with ETH/BTC and breadth as secondary reads."""
    if btc_frame.empty or len(btc_frame) < 50:
        return MarketRegime(
            state=RegimeState.NEUTRAL,
            btc_trend=Trend.RANGE,
            btc_rsi=None,
            btc_atr_pct=None,
            eth_btc_trend=None,
            btc_above_200ema=None,
            notes=("insufficient BTC history to classify regime",),
        )

    structure = classify_structure(btc_frame)
    btc_rsi = _last(rsi(btc_frame, 14))
    btc_atr = _last(atr_pct(btc_frame, 14))

    above_200 = None
    if len(btc_frame) >= 200:
        ema_200 = _last(ema(btc_frame["close"], 200))
        if ema_200 is not None:
            above_200 = float(btc_frame["close"].iloc[-1]) > ema_200

    eth_btc_trend: Trend | None = None
    if eth_frame is not None and not eth_frame.empty:
        eth_btc_trend = _ratio_trend(eth_frame, btc_frame)

    notes: list[str] = []
    volatile = btc_atr is not None and btc_atr >= high_volatility_atr_pct
    if volatile:
        notes.append(f"BTC ATR {btc_atr:.2f}% is elevated; stops need more room")

    # Order matters: volatility is checked before direction, because a strong
    # trend in a violently volatile market still stops out ordinary-sized
    # positions before it pays.
    if volatile and structure.trend in (Trend.RANGE, Trend.UP, Trend.DOWN):
        state = RegimeState.VOLATILE
    elif structure.trend in (Trend.STRONG_DOWN, Trend.DOWN) and above_200 is False:
        state = RegimeState.RISK_OFF
        notes.append("BTC below its 200 EMA and trending down")
    elif structure.trend in (Trend.STRONG_UP, Trend.UP) and above_200 is not False:
        state = RegimeState.RISK_ON
    elif structure.trend is Trend.STRONG_DOWN:
        state = RegimeState.RISK_OFF
    else:
        state = RegimeState.NEUTRAL

    if btc_rsi is not None and btc_rsi > 78:
        notes.append(f"BTC RSI {btc_rsi:.0f} is stretched; chasing longs is late")
    if btc_rsi is not None and btc_rsi < 22:
        notes.append(f"BTC RSI {btc_rsi:.0f} is deeply oversold; shorts are late")
    if eth_btc_trend is not None and eth_btc_trend.is_down and state is RegimeState.RISK_ON:
        notes.append("ETH/BTC falling: BTC is leading, so alt longs underperform")

    breadth = alt_breadth(alt_frames, btc_frame) if alt_frames else None
    if breadth is not None:
        percent = breadth * 100
        if breadth < 0.25:
            notes.append(
                f"only {percent:.0f}% of alts are outperforming BTC: capital is "
                "concentrating in BTC, so satellite longs are blocked"
            )
        elif breadth >= 0.75:
            notes.append(f"{percent:.0f}% of alts are outperforming BTC: alts are bid")

    return MarketRegime(
        state=state,
        btc_trend=structure.trend,
        btc_rsi=btc_rsi,
        btc_atr_pct=btc_atr,
        eth_btc_trend=eth_btc_trend,
        btc_above_200ema=above_200,
        alt_breadth=breadth,
        notes=tuple(notes),
    )


def _ratio_trend(numerator: pd.DataFrame, denominator: pd.DataFrame) -> Trend | None:
    """Trend of one series divided by another (e.g. ETH/BTC).

    Joined on the index rather than zipped: the two frames can have different
    lengths or missing bars, and positionally pairing them would silently
    compare different points in time.
    """
    joined = pd.DataFrame(
        {"num": numerator["close"], "den": denominator["close"]}
    ).dropna()
    if len(joined) < 50:
        return None

    ratio = joined["num"] / joined["den"]
    synthetic = pd.DataFrame(
        {
            "open": ratio,
            "high": ratio,
            "low": ratio,
            "close": ratio,
            "volume": 1.0,
        },
        index=joined.index,
    )
    return classify_structure(synthetic).trend


def estimate_beta(
    symbol_frame: pd.DataFrame,
    btc_frame: pd.DataFrame,
    *,
    symbol: str = "",
    lookback: int = 120,
) -> BetaEstimate:
    """Realised beta to BTC from aligned log returns.

    Log returns, not percentage changes: they are additive across time and
    symmetric, which matters when a satellite token can move -40%/+60% in a
    week. Beta is cov/var of the aligned series.
    """
    joined = pd.DataFrame(
        {"symbol": symbol_frame["close"], "btc": btc_frame["close"]}
    ).dropna()
    if len(joined) < 30:
        return BetaEstimate(symbol, beta=float("nan"), correlation=0.0, samples=len(joined))

    joined = joined.tail(lookback)
    returns = np.log(joined / joined.shift(1)).dropna()
    if len(returns) < 20:
        return BetaEstimate(symbol, float("nan"), 0.0, len(returns))

    symbol_returns = returns["symbol"].to_numpy()
    btc_returns = returns["btc"].to_numpy()
    btc_variance = float(np.var(btc_returns, ddof=1))
    if btc_variance <= 0:
        return BetaEstimate(symbol, float("nan"), 0.0, len(returns))

    covariance = float(np.cov(symbol_returns, btc_returns, ddof=1)[0, 1])
    correlation = float(np.corrcoef(symbol_returns, btc_returns)[0, 1])
    return BetaEstimate(
        symbol=symbol,
        beta=covariance / btc_variance,
        correlation=0.0 if np.isnan(correlation) else correlation,
        samples=len(returns),
    )


def effective_beta(
    measured: BetaEstimate | None, configured: float
) -> tuple[float, str]:
    """Reconcile measured beta with the configured prior.

    Takes the MAXIMUM of the two when the measurement is reliable. Deliberately
    conservative: this number feeds the portfolio exposure cap, and the failure
    mode being guarded against is a book that looks diversified but is three
    correlated BTC bets wearing different tickers. Underestimating beta there
    is how a 1.5% risk budget becomes a 5% loss in one candle.
    """
    if measured is None or not measured.is_reliable or np.isnan(measured.beta):
        return configured, "configured (measurement unreliable)"
    if measured.beta > configured:
        return measured.beta, f"measured {measured.beta:.2f} exceeds configured {configured:.2f}"
    return configured, f"configured {configured:.2f} is the conservative floor"


def _last(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    return float(cleaned.iloc[-1]) if not cleaned.empty else None
