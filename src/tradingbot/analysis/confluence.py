"""The confluence engine: many pieces of evidence folded into one score.

This is the deterministic heart of the bot. Everything upstream measures; this
module decides. It is also the Phase 1 baseline that the LLM layer has to beat,
so it is built to stand alone rather than to be a feature extractor.

**How the score works.** Each check returns a signed contribution in [-1, 1]
(positive = bullish) and carries a weight. The total is the weighted mean, so
the result is always in [-1, 1] and adding a new check cannot inflate the
scale. Every contribution is recorded with its reason, which is what makes a
signal auditable six months later instead of being an unexplained number.

**Why weights, not rules.** A rule chain ("if trend up AND rsi<70 AND...")
fails the moment one condition is marginally missed, and it cannot express that
four weak confirmations outweigh one strong objection. The weights below are
priors, deliberately not fitted to history: tuning them on the same data used
to evaluate them is how a backtest becomes fiction.

**The non-negotiables.** Some conditions are not weighted evidence but hard
blocks - `blockers` - for example a long into a resistance level a few ticks
overhead. A high score cannot buy its way past one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..models import Side, Trend
from .indicators import (
    atr,
    atr_pct,
    bollinger,
    macd,
    mfi,
    relative_volume,
    rsi,
    stoch_rsi,
    supertrend,
)
from .levels import LevelMap
from .patterns import Divergence, Pattern, PatternDirection
from .regime import MarketRegime, RegimeState
from .structure import StructureState

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Evidence:
    """One scored observation."""

    name: str
    # Signed: positive is bullish, negative bearish. Always in [-1, 1].
    contribution: float
    weight: float
    reason: str

    @property
    def weighted(self) -> float:
        return self.contribution * self.weight


@dataclass(frozen=True)
class ConfluenceResult:
    """The verdict for one symbol on one bar."""

    # Weighted mean of all evidence, in [-1, 1].
    score: float
    side: Side
    evidence: tuple[Evidence, ...]
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    timeframe_alignment: float = 0.0

    @property
    def is_blocked(self) -> bool:
        return bool(self.blockers)

    @property
    def conviction(self) -> float:
        """Absolute strength, 0..1, ignoring direction."""
        return abs(self.score)

    def qualifies(self, threshold: float = 0.25) -> bool:
        """Whether this is worth proposing a trade on at all.

        A blocker overrides the score completely. That asymmetry is
        intentional: the cost of skipping a good trade is an opportunity, and
        the cost of taking a blocked one is real money.
        """
        return (
            not self.is_blocked
            and self.side is not Side.FLAT
            and self.conviction >= threshold
        )

    def top_reasons(self, count: int = 5) -> tuple[str, ...]:
        """The evidence that actually drove the decision, strongest first."""
        ranked = sorted(
            self.evidence, key=lambda item: abs(item.weighted), reverse=True
        )
        return tuple(
            f"{item.reason} ({item.weighted:+.2f})"
            for item in ranked[:count]
            if abs(item.weighted) > 0.01
        )

    def to_dict(self) -> dict[str, object]:
        """Serialised for the journal's `features` column."""
        return {
            "score": round(self.score, 4),
            "side": self.side.value,
            "conviction": round(self.conviction, 4),
            "timeframe_alignment": round(self.timeframe_alignment, 4),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "evidence": [
                {
                    "name": item.name,
                    "contribution": round(item.contribution, 4),
                    "weight": item.weight,
                    "weighted": round(item.weighted, 4),
                    "reason": item.reason,
                }
                for item in self.evidence
            ],
        }


# Priors, not fitted parameters. Structure and trend dominate because on a 4h
# timeframe they are the most persistent signals; oscillators are timing tools
# and are weighted accordingly.
WEIGHTS: dict[str, float] = {
    "structure": 2.0,
    "trend_alignment": 1.8,
    "higher_timeframe": 1.6,
    "regime": 1.5,
    "chart_pattern": 1.3,
    "level_proximity": 1.2,
    "momentum": 1.0,
    "supertrend": 1.0,
    "divergence": 0.9,
    "volume": 0.8,
    "candlestick": 0.7,
    "volatility": 0.6,
    "mean_reversion": 0.6,
}


@dataclass
class ConfluenceEngine:
    """Scores one symbol from its pre-computed analysis.

    Takes already-computed inputs rather than raw candles so that the same
    engine serves the live pipeline and the backtester with no divergence
    between them. Two code paths computing "the same" features is how a
    backtest and live trading quietly stop agreeing.
    """

    min_score: float = 0.25
    # A level this close overhead blocks a long outright, measured in ATR so it
    # scales across the universe.
    level_block_atr: float = 0.5

    def evaluate(
        self,
        frame: pd.DataFrame,
        *,
        structure: StructureState,
        levels: LevelMap,
        regime: MarketRegime,
        chart_patterns: tuple[Pattern, ...] = (),
        candle_patterns: tuple[Pattern, ...] = (),
        divergences: tuple[Divergence, ...] = (),
        higher_structure: StructureState | None = None,
        lower_structure: StructureState | None = None,
        is_satellite: bool = False,
    ) -> ConfluenceResult:
        if frame.empty or len(frame) < 30:
            return ConfluenceResult(
                score=0.0,
                side=Side.FLAT,
                evidence=(),
                blockers=("insufficient history to analyse",),
            )

        evidence: list[Evidence] = []
        warnings: list[str] = []
        price = float(frame["close"].iloc[-1])
        atr_value = _last(atr(frame, 14)) or 0.0

        evidence.append(self._score_structure(structure))
        evidence.append(self._score_trend(frame, structure))
        evidence.extend(self._score_momentum(frame))
        evidence.append(self._score_supertrend(frame))
        evidence.append(self._score_volume(frame))
        evidence.append(self._score_volatility(frame, warnings))

        alignment = 0.0
        if higher_structure is not None:
            htf = self._score_higher_timeframe(structure, higher_structure)
            evidence.append(htf)
            alignment = htf.contribution

        if chart_patterns:
            evidence.append(self._score_patterns(chart_patterns, "chart_pattern"))
        if candle_patterns:
            evidence.append(self._score_patterns(candle_patterns, "candlestick"))
        if divergences:
            evidence.append(self._score_divergence(divergences))

        evidence.append(self._score_levels(levels, price, atr_value))
        evidence.append(self._score_regime(regime, is_satellite))
        evidence.append(self._score_mean_reversion(frame, structure))

        # Weighted mean, so the scale is independent of how many checks ran.
        scored = [item for item in evidence if item.weight > 0]
        total_weight = sum(item.weight for item in scored)
        score = (
            sum(item.weighted for item in scored) / total_weight
            if total_weight > 0
            else 0.0
        )

        side = (
            Side.LONG if score >= self.min_score
            else Side.SHORT if score <= -self.min_score
            else Side.FLAT
        )

        blockers = self._find_blockers(
            side=side,
            price=price,
            atr_value=atr_value,
            levels=levels,
            regime=regime,
            structure=structure,
            is_satellite=is_satellite,
            frame=frame,
        )

        return ConfluenceResult(
            score=score,
            side=side,
            evidence=tuple(evidence),
            blockers=tuple(blockers),
            warnings=tuple(warnings),
            timeframe_alignment=alignment,
        )

    # -- individual checks -------------------------------------------------

    def _score_structure(self, structure: StructureState) -> Evidence:
        """Swing sequence: the most direct read of who is in control."""
        bullish = structure.higher_highs + structure.higher_lows
        bearish = structure.lower_highs + structure.lower_lows
        total = bullish + bearish
        if total == 0:
            return Evidence("structure", 0.0, WEIGHTS["structure"], "no clear swing structure")

        contribution = (bullish - bearish) / total
        parts = [f"HH{structure.higher_highs}/HL{structure.higher_lows}"
                 f" vs LH{structure.lower_highs}/LL{structure.lower_lows}"]

        # A change of character is the earliest evidence a trend is over, so it
        # pulls the score toward the new direction rather than merely reducing
        # confidence in the old one.
        if structure.last_choch == "bearish":
            contribution -= 0.4
            parts.append("bearish CHoCH")
        elif structure.last_choch == "bullish":
            contribution += 0.4
            parts.append("bullish CHoCH")
        if structure.last_bos == "bullish":
            contribution += 0.2
            parts.append("bullish BOS")
        elif structure.last_bos == "bearish":
            contribution -= 0.2
            parts.append("bearish BOS")

        return Evidence(
            "structure",
            _clamp(contribution),
            WEIGHTS["structure"],
            "structure: " + ", ".join(parts),
        )

    def _score_trend(self, frame: pd.DataFrame, structure: StructureState) -> Evidence:
        """Trend label scaled by how organised the move is.

        ADX acts as a multiplier rather than a separate vote: direction without
        organisation is a range, and in a range a trend-following score should
        be small regardless of which way the EMAs happen to point.
        """
        direction = structure.trend.direction / 2.0  # Trend.direction is -2..2
        adx_value = structure.adx_value
        if adx_value is None:
            organisation = 0.5
            note = "ADX unavailable"
        else:
            # 15 ADX -> ~0.19, 25 -> 0.5, 40+ -> 1.0
            organisation = _clamp((adx_value - 10.0) / 30.0, low=0.0)
            note = f"ADX {adx_value:.0f}"

        return Evidence(
            "trend_alignment",
            _clamp(direction * organisation),
            WEIGHTS["trend_alignment"],
            f"trend {structure.trend.value} ({note}, EMA stack {structure.ema_stack})",
        )

    def _score_momentum(self, frame: pd.DataFrame) -> list[Evidence]:
        """RSI, MACD and StochRSI as one combined momentum vote."""
        rsi_value = _last(rsi(frame, 14))
        macd_frame = macd(frame)
        hist = _last(macd_frame["macd_hist"])
        hist_previous = _nth_last(macd_frame["macd_hist"], 2)
        stoch = _last(stoch_rsi(frame)["stochrsi_k"])

        parts: list[str] = []
        votes: list[float] = []

        if rsi_value is not None:
            # Centred on 50 and scaled so 70/30 reach about +/-0.67. Not
            # treated as overbought/oversold here: in a trend RSI stays
            # extended, and fading that is the classic way to lose money
            # fighting a trend. Exhaustion is handled in mean_reversion.
            votes.append(_clamp((rsi_value - 50.0) / 30.0))
            parts.append(f"RSI {rsi_value:.0f}")

        if hist is not None:
            votes.append(_clamp(hist / (abs(hist) + 1e-9) * 0.5))
            direction = "rising" if (hist_previous is not None and hist > hist_previous) else "falling"
            if hist_previous is not None:
                # Acceleration matters more than level: a shrinking positive
                # histogram is momentum fading, not momentum present.
                votes.append(0.3 if hist > hist_previous else -0.3)
            parts.append(f"MACD hist {hist:+.4f} {direction}")

        if stoch is not None:
            votes.append(_clamp((stoch - 50.0) / 50.0) * 0.5)
            parts.append(f"StochRSI {stoch:.0f}")

        contribution = sum(votes) / len(votes) if votes else 0.0
        return [
            Evidence(
                "momentum",
                _clamp(contribution),
                WEIGHTS["momentum"],
                "momentum: " + ", ".join(parts) if parts else "momentum unavailable",
            )
        ]

    def _score_supertrend(self, frame: pd.DataFrame) -> Evidence:
        trend = _last(supertrend(frame)["st_trend"])
        if trend is None:
            return Evidence("supertrend", 0.0, WEIGHTS["supertrend"], "SuperTrend unavailable")
        return Evidence(
            "supertrend",
            1.0 if trend > 0 else -1.0,
            WEIGHTS["supertrend"],
            f"SuperTrend {'bullish' if trend > 0 else 'bearish'}",
        )

    def _score_volume(self, frame: pd.DataFrame) -> Evidence:
        """Volume confirmation, signed by the direction of the move it confirms.

        Volume is directionless on its own: a spike confirms whatever just
        happened. So the magnitude comes from relative volume and the sign from
        the bar's own close.
        """
        rvol = _last(relative_volume(frame, 20))
        money_flow = _last(mfi(frame, 14))
        if rvol is None:
            return Evidence("volume", 0.0, WEIGHTS["volume"], "volume baseline unavailable")

        close = float(frame["close"].iloc[-1])
        open_ = float(frame["open"].iloc[-1])
        sign = 1.0 if close > open_ else -1.0 if close < open_ else 0.0
        magnitude = _clamp((rvol - 1.0) / 1.5, low=0.0)

        parts = [f"rVol {rvol:.2f}x"]
        contribution = sign * magnitude
        if money_flow is not None:
            contribution = (contribution + _clamp((money_flow - 50.0) / 40.0)) / 2
            parts.append(f"MFI {money_flow:.0f}")

        return Evidence(
            "volume", _clamp(contribution), WEIGHTS["volume"], "volume: " + ", ".join(parts)
        )

    def _score_volatility(
        self, frame: pd.DataFrame, warnings: list[str]
    ) -> Evidence:
        """Volatility is directionless, so this scores *tradeability*.

        Returns 0 contribution deliberately - it exists to raise warnings and
        to detect the Bollinger squeeze, which is a precondition for breakouts
        rather than a direction.
        """
        atr_percentage = _last(atr_pct(frame, 14))
        bands = bollinger(frame, 20)
        bandwidth = _last(bands["bb_bandwidth"])

        if atr_percentage is not None and atr_percentage > 8.0:
            warnings.append(
                f"ATR {atr_percentage:.1f}% of price: stops must be wide, so size will be small"
            )
        if atr_percentage is not None and atr_percentage < 0.5:
            warnings.append(
                f"ATR {atr_percentage:.2f}% is unusually tight; expect expansion"
            )

        reason = f"ATR {atr_percentage:.2f}% of price" if atr_percentage else "volatility unknown"
        if bandwidth is not None:
            history = bands["bb_bandwidth"].dropna()
            if len(history) >= 50:
                percentile = float((history.tail(100) < bandwidth).mean())
                if percentile < 0.15:
                    reason += f", Bollinger squeeze (bandwidth in bottom {percentile*100:.0f}%)"
                    warnings.append("Bollinger squeeze: expansion likely, direction unproven")

        return Evidence("volatility", 0.0, WEIGHTS["volatility"], reason)

    def _score_higher_timeframe(
        self, primary: StructureState, higher: StructureState
    ) -> Evidence:
        """Agreement between the decision timeframe and the one above it.

        The highest-value filter in the whole engine. Taking a 4h long against
        a daily downtrend is the most reliable way to lose money with a
        technically correct setup.
        """
        primary_direction = primary.trend.direction
        higher_direction = higher.trend.direction

        if higher_direction == 0:
            return Evidence(
                "higher_timeframe", 0.0, WEIGHTS["higher_timeframe"],
                "higher timeframe is ranging: no directional support",
            )

        # The higher timeframe is the authority, so the contribution simply
        # points in its direction with a magnitude set by its strength. There
        # is no separate "they disagree" penalty: a 4h uptrend under a daily
        # downtrend already gets a negative contribution here, which is exactly
        # the correction intended. Agreement only affects the explanation.
        agree = (primary_direction > 0) == (higher_direction > 0)
        magnitude = min(1.0, abs(higher_direction) / 2.0)
        contribution = magnitude * (1 if higher_direction > 0 else -1)

        return Evidence(
            "higher_timeframe",
            _clamp(contribution),
            WEIGHTS["higher_timeframe"],
            f"higher timeframe {higher.trend.value}"
            + ("" if agree else " (conflicts with primary)"),
        )

    def _score_patterns(
        self, patterns: tuple[Pattern, ...], name: str
    ) -> Evidence:
        """Fold detected patterns into one vote, strongest pattern dominating."""
        actionable = [p for p in patterns if p.direction is not PatternDirection.NEUTRAL]
        if not actionable:
            neutral = [p.name for p in patterns]
            return Evidence(
                name, 0.0, WEIGHTS[name],
                f"{name}: only neutral formations ({', '.join(neutral)})" if neutral
                else f"{name}: none detected",
            )

        strongest = max(actionable, key=lambda p: p.confidence)
        # Average so several agreeing patterns reinforce, but weight the
        # strongest most heavily.
        contribution = sum(p.direction.sign * p.confidence for p in actionable) / len(
            actionable
        )
        contribution = (contribution + strongest.direction.sign * strongest.confidence) / 2

        return Evidence(
            name,
            _clamp(contribution),
            WEIGHTS[name],
            f"{name}: {strongest.name} ({strongest.direction.value}, "
            f"confidence {strongest.confidence:.2f})"
            + (f" +{len(actionable)-1} more" if len(actionable) > 1 else ""),
        )

    def _score_divergence(self, divergences: tuple[Divergence, ...]) -> Evidence:
        """Regular divergence warns of reversal; hidden suggests continuation."""
        if not divergences:
            return Evidence("divergence", 0.0, WEIGHTS["divergence"], "no divergence")

        contribution = 0.0
        for divergence in divergences:
            # Hidden divergence is weaker evidence and points the other way.
            factor = 1.0 if divergence.kind == "regular" else 0.5
            contribution += divergence.direction.sign * divergence.strength * factor
        contribution /= len(divergences)

        strongest = max(divergences, key=lambda d: d.strength)
        return Evidence(
            "divergence",
            _clamp(contribution),
            WEIGHTS["divergence"],
            f"{strongest.kind} {strongest.direction.value} divergence on "
            f"{strongest.indicator} (strength {strongest.strength:.2f})",
        )

    def _score_levels(
        self, levels: LevelMap, price: float, atr_value: float
    ) -> Evidence:
        """Score position relative to support and resistance.

        Rewards buying with room overhead and penalises buying into a ceiling.
        This is the check that stops the bot taking a perfectly valid momentum
        long two ticks below a 4-touch resistance.
        """
        support = levels.nearest_support(price)
        resistance = levels.nearest_resistance(price)

        if support is None and resistance is None:
            return Evidence("level_proximity", 0.0, WEIGHTS["level_proximity"], "no levels mapped")

        parts: list[str] = []
        contribution = 0.0
        scale = atr_value if atr_value > 0 else price * 0.01

        if resistance is not None:
            distance = (resistance.price - price) / scale
            parts.append(f"resistance {resistance.price:.6g} at {distance:.1f} ATR ({resistance.touches} touches)")
            # Close overhead resistance is bearish for a long.
            contribution -= _clamp(1.0 - distance / 3.0, low=0.0) * 0.8
        if support is not None:
            distance = (price - support.price) / scale
            parts.append(f"support {support.price:.6g} at {distance:.1f} ATR ({support.touches} touches)")
            contribution += _clamp(1.0 - distance / 3.0, low=0.0) * 0.8

        headroom = levels.headroom_ratio(price)
        if headroom is not None:
            parts.append(f"headroom {headroom:.2f}x")

        if levels.profile is not None:
            poc = levels.profile.poc
            parts.append(f"POC {poc:.6g}")
            if not levels.profile.in_value_area(price):
                parts.append("outside value area")

        return Evidence(
            "level_proximity",
            _clamp(contribution),
            WEIGHTS["level_proximity"],
            "levels: " + ", ".join(parts),
        )

    def _score_regime(self, regime: MarketRegime, is_satellite: bool) -> Evidence:
        """Market backdrop, weighted harder for satellite names."""
        base = {
            RegimeState.RISK_ON: 0.6,
            RegimeState.NEUTRAL: 0.0,
            RegimeState.VOLATILE: -0.2,
            RegimeState.RISK_OFF: -0.7,
        }[regime.state]

        # Satellites live or die by the backdrop: when BTC turns, thin books
        # gap rather than trend.
        if is_satellite:
            base *= 1.4

        return Evidence(
            "regime",
            _clamp(base),
            WEIGHTS["regime"],
            f"regime {regime.describe()}"
            + (" [satellite: amplified]" if is_satellite else ""),
        )

    def _score_mean_reversion(
        self, frame: pd.DataFrame, structure: StructureState
    ) -> Evidence:
        """Counter-trend exhaustion, but only where it is justified.

        Only contributes in a RANGE or when price is at a band extreme in a
        non-trending market. Applying mean reversion inside a strong trend is
        the single most expensive mistake a retail system makes, so the check
        disables itself there rather than being merely down-weighted.
        """
        if structure.trend in (Trend.STRONG_UP, Trend.STRONG_DOWN):
            return Evidence(
                "mean_reversion", 0.0, WEIGHTS["mean_reversion"],
                "mean reversion disabled: trend is too strong to fade",
            )

        percent_b = _last(bollinger(frame, 20)["bb_percent_b"])
        rsi_value = _last(rsi(frame, 14))
        if percent_b is None or rsi_value is None:
            return Evidence("mean_reversion", 0.0, WEIGHTS["mean_reversion"], "unavailable")

        contribution = 0.0
        parts: list[str] = []
        if percent_b > 1.0 and rsi_value > 70:
            contribution = -0.7
            parts.append(f"above upper band with RSI {rsi_value:.0f}: stretched")
        elif percent_b < 0.0 and rsi_value < 30:
            contribution = 0.7
            parts.append(f"below lower band with RSI {rsi_value:.0f}: washed out")
        else:
            parts.append(f"%B {percent_b:.2f}, RSI {rsi_value:.0f}: no extreme")

        return Evidence(
            "mean_reversion",
            _clamp(contribution),
            WEIGHTS["mean_reversion"],
            "mean reversion: " + ", ".join(parts),
        )

    # -- hard blocks -------------------------------------------------------

    def _find_blockers(
        self,
        *,
        side: Side,
        price: float,
        atr_value: float,
        levels: LevelMap,
        regime: MarketRegime,
        structure: StructureState,
        is_satellite: bool,
        frame: pd.DataFrame,
    ) -> list[str]:
        """Conditions no score may override.

        Kept separate from the weighted evidence on purpose. These are not
        "strong negatives" to be outvoted by six weak positives; they are
        situations where the trade is wrong regardless of how good the rest of
        the chart looks.
        """
        blockers: list[str] = []
        if side is Side.FLAT:
            return blockers

        scale = atr_value if atr_value > 0 else price * 0.01

        # 1. Immediate opposing level. Entering here means the stop and the
        #    obstacle are the same distance away.
        if side is Side.LONG:
            resistance = levels.nearest_resistance(price)
            if resistance is not None:
                distance = (resistance.price - price) / scale
                if distance < self.level_block_atr and resistance.touches >= 2:
                    blockers.append(
                        f"long blocked: {resistance.touches}-touch resistance at "
                        f"{resistance.price:.6g}, only {distance:.2f} ATR overhead"
                    )
        else:
            support = levels.nearest_support(price)
            if support is not None:
                distance = (price - support.price) / scale
                if distance < self.level_block_atr and support.touches >= 2:
                    blockers.append(
                        f"short blocked: {support.touches}-touch support at "
                        f"{support.price:.6g}, only {distance:.2f} ATR below"
                    )

        # 2. Satellite longs in a hostile regime. This is the rule that
        #    protects a small account from its worst single outcome.
        if (
            is_satellite
            and side is Side.LONG
            and not regime.allows_satellite_longs
        ):
            blockers.append(
                f"satellite long blocked: regime is {regime.state.value}, "
                "thin books gap rather than trend when BTC turns"
            )

        # 3. Trading directly against a strong higher-level trend.
        if side is Side.LONG and structure.trend is Trend.STRONG_DOWN:
            blockers.append("long blocked: primary trend is strongly down")
        if side is Side.SHORT and structure.trend is Trend.STRONG_UP:
            blockers.append("short blocked: primary trend is strongly up")

        # 4. Buying the very top of a range, or selling the very bottom.
        position = structure.range_position
        if position is not None:
            if side is Side.LONG and position > 0.95 and not structure.trend.is_up:
                blockers.append(
                    f"long blocked: price at {position*100:.0f}% of its range with no uptrend"
                )
            if side is Side.SHORT and position < 0.05 and not structure.trend.is_down:
                blockers.append(
                    f"short blocked: price at {position*100:.0f}% of its range with no downtrend"
                )

        return blockers


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    if value != value:  # NaN
        return 0.0
    return max(low, min(high, value))


def _last(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    return float(cleaned.iloc[-1]) if not cleaned.empty else None


def _nth_last(series: pd.Series, n: int) -> float | None:
    cleaned = series.dropna()
    return float(cleaned.iloc[-n]) if len(cleaned) >= n else None
