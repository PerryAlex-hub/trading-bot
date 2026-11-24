"""The feature snapshot: everything known about one symbol at one instant.

This is the project's central boundary. The README states the governing rule as
"the LLM does judgment, code does math", and this object is where that rule is
enforced: the reasoning layer receives a `FeatureSnapshot` and nothing else. It
never sees raw candles, never calls an indicator, and never computes a level.

Three consequences worth stating plainly:

* **Auditability.** The snapshot is serialised into `signals.features`, so a
  decision made months ago can be re-read with the exact numbers it was made
  from, rather than numbers recomputed from a since-changed codebase.
* **Testability.** Because the snapshot is a plain value, the entire decision
  path downstream of it can be tested without network, database or candles.
* **Contamination control.** The README flags that an LLM asked about SOL in
  March 2025 already knows what happened. `anonymised()` exists for that:
  it strips the symbol and absolute dates so a backtest prompt cannot be
  answered from memory instead of from the chart.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from ..models import Side, Tier, Timeframe, Trend
from .confluence import ConfluenceResult
from .frame import SeriesHealth
from .indicators import (
    adx,
    atr,
    atr_pct,
    bollinger,
    ema,
    macd,
    mfi,
    relative_volume,
    rsi,
    stoch_rsi,
    supertrend,
    vwap,
)
from .levels import LevelMap
from .patterns import Divergence, Pattern
from .regime import BetaEstimate, MarketRegime
from .structure import StructureState

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndicatorReadings:
    """Indicator values at the snapshot bar.

    Floats, and every one optional: a 30-bar series genuinely has no 200 EMA,
    and inventing one would be worse than admitting it is unknown.
    """

    close: float
    rsi_14: float | None = None
    atr_14: float | None = None
    atr_pct: float | None = None
    adx_14: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    bb_percent_b: float | None = None
    bb_bandwidth: float | None = None
    stoch_rsi_k: float | None = None
    mfi_14: float | None = None
    relative_volume: float | None = None
    vwap_20: float | None = None
    supertrend_direction: float | None = None
    supertrend_line: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            key: (round(value, 10) if isinstance(value, float) else value)
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class TimeframeView:
    """One timeframe's reading of the same symbol."""

    timeframe: str
    trend: Trend
    indicators: IndicatorReadings
    structure_direction: int
    adx: float | None
    bars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "trend": self.trend.value,
            "structure_direction": self.structure_direction,
            "adx": _round(self.adx),
            "bars": self.bars,
            "indicators": self.indicators.to_dict(),
        }


@dataclass(frozen=True)
class FeatureSnapshot:
    """Everything the decision layer is allowed to know.

    Frozen, and the only thing passed downstream. If a field is not here, the
    reasoning layer cannot use it - which is the point.
    """

    symbol: str
    tier: Tier
    timeframe: Timeframe
    bar_time: datetime
    computed_at: datetime

    price: float
    indicators: IndicatorReadings
    structure: StructureState
    levels: LevelMap
    regime: MarketRegime
    confluence: ConfluenceResult

    chart_patterns: tuple[Pattern, ...] = ()
    candle_patterns: tuple[Pattern, ...] = ()
    divergences: tuple[Divergence, ...] = ()
    higher_timeframes: tuple[TimeframeView, ...] = ()
    lower_timeframes: tuple[TimeframeView, ...] = ()

    beta_to_btc: BetaEstimate | None = None
    health: SeriesHealth | None = None
    data_warnings: tuple[str, ...] = field(default_factory=tuple)

    # -- derived -----------------------------------------------------------

    @property
    def side(self) -> Side:
        return self.confluence.side

    @property
    def score(self) -> float:
        return self.confluence.score

    @property
    def is_tradeable(self) -> bool:
        """Whether this snapshot may produce a live order.

        PAPER-tier symbols are analysed and journaled but never sized, exactly
        as `universe.yaml` specifies for AXISUSDT.
        """
        return self.tier.tradeable and not self.confluence.is_blocked

    def fingerprint(self) -> str:
        """Stable hash of the inputs that produced this decision.

        Two snapshots with the same fingerprint were computed from the same
        numbers, which makes a duplicated cron run detectable even when the
        timestamps differ.
        """
        payload = json.dumps(
            {
                "symbol": self.symbol,
                "bar": self.bar_time.isoformat(),
                "price": round(self.price, 12),
                "score": round(self.confluence.score, 8),
                "side": self.confluence.side.value,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Full serialisation, for the journal's `features` column."""
        return {
            "symbol": self.symbol,
            "tier": self.tier.value,
            "timeframe": self.timeframe.value,
            "bar_time": self.bar_time.isoformat(),
            "computed_at": self.computed_at.isoformat(),
            "fingerprint": self.fingerprint(),
            "price": _round(self.price),
            "indicators": self.indicators.to_dict(),
            "structure": {
                "trend": self.structure.trend.value,
                "ema_stack": self.structure.ema_stack,
                "adx": _round(self.structure.adx_value),
                "higher_highs": self.structure.higher_highs,
                "higher_lows": self.structure.higher_lows,
                "lower_highs": self.structure.lower_highs,
                "lower_lows": self.structure.lower_lows,
                "last_bos": self.structure.last_bos,
                "last_choch": self.structure.last_choch,
                "range_position": _round(self.structure.range_position),
                "swing_count": len(self.structure.swings),
                "last_swing_high": _round(
                    self.structure.last_high.price if self.structure.last_high else None
                ),
                "last_swing_low": _round(
                    self.structure.last_low.price if self.structure.last_low else None
                ),
            },
            "levels": {
                "supports": [
                    {
                        "price": _round(level.price),
                        "touches": level.touches,
                        "strength": _round(level.strength),
                        "width_atr": _round(level.width_atr),
                    }
                    for level in self.levels.supports
                ],
                "resistances": [
                    {
                        "price": _round(level.price),
                        "touches": level.touches,
                        "strength": _round(level.strength),
                        "width_atr": _round(level.width_atr),
                    }
                    for level in self.levels.resistances
                ],
                "headroom_ratio": _round(self.levels.headroom_ratio(self.price)),
                "round_numbers": [_round(value) for value in self.levels.round_numbers],
                "poc": _round(self.levels.profile.poc if self.levels.profile else None),
                "value_area": (
                    [
                        _round(self.levels.profile.value_area_low),
                        _round(self.levels.profile.value_area_high),
                    ]
                    if self.levels.profile
                    else None
                ),
            },
            "regime": {
                "state": self.regime.state.value,
                "btc_trend": self.regime.btc_trend.value,
                "btc_rsi": _round(self.regime.btc_rsi),
                "btc_atr_pct": _round(self.regime.btc_atr_pct),
                "eth_btc_trend": (
                    self.regime.eth_btc_trend.value if self.regime.eth_btc_trend else None
                ),
                "btc_above_200ema": self.regime.btc_above_200ema,
                "alt_breadth": _round(self.regime.alt_breadth),
                "allows_satellite_longs": self.regime.allows_satellite_longs,
                "notes": list(self.regime.notes),
            },
            "patterns": {
                "chart": [_pattern_dict(p) for p in self.chart_patterns],
                "candlestick": [_pattern_dict(p) for p in self.candle_patterns],
                "divergences": [
                    {
                        "indicator": d.indicator,
                        "kind": d.kind,
                        "direction": d.direction.value,
                        "strength": _round(d.strength),
                    }
                    for d in self.divergences
                ],
            },
            "timeframes": {
                "higher": [view.to_dict() for view in self.higher_timeframes],
                "lower": [view.to_dict() for view in self.lower_timeframes],
            },
            "beta_to_btc": (
                {
                    "beta": _round(self.beta_to_btc.beta),
                    "correlation": _round(self.beta_to_btc.correlation),
                    "samples": self.beta_to_btc.samples,
                    "reliable": self.beta_to_btc.is_reliable,
                }
                if self.beta_to_btc
                else None
            ),
            "confluence": self.confluence.to_dict(),
            "data_health": (
                {
                    "bars": self.health.bars,
                    "gaps": self.health.gaps,
                    "largest_gap_bars": self.health.largest_gap_bars,
                }
                if self.health
                else None
            ),
            "data_warnings": list(self.data_warnings),
        }

    def to_prompt_dict(self, *, anonymise: bool = False) -> dict[str, Any]:
        """A compact view for the reasoning layer.

        Deliberately smaller than `to_dict()`. The analyst is asked to exercise
        judgment on an already-computed picture, not to re-derive the analysis,
        so raw indicator dumps are trimmed to what a judgment call needs.
        """
        payload = self.to_dict()
        for key in ("computed_at", "fingerprint", "data_health"):
            payload.pop(key, None)

        if anonymise:
            # Contamination control, per the README's backtest-honesty section.
            # A model that recognises "SOLUSDT, 2025-03-14" can answer from
            # memory; strip identity and absolute time so it has to read the
            # numbers in front of it.
            payload["symbol"] = "ASSET"
            payload["bar_time"] = "REDACTED"
            payload["tier"] = "REDACTED"
            regime = payload.get("regime")
            if isinstance(regime, dict):
                regime.pop("btc_trend", None)
                regime.pop("notes", None)
        return payload


def build_snapshot(
    *,
    symbol: str,
    tier: Tier,
    timeframe: Timeframe,
    frame: pd.DataFrame,
    structure: StructureState,
    levels: LevelMap,
    regime: MarketRegime,
    confluence: ConfluenceResult,
    chart_patterns: tuple[Pattern, ...] = (),
    candle_patterns: tuple[Pattern, ...] = (),
    divergences: tuple[Divergence, ...] = (),
    higher_timeframes: tuple[TimeframeView, ...] = (),
    lower_timeframes: tuple[TimeframeView, ...] = (),
    beta_to_btc: BetaEstimate | None = None,
    health: SeriesHealth | None = None,
    computed_at: datetime | None = None,
) -> FeatureSnapshot:
    """Assemble the snapshot and collect any data-quality warnings."""
    if frame.empty:
        raise ValueError(f"Cannot build a snapshot for {symbol} from an empty frame")

    warnings: list[str] = []
    if health is not None:
        if health.gaps > 0:
            warnings.append(
                f"{health.gaps} gap(s) in the candle series, largest "
                f"{health.largest_gap_bars} bars: rolling indicators are "
                "computed over non-consecutive bars"
            )
        if health.bars < 200:
            warnings.append(
                f"only {health.bars} bars stored: long-period indicators "
                "(200 EMA) are unavailable and levels are provisional"
            )

    return FeatureSnapshot(
        symbol=symbol,
        tier=tier,
        timeframe=timeframe,
        bar_time=frame.index[-1].to_pydatetime(),
        computed_at=computed_at or datetime.now(tz=frame.index.tz),
        price=float(frame["close"].iloc[-1]),
        indicators=read_indicators(frame),
        structure=structure,
        levels=levels,
        regime=regime,
        confluence=confluence,
        chart_patterns=chart_patterns,
        candle_patterns=candle_patterns,
        divergences=divergences,
        higher_timeframes=higher_timeframes,
        lower_timeframes=lower_timeframes,
        beta_to_btc=beta_to_btc,
        health=health,
        data_warnings=tuple(warnings),
    )


def read_indicators(frame: pd.DataFrame) -> IndicatorReadings:
    """Compute every indicator once and read the final bar.

    One pass, one place. Previously each consumer recomputed what it needed,
    which on a 4-timeframe, 6-symbol run meant recomputing ATR dozens of times
    per cycle - and, worse, risked two consumers disagreeing about the value.
    """
    close = float(frame["close"].iloc[-1])
    if len(frame) < 20:
        return IndicatorReadings(close=close)

    macd_frame = macd(frame)
    bands = bollinger(frame, 20)
    adx_frame = adx(frame, 14)
    st_frame = supertrend(frame)

    return IndicatorReadings(
        close=close,
        rsi_14=_last(rsi(frame, 14)),
        atr_14=_last(atr(frame, 14)),
        atr_pct=_last(atr_pct(frame, 14)),
        adx_14=_last(adx_frame["adx"]),
        plus_di=_last(adx_frame["plus_di"]),
        minus_di=_last(adx_frame["minus_di"]),
        macd=_last(macd_frame["macd"]),
        macd_signal=_last(macd_frame["macd_signal"]),
        macd_hist=_last(macd_frame["macd_hist"]),
        ema_20=_last(ema(frame["close"], 20)),
        ema_50=_last(ema(frame["close"], 50)),
        ema_200=_last(ema(frame["close"], 200)),
        bb_upper=_last(bands["bb_upper"]),
        bb_lower=_last(bands["bb_lower"]),
        bb_percent_b=_last(bands["bb_percent_b"]),
        bb_bandwidth=_last(bands["bb_bandwidth"]),
        stoch_rsi_k=_last(stoch_rsi(frame)["stochrsi_k"]),
        mfi_14=_last(mfi(frame, 14)),
        relative_volume=_last(relative_volume(frame, 20)),
        vwap_20=_last(vwap(frame, 20)),
        supertrend_direction=_last(st_frame["st_trend"]),
        supertrend_line=_last(st_frame["st_line"]),
    )


def build_timeframe_view(
    frame: pd.DataFrame, timeframe: str, structure: StructureState
) -> TimeframeView:
    return TimeframeView(
        timeframe=timeframe,
        trend=structure.trend,
        indicators=read_indicators(frame),
        structure_direction=structure.structure_direction,
        adx=structure.adx_value,
        bars=len(frame),
    )


def _pattern_dict(pattern: Pattern) -> dict[str, Any]:
    return {
        "name": pattern.name,
        "direction": pattern.direction.value,
        "confidence": _round(pattern.confidence),
        "target": _round(pattern.target),
        "invalidation": _round(pattern.invalidation),
        "detail": {key: _round(value) for key, value in pattern.detail.items()},
    }


def _round(value: Any, digits: int = 10) -> Any:
    """Round floats for JSON, and coerce numpy scalars to built-ins.

    numpy types are not JSON-serialisable, and they reach here easily because
    pandas returns them from `.iloc`. Catching that at the serialisation
    boundary is cheaper than auditing every call site.
    """
    if value is None:
        return None
    if isinstance(value, (int, bool)):
        return value
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    if as_float != as_float:  # NaN is not valid JSON
        return None
    return round(as_float, digits)


def _last(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    return float(cleaned.iloc[-1]) if not cleaned.empty else None
