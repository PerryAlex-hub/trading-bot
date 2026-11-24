"""The analysis pipeline: candles in, snapshot out.

One entry point, `analyse_symbol`, used identically by the live cron job, the
CLI and the backtester. That shared path is deliberate and load-bearing: the
most common way a backtest ends up lying is that live and historical code
compute features slightly differently, and the only reliable defence is for
there to be exactly one implementation.

The multi-timeframe read is the main thing this module adds over calling the
analysis modules individually. `universe.yaml` declares 4h as primary with 1d,
1h and 15m alongside, and the hierarchy is used as follows:

* **1d** is the authority. A 4h long against a daily downtrend is penalised.
* **4h** is where the decision is made.
* **1h / 15m** refine entry timing only, and never override the 4h verdict.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ..models import Candle, Tier, Timeframe
from .confluence import ConfluenceEngine
from .frame import check_health, to_frame
from .levels import build_levels
from .patterns import (
    detect_candlestick_patterns,
    detect_chart_patterns,
    detect_divergence,
)
from .regime import MarketRegime, assess_regime, estimate_beta
from .snapshot import FeatureSnapshot, TimeframeView, build_snapshot, build_timeframe_view
from .structure import classify_structure

log = logging.getLogger(__name__)

# Below this, rolling indicators have no warm-up and levels are meaningless.
# Chosen so a 50-period EMA plus a little context exists; AXISUSDT with ~1
# month of 4h bars (~180) clears it, which is why it can be analysed at all.
MIN_BARS_FOR_ANALYSIS = 60

# Higher timeframes legitimately have fewer bars for the same calendar span.
MIN_BARS_FOR_CONTEXT = 30


class InsufficientHistory(RuntimeError):
    """Raised when a symbol cannot be analysed at all.

    A distinct exception rather than returning None: "we could not analyse
    this" and "we analysed it and found nothing" are different outcomes and
    the journal needs to tell them apart.
    """


@dataclass(frozen=True)
class SymbolData:
    """Candles for one symbol across every timeframe, plus market context."""

    symbol: str
    tier: Tier
    frames: Mapping[str, pd.DataFrame]
    btc_frame: pd.DataFrame | None = None
    eth_frame: pd.DataFrame | None = None

    @classmethod
    def from_candles(
        cls,
        symbol: str,
        tier: Tier,
        candles_by_timeframe: Mapping[str, Sequence[Candle]],
        *,
        btc_candles: Sequence[Candle] | None = None,
        eth_candles: Sequence[Candle] | None = None,
    ) -> SymbolData:
        return cls(
            symbol=symbol,
            tier=tier,
            frames={
                timeframe: to_frame(candles)
                for timeframe, candles in candles_by_timeframe.items()
            },
            btc_frame=to_frame(btc_candles) if btc_candles else None,
            eth_frame=to_frame(eth_candles) if eth_candles else None,
        )

    def frame(self, timeframe: str) -> pd.DataFrame | None:
        found = self.frames.get(timeframe)
        return found if found is not None and not found.empty else None


def analyse_symbol(
    data: SymbolData,
    *,
    primary: Timeframe = Timeframe.H4,
    higher: Sequence[str] = ("1d",),
    lower: Sequence[str] = ("1h", "15m"),
    engine: ConfluenceEngine | None = None,
    regime: MarketRegime | None = None,
    computed_at: datetime | None = None,
) -> FeatureSnapshot:
    """Run the full analysis for one symbol and return its snapshot.

    `regime` can be passed in so a multi-symbol run assesses the market once
    rather than per symbol. It is the same for every symbol by definition, and
    recomputing it six times per cycle wastes the compute budget.
    """
    frame = data.frame(primary.value)
    if frame is None:
        raise InsufficientHistory(
            f"{data.symbol}: no candles stored for the primary {primary.value} timeframe"
        )
    if len(frame) < MIN_BARS_FOR_ANALYSIS:
        raise InsufficientHistory(
            f"{data.symbol}: only {len(frame)} {primary.value} bars, "
            f"need at least {MIN_BARS_FOR_ANALYSIS} for indicators to warm up"
        )

    engine = engine or ConfluenceEngine()
    health = check_health(frame, data.symbol, primary)

    structure = classify_structure(frame)
    levels = build_levels(frame)
    chart_patterns = tuple(detect_chart_patterns(frame))
    candle_patterns = tuple(detect_candlestick_patterns(frame))
    divergences = tuple(detect_divergence(frame))

    if regime is None:
        regime = (
            assess_regime(data.btc_frame, eth_frame=data.eth_frame)
            if data.btc_frame is not None
            else _unknown_regime()
        )

    higher_views, higher_structure = _timeframe_views(data, higher)
    lower_views, _ = _timeframe_views(data, lower)

    beta = None
    if data.btc_frame is not None and data.symbol != "BTCUSDT":
        beta = estimate_beta(frame, data.btc_frame, symbol=data.symbol)

    confluence = engine.evaluate(
        frame,
        structure=structure,
        levels=levels,
        regime=regime,
        chart_patterns=chart_patterns,
        candle_patterns=candle_patterns,
        divergences=divergences,
        higher_structure=higher_structure,
        is_satellite=data.tier in (Tier.SATELLITE, Tier.PAPER),
    )

    return build_snapshot(
        symbol=data.symbol,
        tier=data.tier,
        timeframe=primary,
        frame=frame,
        structure=structure,
        levels=levels,
        regime=regime,
        confluence=confluence,
        chart_patterns=chart_patterns,
        candle_patterns=candle_patterns,
        divergences=divergences,
        higher_timeframes=higher_views,
        lower_timeframes=lower_views,
        beta_to_btc=beta,
        health=health,
        computed_at=computed_at,
    )


def _timeframe_views(
    data: SymbolData, timeframes: Sequence[str]
) -> tuple[tuple[TimeframeView, ...], object | None]:
    """Build context views, returning the first usable structure as authority.

    A missing higher timeframe degrades the analysis rather than failing it:
    the confluence engine treats an absent higher-timeframe view as neutral,
    which is the honest representation of "we do not know".
    """
    views: list[TimeframeView] = []
    authority = None
    for timeframe in timeframes:
        frame = data.frame(timeframe)
        if frame is None or len(frame) < MIN_BARS_FOR_CONTEXT:
            log.debug(
                "%s: skipping %s context (%s bars)",
                data.symbol,
                timeframe,
                0 if frame is None else len(frame),
            )
            continue
        structure = classify_structure(frame)
        views.append(build_timeframe_view(frame, timeframe, structure))
        if authority is None:
            authority = structure
    return tuple(views), authority


def _unknown_regime() -> MarketRegime:
    """Neutral regime when BTC context is unavailable.

    Neutral rather than risk-on: without context the bot should not be granted
    permission to take satellite longs, since `RegimeState.NEUTRAL` denies
    them. Failing closed is the correct default for a risk gate.
    """
    from ..models import Trend
    from .regime import RegimeState

    return MarketRegime(
        state=RegimeState.NEUTRAL,
        btc_trend=Trend.RANGE,
        btc_rsi=None,
        btc_atr_pct=None,
        eth_btc_trend=None,
        btc_above_200ema=None,
        notes=("BTC context unavailable; regime assumed neutral",),
    )


def analyse_universe(
    symbols: Sequence[SymbolData],
    *,
    primary: Timeframe = Timeframe.H4,
    engine: ConfluenceEngine | None = None,
) -> tuple[list[FeatureSnapshot], list[tuple[str, str]]]:
    """Analyse several symbols against one shared regime read.

    Returns `(snapshots, failures)` rather than raising. One symbol with a
    broken data feed must not prevent the other five from being analysed - on a
    cron schedule, a raised exception means the whole cycle is skipped.
    """
    engine = engine or ConfluenceEngine()

    # Breadth needs every symbol at once, which is why it is computed here
    # rather than inside assess_regime's per-symbol view. BTC itself is
    # excluded: whether BTC outperforms BTC is not a question.
    alt_frames = {
        candidate.symbol: candidate.frames[primary.value]
        for candidate in symbols
        if not candidate.symbol.startswith("BTC")
        and primary.value in candidate.frames
        and not candidate.frames[primary.value].empty
    }

    regime = _unknown_regime()
    for candidate in symbols:
        if candidate.btc_frame is not None and len(candidate.btc_frame) >= 50:
            regime = assess_regime(
                candidate.btc_frame,
                eth_frame=candidate.eth_frame,
                alt_frames=alt_frames or None,
            )
            break

    snapshots: list[FeatureSnapshot] = []
    failures: list[tuple[str, str]] = []
    for data in symbols:
        try:
            snapshots.append(
                analyse_symbol(data, primary=primary, engine=engine, regime=regime)
            )
        except InsufficientHistory as exc:
            log.info("Skipping %s: %s", data.symbol, exc)
            failures.append((data.symbol, str(exc)))
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Analysis failed for %s", data.symbol)
            failures.append((data.symbol, f"{type(exc).__name__}: {exc}"))

    return snapshots, failures
