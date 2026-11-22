"""Deterministic analysis layer.

Everything here is pure: candles in, numbers out, no network and no database.
That is what makes the whole decision path testable and what lets the
backtester and the live cron job share one implementation.

The governing rule from the README - "the LLM does judgment, code does math" -
is enforced at the `FeatureSnapshot` boundary. The reasoning layer receives a
snapshot and never touches anything else in this package.
"""

from .confluence import ConfluenceEngine, ConfluenceResult, Evidence
from .frame import SeriesHealth, check_health, to_decimal, to_frame
from .levels import Level, LevelMap, VolumeProfile, build_levels
from .patterns import (
    Divergence,
    Pattern,
    PatternDirection,
    detect_candlestick_patterns,
    detect_chart_patterns,
    detect_divergence,
)
from .pipeline import (
    InsufficientHistory,
    SymbolData,
    analyse_symbol,
    analyse_universe,
)
from .regime import (
    BetaEstimate,
    MarketRegime,
    RegimeState,
    assess_regime,
    effective_beta,
    estimate_beta,
)
from .snapshot import FeatureSnapshot, IndicatorReadings, TimeframeView, build_snapshot
from .structure import StructureState, Swing, SwingType, classify_structure, find_swings

__all__ = [
    "BetaEstimate",
    "ConfluenceEngine",
    "ConfluenceResult",
    "Divergence",
    "Evidence",
    "FeatureSnapshot",
    "IndicatorReadings",
    "InsufficientHistory",
    "Level",
    "LevelMap",
    "MarketRegime",
    "Pattern",
    "PatternDirection",
    "RegimeState",
    "SeriesHealth",
    "StructureState",
    "Swing",
    "SwingType",
    "SymbolData",
    "TimeframeView",
    "VolumeProfile",
    "analyse_symbol",
    "analyse_universe",
    "assess_regime",
    "build_levels",
    "build_snapshot",
    "check_health",
    "classify_structure",
    "detect_candlestick_patterns",
    "detect_chart_patterns",
    "detect_divergence",
    "effective_beta",
    "estimate_beta",
    "find_swings",
    "to_decimal",
    "to_frame",
]
