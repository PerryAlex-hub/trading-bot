"""Historical replay of the live decision path.

The point of this package is to answer one question the rest of the project
cannot: does the analysis actually make money. Until now the only evidence was
a journal with no resolved trades in it.

**It replays the real code.** The loop below calls `analyse_universe`,
`build_setup` and `RiskGate` - the same functions the scheduled job calls - on
candle frames truncated at each decision bar. A backtester that reimplements
the strategy measures the reimplementation, and the two drift apart on the
first change to either.
"""

from .engine import (
    BacktestResult,
    SimulatedTrade,
    TradeOutcome,
    run_backtest,
    summarise,
)

__all__ = [
    "BacktestResult",
    "SimulatedTrade",
    "TradeOutcome",
    "run_backtest",
    "summarise",
]
