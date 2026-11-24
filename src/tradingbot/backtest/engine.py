"""The replay loop, the fill model, and the metrics.

Three decisions here decide whether the output is worth reading at all, so they
are stated plainly rather than buried.

**1. Point-in-time frames.** Every decision bar sees candles truncated at that
bar, the same way `CandleRepo.latest(as_of=...)` truncates for the live
catch-up path. Indicators are recomputed per bar on the truncated series, which
is slow and is the only honest way: handing the analysis a full series and
asking about bar 300 lets a 200-EMA see the future.

**2. Pessimistic intrabar resolution.** OHLC cannot say whether the low or the
high came first. When a bar's range contains both the stop and the target, this
engine books a LOSS. That is not conservatism for its own sake - the opposite
assumption turns every wide-range bar into a free win and is the single most
common way a backtest manufactures an edge that does not exist.

**3. No model, so no contamination.** The usual objection to an LLM backtest -
that the model already knows what happened next - does not apply, because
nothing in this decision path is a model. The pipeline is deterministic
arithmetic over the truncated frames. That makes these numbers unusually
trustworthy for this kind of system, and it is the reason the reasoning layer
was left for later.

What this does NOT model: partial fills, slippage beyond the spread, funding
changes over the hold, a real order book, or the circuit breakers. Those make
the result optimistic at the margin, and each is noted where it bites.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

import pandas as pd

from ..analysis.pipeline import MIN_BARS_FOR_ANALYSIS, SymbolData, analyse_universe
from ..config import RiskConfig, Universe
from ..models import PositionSize, Side, Tier, Timeframe, TradeSetup
from ..risk.gate import PortfolioState, RiskGate
from ..risk.sizing import friction_cost
from ..strategy.setup import build_setup

log = logging.getLogger(__name__)


class TradeOutcome(str, Enum):
    TARGET = "target"
    STOP = "stop"
    TIMEOUT = "timeout"          # held past max_hold_bars, closed at the close
    AMBIGUOUS_STOP = "ambiguous" # stop and target in one bar; booked as a stop
    NEVER_FILLED = "never_filled"

    @property
    def is_win(self) -> bool:
        return self is TradeOutcome.TARGET


@dataclass
class SimulatedTrade:
    symbol: str
    side: Side
    decision_bar: datetime
    entry: Decimal
    stop: Decimal
    target: Decimal
    quantity: Decimal
    risk_amount: Decimal
    outcome: TradeOutcome
    fill_bar: datetime | None = None
    exit_bar: datetime | None = None
    exit_price: Decimal | None = None
    friction: Decimal = Decimal("0")
    reasons: tuple[str, ...] = ()

    @property
    def filled(self) -> bool:
        return self.fill_bar is not None

    @property
    def pnl(self) -> Decimal:
        """Realised profit in quote currency, net of friction."""
        if self.exit_price is None or not self.filled:
            return Decimal("0")
        direction = Decimal("1") if self.side is Side.LONG else Decimal("-1")
        gross = (self.exit_price - self.entry) * direction * self.quantity
        return gross - self.friction

    @property
    def resolved(self) -> bool:
        """Filled AND closed. A trade still open when history ran out is not a
        result, and must not be averaged in as one."""
        return self.filled and self.exit_price is not None

    @property
    def r_multiple(self) -> Decimal | None:
        """Result in units of risk, the only cross-symbol comparison.

        None unless the trade actually closed. Returning 0 for an unresolved
        trade - which an earlier version did, since `pnl` is 0 without an exit
        price - would quietly count every trade the replay ran out of history
        on as a break-even, diluting the losses and flattering expectancy.
        """
        if not self.resolved or self.risk_amount <= 0:
            return None
        return self.pnl / self.risk_amount


@dataclass
class BacktestResult:
    trades: list[SimulatedTrade] = field(default_factory=list)
    bars_examined: int = 0
    signals_approved: int = 0
    signals_rejected: int = 0
    veto_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def filled(self) -> list[SimulatedTrade]:
        return [t for t in self.trades if t.filled]


def run_backtest(
    *,
    candles_by_symbol: dict[str, dict[str, pd.DataFrame]],
    universe: Universe,
    risk_config: RiskConfig,
    primary: Timeframe = Timeframe.H4,
    start_index: int | None = None,
    max_wait_bars: int = 3,
    max_hold_bars: int = 30,
    equity: Decimal | None = None,
) -> BacktestResult:
    """Replay every decision bar in the stored history.

    `candles_by_symbol` maps symbol -> timeframe -> full frame. Frames are
    sliced in memory rather than re-queried per bar: a 540-day 4h history is
    ~3,200 decision bars, and a database round trip per symbol per timeframe
    per bar would dominate the runtime without changing a single number.

    `max_wait_bars` expires an unfilled limit entry. The live bot re-analyses
    each bar and would not leave a stale order resting indefinitely, so neither
    does this.

    `max_hold_bars` closes a position that has gone nowhere, at the close. Real
    money would not sit in a 4h trade for a month; counting it as open forever
    would flatter the result by deferring every loss.
    """
    result = BacktestResult()
    gate = RiskGate(risk_config)
    equity = equity if equity is not None else risk_config.assumed_equity
    tiers = {config.symbol: config.tier for config in universe.symbols}

    primary_key = primary.value
    index = _shared_index(candles_by_symbol, primary_key)
    if index is None or len(index) <= MIN_BARS_FOR_ANALYSIS + 1:
        result.errors.append(
            f"not enough shared {primary_key} history to replay "
            f"(need more than {MIN_BARS_FOR_ANALYSIS} bars)"
        )
        return result

    first = start_index if start_index is not None else MIN_BARS_FOR_ANALYSIS
    first = max(first, MIN_BARS_FOR_ANALYSIS)

    # One open trade per symbol at a time, mirroring max_per_base_asset.
    open_until: dict[str, int] = {}

    for position in range(first, len(index) - 1):
        bar_time = index[position]
        result.bars_examined += 1

        payloads = _slice_universe(
            candles_by_symbol, universe, bar_time, primary_key
        )
        if not payloads:
            continue

        try:
            snapshots, _failures = analyse_universe(payloads, primary=primary)
        except Exception as exc:  # pragma: no cover - defensive
            result.errors.append(f"{bar_time:%Y-%m-%d %H:%M}: {type(exc).__name__}: {exc}")
            continue

        for snapshot in snapshots:
            if snapshot.side is Side.FLAT or snapshot.confluence.is_blocked:
                continue
            if open_until.get(snapshot.symbol, -1) > position:
                continue  # already in a trade on this symbol

            proposal = build_setup(
                snapshot,
                atr_multiple=risk_config.decimal(
                    "per_trade", "stop", "atr_multiple", default="2.0"
                ),
                satellite_atr_multiple=risk_config.decimal(
                    "per_trade", "stop", "satellite_atr_multiple", default="3.0"
                ),
                min_risk_reward=risk_config.min_risk_reward,
            )
            if not proposal.ok or proposal.setup is None:
                continue

            setup = proposal.setup
            tier = tiers.get(snapshot.symbol, Tier.SATELLITE)

            # A flat portfolio each bar: this measures the STRATEGY, not a
            # particular sequence of concurrent positions. Portfolio limits are
            # a separate question and modelling them here would entangle the
            # two, so the only concurrency rule kept is one trade per symbol.
            decision = gate.evaluate(
                setup,
                tier=tier,
                portfolio=PortfolioState(equity=equity),
                spec=None,
                quote=None,
                funding=None,
                regime=snapshot.regime,
                beta=snapshot.beta_to_btc,
                tiers_by_symbol=tiers,
                quote_volume_24h=None,
                require_market_data=False,
            )

            if not decision.approved or decision.size is None:
                result.signals_rejected += 1
                for veto in decision.vetoes:
                    result.veto_counts[veto.rule] = result.veto_counts.get(veto.rule, 0) + 1
                continue

            result.signals_approved += 1
            trade = _simulate(
                setup=setup,
                size=decision.size,
                frame=candles_by_symbol[snapshot.symbol][primary_key],
                decision_position=position,
                bar_time=bar_time,
                max_wait_bars=max_wait_bars,
                max_hold_bars=max_hold_bars,
                risk_config=risk_config,
                reasons=tuple(snapshot.confluence.top_reasons()),
            )
            result.trades.append(trade)
            if trade.filled:
                closed_at = _position_of(
                    candles_by_symbol[snapshot.symbol][primary_key], trade.exit_bar
                )
                open_until[snapshot.symbol] = closed_at if closed_at is not None else position

    return result


def _simulate(
    *,
    setup: TradeSetup,
    size: PositionSize,
    frame: pd.DataFrame,
    decision_position: int,
    bar_time: datetime,
    max_wait_bars: int,
    max_hold_bars: int,
    risk_config: RiskConfig,
    reasons: tuple[str, ...],
) -> SimulatedTrade:
    """Walk forward bar by bar and resolve one setup."""
    target = setup.targets[0]
    long = setup.side is Side.LONG

    friction = friction_cost(
        notional=size.notional,
        taker_fee_pct=risk_config.decimal("costs", "assumed_taker_fee_pct", default="0.06"),
        # No stored historical spread, so the configured assumption is used.
        # This is the main way the result is optimistic: a real thin-book fill
        # on a satellite would be worse than a tier-average assumption.
        spread_pct=risk_config.assumed_spread_pct(Tier.SATELLITE),
        funding_rate_8h_pct=Decimal("0"),
        hold_intervals=0,
    )

    trade = SimulatedTrade(
        symbol=setup.symbol,
        side=setup.side,
        decision_bar=bar_time,
        entry=setup.entry,
        stop=setup.stop,
        target=target,
        quantity=size.quantity,
        risk_amount=size.quantity * setup.risk_per_unit,
        outcome=TradeOutcome.NEVER_FILLED,
        friction=friction,
        reasons=reasons,
    )

    fill_position: int | None = None
    last = min(decision_position + 1 + max_wait_bars + max_hold_bars, len(frame) - 1)

    for position in range(decision_position + 1, last + 1):
        row = frame.iloc[position]
        when = frame.index[position]
        high = Decimal(str(row["high"]))
        low = Decimal(str(row["low"]))

        if fill_position is None:
            touched = low <= setup.entry if long else high >= setup.entry
            if not touched:
                if position - decision_position > max_wait_bars:
                    return trade  # expired unfilled
                continue
            fill_position = position
            trade.fill_bar = when

        stop_hit = low <= setup.stop if long else high >= setup.stop
        target_hit = high >= target if long else low <= target

        if stop_hit and target_hit:
            # Both inside one bar. OHLC cannot order them, so book the loss.
            trade.outcome = TradeOutcome.AMBIGUOUS_STOP
            trade.exit_bar, trade.exit_price = when, setup.stop
            return trade
        if stop_hit:
            trade.outcome = TradeOutcome.STOP
            trade.exit_bar, trade.exit_price = when, setup.stop
            return trade
        if target_hit:
            trade.outcome = TradeOutcome.TARGET
            trade.exit_bar, trade.exit_price = when, target
            return trade

        if position - fill_position >= max_hold_bars:
            trade.outcome = TradeOutcome.TIMEOUT
            trade.exit_bar = when
            trade.exit_price = Decimal(str(row["close"]))
            return trade

    # Ran out of history while still open: deliberately left unresolved rather
    # than marked to the last close, so an unfinished trade cannot be counted
    # as a result.
    return trade


def summarise(result: BacktestResult) -> dict[str, object]:
    """Metrics in R, because that is the only cross-symbol comparison.

    Win rate is reported but is NOT the headline. With a 1:2 floor the
    break-even hit rate is about 38%, so a 45% win rate can be excellent and a
    65% one can still lose money. Expectancy per trade is the number that pays.
    """
    filled = [t for t in result.trades if t.resolved and t.r_multiple is not None]
    if not filled:
        return {
            "trades": 0,
            "signals_approved": result.signals_approved,
            "signals_rejected": result.signals_rejected,
            "bars_examined": result.bars_examined,
            "top_vetoes": _top_vetoes(result),
        }

    r_values = [t.r_multiple for t in filled if t.r_multiple is not None]
    wins = [t for t in filled if t.pnl > 0]
    losses = [t for t in filled if t.pnl <= 0]
    gross_win = sum((t.pnl for t in wins), Decimal("0"))
    gross_loss = abs(sum((t.pnl for t in losses), Decimal("0")))
    total_r = sum(r_values, Decimal("0"))

    return {
        "trades": len(filled),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(float(len(wins) / len(filled) * 100), 2),
        "total_r": round(float(total_r), 3),
        "expectancy_r": round(float(total_r / len(r_values)), 4),
        "profit_factor": (
            round(float(gross_win / gross_loss), 3) if gross_loss > 0 else None
        ),
        "max_drawdown_r": round(float(_max_drawdown_r(r_values)), 3),
        "longest_losing_streak": _longest_losing_streak(filled),
        "outcomes": _outcome_counts(result),
        "never_filled": sum(
            1 for t in result.trades if t.outcome is TradeOutcome.NEVER_FILLED
        ),
        "signals_approved": result.signals_approved,
        "signals_rejected": result.signals_rejected,
        "bars_examined": result.bars_examined,
        "top_vetoes": _top_vetoes(result),
    }


def _max_drawdown_r(r_values: list[Decimal]) -> Decimal:
    """Largest peak-to-trough fall of the cumulative R curve."""
    peak = Decimal("0")
    equity = Decimal("0")
    worst = Decimal("0")
    for value in r_values:
        equity += value
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return abs(worst)


def _longest_losing_streak(trades: list[SimulatedTrade]) -> int:
    longest = current = 0
    for trade in sorted(trades, key=lambda t: t.exit_bar or t.decision_bar):
        if trade.pnl <= 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _outcome_counts(result: BacktestResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in result.trades:
        counts[trade.outcome.value] = counts.get(trade.outcome.value, 0) + 1
    return counts


def _top_vetoes(result: BacktestResult, limit: int = 6) -> list[tuple[str, int]]:
    return sorted(result.veto_counts.items(), key=lambda item: -item[1])[:limit]


def _shared_index(
    candles_by_symbol: dict[str, dict[str, pd.DataFrame]], primary: str
) -> pd.DatetimeIndex | None:
    """The union of primary-timeframe bar times, sorted.

    A union rather than an intersection: a symbol added later should not
    truncate the replay for the ones with longer history. Symbols without a
    bar at a given time are simply skipped there.
    """
    indexes = [
        frames[primary].index
        for frames in candles_by_symbol.values()
        if primary in frames and not frames[primary].empty
    ]
    if not indexes:
        return None
    combined = indexes[0]
    for other in indexes[1:]:
        combined = combined.union(other)
    return combined.sort_values()


def _slice_universe(
    candles_by_symbol: dict[str, dict[str, pd.DataFrame]],
    universe: Universe,
    bar_time: datetime,
    primary: str,
) -> list[SymbolData]:
    """Build point-in-time payloads for one decision bar."""
    btc = _slice(candles_by_symbol.get("BTCUSDT", {}).get(primary), bar_time)
    eth = _slice(candles_by_symbol.get("ETHUSDT", {}).get(primary), bar_time)

    payloads: list[SymbolData] = []
    for config in universe.symbols:
        frames = candles_by_symbol.get(config.symbol)
        if not frames:
            continue
        sliced = {
            timeframe: _slice(frame, bar_time)
            for timeframe, frame in frames.items()
        }
        primary_frame = sliced.get(primary)
        if primary_frame is None or len(primary_frame) < MIN_BARS_FOR_ANALYSIS:
            continue
        payloads.append(
            SymbolData(
                symbol=config.symbol,
                tier=config.tier,
                frames={k: v for k, v in sliced.items() if v is not None},
                btc_frame=btc,
                eth_frame=eth,
            )
        )
    return payloads


def _slice(frame: pd.DataFrame | None, bar_time: datetime) -> pd.DataFrame | None:
    """Everything up to and including `bar_time`. The no-lookahead guarantee."""
    if frame is None or frame.empty:
        return None
    return frame.loc[frame.index <= bar_time]


def _position_of(frame: pd.DataFrame, when: datetime | None) -> int | None:
    if when is None:
        return None
    matches = frame.index.get_indexer([when])
    return int(matches[0]) if len(matches) and matches[0] >= 0 else None
