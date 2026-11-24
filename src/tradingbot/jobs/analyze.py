"""The main analysis cycle: the job the scheduler actually runs.

This is where every layer meets. The order is fixed and each step can veto what
follows:

    ingest -> analyse -> build setup -> risk gate -> (reasoning) -> execute -> notify

Properties that come from the deployment rather than from trading theory:

**Idempotent per bar.** The job records the bar it is analysing and refuses to
re-run a bar it already completed. GitHub Actions can fire a schedule twice,
and a manual run can overlap a cron one; without this, duplicate signals and
duplicate execution attempts follow.

**Everything is journaled, including refusals.** A rejected candidate writes a
row with the failing rule. The README's self-review loop depends on this, and
so does answering "why has it not traded in three weeks?".

**Partial failure is normal.** One symbol with a bad feed must not stop the
other five. Exceptions are caught per symbol and recorded on the run.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from ..analysis import SymbolData, analyse_universe
from ..analysis.snapshot import FeatureSnapshot
from ..config import RiskConfig, Universe, load_risk, load_universe
from ..db.engine import dispose_engine, session_scope, wait_for_database
from ..db.repo import (
    AccountRepo,
    CandleRepo,
    CircuitBreakerRepo,
    ContractSpecRepo,
    PositionRepo,
    RunRepo,
    SignalRepo,
)
from ..execution.tracker import PaperTracker
from ..models import Side, SignalStatus, Tier, Timeframe
from ..notify.telegram import TelegramNotifier
from ..risk.gate import PortfolioState, RiskGate
from ..settings import Settings, TradingMode, get_settings
from ..strategy.setup import build_setup

log = logging.getLogger(__name__)

# How many bars each timeframe needs loaded for analysis. 400 gives a 200-period
# indicator a full warm-up plus history for level clustering.
BARS_TO_LOAD = 400


@dataclass
class CycleResult:
    bar_time: datetime | None = None
    scanned: int = 0
    candidates: int = 0
    approved: int = 0
    rejected: int = 0
    executed: int = 0
    skipped_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.skipped_reason is None and not self.errors

    def describe(self) -> str:
        if self.skipped_reason:
            return f"skipped: {self.skipped_reason}"
        return (
            f"scanned {self.scanned}, candidates {self.candidates}, "
            f"approved {self.approved}, rejected {self.rejected}, "
            f"executed {self.executed}"
        )


def run_cycle(
    *,
    settings: Settings | None = None,
    trigger: str = "cron",
    force: bool = False,
    notify: bool = True,
    primary: Timeframe = Timeframe.H4,
    as_of: datetime | None = None,
    track_paper: bool = True,
    extra_errors: Sequence[str] = (),
) -> CycleResult:
    """Run one full analysis cycle.

    `force` bypasses the duplicate-bar guard, for manual re-runs.

    `as_of` analyses a specific past bar instead of the current one, using only
    the data available at that bar. Used by `run_catch_up`.

    `extra_errors` carries problems detected before the cycle started, most
    importantly failed candle ingest. They are attached to the result so they
    reach the Telegram run summary, which already fires whenever a cycle
    reports errors. Discarding them is how two symbols failed to ingest on
    every run for days without anyone being told.
    """
    settings = settings or get_settings()
    universe = load_universe()
    risk_config = load_risk()
    result = CycleResult()
    result.errors.extend(extra_errors)

    wait_for_database()

    bar_time = as_of or _current_bar(primary)
    result.bar_time = bar_time

    with session_scope() as session:
        if not force and RunRepo(session).already_ran(primary.value, bar_time):
            result.skipped_reason = (
                f"bar {bar_time:%Y-%m-%d %H:%M} has already been analysed; "
                "pass force=True to re-run"
            )
            log.info(result.skipped_reason)
            return result

    notifier = TelegramNotifier(
        bot_token=settings.telegram_bot_token.get_secret_value()
        if settings.telegram_bot_token
        else None,
        chat_id=settings.telegram_chat_id,
    )

    with session_scope() as session:
        run = RunRepo(session).start(
            trigger=trigger,
            bar_time=bar_time,
            timeframe=primary.value,
            git_sha=_git_sha(),
            config_fingerprint=_config_fingerprint(risk_config),
        )
        run_id = run.id

    try:
        snapshots, failures = _analyse(universe, primary, as_of=as_of)
        result.scanned = len(snapshots)
        result.errors.extend(f"{symbol}: {reason}" for symbol, reason in failures)

        for snapshot in snapshots:
            try:
                _process(
                    snapshot,
                    settings=settings,
                    risk_config=risk_config,
                    universe=universe,
                    run_id=run_id,
                    result=result,
                    notifier=notifier if notify else None,
                    track_paper=track_paper,
                )
            except Exception as exc:
                log.exception("Processing failed for %s", snapshot.symbol)
                result.errors.append(f"{snapshot.symbol}: {type(exc).__name__}: {exc}")

        with session_scope() as session:
            run_repo = RunRepo(session)
            stored = session.get(type(run), run_id)
            if stored is not None:
                run_repo.finish(
                    stored,
                    ok=result.ok,
                    symbols_scanned=result.scanned,
                    candidates_found=result.candidates,
                    signals_approved=result.approved,
                    error="; ".join(result.errors) if result.errors else None,
                )

        if notify and notifier.enabled and (result.approved or result.errors):
            notifier.send(
                notifier.run_summary(
                    scanned=result.scanned,
                    candidates=result.candidates,
                    approved=result.approved,
                    rejected=result.rejected,
                    errors=tuple(result.errors),
                )
            )

    finally:
        # Release pooled connections so Neon's compute can suspend promptly,
        # which directly protects the 100 CU-hour monthly budget.
        dispose_engine()

    log.info("Cycle complete: %s", result.describe())
    return result


@dataclass
class CatchUpResult:
    """Outcome of a catch-up pass over bars whose trigger never arrived."""

    current: CycleResult | None = None
    backfilled: list[CycleResult] = field(default_factory=list)
    missed_bars: list[datetime] = field(default_factory=list)

    def describe(self) -> str:
        if not self.backfilled:
            return "no missed bars"
        bars = ", ".join(f"{bar:%m-%d %H:%M}" for bar in self.missed_bars)
        approved = sum(cycle.approved for cycle in self.backfilled)
        return (
            f"backfilled {len(self.backfilled)} bar(s) [{bars}], "
            f"{approved} would have been approved (journalled, not alerted)"
        )


def run_catch_up(
    *,
    settings: Settings | None = None,
    primary: Timeframe = Timeframe.H4,
    max_bars: int = 6,
    notify: bool = True,
    force: bool = False,
    extra_errors: Sequence[str] = (),
) -> CatchUpResult:
    """Analyse the current bar, then any recent bar that was never analysed.

    This exists because GitHub's scheduler is not a clock. Measured over the
    first 37 hours of deployment, 7 of roughly 46 scheduled triggers arrived,
    so the 4h analysis ran 3 times instead of 9. Before this function, a
    trigger that never fired meant that bar was skipped permanently: `run_cycle`
    only ever looked at `_current_bar`, so two thirds of bars were never
    examined at all and the journal recorded nothing about them.

    Two deliberate asymmetries between the current bar and a backfilled one:

    * **Only the current bar is alerted.** A signal for a bar that closed eight
      hours ago is not actionable, and pushing it to Telegram would train you
      to ignore the channel. Backfilled bars are journalled silently.
    * **Only the current bar opens a paper position.** See the note in
      `_process`: a simulated fill on a stale limit order would be fiction.

    Bars are processed oldest-first so the journal reads in chronological
    order, and the current bar runs last so its Telegram alert is the final
    thing to arrive.
    """
    settings = settings or get_settings()
    result = CatchUpResult()

    wait_for_database()

    current = _current_bar(primary)
    step = timedelta(seconds=primary.seconds)

    with session_scope() as session:
        result.missed_bars = _missed_bars(
            RunRepo(session),
            timeframe=primary.value,
            current=current,
            step=step,
            max_bars=max_bars,
        )

    for bar in result.missed_bars:
        log.info("Backfilling missed bar %s", bar)
        cycle = run_cycle(
            settings=settings,
            trigger="backfill",
            primary=primary,
            as_of=bar,
            notify=False,
            track_paper=False,
        )
        result.backfilled.append(cycle)

    # The ingest errors ride on the current bar's cycle, not the backfilled
    # ones: they describe the state of the data now, and the current cycle is
    # the one that notifies.
    result.current = run_cycle(
        settings=settings,
        trigger="cron",
        primary=primary,
        notify=notify,
        force=force,
        extra_errors=extra_errors,
    )
    return result


def _missed_bars(
    runs: RunRepo,
    *,
    timeframe: str,
    current: datetime,
    step: timedelta,
    max_bars: int,
) -> list[datetime]:
    """Recent bars with no successful run, oldest-first.

    Walks backwards from the bar before `current` and stops at the first bar
    that has been analysed. Stopping rather than continuing is deliberate:
    the bars behind an analysed one were either done too or fell outside an
    earlier window, and scanning past it would make every run re-examine the
    whole table for the sake of a gap that will never be filled.

    `max_bars` bounds the replay so one cron tick cannot turn a long outage, or
    a freshly seeded database, into an unbounded backfill.
    """
    if max_bars <= 0:
        return []

    missed: list[datetime] = []
    for index in range(1, max_bars + 1):
        bar = current - step * index
        if runs.already_ran(timeframe, bar):
            break
        missed.append(bar)
    return sorted(missed)


def _analyse(
    universe: Universe, primary: Timeframe, as_of: datetime | None = None
) -> tuple[list[FeatureSnapshot], list[tuple[str, str]]]:
    """Load candles from the database and run the analysis pipeline.

    `as_of` truncates every series at that bar, so a missed bar is analysed
    with the information available at the time and nothing later.
    """
    timeframes = list(universe.all_timeframes)

    with session_scope() as session:
        repo = CandleRepo(session)
        btc = repo.latest("BTCUSDT", primary.value, limit=BARS_TO_LOAD, as_of=as_of)
        eth = repo.latest("ETHUSDT", primary.value, limit=BARS_TO_LOAD, as_of=as_of)

        payloads: list[SymbolData] = []
        for config in universe.symbols:
            frames = {
                timeframe: repo.latest(
                    config.symbol, timeframe, limit=BARS_TO_LOAD, as_of=as_of
                )
                for timeframe in timeframes
            }
            if not frames.get(primary.value):
                log.info("No %s candles stored for %s", primary.value, config.symbol)
                continue
            payloads.append(
                SymbolData.from_candles(
                    config.symbol,
                    config.tier,
                    frames,
                    btc_candles=btc or None,
                    eth_candles=eth or None,
                )
            )

    return analyse_universe(payloads, primary=primary)


def _process(
    snapshot: FeatureSnapshot,
    *,
    settings: Settings,
    risk_config: RiskConfig,
    universe: Universe,
    run_id: int | None,
    result: CycleResult,
    notifier: TelegramNotifier | None,
    track_paper: bool = True,
) -> None:
    """Take one snapshot from verdict to journal entry (and maybe an order)."""
    features = snapshot.to_dict()

    # --- no directional call: journal and move on ------------------------
    if snapshot.side is Side.FLAT or snapshot.confluence.is_blocked:
        status = (
            SignalStatus.REJECTED_RISK
            if snapshot.confluence.is_blocked
            else SignalStatus.CANDIDATE
        )
        with session_scope() as session:
            SignalRepo(session).record(
                symbol=snapshot.symbol,
                tier=snapshot.tier.value,
                side=snapshot.side if snapshot.side is not Side.FLAT else Side.FLAT,
                status=status,
                bar_time=snapshot.bar_time,
                timeframe=snapshot.timeframe,
                score=Decimal(str(round(snapshot.score, 6))),
                features=features,
                reasons={"top": list(snapshot.confluence.top_reasons())},
                vetoes={"blockers": list(snapshot.confluence.blockers)},
                run_id=run_id,
            )
        if snapshot.confluence.is_blocked:
            result.rejected += 1
        return

    result.candidates += 1

    # --- build the setup --------------------------------------------------
    proposal = build_setup(
        snapshot,
        atr_multiple=risk_config.decimal("per_trade", "stop", "atr_multiple", default="2.0"),
        satellite_atr_multiple=risk_config.decimal(
            "per_trade", "stop", "satellite_atr_multiple", default="3.0"
        ),
        min_risk_reward=risk_config.min_risk_reward,
    )
    if not proposal.ok or proposal.setup is None:
        with session_scope() as session:
            SignalRepo(session).record(
                symbol=snapshot.symbol,
                tier=snapshot.tier.value,
                side=snapshot.side,
                status=SignalStatus.REJECTED_RISK,
                bar_time=snapshot.bar_time,
                timeframe=snapshot.timeframe,
                score=Decimal(str(round(snapshot.score, 6))),
                features=features,
                reasons={"top": list(snapshot.confluence.top_reasons())},
                vetoes={"setup": proposal.reason},
                run_id=run_id,
            )
        result.rejected += 1
        log.info("%s: no viable setup (%s)", snapshot.symbol, proposal.reason)
        return

    setup = proposal.setup

    # --- risk gate --------------------------------------------------------
    with session_scope() as session:
        portfolio = _portfolio_state(session, risk_config)
        spec = ContractSpecRepo(session).get(snapshot.symbol)
        history_days = CandleRepo(session).history_days(
            snapshot.symbol, snapshot.timeframe.value
        )

    quote, funding = _market_data(snapshot.symbol, settings)

    gate = RiskGate(risk_config)
    decision = gate.evaluate(
        setup,
        tier=snapshot.tier,
        portfolio=portfolio,
        spec=spec,
        quote=quote,
        funding=funding,
        regime=snapshot.regime,
        beta=snapshot.beta_to_btc,
        tiers_by_symbol={config.symbol: config.tier for config in universe.symbols},
        history_days=history_days,
        # Signal-only mode never places an order, so a missing quote should not
        # suppress the analysis. Live and paper modes require it.
        require_market_data=settings.trading_mode is not TradingMode.SIGNAL_ONLY,
    )

    if not decision.approved:
        with session_scope() as session:
            SignalRepo(session).record(
                symbol=snapshot.symbol,
                tier=snapshot.tier.value,
                side=snapshot.side,
                status=SignalStatus.REJECTED_RISK,
                bar_time=snapshot.bar_time,
                timeframe=snapshot.timeframe,
                setup=setup,
                score=Decimal(str(round(snapshot.score, 6))),
                features=features,
                reasons={"top": list(snapshot.confluence.top_reasons())},
                vetoes={
                    "rules": [
                        {"rule": veto.rule, "detail": veto.detail}
                        for veto in decision.vetoes
                    ]
                },
                run_id=run_id,
            )
        result.rejected += 1
        log.info("%s rejected: %s", snapshot.symbol, decision.veto_summary)
        if notifier is not None and notifier.enabled:
            notifier.send(
                notifier.rejection_message(
                    symbol=snapshot.symbol,
                    side=snapshot.side,
                    decision=decision,
                    score=snapshot.score,
                    setup=setup,
                    reasons=tuple(snapshot.confluence.top_reasons()),
                )
            )
        return

    # --- approved ---------------------------------------------------------
    size = decision.size
    result.approved += 1

    with session_scope() as session:
        signal = SignalRepo(session).record(
            symbol=snapshot.symbol,
            tier=snapshot.tier.value,
            side=snapshot.side,
            status=SignalStatus.APPROVED,
            bar_time=snapshot.bar_time,
            timeframe=snapshot.timeframe,
            setup=setup,
            score=Decimal(str(round(snapshot.score, 6))),
            features=features,
            reasons={
                "top": list(snapshot.confluence.top_reasons()),
                "entry_basis": proposal.entry_basis,
                "stop_basis": proposal.stop_basis,
                "target_basis": proposal.target_basis,
            },
            vetoes=None,
            quantity=size.quantity if size else None,
            notional=size.notional if size else None,
            margin=size.margin if size else None,
            leverage=size.leverage if size else None,
            risk_amount=size.risk_amount if size else None,
            run_id=run_id,
        )
        signal_id = signal.id

    tracked = False
    detail = ""

    # --- paper tracking ---------------------------------------------------
    # Not execution: nothing leaves this process and no venue is contacted. A
    # simulated position is opened so the journal can later score this
    # prediction in R, which is the only way to learn whether the analysis is
    # worth anything.
    # A backfilled bar passes track_paper=False. Opening a simulated position
    # from a bar that closed hours ago would invent a fill: the entry is a
    # limit order that may never have been touched, and reconstructing whether
    # it filled needs a tick-level replay, which is the backtest harness's job
    # rather than something to fake here. The signal and its reasoning are
    # still journalled, which is the part worth having.
    if track_paper and settings.tracks_paper_positions and size is not None:
        with session_scope() as session:
            stored_signal = SignalRepo(session).get(signal_id)
            if stored_signal is not None:
                tracker = PaperTracker(
                    session=session,
                    taker_fee_pct=risk_config.decimal(
                        "costs", "assumed_taker_fee_pct", default="0.06"
                    ),
                    satellite_symbols=satellite_symbols(universe),
                )
                track = tracker.open(
                    signal=stored_signal,
                    setup=setup,
                    size=size,
                    quote=quote,
                    spec=spec,
                )
                tracked = track.ok
                detail = track.detail
                if track.ok:
                    stored_signal.status = SignalStatus.EXECUTED.value
                    result.executed += 1

    log.info(
        "%s APPROVED %s score=%+.3f%s",
        snapshot.symbol,
        snapshot.side.value,
        snapshot.score,
        f" | {detail}" if detail else "",
    )

    if notifier is not None and notifier.enabled:
        sent = notifier.send(
            notifier.signal_message(
                symbol=snapshot.symbol,
                tier=snapshot.tier.value,
                setup=setup,
                size=size,
                score=snapshot.score,
                reasons=snapshot.confluence.top_reasons(),
                warnings=decision.warnings,
                is_paper=True,
                executed=tracked,
            )
        )
        if sent:
            with session_scope() as session:
                stored = SignalRepo(session).get(signal_id)
                if stored is not None:
                    SignalRepo(session).mark_notified(stored)


def _portfolio_state(session, risk_config: RiskConfig) -> PortfolioState:
    """Assemble live portfolio state for the risk gate."""
    positions = PositionRepo(session)
    account = AccountRepo(session)
    breaker = CircuitBreakerRepo(session).active()

    latest = account.latest()
    equity = latest.equity if latest is not None else risk_config.assumed_equity

    day_start = datetime.now(tz=UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    return PortfolioState(
        equity=equity,
        open_positions=tuple(positions.open_positions()),
        consecutive_losses=positions.consecutive_losses(),
        realised_pnl_today=positions.realised_pnl_since(day_start),
        equity_at_day_start=account.equity_at_day_start() or equity,
        halted_until=breaker.halted_until if breaker else None,
        halt_reason=breaker.reason if breaker else None,
    )


def _market_data(symbol: str, settings: Settings):
    """Fetch a public quote and funding rate from WEEX.

    No credentials: these are public endpoints. Returns `(None, None)` on any
    failure, and the risk gate decides what that means - a veto when a paper
    position would be opened, tolerated in signal-only mode.
    """
    from ..exchanges.weex_market import fetch_market_data

    return fetch_market_data(symbol)


def satellite_symbols(universe: Universe) -> frozenset[str]:
    """Symbols that get wider assumed slippage in paper fills."""
    return frozenset(
        config.symbol
        for config in universe.symbols
        if config.tier in (Tier.SATELLITE, Tier.PAPER)
    )


def _current_bar(timeframe: Timeframe) -> datetime:
    """The open time of the most recently CLOSED bar.

    Uses the closed bar, never the in-progress one. A cron that fires at 16:03
    must analyse the 12:00 bar, because the 16:00 bar has not finished and
    feeding a partial candle to an indicator is lookahead bias.
    """
    now = datetime.now(tz=UTC)
    seconds = timeframe.seconds
    epoch_seconds = int(now.timestamp())
    current_open = epoch_seconds - (epoch_seconds % seconds)
    return datetime.fromtimestamp(current_open - seconds, tz=UTC)


def _git_sha() -> str | None:
    """Record which commit produced a signal.

    Reads the CI-provided environment variable first: there is no git
    repository in a built container, so `git rev-parse` only works locally.
    """
    for variable in ("GIT_SHA", "GITHUB_SHA", "RENDER_GIT_COMMIT", "SOURCE_COMMIT"):
        value = os.getenv(variable)
        if value:
            return value[:40]
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
            )
            .decode()
            .strip()[:40]
        )
    except Exception:
        return None


def _config_fingerprint(risk_config: RiskConfig) -> str:
    """Hash the risk rules in force.

    Without this, a journal spanning months of config edits cannot be segmented
    by what the rules actually were, which makes the self-review loop compare
    results produced under different regimes as if they were the same.
    """
    import hashlib
    import json

    payload = json.dumps(risk_config.raw, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
