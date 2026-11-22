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
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
from ..execution.engine import ExecutionEngine
from ..execution.paper import PaperBroker
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
) -> CycleResult:
    """Run one full analysis cycle.

    `force` bypasses the duplicate-bar guard, for manual re-runs.
    """
    settings = settings or get_settings()
    universe = load_universe()
    risk_config = load_risk()
    result = CycleResult()

    wait_for_database()

    bar_time = _current_bar(primary)
    result.bar_time = bar_time

    with session_scope() as session:
        if not force and RunRepo(session).already_ran(primary.value, bar_time, trigger):
            result.skipped_reason = (
                f"bar {bar_time:%Y-%m-%d %H:%M} has already been analysed "
                f"(trigger={trigger}); pass force=True to re-run"
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
        snapshots, failures = _analyse(universe, primary)
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


def _analyse(
    universe: Universe, primary: Timeframe
) -> tuple[list[FeatureSnapshot], list[tuple[str, str]]]:
    """Load candles from the database and run the analysis pipeline."""
    timeframes = list(universe.all_timeframes)

    with session_scope() as session:
        repo = CandleRepo(session)
        btc = repo.latest("BTCUSDT", primary.value, limit=BARS_TO_LOAD)
        eth = repo.latest("ETHUSDT", primary.value, limit=BARS_TO_LOAD)

        payloads: list[SymbolData] = []
        for config in universe.symbols:
            frames = {
                timeframe: repo.latest(config.symbol, timeframe, limit=BARS_TO_LOAD)
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

    executed = False
    detail = ""

    # --- execution --------------------------------------------------------
    if settings.trading_mode is not TradingMode.SIGNAL_ONLY and size is not None:
        with session_scope() as session:
            stored_signal = SignalRepo(session).get(signal_id)
            if stored_signal is not None:
                venue = _build_venue(settings, universe)
                engine = ExecutionEngine(session=session, venue=venue, spec=spec)
                report = engine.execute(
                    signal=stored_signal, setup=setup, size=size, quote=quote
                )
                executed = report.ok
                detail = report.detail
                if report.ok:
                    stored_signal.status = SignalStatus.EXECUTED.value
                    result.executed += 1
                if report.needs_attention and notifier is not None:
                    notifier.alert(
                        "UNPROTECTED POSITION",
                        f"{snapshot.symbol}: {report.detail}\n\n"
                        "A position is open with no stop at the venue. "
                        "Intervene manually.",
                    )

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
                is_paper=settings.trading_mode is not TradingMode.LIVE,
                executed=executed,
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
    """Fetch a live quote and funding rate from WEEX.

    Returns `(None, None)` on any failure. The risk gate decides what that
    means: a veto in paper/live mode, tolerated in signal-only mode.
    """
    from ..exchanges.weex_private import WeexTradingClient

    try:
        with WeexTradingClient(
            api_key=_secret(settings.weex_api_key),
            api_secret=_secret(settings.weex_api_secret),
            api_passphrase=_secret(settings.weex_api_passphrase),
        ) as client:
            return client.quote(symbol), client.funding(symbol)
    except Exception as exc:
        log.warning("Could not fetch WEEX market data for %s: %s", symbol, exc)
        return None, None


def _build_venue(settings: Settings, universe: Universe):
    """Return the paper broker or the verified live client.

    Live mode calls `verify_credentials()` first and falls back to paper if it
    fails. The WEEX signing scheme in this codebase is inferred, so attempting
    a real order against an unverified implementation is exactly the thing to
    avoid.
    """
    satellites = frozenset(
        config.symbol
        for config in universe.symbols
        if config.tier in (Tier.SATELLITE, Tier.PAPER)
    )

    if not settings.can_place_real_orders:
        return PaperBroker(
            equity=load_risk().assumed_equity, satellite_symbols=satellites
        )

    from ..exchanges.weex_private import WeexTradingClient

    client = WeexTradingClient(
        api_key=_secret(settings.weex_api_key),
        api_secret=_secret(settings.weex_api_secret),
        api_passphrase=_secret(settings.weex_api_passphrase),
    )
    verified, detail = client.verify_credentials()
    if not verified:
        log.error(
            "WEEX credential verification FAILED (%s). Falling back to paper "
            "execution rather than sending unverified live orders.",
            detail,
        )
        return PaperBroker(
            equity=load_risk().assumed_equity, satellite_symbols=satellites
        )
    log.info("WEEX credentials %s", detail)
    return client


def _secret(value) -> str | None:
    return value.get_secret_value() if value is not None else None


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
