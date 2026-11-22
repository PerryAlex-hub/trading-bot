"""Paper position resolution and circuit breakers.

This job answers the only question that matters about a signal bot: **were the
signals right?** There is no venue to reconcile against, because the bot places
no orders. Instead it walks the candles printed since each simulated entry and
decides whether the stop or the target was reached.

Cheap by design: it exits in seconds when nothing is open, which is what makes
a frequent schedule affordable inside GitHub Actions' free minutes.

Two honesty rules, both enforced in `execution/tracker.py`:

* If one bar's range contains both the stop and the target, the **stop** is
  assumed to have filled. OHLC cannot order two touches inside a bar, and
  assuming the favourable one is how paper results stop predicting reality.
* Round-trip taker fees are deducted, so a reported R-multiple is net.

Circuit breakers trip here rather than at signal time, because that is when
losses are discovered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from ..config import load_risk, load_universe
from ..db.engine import dispose_engine, session_scope, wait_for_database
from ..db.repo import (
    AccountRepo,
    CandleRepo,
    CircuitBreakerRepo,
    PositionRepo,
)
from ..notify.telegram import TelegramNotifier
from ..risk.gate import check_circuit_breakers_after_close
from ..settings import Settings, get_settings

log = logging.getLogger(__name__)

# A position whose stop and target are both never touched would otherwise hold
# one of the two portfolio slots risk.yaml allows, forever, blocking every
# later signal on that symbol.
DEFAULT_MAX_HOLD_HOURS = 24 * 14

# Enough 4h bars to cover the maximum hold with room to spare.
BARS_FOR_RESOLUTION = 200


@dataclass
class MonitorResult:
    open_positions: int = 0
    resolved: int = 0
    expired: int = 0
    ambiguous: int = 0
    breaker_tripped: str | None = None
    resolutions: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)

    @property
    def quiet(self) -> bool:
        """Nothing happened, so the caller can skip notifying."""
        return (
            self.resolved == 0
            and self.expired == 0
            and self.breaker_tripped is None
            and not self.alerts
        )

    def describe(self) -> str:
        return (
            f"{self.open_positions} open, {self.resolved} resolved, "
            f"{self.expired} expired"
            + (f", {self.ambiguous} same-bar" if self.ambiguous else "")
        )


def run_monitor(
    *, settings: Settings | None = None, notify: bool = True
) -> MonitorResult:
    """Resolve open paper positions against recent candles."""
    settings = settings or get_settings()
    risk_config = load_risk()
    universe = load_universe()
    result = MonitorResult()

    wait_for_database()
    try:
        with session_scope() as session:
            open_positions = PositionRepo(session).open_positions()
            result.open_positions = len(open_positions)

        # Fast exit keeps a frequent schedule inside the free minute budget.
        if not open_positions:
            log.info("No open positions; monitor exiting early")
            return result

        notifier = TelegramNotifier(
            bot_token=(
                settings.telegram_bot_token.get_secret_value()
                if settings.telegram_bot_token
                else None
            ),
            chat_id=settings.telegram_chat_id,
        )

        _resolve_positions(
            result,
            notifier if notify else None,
            primary_timeframe=universe.primary_timeframe,
            taker_fee_pct=risk_config.decimal(
                "costs", "assumed_taker_fee_pct", default="0.06"
            ),
        )
        _maybe_trip_breaker(result, notifier if notify else None)

        if notify and notifier.enabled and not result.quiet:
            notifier.send(f"Monitor: {result.describe()}")

    finally:
        # Let Neon's compute suspend promptly, which protects the free
        # 100 CU-hour monthly budget.
        dispose_engine()

    log.info("Monitor complete: %s", result.describe())
    return result


def _resolve_positions(
    result: MonitorResult,
    notifier: TelegramNotifier | None,
    *,
    primary_timeframe: str,
    taker_fee_pct: Decimal,
) -> None:
    """Check each open paper position against the bars since it opened."""
    from ..execution.tracker import PaperTracker

    with session_scope() as session:
        positions = PositionRepo(session)
        candles = CandleRepo(session)
        tracker = PaperTracker(session=session, taker_fee_pct=taker_fee_pct)

        for position in positions.open_positions():
            if not position.is_paper:
                # Should not exist: nothing in this codebase can open a live
                # position. Flag it rather than silently ignoring it.
                result.alerts.append(
                    f"{position.symbol}: a NON-PAPER position exists, which this "
                    "bot cannot create. Investigate manually."
                )
                continue

            series = candles.latest(
                position.symbol, primary_timeframe, limit=BARS_FOR_RESOLUTION
            )
            if not series:
                log.warning(
                    "No %s candles stored for %s; cannot resolve",
                    primary_timeframe,
                    position.symbol,
                )
                continue

            resolution = tracker.resolve(position, series)
            if resolution is None:
                # Still running. Expire it if it has gone nowhere for too long.
                resolution = tracker.expire_stale(
                    position,
                    max_age_hours=DEFAULT_MAX_HOLD_HOURS,
                    last_price=series[-1].close,
                )
                if resolution is not None:
                    result.expired += 1
            else:
                result.resolved += 1
                if resolution.ambiguous:
                    result.ambiguous += 1

            if resolution is None:
                continue

            result.resolutions.append(resolution.describe())
            log.info("Resolved %s", resolution.describe())

            if notifier is not None and notifier.enabled:
                notifier.send(
                    notifier.position_closed_message(
                        symbol=position.symbol,
                        side=position.side,
                        exit_price=resolution.exit_price,
                        pnl=position.realised_pnl,
                        pnl_pct=position.realised_pnl_pct,
                        r_multiple=position.r_multiple,
                        reason=(
                            f"{resolution.reason} hit"
                            + (
                                " (stop and target in one bar: stop assumed)"
                                if resolution.ambiguous
                                else ""
                            )
                        ),
                        is_paper=True,
                    )
                )


def _maybe_trip_breaker(result: MonitorResult, notifier: TelegramNotifier | None) -> None:
    """Trip a circuit breaker if the day's losses breached a limit.

    Checked here because losses are discovered at resolution time, not when the
    signal was produced.
    """
    risk_config = load_risk()
    with session_scope() as session:
        positions = PositionRepo(session)
        account = AccountRepo(session)
        breaker_repo = CircuitBreakerRepo(session)

        if breaker_repo.active() is not None:
            return

        day_start = datetime.now(tz=UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        realised = positions.realised_pnl_since(day_start)
        start_equity = account.equity_at_day_start() or risk_config.assumed_equity
        daily_loss_pct = (
            abs(realised) / start_equity * Decimal("100")
            if realised < 0 and start_equity > 0
            else Decimal("0")
        )

        breach = check_circuit_breakers_after_close(
            risk_config,
            consecutive_losses=positions.consecutive_losses(),
            daily_loss_pct=daily_loss_pct,
        )
        if breach is None:
            return

        reason, detail, hours = breach
        breaker_repo.trip(reason=reason, detail=detail, hours=hours)
        result.breaker_tripped = f"{reason}: {detail}"
        log.warning("Circuit breaker tripped: %s", result.breaker_tripped)

        if notifier is not None and notifier.enabled:
            notifier.alert(
                "CIRCUIT BREAKER TRIPPED",
                f"{detail}\n\nNo new signals will be approved for {hours:.0f}h.",
            )


def snapshot_account(*, settings: Settings | None = None) -> Decimal | None:
    """Record simulated equity.

    The daily-loss breaker measures against the day's opening equity, so a
    snapshot must exist before the first resolution of the day or the breaker
    has no baseline. Equity is derived from the assumed starting balance plus
    realised paper P&L; there is no account to query.
    """
    settings = settings or get_settings()
    risk_config = load_risk()

    wait_for_database()
    with session_scope() as session:
        positions = PositionRepo(session)
        account = AccountRepo(session)

        latest = account.latest()
        baseline = latest.equity if latest is not None else risk_config.assumed_equity

        # Only P&L realised since the last snapshot, so equity is not
        # double-counted every time this runs.
        since = (
            latest.at
            if latest is not None
            else datetime.now(tz=UTC).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        )
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
        equity = baseline + positions.realised_pnl_since(since)

        account.snapshot(
            equity=equity,
            open_positions=len(positions.open_positions()),
            is_paper=True,
        )
    log.info("Recorded simulated equity snapshot: %s USDT", equity)
    return equity
