"""Position monitoring and venue reconciliation.

Runs far more often than the analysis cycle and is deliberately cheap: it exits
in seconds when nothing is open. That is what makes a frequent schedule
affordable on GitHub Actions' 2,000 free private-repo minutes.

It exists because of one consequence of the free-tier design: **the bot is
asleep between runs, so the database's idea of the world can be wrong.** A stop
can trigger at 03:00 and the bot will not know until it next wakes. Three
things follow:

1. **Reconcile before anything else.** The venue is the source of truth. If the
   database says a position is open and WEEX says it is not, the stop filled,
   and the position must be closed in the journal with the correct exit.
2. **An open position without a resting stop is an incident.** Nothing is
   watching it. The monitor flattens it rather than hoping the next run is soon.
3. **Circuit breakers trip here.** Losses are discovered at reconciliation, not
   at execution, so the breaker check belongs on this path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from ..config import load_risk
from ..db.engine import dispose_engine, session_scope, wait_for_database
from ..db.repo import (
    AccountRepo,
    CircuitBreakerRepo,
    OrderRepo,
    PositionRepo,
)
from ..models import OrderStatus
from ..notify.telegram import TelegramNotifier
from ..risk.gate import check_circuit_breakers_after_close
from ..settings import Settings, TradingMode, get_settings

log = logging.getLogger(__name__)


@dataclass
class MonitorResult:
    open_positions: int = 0
    closed: int = 0
    unprotected_found: int = 0
    unprotected_fixed: int = 0
    orders_reconciled: int = 0
    breaker_tripped: str | None = None
    alerts: list[str] = field(default_factory=list)

    @property
    def quiet(self) -> bool:
        """Whether nothing happened. Lets the caller skip a notification."""
        return (
            self.closed == 0
            and self.unprotected_found == 0
            and self.breaker_tripped is None
            and not self.alerts
        )

    def describe(self) -> str:
        return (
            f"{self.open_positions} open, {self.closed} closed, "
            f"{self.unprotected_found} unprotected, "
            f"{self.orders_reconciled} orders reconciled"
        )


def run_monitor(
    *, settings: Settings | None = None, notify: bool = True
) -> MonitorResult:
    """Reconcile against the venue and act on anything that needs attention."""
    settings = settings or get_settings()
    result = MonitorResult()

    wait_for_database()
    try:
        with session_scope() as session:
            open_positions = PositionRepo(session).open_positions()
            result.open_positions = len(open_positions)
            pending = OrderRepo(session).open_orders()

        # Fast exit. This is what keeps a frequent schedule inside the free
        # Actions minute budget: with no positions the job costs seconds.
        if not open_positions and not pending:
            log.info("Nothing open; monitor exiting early")
            return result

        notifier = TelegramNotifier(
            bot_token=settings.telegram_bot_token.get_secret_value()
            if settings.telegram_bot_token
            else None,
            chat_id=settings.telegram_chat_id,
        )

        venue = _venue(settings)
        if venue is None:
            # Paper positions cannot be reconciled against a real venue, but
            # the unprotected check still applies to live ones.
            log.info("No live venue available; skipping venue reconciliation")
        else:
            _reconcile(venue, result, notifier if notify else None, settings)

        _check_unprotected(result, notifier if notify else None, settings)
        _maybe_trip_breaker(result, notifier if notify else None)

        if notify and notifier.enabled and not result.quiet:
            notifier.send(f"Monitor: {result.describe()}")

    finally:
        dispose_engine()

    log.info("Monitor complete: %s", result.describe())
    return result


def _venue(settings: Settings):
    """A verified live client, or None.

    Returns None rather than a paper broker: reconciliation compares the
    database against reality, and a paper broker is not reality.
    """
    if settings.trading_mode is not TradingMode.LIVE:
        return None

    from ..exchanges.weex_private import WeexTradingClient

    client = WeexTradingClient(
        api_key=_secret(settings.weex_api_key),
        api_secret=_secret(settings.weex_api_secret),
        api_passphrase=_secret(settings.weex_api_passphrase),
    )
    verified, detail = client.verify_credentials()
    if not verified:
        log.error("Cannot reconcile: WEEX credentials failed verification (%s)", detail)
        return None
    return client


def _reconcile(venue, result: MonitorResult, notifier, settings: Settings) -> None:
    """Make the database agree with the venue.

    The venue wins every disagreement. The database is a record of what this
    bot believes; WEEX is a record of what is true.
    """
    try:
        venue_positions = {
            position.symbol: position for position in venue.open_positions()
        }
    except Exception as exc:
        log.error("Could not read positions from the venue: %s", exc)
        result.alerts.append(f"venue position read failed: {exc}")
        return

    with session_scope() as session:
        repo = PositionRepo(session)
        for position in repo.open_positions():
            if position.is_paper:
                continue

            live = venue_positions.get(position.symbol)
            if live is not None:
                # Still open. Flag a size mismatch, which usually means a
                # partial fill or a manual intervention.
                if abs(live.quantity - position.quantity) > position.quantity * Decimal("0.01"):
                    message = (
                        f"{position.symbol}: database says {position.quantity}, "
                        f"venue says {live.quantity}"
                    )
                    log.warning("Position size mismatch: %s", message)
                    result.alerts.append(message)
                continue

            # The venue has no such position: a protective order filled while
            # the bot was asleep. This is the normal, expected case.
            exit_price = _infer_exit_price(position)
            repo.close(
                position,
                exit_price=exit_price,
                reason="closed_at_venue",
            )
            result.closed += 1
            log.info(
                "Reconciled %s as closed at the venue: exit %s, R=%s",
                position.symbol,
                exit_price,
                position.r_multiple,
            )
            if notifier is not None and notifier.enabled:
                notifier.send(
                    notifier.position_closed_message(
                        symbol=position.symbol,
                        side=position.side,
                        exit_price=exit_price,
                        pnl=position.realised_pnl,
                        pnl_pct=position.realised_pnl_pct,
                        r_multiple=position.r_multiple,
                        reason="stop or target filled while the bot was asleep",
                        is_paper=False,
                    )
                )

    # Settle any order rows left PENDING by a crashed run.
    with session_scope() as session:
        order_repo = OrderRepo(session)
        for order in order_repo.open_orders():
            if order.is_paper:
                continue
            if order.venue_order_id is None and order.status == OrderStatus.PENDING.value:
                # Recorded but never confirmed. Mark it FAILED so it does not
                # linger as "maybe live" forever; the position reconciliation
                # above is what actually establishes the truth.
                order.status = OrderStatus.FAILED.value
                order.error = (
                    "no venue order id; the submitting run did not complete. "
                    "Resolved by reconciliation."
                )
                result.orders_reconciled += 1
        session.flush()


def _infer_exit_price(position) -> Decimal:
    """Best available exit price for a position closed while the bot slept.

    Assumes the stop filled, which is the conservative choice: it records the
    loss rather than guessing at a profit. The venue's fill history would be
    better, but its endpoint shape is unverified in this codebase, so this
    errs toward understating results instead of flattering them.
    """
    if position.stop_price is not None:
        return position.stop_price
    return position.entry_price


def _check_unprotected(result: MonitorResult, notifier, settings: Settings) -> None:
    """Find live positions with no stop at the venue and flatten them.

    The alarm condition of the whole design. On free infrastructure nothing is
    watching between runs, so an unprotected position is uncapped risk.
    """
    with session_scope() as session:
        exposed = PositionRepo(session).open_without_stop()
        result.unprotected_found = len(exposed)

        if not exposed:
            return

        for position in exposed:
            message = (
                f"{position.symbol} {position.side} {position.quantity} is OPEN "
                f"with no stop order at the venue"
            )
            log.error(message)
            result.alerts.append(message)

            if settings.can_place_real_orders:
                from ..execution.engine import ExecutionEngine

                venue = _venue(settings)
                if venue is None:
                    continue
                engine = ExecutionEngine(session=session, venue=venue)
                report = engine.close_position(position, reason="unprotected")
                if report.ok:
                    result.unprotected_fixed += 1
                    log.info("Flattened unprotected position %s", position.symbol)

        if notifier is not None and notifier.enabled:
            notifier.alert(
                "UNPROTECTED POSITION(S)",
                "\n".join(result.alerts[:5])
                + f"\n\nFlattened: {result.unprotected_fixed}/{len(exposed)}",
            )


def _maybe_trip_breaker(result: MonitorResult, notifier) -> None:
    """Trip a circuit breaker if the day's losses breached a limit.

    Checked here rather than at execution because losses are *discovered* at
    reconciliation: the stop filled hours ago while the bot was asleep.
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
                f"{detail}\n\nNo new positions for {hours:.0f}h.",
            )


def snapshot_account(*, settings: Settings | None = None) -> Decimal | None:
    """Record current equity.

    The daily-loss breaker is defined against the day's opening equity, so a
    snapshot has to exist before the first trade of the day or the breaker has
    no baseline to measure against.
    """
    settings = settings or get_settings()
    risk_config = load_risk()

    wait_for_database()
    equity: Decimal | None = None
    is_paper = settings.trading_mode is not TradingMode.LIVE

    if not is_paper:
        venue = _venue(settings)
        if venue is not None:
            try:
                equity = venue.account_equity()
            except Exception as exc:
                log.warning("Could not read account equity: %s", exc)

    with session_scope() as session:
        if equity is None:
            latest = AccountRepo(session).latest()
            equity = latest.equity if latest else risk_config.assumed_equity
        open_count = len(PositionRepo(session).open_positions())
        AccountRepo(session).snapshot(
            equity=equity, open_positions=open_count, is_paper=is_paper
        )
    log.info("Recorded equity snapshot: %s USDT", equity)
    return equity


def _secret(value) -> str | None:
    return value.get_secret_value() if value is not None else None
