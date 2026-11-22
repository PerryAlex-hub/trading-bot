"""Command-line interface.

Every scheduled job is also a CLI command, deliberately. The GitHub Actions
workflows invoke exactly these commands, so anything the scheduler does can be
reproduced locally with the same entry point - no separate script that might
drift from what production runs.

    tradingbot init-db              create tables
    tradingbot ingest               sync candles from Binance
    tradingbot analyse              run one analysis cycle
    tradingbot monitor              reconcile positions, check breakers
    tradingbot verify-credentials   prove WEEX signing works (required for live)
    tradingbot status               account, positions, recent signals
    tradingbot journal              recent signals with their reasons
    tradingbot doctor               check the deployment is wired up correctly
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import typer
from rich.console import Console
from rich.table import Table

from .settings import configure_logging, get_settings

app = typer.Typer(
    add_completion=False,
    help="Crypto signal-analysis bot: deterministic analysis, gated execution.",
)
console = Console()
log = logging.getLogger(__name__)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    settings = get_settings()
    if verbose:
        object.__setattr__(settings, "log_level", "DEBUG")
    configure_logging(settings)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


@app.command("init-db")
def init_db(
    drop: bool = typer.Option(False, "--drop", help="DROP every table first"),
) -> None:
    """Create the schema.

    Uses the direct (unpooled) Neon endpoint: PgBouncer in transaction mode
    cannot reliably run DDL.
    """
    from .db.engine import get_migration_engine
    from .db.schema import Base

    engine = get_migration_engine()
    if drop:
        confirm = typer.confirm(
            "This DROPS every table, including the signal journal. Continue?"
        )
        if not confirm:
            raise typer.Abort()
        Base.metadata.drop_all(engine)
        console.print("[yellow]Dropped all tables[/yellow]")

    Base.metadata.create_all(engine)
    console.print(
        f"[green]Schema ready[/green] ({len(Base.metadata.tables)} tables)"
    )
    for name in sorted(Base.metadata.tables):
        console.print(f"  - {name}")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    timeframe: list[str] = typer.Option(None, "--timeframe", "-t"),
    no_prune: bool = typer.Option(False, "--no-prune"),
) -> None:
    """Sync candles from Binance into the database."""
    from .jobs.ingest import ingest_candles

    result = ingest_candles(
        timeframes=tuple(timeframe) if timeframe else None, prune=not no_prune
    )
    console.print(f"[green]Ingest:[/green] {result.describe()}")
    for symbol, tf, error in result.failures:
        console.print(f"  [red]{symbol} {tf}[/red]: {error}")
    if not result.ok:
        raise typer.Exit(code=1)


@app.command("analyse")
def analyse(
    force: bool = typer.Option(False, "--force", help="Re-run an already-analysed bar"),
    no_notify: bool = typer.Option(False, "--no-notify"),
    skip_ingest: bool = typer.Option(False, "--skip-ingest"),
) -> None:
    """Run one full analysis cycle."""
    from .jobs.analyze import run_cycle
    from .jobs.ingest import ingest_candles

    settings = get_settings()
    console.print(
        f"[bold]mode:[/bold] {settings.trading_mode.value} | "
        "orders: never (this project has no execution capability)"
    )

    if not skip_ingest:
        ingest_candles()

    result = run_cycle(settings=settings, trigger="manual", force=force, notify=not no_notify)
    if result.skipped_reason:
        console.print(f"[yellow]{result.skipped_reason}[/yellow]")
        return

    console.print(f"[green]Cycle:[/green] {result.describe()}")
    for error in result.errors:
        console.print(f"  [red]error[/red]: {error}")


@app.command()
def monitor(no_notify: bool = typer.Option(False, "--no-notify")) -> None:
    """Reconcile positions against the venue and check circuit breakers."""
    from .jobs.monitor import run_monitor, snapshot_account

    snapshot_account()
    result = run_monitor(notify=not no_notify)
    console.print(f"[green]Monitor:[/green] {result.describe()}")
    for alert in result.alerts:
        console.print(f"  [red]ALERT[/red]: {alert}")
    if result.breaker_tripped:
        console.print(f"  [red]BREAKER[/red]: {result.breaker_tripped}")


@app.command("refresh-specs")
def refresh_specs() -> None:
    """Refresh cached WEEX contract minimums."""
    from .jobs.ingest import refresh_contract_specs

    count = refresh_contract_specs(use_cache=False)
    if count:
        console.print(f"[green]Cached {count} contract specs[/green]")
    else:
        console.print("[yellow]Could not reach WEEX; existing cache retained[/yellow]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


@app.command("clear-breaker")
def clear_breaker() -> None:
    """Manually clear an active circuit breaker."""
    from .db.engine import session_scope
    from .db.repo import CircuitBreakerRepo

    with session_scope() as session:
        repo = CircuitBreakerRepo(session)
        active = repo.active()
        if active is None:
            console.print("No active circuit breaker")
            return
        console.print(f"[yellow]Active:[/yellow] {active.reason} - {active.detail}")
        if not typer.confirm("Clear it?"):
            raise typer.Abort()
        count = repo.clear_all()
    console.print(f"[green]Cleared {count} breaker(s)[/green]")


@app.command("close-paper")
def close_paper(
    symbol: str = typer.Argument(..., help="e.g. BTCUSDT"),
    reason: str = typer.Option("manual", "--reason"),
) -> None:
    """Close a tracked paper position at the last stored price.

    Bookkeeping only. No order is placed anywhere, because this project has no
    execution capability.
    """
    from .config import load_universe
    from .db.engine import session_scope
    from .db.repo import CandleRepo, PositionRepo
    from .execution.tracker import PaperTracker

    with session_scope() as session:
        position = PositionRepo(session).open_for_symbol(symbol.upper())
        if position is None:
            console.print(f"[yellow]No open paper position in {symbol}[/yellow]")
            raise typer.Exit(code=1)

        series = CandleRepo(session).latest(
            position.symbol, load_universe().primary_timeframe, limit=1
        )
        if not series:
            console.print("[red]No stored candles; cannot price the close[/red]")
            raise typer.Exit(code=1)

        last = series[-1].close
        console.print(
            f"{position.side} {position.quantity} {position.symbol} "
            f"from {position.entry_price}, last price {last}"
        )
        if not typer.confirm("Close it in the journal?"):
            raise typer.Abort()

        resolution = PaperTracker(session=session).expire_stale(
            position, max_age_hours=0, last_price=last
        )

    if resolution is None:
        console.print("[red]Could not close the position[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]{resolution.describe()}[/green]")


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


@app.command("telegram-setup")
def telegram_setup() -> None:
    """Register the Telegram webhook and the command menu.

    Run once after deploying the API, and again if PUBLIC_BASE_URL changes.
    Telegram requires HTTPS, so this cannot target a local server.
    """
    from .notify.bot import register_webhook, set_command_menu, webhook_info

    settings = get_settings()
    ok, detail = register_webhook(settings)
    if not ok:
        console.print(f"[red]Webhook registration failed:[/red] {detail}")
        raise typer.Exit(code=1)

    console.print(f"[green]{detail}[/green]")
    if set_command_menu(settings):
        console.print("[green]Command menu registered[/green]")

    info = webhook_info(settings)
    if info:
        console.print(f"  url           {info.get('url')}")
        console.print(f"  pending       {info.get('pending_update_count')}")
        if info.get("last_error_message"):
            console.print(
                f"  [yellow]last error  {info['last_error_message']}[/yellow]"
            )
    allowed = sorted(settings.allowed_chat_ids())
    console.print()
    if allowed:
        console.print(f"Authorised chat ids: {allowed}")
    else:
        console.print(
            "[red]No authorised chat ids.[/red] Set TELEGRAM_CHAT_ID, or the "
            "bot will ignore every command it receives."
        )


@app.command("telegram-test")
def telegram_test(
    message: str = typer.Option("Test from trading-bot", "--message", "-m"),
) -> None:
    """Send a test message to confirm delivery works."""
    from .notify.telegram import TelegramNotifier

    settings = get_settings()
    notifier = TelegramNotifier(
        bot_token=(
            settings.telegram_bot_token.get_secret_value()
            if settings.telegram_bot_token
            else None
        ),
        chat_id=settings.telegram_chat_id,
    )
    if not notifier.enabled:
        console.print("[red]Telegram is not configured[/red]")
        raise typer.Exit(code=1)
    if notifier.send(message):
        console.print("[green]Sent[/green]")
    else:
        console.print("[red]Send failed; check the token and chat id[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """Account, open positions and the last few runs."""
    from .db.engine import session_scope
    from .db.repo import AccountRepo, CircuitBreakerRepo, PositionRepo, RunRepo

    settings = get_settings()
    console.print(
        f"[bold]environment[/bold] {settings.environment.value} | "
        f"[bold]mode[/bold] {settings.trading_mode.value} | "
        "[bold]orders[/bold] never"
    )

    with session_scope() as session:
        account = AccountRepo(session).latest()
        breaker = CircuitBreakerRepo(session).active()
        positions = PositionRepo(session).open_positions()
        closed = PositionRepo(session).recent_closed(limit=10)
        runs = RunRepo(session).recent(limit=5)

        if account is not None:
            console.print(
                f"\nequity [bold]{account.equity}[/bold] USDT "
                f"(recorded {account.at:%Y-%m-%d %H:%M} UTC)"
            )
        if breaker is not None:
            console.print(
                f"[red]CIRCUIT BREAKER ACTIVE[/red]: {breaker.reason} - "
                f"{breaker.detail} (until {breaker.halted_until:%Y-%m-%d %H:%M} UTC)"
            )

        if positions:
            table = Table(title="Open positions")
            for column in ("symbol", "side", "qty", "entry", "stop", "stop@venue", "paper"):
                table.add_column(column)
            for position in positions:
                table.add_row(
                    position.symbol,
                    position.side,
                    str(position.quantity),
                    str(position.entry_price),
                    str(position.stop_price),
                    "[green]yes[/green]" if position.stop_order_id else "[red]NO[/red]",
                    "yes" if position.is_paper else "[red]LIVE[/red]",
                )
            console.print(table)
        else:
            console.print("\nno open positions")

        if closed:
            wins = sum(1 for p in closed if (p.realised_pnl or 0) > 0)
            total_r = sum((p.r_multiple or Decimal("0")) for p in closed)
            console.print(
                f"\nlast {len(closed)} closed: {wins} wins, "
                f"{len(closed)-wins} losses, total {total_r:+.2f}R"
            )

        if runs:
            table = Table(title="Recent runs")
            for column in ("started", "trigger", "bar", "ok", "scanned", "approved"):
                table.add_column(column)
            for run in runs:
                table.add_row(
                    f"{run.started_at:%m-%d %H:%M}",
                    run.trigger,
                    f"{run.bar_time:%m-%d %H:%M}" if run.bar_time else "-",
                    "[green]yes[/green]" if run.ok else "[red]no[/red]",
                    str(run.symbols_scanned),
                    str(run.signals_approved),
                )
            console.print(table)


@app.command()
def journal(
    limit: int = typer.Option(20, "--limit", "-n"),
    symbol: str = typer.Option(None, "--symbol", "-s"),
    days: int = typer.Option(None, "--days", "-d"),
) -> None:
    """Recent signals and why each was approved or rejected."""
    from .db.engine import session_scope
    from .db.repo import SignalRepo

    since = (
        datetime.now(tz=UTC) - timedelta(days=days) if days else None
    )
    with session_scope() as session:
        signals = SignalRepo(session).recent(
            limit=limit, symbol=symbol.upper() if symbol else None, since=since
        )

        if not signals:
            console.print("no signals recorded")
            return

        table = Table(title=f"Journal ({len(signals)} signals)")
        for column in ("bar", "symbol", "side", "status", "score", "R:R", "detail"):
            table.add_column(column, overflow="fold")

        for signal in signals:
            detail = ""
            if signal.vetoes:
                rules = signal.vetoes.get("rules") or []
                if rules:
                    detail = "; ".join(rule.get("rule", "") for rule in rules[:2])
                elif signal.vetoes.get("setup"):
                    detail = str(signal.vetoes["setup"])[:70]
                elif signal.vetoes.get("blockers"):
                    detail = "; ".join(signal.vetoes["blockers"])[:70]
            elif signal.reasons:
                top = signal.reasons.get("top") or []
                detail = top[0][:70] if top else ""

            colour = {
                "approved": "green",
                "executed": "bold green",
                "rejected_risk": "yellow",
                "candidate": "dim",
            }.get(signal.status, "white")

            table.add_row(
                f"{signal.bar_time:%m-%d %H:%M}",
                signal.symbol,
                signal.side,
                f"[{colour}]{signal.status}[/{colour}]",
                f"{signal.score:+.3f}" if signal.score is not None else "-",
                f"{signal.risk_reward:.2f}" if signal.risk_reward is not None else "-",
                detail,
            )
        console.print(table)


@app.command()
def doctor() -> None:
    """Check the deployment is wired up correctly.

    Run this first after deploying. It verifies the things that actually break
    in practice rather than the things that are easy to check.
    """
    settings = get_settings()
    problems: list[str] = []
    warnings: list[str] = []

    console.print("[bold]Configuration[/bold]")
    console.print(f"  environment      {settings.environment.value}")
    console.print(f"  trading mode     {settings.trading_mode.value}")
    console.print("  orders           never (no execution code, no credentials)")

    # --- database ---------------------------------------------------------
    if settings.database_url is None:
        problems.append("DATABASE_URL is not set")
        console.print("  database         [red]not configured[/red]")
    else:
        url = settings.require_database_url()
        pooled = "-pooler" in url
        console.print(
            f"  database         configured ({'pooled' if pooled else 'DIRECT'})"
        )
        if not pooled:
            warnings.append(
                "DATABASE_URL is the direct endpoint. Use the '-pooler' host for "
                "the app: Neon's free compute suspends after 5 idle minutes and "
                "the pooler absorbs the reconnect."
            )
        try:
            from sqlalchemy import text

            from .db.engine import get_engine

            with get_engine().connect() as conn:
                version = conn.execute(text("SELECT version()")).scalar_one()
            console.print(f"  connection       [green]ok[/green] ({str(version)[:40]})")

            from sqlalchemy import inspect

            from .db.schema import Base

            existing = set(inspect(get_engine()).get_table_names())
            missing = set(Base.metadata.tables) - existing
            if missing:
                problems.append(
                    f"missing tables: {', '.join(sorted(missing))}. Run `tradingbot init-db`."
                )
            else:
                console.print(f"  schema           [green]all {len(Base.metadata.tables)} tables present[/green]")
        except Exception as exc:
            problems.append(f"cannot connect to the database: {exc}")
            console.print(f"  connection       [red]failed[/red]: {exc}")

    # --- config files -----------------------------------------------------
    try:
        from .config import load_risk, load_universe

        universe = load_universe()
        risk = load_risk()
        console.print(
            f"  universe         {len(universe.symbols)} symbols, "
            f"primary {universe.primary_timeframe}"
        )

        # The config contradiction found during development: a symbol whose own
        # beta exceeds the portfolio cap can never be traded at all.
        cap = risk.decimal("portfolio", "max_portfolio_btc_beta", default="2.0")
        unreachable = [
            config.symbol
            for config in universe.symbols
            if config.tier.tradeable and risk.beta_to_btc(config.symbol) > cap
        ]
        if unreachable:
            warnings.append(
                f"these symbols can NEVER trade: {', '.join(unreachable)}. Their "
                f"configured beta_to_btc exceeds max_portfolio_btc_beta ({cap}), "
                "so the gate vetoes them even on an empty book."
            )
    except Exception as exc:
        problems.append(f"config failed to load: {exc}")

    # --- Telegram (the user interface, so this is load-bearing) ----------
    if settings.telegram_bot_token and settings.telegram_chat_id:
        console.print("  telegram         configured")
    else:
        problems.append(
            "Telegram is not configured. It is the only delivery channel AND "
            "the only user interface, so without it the bot runs silently."
        )

    if settings.telegram_bot_token is not None:
        if settings.telegram_webhook_secret is None:
            warnings.append(
                "TELEGRAM_WEBHOOK_SECRET is not set, so the webhook rejects all "
                "traffic and interactive commands will not work. Push signals "
                "still work. Generate one with: python -c "
                "\"import secrets; print(secrets.token_urlsafe(32))\""
            )
        if not settings.public_base_url:
            warnings.append(
                "PUBLIC_BASE_URL is not set, so `tradingbot telegram-setup` "
                "cannot register the webhook."
            )
        if not settings.allowed_chat_ids():
            warnings.append(
                "No authorised chat ids, so the bot will ignore every command."
            )
        else:
            console.print(
                f"  bot commands     {len(settings.allowed_chat_ids())} authorised chat(s)"
            )

    # --- verdict ----------------------------------------------------------
    console.print()
    for warning in warnings:
        console.print(f"[yellow]warning[/yellow] {warning}")
    for problem in problems:
        console.print(f"[red]problem[/red] {problem}")

    if problems:
        console.print(f"\n[red]{len(problems)} problem(s) found[/red]")
        raise typer.Exit(code=1)
    console.print("\n[green]Deployment looks healthy[/green]")


if __name__ == "__main__":
    app()
