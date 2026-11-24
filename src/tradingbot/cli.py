"""Command-line interface.

Every scheduled job is also a CLI command, deliberately. The GitHub Actions
workflows invoke exactly these commands, so anything the scheduler does can be
reproduced locally with the same entry point - no separate script that might
drift from what production runs.

    tradingbot init-db              create tables
    tradingbot ingest               sync candles from Binance
    tradingbot analyse              run one analysis cycle
    tradingbot monitor              reconcile positions, check breakers
    tradingbot refresh-specs        reload WEEX contract minimums
    tradingbot status               account, positions, recent signals
    tradingbot journal              recent signals with their reasons
    tradingbot doctor               check the deployment is wired up correctly
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

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
    backfill: int = typer.Option(
        6,
        "--backfill",
        help=(
            "How many recent bars to catch up on if their scheduled run never "
            "fired. 0 disables catch-up and analyses only the current bar."
        ),
    ),
) -> None:
    """Run one full analysis cycle, catching up on any missed bars first."""
    from .jobs.analyze import run_catch_up, run_cycle
    from .jobs.ingest import ingest_candles

    settings = get_settings()
    console.print(
        f"[bold]mode:[/bold] {settings.trading_mode.value} | "
        "orders: never (this project has no execution capability)"
    )

    # Ingest failures used to be discarded here. A symbol that is not listed on
    # the data venue fails on every timeframe, every run, and the only trace
    # was a log line nobody reads - so the universe silently shrank and the
    # run summary still said everything was fine. They are now carried into the
    # cycle, which puts them in the Telegram summary.
    ingest_errors: list[str] = []
    if not skip_ingest:
        ingest = ingest_candles()
        ingest_errors = [
            f"ingest {symbol} {timeframe}: {error}"
            for symbol, timeframe, error in ingest.failures
        ]
        if ingest_errors:
            console.print(
                f"[red]Ingest had {len(ingest.failures)} failure(s)[/red] "
                f"across {len(set(f[0] for f in ingest.failures))} symbol(s)"
            )
            for line in ingest_errors[:8]:
                console.print(f"  [red]{line}[/red]")

    if backfill <= 0:
        result = run_cycle(
            settings=settings,
            trigger="manual",
            force=force,
            notify=not no_notify,
            extra_errors=ingest_errors,
        )
        _report_cycle(result)
        return

    catch_up = run_catch_up(
        settings=settings,
        max_bars=backfill,
        notify=not no_notify,
        force=force,
        extra_errors=ingest_errors,
    )
    if catch_up.backfilled:
        console.print(f"[yellow]Catch-up:[/yellow] {catch_up.describe()}")
        for cycle in catch_up.backfilled:
            console.print(f"  [dim]{cycle.bar_time:%Y-%m-%d %H:%M}[/dim] {cycle.describe()}")
    if catch_up.current is not None:
        _report_cycle(catch_up.current)


def _report_cycle(result) -> None:
    if result.skipped_reason:
        console.print(f"[yellow]{result.skipped_reason}[/yellow]")
        return
    console.print(f"[green]Cycle:[/green] {result.describe()}")
    for error in result.errors:
        console.print(f"  [red]error[/red]: {error}")


@app.command()
def monitor(no_notify: bool = typer.Option(False, "--no-notify")) -> None:
    """Resolve open paper positions against recent candles."""
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


def _dump_payload(path: Path | None, payload: dict[str, object]) -> None:
    """Write the raw venue response, so a failed parse is still diagnostic."""
    if path is None or not payload:
        return
    try:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        console.print(f"[yellow]Could not write {path}:[/yellow] {exc}")
        return
    console.print(f"[dim]Raw venue payload written to {path}[/dim]")


@app.command()
def minimums(
    max_usdt: float = typer.Option(
        2.0, "--max-usdt", help="Only list contracts cheaper than this to open"
    ),
    limit: int = typer.Option(25, "--limit", help="How many to list"),
    leverage: int = typer.Option(10, "--leverage", help="Leverage for the margin column"),
    dump: Path | None = typer.Option(
        None,
        "--dump",
        help="Write the raw venue payloads here, whether or not parsing succeeds",
    ),
) -> None:
    """Rank WEEX contracts by the smallest position you can open.

    Answers "which coin can I test live with for cents" with data instead of
    guesswork. WEEX publishes its minimum in the BASE asset, so the cost in
    USDT is `min_order_size x price`: BTC's 0.0001 minimum is ~6 USDT only
    because BTC is expensive, while a sub-cent token with a minimum of 1 unit
    costs a fraction of a cent. There is no venue-wide USDT floor to escape.

    Needs no database and no credentials - the market endpoints are public.
    """
    from .config import load_universe
    from .exchanges.weex_market import WeexMarketData

    # The raw payload is dumped unconditionally, because this command is
    # routinely run on a machine that can reach WEEX by someone whose helper
    # cannot. The field names here were taken from WEEX's published docs
    # rather than observed, so if they are wrong the parsed table comes back
    # empty and the dump is the only way to find out why without another
    # round trip.
    raw: dict[str, object] = {}
    try:
        with WeexMarketData() as client:
            raw["contracts"] = client.fetch_contracts_raw(use_cache=False)
            specs = client.contract_specs(use_cache=True)
            marks = client.all_marks()
            raw["parsed_symbols"] = sorted(specs)[:50]
            raw["parsed_marks"] = len(marks)
    except Exception as exc:
        _dump_payload(dump, raw)
        console.print(f"[red]Could not reach WEEX:[/red] {exc}")
        console.print(
            "\nThe market endpoints are public, so this is a network or a "
            "moved-path problem rather than an auth one. Check "
            "exchanges/weex_market.py TICKER_PATHS against the current docs."
        )
        raise typer.Exit(code=1) from exc

    _dump_payload(dump, raw)

    if not marks:
        console.print(
            "[yellow]Contract specs loaded but no prices came back[/yellow], so "
            "minimums cannot be converted to USDT. The ticker path has probably "
            "moved; the spec list is fine."
        )
        raise typer.Exit(code=1)

    if not specs:
        console.print(
            "[yellow]The venue answered but no contracts could be parsed.[/yellow] "
            "That means the field names differ from the documented ones. Re-run "
            "with --dump payload.json and send that file."
        )
        raise typer.Exit(code=1)

    rows: list[tuple[Decimal, str, Decimal, Decimal]] = []
    for symbol, spec in specs.items():
        mark = marks.get(symbol)
        if mark is None or spec.min_order_size is None:
            continue
        notional = spec.effective_min_notional(mark)
        if notional is None:
            continue
        rows.append((notional, symbol, spec.min_order_size, mark))
    rows.sort(key=lambda row: row[0])

    universe_symbols = {
        config.symbol for config in load_universe().symbols
    }

    table = Table(title=f"Smallest openable position (of {len(rows)} contracts priced)")
    table.add_column("Symbol")
    table.add_column("Min size", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Min order (USDT)", justify="right")
    table.add_column(f"Margin @{leverage}x", justify="right")

    shown = 0
    for notional, symbol, min_size, mark in rows:
        if notional > Decimal(str(max_usdt)):
            break
        if shown >= limit:
            break
        margin = notional / Decimal(leverage)
        table.add_row(
            f"{symbol}{' *' if symbol in universe_symbols else ''}",
            f"{min_size.normalize():f}",
            f"{mark.normalize():f}",
            f"{notional:.4f}",
            f"{margin:.4f}",
        )
        shown += 1

    console.print(table)
    if shown == 0:
        console.print(
            f"[yellow]Nothing under {max_usdt} USDT.[/yellow] The cheapest is "
            f"{rows[0][1]} at {rows[0][0]:.4f} USDT."
        )
    console.print("[dim]* already in universe.yaml[/dim]")

    # The symbols actually configured, whether or not they are cheap, because
    # this is the number risk.yaml's fallback_min_notional_usdt is guessing at.
    console.print("\n[bold]Your universe[/bold]")
    for symbol in sorted(universe_symbols):
        spec = specs.get(symbol)
        mark = marks.get(symbol)
        if spec is None:
            console.print(f"  {symbol:<10} [red]not listed on WEEX futures[/red]")
            continue
        if mark is None or spec.min_order_size is None:
            console.print(f"  {symbol:<10} [yellow]no price or no minimum reported[/yellow]")
            continue
        notional = spec.effective_min_notional(mark) or Decimal("0")
        console.print(
            f"  {symbol:<10} min {spec.min_order_size.normalize():f} "
            f"= {notional:.4f} USDT, margin {notional / Decimal(leverage):.4f} "
            f"at {leverage}x"
        )


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

    # --- data coverage ----------------------------------------------------
    # The check that would have caught the real failure on day one: a symbol
    # that the data venue does not list ingests nothing, is skipped by the
    # analysis with an INFO log, and never appears in any output again. Two of
    # six symbols were in that state for days while every run reported success.
    console.print("\n[bold]Data coverage[/bold]")
    try:
        from .config import load_risk, load_universe
        from .db.engine import session_scope
        from .db.repo import CandleRepo

        universe = load_universe()
        primary = universe.primary_timeframe
        floor = load_risk().decimal("gates", "min_24h_quote_volume", default="0")
        empty: list[str] = []
        thin: list[str] = []
        with session_scope() as session:
            repo = CandleRepo(session)
            for symbol in universe.symbols_requiring_data():
                count = repo.count(symbol, primary)
                days = repo.history_days(symbol, primary)
                if count == 0:
                    empty.append(symbol)
                    console.print(f"  {symbol:<10} [red]no {primary} candles[/red]")
                    continue
                volume = repo.quote_volume_24h(symbol, primary)
                line = f"  {symbol:<10} {count} {primary} bars, {days:.0f} days"
                if volume is None:
                    line += ", volume unknown"
                else:
                    line += f", 24h vol {volume:,.0f}"
                    if floor > 0 and volume < floor:
                        thin.append(f"{symbol} ({volume:,.0f})")
                        line += " [red](below floor)[/red]"
                console.print(line)
        if thin:
            warnings.append(
                f"below the {floor:,.0f} liquidity floor: {', '.join(thin)}. The "
                "risk gate vetoes these on gates.min_24h_quote_volume, so they "
                "will never produce a signal while they stay this thin."
            )
        if empty:
            problems.append(
                f"no candles stored for {', '.join(empty)}. These symbols cannot "
                "produce a signal: the analysis skips any symbol with no primary "
                "timeframe data. Either they are not listed on the data venue "
                "(check with `tradingbot ingest`) or ingest has never succeeded "
                "for them. Remove them from universe.yaml or source their "
                "candles elsewhere."
            )
    except Exception as exc:
        warnings.append(f"data coverage check could not run: {exc}")

    # --- venues -----------------------------------------------------------
    # The failure this exists to diagnose: every signal on the first deployment
    # was rejected with `costs.spread_unknown` because no WEEX quote came back,
    # and nothing in the output said whether the venue was unreachable, the
    # endpoint path had moved, or the symbol was simply not listed. Those need
    # three different fixes, so guessing is expensive.
    console.print("\n[bold]Venues[/bold]")
    try:
        from .config import load_universe
        from .exchanges.weex_market import WeexMarketData

        probe_symbol = next(
            (c.symbol for c in load_universe().symbols if c.tier.tradeable),
            "BTCUSDT",
        )
        with WeexMarketData() as client:
            try:
                specs = client.contract_specs(use_cache=False)
                console.print(
                    f"  weex specs       [green]ok[/green] ({len(specs)} contracts)"
                )
                missing = [
                    c.symbol
                    for c in load_universe().symbols
                    if c.symbol.upper() not in specs
                ]
                if missing:
                    warnings.append(
                        f"not listed on WEEX contracts: {', '.join(missing)}. "
                        "Sizing falls back to the configured minimum for these, "
                        "so their tickets use an assumed floor."
                    )
            except Exception as exc:
                warnings.append(
                    f"WEEX contract info unavailable ({str(exc)[:120]}). Sizing "
                    "uses venue.fallback_min_notional_usdt instead of real "
                    "minimums. Run `tradingbot refresh-specs` once it is back."
                )
                console.print("  weex specs       [yellow]unavailable[/yellow]")

            quote = client.quote(probe_symbol)
            if quote is None:
                warnings.append(
                    f"no WEEX quote for {probe_symbol}, so spreads cannot be "
                    "verified and friction is charged at the assumed rate in "
                    "costs.assumed_spread_pct. Signals still issue. If this "
                    "persists, the ticker path in exchanges/weex_market.py may "
                    "have moved, or the host may be blocked from this network."
                )
                console.print(f"  weex quote       [yellow]none for {probe_symbol}[/yellow]")
            else:
                console.print(
                    f"  weex quote       [green]ok[/green] ({probe_symbol} "
                    f"spread {quote.spread_pct:.4f}%)"
                )
    except Exception as exc:
        warnings.append(f"venue probe could not run: {exc}")

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
