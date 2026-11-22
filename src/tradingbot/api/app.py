"""HTTP service: the Telegram webhook, plus a read-only JSON API.

Deployment shape: a Render free web service kept awake by an external pinger
(cron-job.org hitting `/ping` every ~10 minutes). Three things follow.

**`/ping` must be as close to free as possible.** It touches no database and
allocates almost nothing. This matters because Neon's free plan allows 100
CU-hours per month, and a pinger that woke the database every 10 minutes would
hold the compute on permanently and exhaust that budget. Keeping Render awake
and keeping Neon asleep are both requirements, and they conflict unless the
ping endpoint is deliberately inert.

**The webhook is the user interface.** There is no frontend. `/telegram/webhook`
receives commands and the bot replies in chat. The keep-alive is what makes
those replies instant instead of waiting ~60s for a cold start.

**Nothing here can trade.** The bot holds no exchange credentials and there is
no execution code in the project, so this is a structural property rather than
a configuration choice.

Render's free allowance is 750 instance-hours per workspace per month and a
31-day month is 744 hours, so an always-awake service consumes essentially all
of it. If it runs out, every free web service in the workspace is suspended
until the next month - but the GitHub Actions scheduler is independent, so
signals keep arriving in Telegram and only these endpoints go dark.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..__init__ import __version__
from ..db.engine import get_engine, session_scope
from ..db.repo import (
    AccountRepo,
    CircuitBreakerRepo,
    PositionRepo,
    RunRepo,
)
from ..db.schema import Position, Signal
from ..settings import configure_logging, get_settings

log = logging.getLogger(__name__)

app = FastAPI(
    title="trading-bot",
    version=__version__,
    description="Crypto signal-analysis bot. Read-mostly API over the signal journal.",
)

# The frontend is deployed separately (Vercel), so cross-origin reads are
# required. Credentials are not used - the API is read-mostly and token-guarded
# for writes - so a permissive origin list is acceptable here and avoids
# redeploying the backend every time a Vercel preview URL changes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_session() -> Session:
    with session_scope() as session:
        yield session


@app.on_event("startup")
def _startup() -> None:
    configure_logging()
    settings = get_settings()
    log.info(
        "API starting: env=%s mode=%s (no execution capability)",
        settings.environment.value,
        settings.trading_mode.value,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness. Deliberately does NOT touch the database.

    An uptime pinger hitting this every few minutes would otherwise keep Neon's
    compute awake continuously and consume the free 100 CU-hours.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "environment": settings.environment.value,
        "trading_mode": settings.trading_mode.value,
        # Always false, and reported so a consumer never has to infer it.
        "live_orders_enabled": False,
    }


@app.get("/ping")
def ping() -> dict[str, str]:
    """Keep-alive target for an external pinger.

    Deliberately the cheapest endpoint in the service: no database, no config
    load, no settings read. Render free services spin down after 15 minutes
    idle, so a ping every ~10 minutes holds it awake with margin for the
    4-40 second jitter free cron services add.

    Do NOT point a pinger at an endpoint that queries the database. Neon's free
    compute suspends after 5 idle minutes, and waking it every 10 minutes
    around the clock would burn the entire 100 CU-hour monthly allowance.
    """
    return {"status": "awake"}


@app.get("/health/db")
def health_db() -> dict[str, Any]:
    """Readiness, including the database. Use sparingly."""
    from sqlalchemy import text

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "reachable"}
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"database unreachable: {type(exc).__name__}"
        ) from exc


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------


class SignalOut(BaseModel):
    id: int
    symbol: str
    tier: str
    timeframe: str
    bar_time: datetime
    side: str
    status: str
    score: Decimal | None = None
    entry: Decimal | None = None
    stop: Decimal | None = None
    target_1: Decimal | None = None
    risk_reward: Decimal | None = None
    quantity: Decimal | None = None
    notional: Decimal | None = None
    reasons: dict[str, Any] | None = None
    vetoes: dict[str, Any] | None = None
    notified_at: datetime | None = None

    class Config:
        from_attributes = True


class PositionOut(BaseModel):
    id: int
    symbol: str
    side: str
    is_open: bool
    is_paper: bool
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal | None = None
    stop_price: Decimal | None = None
    target_price: Decimal | None = None
    # Exposed because a live position without this is unprotected capital, and
    # the dashboard should show that in red.
    stop_order_id: str | None = None
    opened_at: datetime
    closed_at: datetime | None = None
    close_reason: str | None = None
    realised_pnl: Decimal | None = None
    realised_pnl_pct: Decimal | None = None
    r_multiple: Decimal | None = None

    class Config:
        from_attributes = True


@app.get("/signals", response_model=list[SignalOut])
def list_signals(
    session: Session = Depends(get_session),
    limit: int = Query(50, ge=1, le=500),
    symbol: str | None = None,
    status: str | None = None,
    days: int | None = Query(None, ge=1, le=365),
) -> list[Signal]:
    """The journal, newest first. Includes rejections."""
    statement = select(Signal)
    if symbol:
        statement = statement.where(Signal.symbol == symbol.upper())
    if status:
        statement = statement.where(Signal.status == status)
    if days:
        statement = statement.where(
            Signal.bar_time >= datetime.now(tz=UTC) - timedelta(days=days)
        )
    return list(
        session.execute(
            statement.order_by(Signal.bar_time.desc()).limit(limit)
        ).scalars()
    )


@app.get("/signals/{signal_id}")
def get_signal(signal_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    """One signal including its full feature snapshot.

    The snapshot is the audit record: it holds the exact numbers the decision
    was made from, so a signal can be reviewed without recomputing anything
    from a since-changed codebase.
    """
    signal = session.get(Signal, signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail="signal not found")
    return {
        "signal": SignalOut.model_validate(signal).model_dump(mode="json"),
        "features": signal.features,
        "analyst": {
            "verdict": signal.analyst_verdict,
            "confidence": signal.analyst_confidence,
            "rationale": signal.analyst_rationale,
            "model": signal.analyst_model,
        },
    }


@app.get("/positions", response_model=list[PositionOut])
def list_positions(
    session: Session = Depends(get_session),
    open_only: bool = False,
    limit: int = Query(50, ge=1, le=500),
) -> list[Position]:
    if open_only:
        return PositionRepo(session).open_positions()
    return list(
        session.execute(
            select(Position).order_by(Position.opened_at.desc()).limit(limit)
        ).scalars()
    )


@app.get("/account")
def account(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Equity, open risk, and whether anything is unprotected."""
    settings = get_settings()
    latest = AccountRepo(session).latest()
    positions = PositionRepo(session)
    breaker = CircuitBreakerRepo(session).active()
    exposed = positions.open_without_stop()

    return {
        "equity": latest.equity if latest else None,
        "recorded_at": latest.at if latest else None,
        "open_positions": len(positions.open_positions()),
        "consecutive_losses": positions.consecutive_losses(),
        "trading_mode": settings.trading_mode.value,
        # Always false, and reported so a consumer never has to infer it.
        "live_orders_enabled": False,
        "circuit_breaker": (
            {
                "reason": breaker.reason,
                "detail": breaker.detail,
                "halted_until": breaker.halted_until,
            }
            if breaker
            else None
        ),
        # The alarm the dashboard must surface loudly.
        "unprotected_positions": [
            {"symbol": p.symbol, "side": p.side, "quantity": p.quantity}
            for p in exposed
        ],
    }


@app.get("/performance")
def performance(
    session: Session = Depends(get_session),
    days: int = Query(90, ge=1, le=3650),
) -> dict[str, Any]:
    """Aggregate results in R, which is the only cross-symbol comparison.

    Deliberately reports paper and live separately. Mixing them would make the
    headline number meaningless, since paper fills are modelled and live ones
    are not.
    """
    since = datetime.now(tz=UTC) - timedelta(days=days)
    closed = PositionRepo(session).closed_since(since)

    def summarise(group: list[Position]) -> dict[str, Any]:
        if not group:
            return {"trades": 0}
        wins = [p for p in group if (p.realised_pnl or 0) > 0]
        losses = [p for p in group if (p.realised_pnl or 0) <= 0]
        r_values = [p.r_multiple for p in group if p.r_multiple is not None]
        total_r = sum(r_values) if r_values else Decimal("0")
        gross_win = sum((p.realised_pnl or Decimal("0")) for p in wins)
        gross_loss = abs(sum((p.realised_pnl or Decimal("0")) for p in losses))
        return {
            "trades": len(group),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(group) * 100, 2),
            "total_r": round(float(total_r), 3),
            "avg_r": round(float(total_r) / len(r_values), 3) if r_values else None,
            "total_pnl": sum((p.realised_pnl or Decimal("0")) for p in group),
            # Expectancy in R is the number that tells you whether the strategy
            # works; win rate alone does not.
            "profit_factor": (
                round(float(gross_win / gross_loss), 3) if gross_loss > 0 else None
            ),
        }

    return {
        "window_days": days,
        "paper": summarise([p for p in closed if p.is_paper]),
        "live": summarise([p for p in closed if not p.is_paper]),
    }


@app.get("/runs")
def list_runs(
    session: Session = Depends(get_session), limit: int = Query(20, ge=1, le=200)
) -> list[dict[str, Any]]:
    """Recent analysis runs. Shows whether the scheduler is actually firing."""
    return [
        {
            "id": run.id,
            "trigger": run.trigger,
            "bar_time": run.bar_time,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "ok": run.ok,
            "symbols_scanned": run.symbols_scanned,
            "candidates_found": run.candidates_found,
            "signals_approved": run.signals_approved,
            "git_sha": run.git_sha,
            "error": run.error,
        }
        for run in RunRepo(session).recent(limit=limit)
    ]


@app.get("/universe")
def universe() -> dict[str, Any]:
    """The configured universe and risk rules, for the frontend to render."""
    from ..config import load_risk, load_universe

    uni = load_universe()
    risk = load_risk()
    cap = risk.decimal("portfolio", "max_portfolio_btc_beta", default="2.0")

    return {
        "quote": uni.quote,
        "primary_timeframe": uni.primary_timeframe,
        "timeframes": list(uni.all_timeframes),
        "symbols": [
            {
                "symbol": config.symbol,
                "tier": config.tier.value,
                "tradeable": config.tier.tradeable,
                "notes": config.notes,
                "beta_to_btc": risk.beta_to_btc(config.symbol),
                # Surfaced because this is a config state that silently
                # disables a symbol entirely.
                "blocked_by_beta_cap": risk.beta_to_btc(config.symbol) > cap,
            }
            for config in uni.symbols
        ],
        "risk": {
            "risk_pct": risk.risk_pct,
            "min_risk_reward": risk.min_risk_reward,
            "max_open_positions": risk.get("portfolio", "max_open_positions"),
            "max_portfolio_btc_beta": cap,
        },
    }


@app.get("/stats")
def stats(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Counts for the dashboard header."""
    signal_counts = dict(
        session.execute(
            select(Signal.status, func.count()).group_by(Signal.status)
        ).all()
    )
    return {
        "signals_by_status": signal_counts,
        "signals_total": sum(signal_counts.values()),
        "positions_open": session.execute(
            select(func.count()).select_from(Position).where(Position.is_open.is_(True))
        ).scalar_one(),
        "positions_closed": session.execute(
            select(func.count()).select_from(Position).where(Position.is_open.is_(False))
        ).scalar_one(),
    }


# ---------------------------------------------------------------------------
# Telegram webhook: the user interface
# ---------------------------------------------------------------------------


@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    background: BackgroundTasks,
    secret_token: str | None = Header(default=None, alias="X-Telegram-Bot-Api-Secret-Token"),
) -> dict[str, bool]:
    """Receive a Telegram update and answer it.

    Returns 200 in every case, including rejection. Telegram retries on a
    non-2xx response and will eventually disable a webhook that keeps failing,
    so an error reply here would turn one bad update into an outage. A 200 on
    rejection also avoids confirming to a prober that the endpoint is real.

    The reply is sent in a background task so the handler returns immediately:
    Telegram times out webhook deliveries, and an on-demand `/analyse` of a
    symbol takes longer than that budget allows.
    """
    from ..notify.bot import TelegramBot

    bot = TelegramBot(get_settings())

    if not bot.verify_secret(secret_token):
        log.warning("Rejected a webhook call with a bad or missing secret token")
        return {"ok": True}

    try:
        update = await request.json()
    except Exception:
        log.warning("Webhook payload was not JSON")
        return {"ok": True}

    if not isinstance(update, dict):
        return {"ok": True}

    from ..notify.bot import BotCommand

    command = BotCommand.parse(update)
    if command is None or not bot.is_authorised(command.chat_id):
        if command is not None:
            log.warning("Unauthorised chat id %s", command.chat_id)
        return {"ok": True}

    def respond(chat_id: str, payload: dict) -> None:
        reply = bot.handle(payload)
        if reply:
            bot.reply(chat_id, reply)

    background.add_task(respond, command.chat_id, update)
    return {"ok": True}


@app.get("/telegram/status")
def telegram_status() -> dict[str, Any]:
    """Diagnose the webhook without exposing the token or the secret."""
    from ..notify.bot import webhook_info

    settings = get_settings()
    info = webhook_info(settings)
    return {
        "token_configured": settings.telegram_bot_token is not None,
        "secret_configured": settings.telegram_webhook_secret is not None,
        "authorised_chats": len(settings.allowed_chat_ids()),
        "webhook": (
            {
                "url_registered": bool(info.get("url")),
                "pending_update_count": info.get("pending_update_count"),
                "last_error_message": info.get("last_error_message"),
                "last_error_date": info.get("last_error_date"),
            }
            if info
            else None
        ),
    }
