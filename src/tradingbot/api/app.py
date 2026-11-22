"""Read-mostly HTTP API, for the frontend and for health checks.

Deployment shape this is built for: a Render free web service. Two consequences
drive the design.

**It sleeps.** A free service spins down after 15 minutes idle and takes about
a minute to wake. So the API is *not* on the critical path for trading: the
scheduler runs the jobs, Telegram delivers the signals, and this service exists
to read the journal. If it is asleep when nothing is looking at it, nothing is
lost.

**It must not be the thing that trades.** Write endpoints are limited to
operations a human explicitly triggers, and placing a new position is not among
them. An HTTP endpoint that opens a trade is an HTTP endpoint that can be
called twice.

`/health` is deliberately cheap and does not touch the database, so an uptime
pinger cannot burn the Neon compute budget by keeping the database awake.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
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
        "API starting: env=%s mode=%s live_orders=%s",
        settings.environment.value,
        settings.trading_mode.value,
        settings.can_place_real_orders,
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
        "live_orders_enabled": settings.can_place_real_orders,
    }


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
        "live_orders_enabled": settings.can_place_real_orders,
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
