"""Database engine, tuned for Neon serverless Postgres.

Neon's free tier has two behaviours that break a naively-configured pool, and
both of them bite exactly where this bot runs:

1. **Compute suspends after 5 idle minutes.** The bot wakes on a cron every few
   hours, so *every* run starts against a cold database. The first connection
   has to wait for the compute to resume, and a pooled connection that was
   cached from a previous run is dead. `pool_pre_ping` plus a connect retry is
   what makes this invisible instead of a crashed job.

2. **The pooled and direct URLs are not interchangeable.** The `-pooler` host
   is PgBouncer in transaction mode: it is right for short queries and wrong
   for migrations, which need session-level state for advisory locks.

Driver note: SQLAlchemy defaults the bare `postgresql://` scheme to psycopg2,
which is not installed. Neon hands out URLs in that bare form, so the scheme is
rewritten rather than requiring the URL to be hand-edited after copy-paste.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from ..settings import Settings, get_settings

log = logging.getLogger(__name__)

# Neon resuming a suspended compute usually takes a few hundred ms, but a cold
# start under load can take several seconds. Retry rather than fail the run.
_CONNECT_ATTEMPTS = 4
_CONNECT_BACKOFF_SECONDS = 1.5


def normalise_database_url(url: str) -> str:
    """Force the psycopg3 driver and require TLS.

    Neon only accepts TLS connections, but a URL copied from somewhere other
    than the dashboard may be missing `sslmode`. Being explicit means a
    misconfigured URL fails loudly at connect time instead of silently
    downgrading.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    if "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


def _build_engine(url: str, *, pooled: bool, settings: Settings) -> Engine:
    """Create an engine.

    `pooled=False` uses NullPool for migrations and one-shot scripts: holding a
    pool open against a compute that is about to suspend buys nothing and
    keeps the Neon instance awake, burning the free CU-hour budget.
    """
    kwargs: dict[str, object] = {
        "pool_pre_ping": True,  # non-negotiable against a suspending compute
        "future": True,
        "connect_args": {
            # Server-side cap. A runaway analytical query against the candle
            # table should fail fast rather than consume the monthly compute
            # allowance in one go.
            "options": f"-c statement_timeout={settings.db_statement_timeout_ms}",
            "application_name": "tradingbot",
            "connect_timeout": 15,
        },
    }
    if pooled:
        kwargs.update(
            poolclass=None,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            # Recycle well inside Neon's idle-suspend window so we never hand
            # out a connection whose server has gone away.
            pool_recycle=240,
            pool_timeout=30,
        )
    else:
        kwargs["poolclass"] = NullPool

    engine = create_engine(normalise_database_url(url), **kwargs)  # type: ignore[arg-type]

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, _record):  # type: ignore[no-untyped-def]
        log.debug("Opened a new database connection")

    return engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """The application engine, against the pooled Neon endpoint."""
    settings = get_settings()
    return _build_engine(settings.require_database_url(), pooled=True, settings=settings)


def get_migration_engine() -> Engine:
    """Unpooled engine against the direct endpoint, for Alembic only."""
    settings = get_settings()
    return _build_engine(
        settings.migration_database_url(), pooled=False, settings=settings
    )


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,  # results stay readable after commit
        autoflush=False,
    )


def wait_for_database(*, attempts: int = _CONNECT_ATTEMPTS) -> None:
    """Block until Neon's compute has resumed, or give up with a clear error.

    Called at the start of every scheduled job. Without it the first query of a
    cold run is the thing that fails, which makes the traceback look like a
    bug in whatever ran first rather than an idle database.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            if attempt > 1:
                log.info("Database reachable after %d attempts", attempt)
            return
        except OperationalError as exc:
            last_error = exc
            wait = _CONNECT_BACKOFF_SECONDS * attempt
            log.warning(
                "Database not ready (attempt %d/%d), retrying in %.1fs: %s",
                attempt,
                attempts,
                wait,
                exc.__class__.__name__,
            )
            if attempt < attempts:
                time.sleep(wait)
    raise RuntimeError(
        "Could not reach the database after "
        f"{attempts} attempts. If this is Neon, check the project is not "
        "suspended for exceeding the free compute allowance."
    ) from last_error


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception.

    Every job and request body runs inside one of these. Partial writes are the
    one thing that would make the journal untrustworthy, so there is no API
    here for committing half of a unit of work.
    """
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine() -> None:
    """Drop pooled connections.

    Worth calling at the end of a cron job: it lets Neon's compute suspend
    promptly instead of being held awake by an idle connection, which directly
    protects the free CU-hour budget.
    """
    get_engine().dispose()
