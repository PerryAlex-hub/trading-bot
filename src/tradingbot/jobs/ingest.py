"""Candle ingest.

Runs before every analysis cycle. Pulls only what is missing, which matters for
two independent free-tier budgets:

* **Neon**: 1 GB storage and 100 CU-hours per month. Re-downloading years of
  history on every run would burn both.
* **Binance**: a weight-based rate limit. Backfilling six symbols across four
  timeframes from scratch is thousands of requests; incrementally it is a
  handful.

So ingest is resumable by design: it asks the database for the last stored bar
and fetches forward from there. The first run backfills, every run after is
cheap.

Retention is applied per timeframe. 15m bars are the fastest-growing and least
useful historically, so they are pruned hardest. This is what keeps the project
inside 1 GB indefinitely rather than until it quietly fills up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ..config import Universe, load_universe
from ..db.engine import session_scope
from ..db.repo import CandleRepo
from ..exchanges.binance import BinanceHistory
from ..models import Timeframe

log = logging.getLogger(__name__)

# How far back to go on a first-ever backfill, per timeframe. Chosen so each
# timeframe has enough bars for a 200-period indicator plus room for levels,
# without storing years of 15m data nobody will read.
BACKFILL_DAYS: dict[str, int] = {
    "1d": 1000,
    "4h": 540,
    "1h": 120,
    "15m": 30,
}

# Retention per timeframe. Keeps storage bounded on Neon's free 1 GB.
RETENTION_DAYS: dict[str, int] = {
    "1d": 3650,   # daily bars are tiny and historically valuable: keep them
    "4h": 1095,
    "1h": 180,
    "15m": 45,
}

# Binance returns at most 1000 bars per request.
MAX_CANDLES_PER_RUN = 5000


@dataclass
class IngestResult:
    written: int = 0
    symbols: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    pruned: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    def describe(self) -> str:
        text = f"{self.written} candles across {self.symbols} symbols"
        if self.pruned:
            text += f", pruned {self.pruned}"
        if self.failures:
            text += f", {len(self.failures)} failures"
        return text


def ingest_candles(
    *,
    universe: Universe | None = None,
    timeframes: tuple[str, ...] | None = None,
    prune: bool = True,
) -> IngestResult:
    """Bring every symbol's candle history up to date.

    The data host and spot/futures choice come from settings, because
    Binance's main API is unreachable from some networks and cloud regions.
    """
    universe = universe or load_universe()
    wanted = timeframes or universe.all_timeframes
    result = IngestResult()
    symbols = universe.symbols_requiring_data()

    with BinanceHistory.from_settings() as api:
        for symbol in symbols:
            result.symbols += 1
            for timeframe in wanted:
                try:
                    written = _sync_one(api, symbol, timeframe)
                    result.written += written
                except Exception as exc:
                    # One symbol or timeframe failing must not abort the rest.
                    # On a cron schedule a raised exception means no analysis
                    # happens at all this cycle.
                    log.warning("Ingest failed for %s %s: %s", symbol, timeframe, exc)
                    result.failures.append((symbol, timeframe, str(exc)[:200]))

    if prune:
        result.pruned = prune_old_candles(symbols, wanted)

    log.info("Ingest complete: %s", result.describe())
    return result


def _sync_one(api: BinanceHistory, symbol: str, timeframe: str) -> int:
    """Fetch and store the candles missing for one symbol/timeframe."""
    with session_scope() as session:
        repo = CandleRepo(session)
        last = repo.last_open_time(symbol, timeframe)

    if last is None:
        days = BACKFILL_DAYS.get(timeframe, 365)
        start = datetime.now(tz=UTC) - timedelta(days=days)
        log.info("First backfill for %s %s: %d days", symbol, timeframe, days)
    else:
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        # Resume from the bar after the last stored one. The upsert makes an
        # overlap harmless, so starting slightly early is safer than risking a
        # gap - and a gap silently corrupts every rolling indicator.
        start = last - timedelta(seconds=Timeframe(timeframe).seconds)

    candles = api.candles(
        symbol, timeframe, start=start, max_candles=MAX_CANDLES_PER_RUN
    )
    if not candles:
        log.debug("No new %s %s candles", symbol, timeframe)
        return 0

    with session_scope() as session:
        written = CandleRepo(session).upsert_many(
            symbol, timeframe, candles, source="binance"
        )
    log.debug("Stored %d %s %s candles", written, symbol, timeframe)
    return written


def prune_old_candles(
    symbols: tuple[str, ...], timeframes: tuple[str, ...]
) -> int:
    """Delete candles past their retention window.

    Deliberately conservative on daily bars and aggressive on 15m: the storage
    cost is dominated by intraday data, while the analytical value is
    concentrated in the higher timeframes.
    """
    total = 0
    with session_scope() as session:
        repo = CandleRepo(session)
        for symbol in symbols:
            for timeframe in timeframes:
                days = RETENTION_DAYS.get(timeframe)
                if days is None:
                    continue
                cutoff = datetime.now(tz=UTC) - timedelta(days=days)
                deleted = repo.prune_before(symbol, timeframe, cutoff)
                if deleted:
                    log.info(
                        "Pruned %d %s %s candles older than %d days",
                        deleted, symbol, timeframe, days,
                    )
                total += deleted
    return total


def refresh_contract_specs(*, use_cache: bool = False) -> int:
    """Refresh the cached WEEX contract specs.

    Cached in the database rather than on disk because a GitHub Actions runner
    keeps no filesystem between runs, so the file cache in `exchanges/weex.py`
    never survives in production. Failures are logged, not raised: stale
    minimums degrade sizing accuracy, while a hard failure stops all trading.
    """
    from ..db.repo import ContractSpecRepo
    from ..exchanges.weex import WeexClient, WeexEndpointError

    try:
        with WeexClient() as client:
            specs = client.contract_specs(use_cache=use_cache)
    except (WeexEndpointError, Exception) as exc:
        log.warning("Could not refresh WEEX contract specs: %s", exc)
        return 0

    with session_scope() as session:
        count = ContractSpecRepo(session).put_many(specs.values())
    log.info("Cached %d WEEX contract specs", count)
    return count
