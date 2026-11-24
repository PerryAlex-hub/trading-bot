"""Repositories: every database read and write the rest of the bot performs.

Keeping SQL here rather than in the analysis and execution modules is what lets
the pipeline be tested against in-memory candle lists, and it means a change to
the storage layer cannot quietly alter trading behaviour.

The upserts matter more than usual. This bot runs as a set of cron jobs that
can be retried or double-fired, so "insert" is almost always the wrong verb:
re-running the 4h analysis must converge on the same rows, not accumulate
duplicates.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ..models import Candle as CandleValue
from ..models import (
    ContractSpec,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Side,
    SignalStatus,
    Timeframe,
    TradeSetup,
)
from .schema import (
    AccountSnapshot,
    AnalysisRun,
    Candle,
    CircuitBreakerState,
    ContractSpecCache,
    Order,
    Position,
    Signal,
    utcnow,
)

log = logging.getLogger(__name__)

# Postgres caps parameters per statement at 65535. Each candle row binds 8
# columns, so chunks are kept well under that ceiling.
_UPSERT_CHUNK = 2000


class CandleRepo:
    """Candle persistence and retrieval."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_many(
        self,
        symbol: str,
        timeframe: str,
        candles: Sequence[CandleValue],
        *,
        source: str = "binance",
    ) -> int:
        """Insert candles, overwriting any bar already stored for that time.

        `ON CONFLICT DO UPDATE` rather than `DO NOTHING`: an exchange can
        restate a bar shortly after it closes, and keeping the first version we
        happened to see would leave the stored series disagreeing with the
        venue forever.
        """
        if not candles:
            return 0

        written = 0
        rows = [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "open_time": candle.open_time,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "source": source,
            }
            for candle in candles
        ]

        # Dialect-aware so the suite can exercise ingest end-to-end on
        # in-memory SQLite. Both dialects support ON CONFLICT DO UPDATE, but
        # they are different constructs and SQLite needs the index columns
        # rather than a named constraint.
        dialect = self.session.get_bind().dialect.name
        insert = sqlite_insert if dialect == "sqlite" else pg_insert

        for start in range(0, len(rows), _UPSERT_CHUNK):
            chunk = rows[start : start + _UPSERT_CHUNK]
            statement = insert(Candle).values(chunk)
            updates = {
                "open": statement.excluded.open,
                "high": statement.excluded.high,
                "low": statement.excluded.low,
                "close": statement.excluded.close,
                "volume": statement.excluded.volume,
                "source": statement.excluded.source,
            }
            if dialect == "sqlite":
                statement = statement.on_conflict_do_update(
                    index_elements=["symbol", "timeframe", "open_time"], set_=updates
                )
            else:
                statement = statement.on_conflict_do_update(
                    constraint="uq_candle_bar", set_=updates
                )
            self.session.execute(statement)
            written += len(chunk)

        log.debug("Upserted %d %s %s candles", written, symbol, timeframe)
        return written

    def latest(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        as_of: datetime | None = None,
    ) -> list[CandleValue]:
        """The most recent `limit` closed bars, returned oldest-first.

        Indicators need chronological order, but the database can only serve
        "most recent" efficiently, so the slice is taken descending and then
        reversed.

        `as_of` truncates the series at that bar's open time, which is what
        makes catch-up analysis of a missed bar honest: the indicators then see
        exactly what they would have seen at the time, with no knowledge of
        bars that closed afterwards. Without it, re-analysing an old bar with
        today's data is lookahead bias dressed up as a backfill, and every
        journalled result from it would be worthless.
        """
        statement = select(Candle).where(
            Candle.symbol == symbol, Candle.timeframe == timeframe
        )
        if as_of is not None:
            statement = statement.where(Candle.open_time <= as_of)
        statement = statement.order_by(Candle.open_time.desc()).limit(limit)
        rows = list(self.session.execute(statement).scalars())
        rows.reverse()
        return [self._to_value(row) for row in rows]

    def range(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[CandleValue]:
        statement = select(Candle).where(
            Candle.symbol == symbol, Candle.timeframe == timeframe
        )
        if start is not None:
            statement = statement.where(Candle.open_time >= start)
        if end is not None:
            statement = statement.where(Candle.open_time <= end)
        rows = self.session.execute(statement.order_by(Candle.open_time)).scalars()
        return [self._to_value(row) for row in rows]

    def last_open_time(self, symbol: str, timeframe: str) -> datetime | None:
        """Where ingest should resume from."""
        return self.session.execute(
            select(func.max(Candle.open_time)).where(
                Candle.symbol == symbol, Candle.timeframe == timeframe
            )
        ).scalar_one_or_none()

    def count(self, symbol: str, timeframe: str) -> int:
        return (
            self.session.execute(
                select(func.count())
                .select_from(Candle)
                .where(Candle.symbol == symbol, Candle.timeframe == timeframe)
            ).scalar_one()
            or 0
        )

    def history_days(self, symbol: str, timeframe: str) -> float:
        """Stored history span in days.

        Drives the paper-to-satellite promotion gate, which is defined in
        `risk.yaml` as a number of days of real stored data rather than a
        listing date we would have to trust an API for.
        """
        result = self.session.execute(
            select(func.min(Candle.open_time), func.max(Candle.open_time)).where(
                Candle.symbol == symbol, Candle.timeframe == timeframe
            )
        ).one_or_none()
        if result is None or result[0] is None or result[1] is None:
            return 0.0
        return (result[1] - result[0]).total_seconds() / 86400

    def prune_before(self, symbol: str, timeframe: str, cutoff: datetime) -> int:
        """Delete bars older than `cutoff`.

        Neon's free plan caps storage at 1 GB per project. Intraday series grow
        fastest and are the least useful historically, so retention is applied
        per timeframe rather than globally.
        """
        result = self.session.execute(
            Candle.__table__.delete().where(
                Candle.symbol == symbol,
                Candle.timeframe == timeframe,
                Candle.open_time < cutoff,
            )
        )
        return result.rowcount or 0

    @staticmethod
    def _to_value(row: Candle) -> CandleValue:
        open_time = row.open_time
        if open_time.tzinfo is None:
            # SQLite loses tzinfo on round-trip; Postgres does not. Normalising
            # here keeps the analysis layer free of storage-engine quirks.
            open_time = open_time.replace(tzinfo=UTC)
        return CandleValue(
            open_time=open_time,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
        )


class RunRepo:
    """Analysis-run bookkeeping."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def start(
        self,
        *,
        trigger: str,
        bar_time: datetime | None,
        timeframe: str = "4h",
        git_sha: str | None = None,
        config_fingerprint: str | None = None,
    ) -> AnalysisRun:
        """Begin (or restart) the run row for this bar and trigger.

        Reuses an existing row rather than inserting a second one. The unique
        constraint on (timeframe, bar_time, trigger) is there to keep one
        record per bar, but inserting blindly made two legitimate operations
        impossible: retrying a FAILED run, and `analyse --force`. In both cases
        `already_ran()` correctly says the bar may be processed, and then the
        insert died on the constraint - so a single transient failure
        permanently poisoned that bar, because every retry crashed the same way.

        Observed in practice on the second manual run of one bar.
        """
        existing = (
            self.session.execute(
                select(AnalysisRun).where(
                    AnalysisRun.timeframe == timeframe,
                    AnalysisRun.bar_time == bar_time,
                    AnalysisRun.trigger == trigger,
                )
            ).scalar_one_or_none()
            if bar_time is not None
            else None
        )

        if existing is not None:
            # Reset it to "in progress" so a stale error or count from the
            # previous attempt cannot be mistaken for this one's result.
            existing.started_at = utcnow()
            existing.finished_at = None
            existing.ok = False
            existing.error = None
            existing.symbols_scanned = 0
            existing.candidates_found = 0
            existing.signals_approved = 0
            existing.git_sha = git_sha
            existing.config_fingerprint = config_fingerprint
            self.session.flush()
            return existing

        run = AnalysisRun(
            trigger=trigger,
            bar_time=bar_time,
            timeframe=timeframe,
            started_at=utcnow(),
            git_sha=git_sha,
            config_fingerprint=config_fingerprint,
        )
        self.session.add(run)
        self.session.flush()
        return run

    def finish(
        self,
        run: AnalysisRun,
        *,
        ok: bool,
        symbols_scanned: int = 0,
        candidates_found: int = 0,
        signals_approved: int = 0,
        error: str | None = None,
    ) -> None:
        run.finished_at = utcnow()
        run.ok = ok
        run.symbols_scanned = symbols_scanned
        run.candidates_found = candidates_found
        run.signals_approved = signals_approved
        # Tracebacks can be enormous; the tail is the useful part and the
        # journal is not a log aggregator.
        run.error = error[-4000:] if error else None
        self.session.flush()

    def already_ran(
        self, timeframe: str, bar_time: datetime, trigger: str | None = None
    ) -> bool:
        """Whether this exact bar has already been analysed.

        GitHub Actions can fire a schedule more than once, and a manual run can
        overlap a cron one. Without this check a duplicate run would re-notify
        and, worse, re-attempt execution.

        `trigger=None` asks whether ANY run covered the bar, which is what the
        catch-up pass needs: a bar analysed by cron must not be re-analysed by
        a backfill, and scoping the check to one trigger would let exactly that
        happen.
        """
        statement = select(AnalysisRun.id).where(
            AnalysisRun.timeframe == timeframe,
            AnalysisRun.bar_time == bar_time,
            AnalysisRun.ok.is_(True),
        )
        if trigger is not None:
            statement = statement.where(AnalysisRun.trigger == trigger)
        return self.session.execute(statement).first() is not None

    def recent(self, limit: int = 20) -> list[AnalysisRun]:
        return list(
            self.session.execute(
                select(AnalysisRun)
                .order_by(AnalysisRun.started_at.desc())
                .limit(limit)
            ).scalars()
        )


class SignalRepo:
    """The journal. Records every verdict, including the refusals."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def record(
        self,
        *,
        symbol: str,
        tier: str,
        side: Side,
        status: SignalStatus,
        bar_time: datetime,
        timeframe: Timeframe | str = Timeframe.H4,
        setup: TradeSetup | None = None,
        score: Decimal | None = None,
        features: dict[str, Any] | None = None,
        reasons: dict[str, Any] | None = None,
        vetoes: dict[str, Any] | None = None,
        quantity: Decimal | None = None,
        notional: Decimal | None = None,
        margin: Decimal | None = None,
        leverage: Decimal | None = None,
        risk_amount: Decimal | None = None,
        run_id: int | None = None,
    ) -> Signal:
        """Upsert one verdict for a symbol/bar/side.

        Idempotent by design: a replayed run overwrites its own previous
        verdict for that bar rather than appending a second opinion.
        """
        tf = timeframe.value if isinstance(timeframe, Timeframe) else timeframe
        existing = self.session.execute(
            select(Signal).where(
                Signal.symbol == symbol,
                Signal.timeframe == tf,
                Signal.bar_time == bar_time,
                Signal.side == side.value,
            )
        ).scalar_one_or_none()

        signal = existing or Signal(
            symbol=symbol, timeframe=tf, bar_time=bar_time, side=side.value
        )
        signal.tier = tier
        signal.status = status.value
        signal.score = score
        signal.features = features
        signal.reasons = reasons
        signal.vetoes = vetoes
        signal.quantity = quantity
        signal.notional = notional
        signal.margin = margin
        signal.leverage = leverage
        signal.risk_amount = risk_amount
        if run_id is not None:
            signal.run_id = run_id

        if setup is not None:
            signal.entry = setup.entry
            signal.stop = setup.stop
            signal.target_1 = setup.targets[0]
            signal.target_2 = setup.targets[1] if len(setup.targets) > 1 else None
            signal.risk_reward = setup.risk_reward()

        if existing is None:
            self.session.add(signal)
        self.session.flush()
        return signal

    def set_analyst_verdict(
        self,
        signal: Signal,
        *,
        verdict: str,
        confidence: Decimal | None,
        rationale: str | None,
        model: str | None,
    ) -> None:
        signal.analyst_verdict = verdict
        signal.analyst_confidence = confidence
        signal.analyst_rationale = rationale
        signal.analyst_model = model
        self.session.flush()

    def mark_notified(self, signal: Signal) -> None:
        signal.notified_at = utcnow()
        self.session.flush()

    def get(self, signal_id: int) -> Signal | None:
        return self.session.get(Signal, signal_id)

    def pending_notification(self, limit: int = 20) -> list[Signal]:
        """Approved or executed signals that have not been delivered yet.

        Separating "decided" from "delivered" means a Telegram outage loses
        nothing: the next run picks up where it left off instead of the signal
        vanishing.
        """
        return list(
            self.session.execute(
                select(Signal)
                .where(
                    Signal.notified_at.is_(None),
                    Signal.status.in_(
                        (SignalStatus.APPROVED.value, SignalStatus.EXECUTED.value)
                    ),
                )
                .order_by(Signal.bar_time.desc())
                .limit(limit)
            ).scalars()
        )

    def recent(
        self,
        *,
        limit: int = 50,
        symbol: str | None = None,
        status: SignalStatus | None = None,
        since: datetime | None = None,
    ) -> list[Signal]:
        statement = select(Signal)
        if symbol:
            statement = statement.where(Signal.symbol == symbol)
        if status:
            statement = statement.where(Signal.status == status.value)
        if since:
            statement = statement.where(Signal.bar_time >= since)
        return list(
            self.session.execute(
                statement.order_by(Signal.bar_time.desc()).limit(limit)
            ).scalars()
        )


class OrderRepo:
    """Order records, keyed on the idempotency token sent to the venue."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def find_by_client_id(
        self, client_order_id: str, venue: str = "weex"
    ) -> Order | None:
        return self.session.execute(
            select(Order).where(
                Order.venue == venue, Order.client_order_id == client_order_id
            )
        ).scalar_one_or_none()

    def reserve(
        self,
        request: OrderRequest,
        *,
        signal_id: int | None,
        position_id: int | None = None,
        venue: str = "weex",
        is_paper: bool = True,
    ) -> tuple[Order, bool]:
        """Write the intent BEFORE contacting the venue.

        Returns `(order, created)`. `created=False` means this exact intent was
        already recorded, so the caller must not place it again.

        Recording first is deliberate. If the process dies between the HTTP
        request and the response, a row that was written beforehand lets the
        next run reconcile against the venue. A row written afterwards would
        leave an untracked live position - the worst failure mode this bot has.
        """
        existing = self.find_by_client_id(request.client_order_id, venue)
        if existing is not None:
            return existing, False

        order = Order(
            signal_id=signal_id,
            position_id=position_id,
            client_order_id=request.client_order_id,
            venue=venue,
            symbol=request.symbol,
            side=request.side.value,
            order_type=request.order_type.value,
            status=OrderStatus.PENDING.value,
            quantity=request.quantity,
            price=request.price,
            trigger_price=request.trigger_price,
            reduce_only=request.reduce_only,
            is_paper=is_paper,
        )
        self.session.add(order)
        self.session.flush()
        return order, True

    def apply_result(self, order: Order, result: OrderResult) -> None:
        order.status = result.status.value
        order.venue_order_id = result.venue_order_id
        order.filled_quantity = result.filled_quantity
        order.average_price = result.average_price
        order.error = result.error[-2000:] if result.error else None
        order.raw_response = result.raw or None
        order.submitted_at = order.submitted_at or utcnow()
        self.session.flush()

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        statement = select(Order).where(
            Order.status.in_(
                (
                    OrderStatus.PENDING.value,
                    OrderStatus.SUBMITTED.value,
                    OrderStatus.PARTIALLY_FILLED.value,
                )
            )
        )
        if symbol:
            statement = statement.where(Order.symbol == symbol)
        return list(self.session.execute(statement).scalars())


class PositionRepo:
    """Open and closed positions, and the portfolio state the risk gate reads."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def open_positions(self) -> list[Position]:
        return list(
            self.session.execute(
                select(Position).where(Position.is_open.is_(True))
            ).scalars()
        )

    def open_for_symbol(self, symbol: str) -> Position | None:
        return self.session.execute(
            select(Position).where(
                Position.symbol == symbol, Position.is_open.is_(True)
            )
        ).scalar_one_or_none()

    def open_without_stop(self) -> list[Position]:
        """Open positions with no protective order recorded at the venue.

        This is the alarm condition of the whole design. The bot sleeps between
        cron runs, so an open position whose stop is not sitting on the
        exchange is unprotected capital, and the monitor job treats it as an
        incident rather than a warning.
        """
        return list(
            self.session.execute(
                select(Position).where(
                    Position.is_open.is_(True),
                    Position.stop_order_id.is_(None),
                    Position.is_paper.is_(False),
                )
            ).scalars()
        )

    def create(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: Decimal,
        entry_price: Decimal,
        stop_price: Decimal | None,
        target_price: Decimal | None,
        leverage: Decimal | None,
        signal_id: int | None,
        is_paper: bool,
    ) -> Position:
        position = Position(
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            leverage=leverage,
            signal_id=signal_id,
            is_paper=is_paper,
            is_open=True,
            opened_at=utcnow(),
        )
        self.session.add(position)
        self.session.flush()
        return position

    def attach_protective_orders(
        self,
        position: Position,
        *,
        stop_order_id: str | None = None,
        target_order_id: str | None = None,
    ) -> None:
        if stop_order_id is not None:
            position.stop_order_id = stop_order_id
        if target_order_id is not None:
            position.target_order_id = target_order_id
        self.session.flush()

    def close(
        self,
        position: Position,
        *,
        exit_price: Decimal,
        reason: str,
        fees: Decimal | None = None,
        funding: Decimal | None = None,
    ) -> Position:
        """Close a position and compute its result in R.

        R-multiple is the number that matters: it is the only way to compare a
        BTC trade against a GOAT trade, and it is what the self-review loop
        aggregates over.
        """
        position.is_open = False
        position.exit_price = exit_price
        position.closed_at = utcnow()
        position.close_reason = reason
        position.fees_paid = fees
        position.funding_paid = funding

        direction = Decimal(Side(position.side).sign)
        gross = (exit_price - position.entry_price) * direction * position.quantity
        costs = (fees or Decimal("0")) + (funding or Decimal("0"))
        position.realised_pnl = gross - costs

        notional = position.entry_price * position.quantity
        if notional > 0:
            position.realised_pnl_pct = position.realised_pnl / notional * Decimal("100")

        if position.stop_price is not None:
            risk_per_unit = abs(position.entry_price - position.stop_price)
            if risk_per_unit > 0:
                risk_total = risk_per_unit * position.quantity
                position.r_multiple = position.realised_pnl / risk_total

        self.session.flush()
        return position

    def closed_since(self, since: datetime) -> list[Position]:
        # `id` breaks ties on closed_at: see recent_closed() for why that
        # matters.
        return list(
            self.session.execute(
                select(Position)
                .where(Position.is_open.is_(False), Position.closed_at >= since)
                .order_by(Position.closed_at, Position.id)
            ).scalars()
        )

    def recent_closed(self, limit: int = 50) -> list[Position]:
        """Closed positions, newest first.

        The `id` tiebreaker is load-bearing, not tidiness. Several positions
        can close in the same microsecond - the monitor's reconcile loop does
        exactly that when it discovers two stops filled while the bot was
        asleep. Ordering on `closed_at` alone leaves their relative order
        arbitrary, and `consecutive_losses()` stops counting at the first win,
        so an arbitrary tie order produces a wrong streak and therefore a
        wrong circuit-breaker decision.
        """
        return list(
            self.session.execute(
                select(Position)
                .where(Position.is_open.is_(False))
                .order_by(Position.closed_at.desc(), Position.id.desc())
                .limit(limit)
            ).scalars()
        )

    def consecutive_losses(self, lookback: int = 20) -> int:
        """Losing streak length, counting back from the most recent close."""
        streak = 0
        for position in self.recent_closed(limit=lookback):
            if position.realised_pnl is not None and position.realised_pnl < 0:
                streak += 1
            else:
                break
        return streak

    def realised_pnl_since(self, since: datetime) -> Decimal:
        total = self.session.execute(
            select(func.coalesce(func.sum(Position.realised_pnl), 0)).where(
                Position.is_open.is_(False), Position.closed_at >= since
            )
        ).scalar_one()
        return Decimal(str(total))


class CircuitBreakerRepo:
    """Trading halts, persisted so they survive the process exiting."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def active(self) -> CircuitBreakerState | None:
        now = utcnow()
        return self.session.execute(
            select(CircuitBreakerState)
            .where(
                CircuitBreakerState.cleared_at.is_(None),
                CircuitBreakerState.halted_until > now,
            )
            .order_by(CircuitBreakerState.halted_until.desc())
        ).scalars().first()

    def trip(self, *, reason: str, detail: str, hours: float) -> CircuitBreakerState:
        state = CircuitBreakerState(
            reason=reason,
            detail=detail,
            tripped_at=utcnow(),
            halted_until=utcnow() + timedelta(hours=hours),
        )
        self.session.add(state)
        self.session.flush()
        log.warning("Circuit breaker tripped (%s): %s", reason, detail)
        return state

    def clear_all(self) -> int:
        """Manual override. Deliberately requires an explicit human action."""
        states = list(
            self.session.execute(
                select(CircuitBreakerState).where(
                    CircuitBreakerState.cleared_at.is_(None)
                )
            ).scalars()
        )
        for state in states:
            state.cleared_at = utcnow()
        self.session.flush()
        return len(states)


class AccountRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def snapshot(
        self,
        *,
        equity: Decimal,
        available: Decimal | None = None,
        used_margin: Decimal | None = None,
        unrealised_pnl: Decimal | None = None,
        open_positions: int = 0,
        is_paper: bool = True,
    ) -> AccountSnapshot:
        record = AccountSnapshot(
            equity=equity,
            available=available,
            used_margin=used_margin,
            unrealised_pnl=unrealised_pnl,
            open_positions=open_positions,
            is_paper=is_paper,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def latest(self) -> AccountSnapshot | None:
        return self.session.execute(
            select(AccountSnapshot).order_by(AccountSnapshot.at.desc()).limit(1)
        ).scalars().first()

    def equity_at_day_start(self) -> Decimal | None:
        """Equity at the start of the current UTC day.

        The daily-loss breaker is defined against this, so it has to come from
        a recorded snapshot rather than from "whatever equity was the last time
        we looked", which drifts.
        """
        start = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        record = self.session.execute(
            select(AccountSnapshot)
            .where(AccountSnapshot.at >= start)
            .order_by(AccountSnapshot.at)
            .limit(1)
        ).scalars().first()
        return record.equity if record else None


class ContractSpecRepo:
    """Venue contract specs, cached in the database.

    A GitHub Actions runner has no filesystem after the job ends, so the disk
    cache in `exchanges/weex.py` does nothing in production. Caching here means
    one WEEX probe serves every later run, and a WEEX outage degrades to
    slightly stale minimums rather than a dead pipeline.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def put_many(self, specs: Iterable[ContractSpec], venue: str = "weex") -> int:
        count = 0
        for spec in specs:
            row = self.session.get(ContractSpecCache, {"symbol": spec.symbol, "venue": venue})
            if row is None:
                row = ContractSpecCache(symbol=spec.symbol, venue=venue)
                self.session.add(row)
            row.min_order_size = spec.min_order_size
            row.min_notional = spec.min_notional
            row.size_step = spec.size_step
            row.price_step = spec.price_step
            row.max_leverage = spec.max_leverage
            row.raw = spec.raw or None
            row.fetched_at = utcnow()
            count += 1
        self.session.flush()
        return count

    def get(self, symbol: str, venue: str = "weex") -> ContractSpec | None:
        row = self.session.get(ContractSpecCache, {"symbol": symbol, "venue": venue})
        if row is None:
            return None
        return ContractSpec(
            symbol=row.symbol,
            min_order_size=row.min_order_size,
            min_notional=row.min_notional,
            size_step=row.size_step,
            price_step=row.price_step,
            max_leverage=row.max_leverage,
            raw=row.raw or {},
        )

    def age_hours(self, symbol: str, venue: str = "weex") -> float | None:
        row = self.session.get(ContractSpecCache, {"symbol": symbol, "venue": venue})
        if row is None:
            return None
        fetched = row.fetched_at
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=UTC)
        return (utcnow() - fetched).total_seconds() / 3600
