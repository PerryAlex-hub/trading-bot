"""Database schema.

Two design constraints shaped this, both from the deployment target:

**Neon free tier gives 1 GB per project.** The candle table is the only one
that can realistically grow, so prices are stored as `NUMERIC` (exact, which
sizing requires) but the table is kept narrow and indexed on exactly one
composite key. Six symbols x four timeframes x a few years of bars is in the
low millions of rows and tens of MB; the headroom is real but not infinite,
which is why `candles` has a retention story and the journal does not.

**The bot runs on a cron that can fire twice.** Every table that records an
external side effect has a natural unique key, so a replayed job updates a row
instead of inserting a duplicate. `orders.client_order_id` is the important
one: it is the same idempotency token sent to the venue.

Money and prices are NUMERIC, never float. Timestamps are all timezone-aware.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Prices need more precision than money: a satellite token can trade at
# 0.00000123, while BTC notional needs room on the left of the decimal point.
PRICE = Numeric(38, 18)
QTY = Numeric(38, 18)
MONEY = Numeric(24, 8)
PCT = Numeric(12, 6)

# JSONB on Postgres, plain JSON on SQLite so the test suite can run without a
# database server. Tests that need real behaviour run against Postgres in CI.
JSON_TYPE = JSONB().with_variant(JSON(), "sqlite")


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Candle(Base):
    """One OHLCV bar for one symbol and timeframe.

    Only closed bars are ever stored. The ingest job drops the in-progress bar
    before writing, because a partial candle that later gets overwritten with
    different values would silently change the result of any backtest that had
    already read it.
    """

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    open: Mapped[Decimal] = mapped_column(PRICE, nullable=False)
    high: Mapped[Decimal] = mapped_column(PRICE, nullable=False)
    low: Mapped[Decimal] = mapped_column(PRICE, nullable=False)
    close: Mapped[Decimal] = mapped_column(PRICE, nullable=False)
    volume: Mapped[Decimal] = mapped_column(QTY, nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="binance")

    __table_args__ = (
        # Makes ingest idempotent: a re-run upserts rather than duplicating.
        UniqueConstraint("symbol", "timeframe", "open_time", name="uq_candle_bar"),
        # Every read is "latest N bars for this symbol/timeframe", so the index
        # is ordered to serve that directly.
        Index("ix_candle_series", "symbol", "timeframe", "open_time"),
        CheckConstraint("high >= low", name="ck_candle_high_ge_low"),
        CheckConstraint("volume >= 0", name="ck_candle_volume_nonneg"),
    )

    def __repr__(self) -> str:
        return f"<Candle {self.symbol} {self.timeframe} {self.open_time:%Y-%m-%d %H:%M}>"


class AnalysisRun(Base, TimestampMixin):
    """One execution of the analysis pipeline.

    Exists so a signal can be traced back to the exact run that produced it,
    and so a scheduler that fires twice in one bar can be detected rather than
    quietly producing two sets of signals.
    """

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trigger: Mapped[str] = mapped_column(String(32), nullable=False)  # cron/manual/api
    bar_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="4h")

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    symbols_scanned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    candidates_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    signals_approved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    error: Mapped[str | None] = mapped_column(Text)
    git_sha: Mapped[str | None] = mapped_column(String(40))
    # Which code produced this. Without it, a journal spanning months of
    # strategy edits cannot be segmented by what the rules actually were.
    config_fingerprint: Mapped[str | None] = mapped_column(String(64))

    signals: Mapped[list[Signal]] = relationship(back_populates="run")

    __table_args__ = (
        UniqueConstraint(
            "timeframe", "bar_time", "trigger", name="uq_run_per_bar_trigger"
        ),
        Index("ix_run_started", "started_at"),
    )


class Signal(Base, TimestampMixin):
    """One analysis verdict, approved or not.

    Rejections are first-class rows. The whole point of the journal is to be
    able to ask "what did the risk gate stop me from doing, and would it have
    worked?" - which is impossible if only approvals are stored.
    """

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("analysis_runs.id", ondelete="SET NULL")
    )

    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    tier: Mapped[str] = mapped_column(String(16), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="4h")
    bar_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    side: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)

    entry: Mapped[Decimal | None] = mapped_column(PRICE)
    stop: Mapped[Decimal | None] = mapped_column(PRICE)
    target_1: Mapped[Decimal | None] = mapped_column(PRICE)
    target_2: Mapped[Decimal | None] = mapped_column(PRICE)
    risk_reward: Mapped[Decimal | None] = mapped_column(PCT)

    quantity: Mapped[Decimal | None] = mapped_column(QTY)
    notional: Mapped[Decimal | None] = mapped_column(MONEY)
    margin: Mapped[Decimal | None] = mapped_column(MONEY)
    leverage: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    risk_amount: Mapped[Decimal | None] = mapped_column(MONEY)

    # Deterministic score from the confluence engine, before any LLM involvement.
    score: Mapped[Decimal | None] = mapped_column(PCT)
    # The full computed snapshot the decision was made from. This is what makes
    # a signal auditable after the fact and is the input to the self-review loop.
    features: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    reasons: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    vetoes: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)

    # Reasoning layer output, kept separate from deterministic fields so the
    # two can always be compared independently.
    analyst_verdict: Mapped[str | None] = mapped_column(String(16))
    analyst_confidence: Mapped[Decimal | None] = mapped_column(PCT)
    analyst_rationale: Mapped[str | None] = mapped_column(Text)
    analyst_model: Mapped[str | None] = mapped_column(String(64))

    notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    run: Mapped[AnalysisRun | None] = relationship(back_populates="signals")
    orders: Mapped[list[Order]] = relationship(back_populates="signal")
    position: Mapped[Position | None] = relationship(
        back_populates="signal", uselist=False
    )

    __table_args__ = (
        # One verdict per symbol per bar. A cron double-fire updates this row.
        UniqueConstraint(
            "symbol", "timeframe", "bar_time", "side", name="uq_signal_per_bar"
        ),
        Index("ix_signal_status_time", "status", "bar_time"),
        Index("ix_signal_symbol_time", "symbol", "bar_time"),
    )

    def __repr__(self) -> str:
        return f"<Signal {self.symbol} {self.side} {self.status}>"


class Order(Base, TimestampMixin):
    """One order intent and whatever the venue did with it.

    `client_order_id` is unique and is the same token sent to WEEX. This is the
    load-bearing piece of the whole execution design: the cron can be retried,
    the runner can be killed mid-request, and a replay still cannot open a
    second position, because both this table and the venue reject the duplicate.
    """

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    signal_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("signals.id", ondelete="SET NULL")
    )
    position_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("positions.id", ondelete="SET NULL")
    )

    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False)
    venue_order_id: Mapped[str | None] = mapped_column(String(64))
    venue: Mapped[str] = mapped_column(String(16), nullable=False, default="weex")

    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    order_type: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)

    quantity: Mapped[Decimal] = mapped_column(QTY, nullable=False)
    price: Mapped[Decimal | None] = mapped_column(PRICE)
    trigger_price: Mapped[Decimal | None] = mapped_column(PRICE)
    filled_quantity: Mapped[Decimal] = mapped_column(QTY, nullable=False, default=0)
    average_price: Mapped[Decimal | None] = mapped_column(PRICE)
    fee: Mapped[Decimal | None] = mapped_column(MONEY)

    reduce_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Distinguishes a simulated fill from a real one. Mixing the two in one
    # table is fine; being unable to tell them apart afterwards is not.
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)

    signal: Mapped[Signal | None] = relationship(back_populates="orders")
    position: Mapped[Position | None] = relationship(back_populates="orders")

    __table_args__ = (
        UniqueConstraint("venue", "client_order_id", name="uq_order_client_id"),
        Index("ix_order_status", "status"),
        Index("ix_order_symbol_time", "symbol", "created_at"),
        CheckConstraint("quantity > 0", name="ck_order_qty_positive"),
    )


class Position(Base, TimestampMixin):
    """An open or closed position.

    The protective stop is recorded here as a venue order id, not as a price we
    intend to watch. On free infrastructure the bot is asleep most of the time,
    so the stop has to live at the exchange; `stop_order_id` being NULL on an
    open position is a reconciliation alarm, not a detail.
    """

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    signal_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("signals.id", ondelete="SET NULL"), unique=True
    )

    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    is_open: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    quantity: Mapped[Decimal] = mapped_column(QTY, nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(PRICE, nullable=False)
    exit_price: Mapped[Decimal | None] = mapped_column(PRICE)
    leverage: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))

    stop_price: Mapped[Decimal | None] = mapped_column(PRICE)
    target_price: Mapped[Decimal | None] = mapped_column(PRICE)
    # Server-side protective orders. These are what actually protect the
    # account between cron runs.
    stop_order_id: Mapped[str | None] = mapped_column(String(64))
    target_order_id: Mapped[str | None] = mapped_column(String(64))

    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    close_reason: Mapped[str | None] = mapped_column(String(32))

    realised_pnl: Mapped[Decimal | None] = mapped_column(MONEY)
    realised_pnl_pct: Mapped[Decimal | None] = mapped_column(PCT)
    fees_paid: Mapped[Decimal | None] = mapped_column(MONEY)
    funding_paid: Mapped[Decimal | None] = mapped_column(MONEY)
    # Realised PnL expressed in units of the risk taken. The only honest way to
    # compare a BTC trade with a GOAT trade.
    r_multiple: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))

    mae_pct: Mapped[Decimal | None] = mapped_column(PCT)  # worst excursion against
    mfe_pct: Mapped[Decimal | None] = mapped_column(PCT)  # best excursion in favour

    signal: Mapped[Signal | None] = relationship(back_populates="position")
    orders: Mapped[list[Order]] = relationship(back_populates="position")

    __table_args__ = (
        Index("ix_position_open", "is_open", "symbol"),
        Index("ix_position_closed_at", "closed_at"),
        CheckConstraint("quantity > 0", name="ck_position_qty_positive"),
    )


class AccountSnapshot(Base):
    """Periodic equity record.

    Circuit breakers are defined against daily loss, so they need a trustworthy
    equity series rather than a number recomputed from trade history each time.
    """

    __tablename__ = "account_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    equity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    available: Mapped[Decimal | None] = mapped_column(MONEY)
    used_margin: Mapped[Decimal | None] = mapped_column(MONEY)
    unrealised_pnl: Mapped[Decimal | None] = mapped_column(MONEY)
    open_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (Index("ix_snapshot_at", "at"),)


class CircuitBreakerState(Base, TimestampMixin):
    """Active trading halts.

    Persisted rather than computed, because a halt has to survive the process
    exiting: the bot is a series of short-lived cron jobs, and a breaker that
    lived in memory would reset itself every few hours.
    """

    __tablename__ = "circuit_breaker_state"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)
    tripped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    halted_until: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_breaker_active", "cleared_at", "halted_until"),)


class ContractSpecCache(Base, TimestampMixin):
    """Last-known venue contract specs.

    Cached in the database rather than on disk: the filesystem of a GitHub
    Actions runner does not exist after the job ends, so a file cache would
    mean re-probing WEEX on every single run. It also means a WEEX outage
    degrades to slightly stale minimums rather than a dead pipeline.
    """

    __tablename__ = "contract_spec_cache"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    venue: Mapped[str] = mapped_column(String(16), primary_key=True, default="weex")

    min_order_size: Mapped[Decimal | None] = mapped_column(QTY)
    min_notional: Mapped[Decimal | None] = mapped_column(MONEY)
    size_step: Mapped[Decimal | None] = mapped_column(QTY)
    price_step: Mapped[Decimal | None] = mapped_column(PRICE)
    max_leverage: Mapped[int | None] = mapped_column(Integer)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)


ALL_TABLES = (
    Candle,
    AnalysisRun,
    Signal,
    Order,
    Position,
    AccountSnapshot,
    CircuitBreakerState,
    ContractSpecCache,
)
