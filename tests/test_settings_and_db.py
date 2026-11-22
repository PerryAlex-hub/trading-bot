"""Settings, the live-trading gate, and persistence.

The gate tests matter most. `trading_mode=live` plus `LIVE_TRADING_ENABLED`
must BOTH be set before a real order is possible, so that a single stray
environment variable in a deploy config cannot arm the bot.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from tradingbot.db.engine import normalise_database_url
from tradingbot.db.repo import (
    AccountRepo,
    CandleRepo,
    CircuitBreakerRepo,
    ContractSpecRepo,
    PositionRepo,
    RunRepo,
    SignalRepo,
)
from tradingbot.models import ContractSpec, Side, SignalStatus, Timeframe, TradeSetup
from tradingbot.settings import Settings, TradingMode

from .conftest import make_candles


class TestLiveTradingGate:
    def test_live_mode_requires_the_second_switch(self) -> None:
        with pytest.raises(ValueError, match="LIVE_TRADING_ENABLED"):
            Settings(trading_mode=TradingMode.LIVE, live_trading_enabled=False)

    def test_live_mode_requires_credentials(self) -> None:
        with pytest.raises(ValueError, match="credentials are missing"):
            Settings(trading_mode=TradingMode.LIVE, live_trading_enabled=True)

    def test_live_mode_with_both_switches_and_credentials_is_allowed(self) -> None:
        settings = Settings(
            trading_mode=TradingMode.LIVE,
            live_trading_enabled=True,
            weex_api_key="k",
            weex_api_secret="s",
            weex_api_passphrase="p",
        )
        assert settings.can_place_real_orders

    def test_the_flag_alone_does_not_arm_anything(self) -> None:
        """The dangerous near-miss: flag set, mode left at the default."""
        settings = Settings(live_trading_enabled=True)
        assert settings.trading_mode is TradingMode.SIGNAL_ONLY
        assert not settings.can_place_real_orders

    def test_paper_mode_never_places_real_orders(self) -> None:
        settings = Settings(trading_mode=TradingMode.PAPER, live_trading_enabled=True)
        assert not settings.can_place_real_orders

    def test_default_is_signal_only(self) -> None:
        settings = Settings()
        assert settings.trading_mode is TradingMode.SIGNAL_ONLY
        assert not settings.can_place_real_orders


class TestDatabaseUrl:
    def test_bare_postgres_scheme_is_given_the_psycopg3_driver(self) -> None:
        """SQLAlchemy defaults the bare scheme to psycopg2, which is not installed.

        Neon hands out URLs in exactly this form.
        """
        url = normalise_database_url("postgresql://u:p@host/db")
        assert url.startswith("postgresql+psycopg://")

    def test_heroku_style_scheme_is_normalised(self) -> None:
        assert normalise_database_url("postgres://u:p@host/db").startswith(
            "postgresql+psycopg://"
        )

    def test_tls_is_required(self) -> None:
        assert "sslmode=require" in normalise_database_url("postgresql://u:p@host/db")

    def test_an_existing_sslmode_is_preserved(self) -> None:
        url = normalise_database_url("postgresql://u:p@host/db?sslmode=verify-full")
        assert url.count("sslmode") == 1
        assert "verify-full" in url

    def test_migration_url_falls_back_to_the_unpooled_host(self) -> None:
        """PgBouncer in transaction mode cannot reliably run DDL."""
        settings = Settings(database_url="postgresql://u:p@ep-x-pooler.neon.tech/db")
        assert "-pooler" not in settings.migration_database_url()


class TestCandleRepo:
    def test_upsert_then_read_round_trips(self, session) -> None:
        candles = make_candles(n=50, seed=1)
        written = CandleRepo(session).upsert_many("BTCUSDT", "4h", candles)
        assert written == 50
        stored = CandleRepo(session).latest("BTCUSDT", "4h", limit=50)
        assert len(stored) == 50
        assert stored[0].open_time < stored[-1].open_time  # oldest first

    def test_upsert_is_idempotent(self, session) -> None:
        """Ingest re-runs constantly; it must converge, not accumulate."""
        candles = make_candles(n=30, seed=2)
        repo = CandleRepo(session)
        repo.upsert_many("BTCUSDT", "4h", candles)
        repo.upsert_many("BTCUSDT", "4h", candles)
        assert repo.count("BTCUSDT", "4h") == 30

    def test_upsert_overwrites_a_restated_bar(self, session) -> None:
        """An exchange can restate a bar shortly after it closes."""
        candles = make_candles(n=10, seed=3)
        repo = CandleRepo(session)
        repo.upsert_many("BTCUSDT", "4h", candles)

        from tradingbot.models import Candle

        amended = Candle(
            open_time=candles[-1].open_time,
            open=candles[-1].open,
            high=Decimal("999999"),
            low=candles[-1].low,
            close=candles[-1].close,
            volume=candles[-1].volume,
        )
        repo.upsert_many("BTCUSDT", "4h", [amended])
        assert repo.count("BTCUSDT", "4h") == 10
        assert repo.latest("BTCUSDT", "4h", limit=1)[0].high == Decimal("999999")

    def test_timezone_survives_the_round_trip(self, session) -> None:
        """SQLite drops tzinfo; the repo must restore it."""
        repo = CandleRepo(session)
        repo.upsert_many("BTCUSDT", "4h", make_candles(n=5, seed=4))
        assert repo.latest("BTCUSDT", "4h")[0].open_time.tzinfo is not None

    def test_last_open_time_drives_resumable_ingest(self, session) -> None:
        candles = make_candles(n=20, seed=5)
        repo = CandleRepo(session)
        repo.upsert_many("BTCUSDT", "4h", candles)
        assert repo.last_open_time("BTCUSDT", "4h") is not None

    def test_history_days_gates_paper_promotion(self, session) -> None:
        repo = CandleRepo(session)
        repo.upsert_many("AXISUSDT", "4h", make_candles(n=180, seed=6))
        days = repo.history_days("AXISUSDT", "4h")
        assert days == pytest.approx(179 * 4 / 24, rel=0.01)

    def test_prune_removes_only_old_bars(self, session) -> None:
        repo = CandleRepo(session)
        candles = make_candles(n=100, seed=7)
        repo.upsert_many("BTCUSDT", "4h", candles)
        cutoff = candles[50].open_time
        deleted = repo.prune_before("BTCUSDT", "4h", cutoff)
        assert deleted == 50
        assert repo.count("BTCUSDT", "4h") == 50


class TestSignalRepo:
    def test_a_rejection_is_a_first_class_row(self, session) -> None:
        """The journal's value comes mostly from the refusals."""
        signal = SignalRepo(session).record(
            symbol="BTCUSDT", tier="core", side=Side.LONG,
            status=SignalStatus.REJECTED_RISK,
            bar_time=dt.datetime(2026, 11, 22, 16, tzinfo=dt.UTC),
            vetoes={"rules": [{"rule": "costs.max_spread_pct", "detail": "too wide"}]},
        )
        assert signal.id is not None
        assert signal.vetoes["rules"][0]["rule"] == "costs.max_spread_pct"

    def test_recording_the_same_bar_twice_updates_rather_than_duplicates(
        self, session
    ) -> None:
        repo = SignalRepo(session)
        bar = dt.datetime(2026, 11, 22, 16, tzinfo=dt.UTC)
        first = repo.record(
            symbol="BTCUSDT", tier="core", side=Side.LONG,
            status=SignalStatus.CANDIDATE, bar_time=bar,
        )
        second = repo.record(
            symbol="BTCUSDT", tier="core", side=Side.LONG,
            status=SignalStatus.APPROVED, bar_time=bar,
        )
        assert first.id == second.id
        assert second.status == SignalStatus.APPROVED.value

    def test_setup_fields_are_persisted(self, session) -> None:
        setup = TradeSetup(
            "BTCUSDT", Side.LONG, Decimal("92000"), Decimal("89200"),
            (Decimal("97600"), Decimal("103200")),
        )
        signal = SignalRepo(session).record(
            symbol="BTCUSDT", tier="core", side=Side.LONG,
            status=SignalStatus.APPROVED,
            bar_time=dt.datetime(2026, 11, 22, 16, tzinfo=dt.UTC),
            setup=setup, timeframe=Timeframe.H4,
        )
        assert signal.entry == Decimal("92000")
        assert signal.target_2 == Decimal("103200")
        assert signal.risk_reward == Decimal("2")

    def test_pending_notification_survives_a_delivery_failure(self, session) -> None:
        repo = SignalRepo(session)
        repo.record(
            symbol="BTCUSDT", tier="core", side=Side.LONG,
            status=SignalStatus.APPROVED,
            bar_time=dt.datetime(2026, 11, 22, 16, tzinfo=dt.UTC),
        )
        assert len(repo.pending_notification()) == 1


class TestRunRepo:
    def test_a_completed_bar_is_not_re_run(self, session) -> None:
        """GitHub Actions can fire one schedule twice."""
        repo = RunRepo(session)
        bar = dt.datetime(2026, 11, 22, 16, tzinfo=dt.UTC)
        run = repo.start(trigger="cron", bar_time=bar)
        repo.finish(run, ok=True)
        assert repo.already_ran("4h", bar, "cron")

    def test_a_failed_run_may_be_retried(self, session) -> None:
        repo = RunRepo(session)
        bar = dt.datetime(2026, 11, 22, 16, tzinfo=dt.UTC)
        run = repo.start(trigger="cron", bar_time=bar)
        repo.finish(run, ok=False, error="boom")
        assert not repo.already_ran("4h", bar, "cron")

    def test_a_long_traceback_is_truncated(self, session) -> None:
        repo = RunRepo(session)
        run = repo.start(trigger="cron", bar_time=None)
        repo.finish(run, ok=False, error="x" * 10_000)
        assert run.error is not None and len(run.error) <= 4000


class TestPositionRepo:
    def test_realised_pnl_and_r_are_computed_for_a_long(self, session) -> None:
        repo = PositionRepo(session)
        position = repo.create(
            symbol="BTCUSDT", side=Side.LONG, quantity=Decimal("1"),
            entry_price=Decimal("100"), stop_price=Decimal("90"),
            target_price=Decimal("120"), leverage=Decimal("1"),
            signal_id=None, is_paper=True,
        )
        repo.close(position, exit_price=Decimal("120"), reason="target")
        assert position.realised_pnl == Decimal("20")
        assert position.r_multiple == Decimal("2")

    def test_realised_pnl_is_computed_for_a_short(self, session) -> None:
        repo = PositionRepo(session)
        position = repo.create(
            symbol="BTCUSDT", side=Side.SHORT, quantity=Decimal("1"),
            entry_price=Decimal("100"), stop_price=Decimal("110"),
            target_price=Decimal("80"), leverage=Decimal("1"),
            signal_id=None, is_paper=True,
        )
        repo.close(position, exit_price=Decimal("80"), reason="target")
        assert position.realised_pnl == Decimal("20")
        assert position.r_multiple == Decimal("2")

    def test_costs_are_deducted_from_pnl(self, session) -> None:
        repo = PositionRepo(session)
        position = repo.create(
            symbol="BTCUSDT", side=Side.LONG, quantity=Decimal("1"),
            entry_price=Decimal("100"), stop_price=Decimal("90"),
            target_price=Decimal("120"), leverage=Decimal("1"),
            signal_id=None, is_paper=True,
        )
        repo.close(
            position, exit_price=Decimal("120"), reason="target",
            fees=Decimal("1"), funding=Decimal("0.5"),
        )
        assert position.realised_pnl == Decimal("18.5")

    def test_consecutive_losses_counts_only_the_current_streak(self, session) -> None:
        repo = PositionRepo(session)
        for exit_price in (Decimal("120"), Decimal("90"), Decimal("90")):
            position = repo.create(
                symbol="BTCUSDT", side=Side.LONG, quantity=Decimal("1"),
                entry_price=Decimal("100"), stop_price=Decimal("90"),
                target_price=Decimal("120"), leverage=Decimal("1"),
                signal_id=None, is_paper=True,
            )
            repo.close(position, exit_price=exit_price, reason="x")
        assert repo.consecutive_losses() == 2

    def test_a_live_position_without_a_stop_is_flagged(self, session) -> None:
        """The alarm condition of the whole design."""
        repo = PositionRepo(session)
        repo.create(
            symbol="BTCUSDT", side=Side.LONG, quantity=Decimal("1"),
            entry_price=Decimal("100"), stop_price=Decimal("90"),
            target_price=None, leverage=Decimal("1"),
            signal_id=None, is_paper=False,
        )
        assert len(repo.open_without_stop()) == 1

    def test_paper_positions_are_not_flagged_as_unprotected(self, session) -> None:
        repo = PositionRepo(session)
        repo.create(
            symbol="BTCUSDT", side=Side.LONG, quantity=Decimal("1"),
            entry_price=Decimal("100"), stop_price=Decimal("90"),
            target_price=None, leverage=Decimal("1"),
            signal_id=None, is_paper=True,
        )
        assert repo.open_without_stop() == []


class TestCircuitBreakerRepo:
    def test_a_tripped_breaker_is_active(self, session) -> None:
        repo = CircuitBreakerRepo(session)
        repo.trip(reason="max_daily_loss_pct", detail="down 4%", hours=24)
        active = repo.active()
        assert active is not None and active.reason == "max_daily_loss_pct"

    def test_clearing_deactivates_it(self, session) -> None:
        repo = CircuitBreakerRepo(session)
        repo.trip(reason="x", detail="y", hours=24)
        assert repo.clear_all() == 1
        assert repo.active() is None

    def test_an_expired_breaker_is_not_active(self, session) -> None:
        repo = CircuitBreakerRepo(session)
        repo.trip(reason="x", detail="y", hours=-1)
        assert repo.active() is None


class TestContractSpecRepo:
    def test_specs_round_trip(self, session) -> None:
        """Cached in the database because an Actions runner keeps no filesystem."""
        repo = ContractSpecRepo(session)
        repo.put_many([
            ContractSpec(
                "BTCUSDT", min_order_size=Decimal("0.0001"),
                size_step=Decimal("0.0001"), min_notional=Decimal("5"),
                max_leverage=125,
            )
        ])
        stored = repo.get("BTCUSDT")
        assert stored is not None
        assert stored.min_notional == Decimal("5")
        assert stored.max_leverage == 125

    def test_a_missing_symbol_returns_none(self, session) -> None:
        assert ContractSpecRepo(session).get("NOPE") is None


class TestAccountRepo:
    def test_day_start_equity_is_the_breaker_baseline(self, session) -> None:
        repo = AccountRepo(session)
        repo.snapshot(equity=Decimal("200"), is_paper=True)
        assert repo.equity_at_day_start() == Decimal("200")
