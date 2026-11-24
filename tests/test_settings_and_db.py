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


class TestNoLiveTrading:
    """The guarantee is now structural rather than a gate.

    Previously two switches had to agree before real orders were possible.
    Removing the capability entirely is a stronger property: there is no code
    path to place an order, no credentials to hold, and therefore nothing for a
    misconfiguration to arm.
    """

    def test_only_signal_only_and_paper_exist(self) -> None:
        assert {mode.value for mode in TradingMode} == {"signal_only", "paper"}

    def test_default_mode_is_signal_only(self) -> None:
        assert Settings().trading_mode is TradingMode.SIGNAL_ONLY

    def test_no_mode_can_place_real_orders(self) -> None:
        for mode in TradingMode:
            assert not Settings(trading_mode=mode).can_place_real_orders

    def test_exchange_credentials_are_not_configurable(self) -> None:
        """Nothing can supply a key, so nothing can authenticate."""
        fields = set(Settings.model_fields)
        assert "weex_api_key" not in fields
        assert "weex_api_secret" not in fields
        assert "weex_api_passphrase" not in fields
        assert "live_trading_enabled" not in fields

    def test_paper_mode_tracks_positions(self) -> None:
        assert Settings(trading_mode=TradingMode.PAPER).tracks_paper_positions
        assert not Settings(trading_mode=TradingMode.SIGNAL_ONLY).tracks_paper_positions


class TestTelegramAuthorisation:
    """Telegram is the user interface, so its access control is load-bearing."""

    def test_the_delivery_chat_is_authorised_by_default(self) -> None:
        assert Settings(telegram_chat_id="12345").allowed_chat_ids() == frozenset({"12345"})

    def test_an_unconfigured_bot_authorises_nobody(self) -> None:
        """Failing closed: an open bot would expose the journal to anyone."""
        assert Settings().allowed_chat_ids() == frozenset()

    def test_multiple_ids_can_be_listed(self) -> None:
        settings = Settings(telegram_chat_id="1", telegram_allowed_chat_ids="1, 2 ,3")
        assert settings.allowed_chat_ids() == frozenset({"1", "2", "3"})

    def test_webhook_secret_is_required_to_accept_traffic(self) -> None:
        from tradingbot.notify.bot import TelegramBot

        bot = TelegramBot(Settings(telegram_chat_id="1"))
        assert not bot.verify_secret("anything")
        assert not bot.verify_secret(None)

    def test_a_matching_secret_is_accepted(self) -> None:
        from tradingbot.notify.bot import TelegramBot

        bot = TelegramBot(
            Settings(telegram_chat_id="1", telegram_webhook_secret="s3cret")
        )
        assert bot.verify_secret("s3cret")
        assert not bot.verify_secret("wrong")


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

    def test_a_failed_run_can_actually_be_restarted(self, session) -> None:
        """REGRESSION: the unique constraint made retries impossible.

        `already_ran()` correctly permits reprocessing a failed bar, but
        `start()` then inserted a second row and died on
        uq_run_per_bar_trigger. One transient failure therefore poisoned that
        bar forever, because every retry crashed identically.
        """
        repo = RunRepo(session)
        bar = dt.datetime(2026, 11, 22, 16, tzinfo=dt.UTC)

        first = repo.start(trigger="cron", bar_time=bar)
        repo.finish(first, ok=False, error="transient network failure")
        assert not repo.already_ran("4h", bar, "cron")

        # Must not raise, and must reuse the same row.
        second = repo.start(trigger="cron", bar_time=bar)
        assert second.id == first.id
        assert second.error is None, "stale error carried into the new attempt"
        assert second.finished_at is None
        assert second.ok is False

        repo.finish(second, ok=True, symbols_scanned=6)
        assert repo.already_ran("4h", bar, "cron")

    def test_restarting_clears_the_previous_counts(self, session) -> None:
        repo = RunRepo(session)
        bar = dt.datetime(2026, 11, 22, 20, tzinfo=dt.UTC)
        first = repo.start(trigger="manual", bar_time=bar)
        repo.finish(first, ok=True, symbols_scanned=6, signals_approved=4)

        second = repo.start(trigger="manual", bar_time=bar)
        assert second.symbols_scanned == 0
        assert second.signals_approved == 0

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

    def test_streak_is_deterministic_when_positions_close_together(self, session) -> None:
        """REGRESSION: several positions can close in the same microsecond.

        The monitor's reconcile loop does exactly that when it finds two stops
        filled while the bot was asleep. Ordering on closed_at alone left their
        order arbitrary, and consecutive_losses() stops at the first win - so
        the streak, and the circuit-breaker decision built on it, was
        non-deterministic.
        """
        repo = PositionRepo(session)
        for exit_price in (Decimal("90"), Decimal("120"), Decimal("90"), Decimal("90")):
            position = repo.create(
                symbol="BTCUSDT", side=Side.LONG, quantity=Decimal("1"),
                entry_price=Decimal("100"), stop_price=Decimal("90"),
                target_price=Decimal("120"), leverage=Decimal("1"),
                signal_id=None, is_paper=True,
            )
            repo.close(position, exit_price=exit_price, reason="x")

        # Two losses after the win, regardless of timestamp collisions.
        assert [repo.consecutive_losses() for _ in range(5)] == [2] * 5

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


class TestAsOfReads:
    """Catch-up analysis must not see bars that closed after the one it is on.

    This is the guard that makes backfilling honest. Re-analysing a missed bar
    with today's candles would score the setup against its own outcome, and
    every journalled result from such a run would be worthless.
    """

    def test_as_of_truncates_the_series(self, session) -> None:
        candles = make_candles(n=40)
        repo = CandleRepo(session)
        repo.upsert_many("BTCUSDT", "4h", candles)

        cutoff = candles[20].open_time
        visible = repo.latest("BTCUSDT", "4h", limit=500, as_of=cutoff)

        assert visible[-1].open_time == cutoff
        assert len(visible) == 21
        assert all(candle.open_time <= cutoff for candle in visible)

    def test_as_of_none_is_unchanged(self, session) -> None:
        candles = make_candles(n=25)
        repo = CandleRepo(session)
        repo.upsert_many("BTCUSDT", "4h", candles)

        assert len(repo.latest("BTCUSDT", "4h", limit=500)) == 25

    def test_as_of_still_respects_the_limit(self, session) -> None:
        """The limit applies to the truncated window, newest-first."""
        candles = make_candles(n=40)
        repo = CandleRepo(session)
        repo.upsert_many("BTCUSDT", "4h", candles)

        cutoff = candles[30].open_time
        visible = repo.latest("BTCUSDT", "4h", limit=5, as_of=cutoff)

        assert len(visible) == 5
        assert visible[-1].open_time == cutoff


class TestBarCompletionIgnoresTrigger:
    """A bar done by cron must not be re-done by a backfill.

    `already_ran` was scoped to one trigger, which is right for its original
    job (letting `--force` and a failed-run retry through) but wrong for the
    catch-up pass: scoping by trigger would make every cron-analysed bar look
    unanalysed to the backfill and the journal would fill with duplicates.
    """

    def _complete(self, session, bar_time, trigger: str) -> None:
        repo = RunRepo(session)
        run = repo.start(trigger=trigger, bar_time=bar_time, timeframe="4h")
        repo.finish(run, ok=True)

    def test_any_trigger_counts_when_trigger_is_unspecified(self, session) -> None:
        bar = dt.datetime(2026, 11, 24, 8, tzinfo=dt.UTC)
        self._complete(session, bar, "cron")

        repo = RunRepo(session)
        assert repo.already_ran("4h", bar) is True
        assert repo.already_ran("4h", bar, "backfill") is False
        assert repo.already_ran("4h", bar, "cron") is True

    def test_an_unanalysed_bar_is_reported_missing(self, session) -> None:
        bar = dt.datetime(2026, 11, 24, 8, tzinfo=dt.UTC)
        assert RunRepo(session).already_ran("4h", bar) is False

    def test_a_failed_run_does_not_count_as_done(self, session) -> None:
        """Otherwise a transient failure would permanently skip that bar."""
        bar = dt.datetime(2026, 11, 24, 12, tzinfo=dt.UTC)
        repo = RunRepo(session)
        run = repo.start(trigger="cron", bar_time=bar, timeframe="4h")
        repo.finish(run, ok=False, error="boom")

        assert repo.already_ran("4h", bar) is False


class TestMissedBarSelection:
    """Which bars a catch-up pass decides to replay.

    The deployed scheduler delivered 7 of ~46 expected triggers in 37 hours,
    so this selection is the difference between analysing 3 bars a day and
    analysing all 6.
    """

    class FakeRuns:
        def __init__(self, done: set) -> None:
            self.done = done

        def already_ran(self, timeframe, bar_time, trigger=None) -> bool:
            return bar_time in self.done

    def _bars(self, done: set, *, max_bars: int = 6):
        from tradingbot.jobs.analyze import _missed_bars

        current = dt.datetime(2026, 11, 24, 12, tzinfo=dt.UTC)
        return _missed_bars(
            self.FakeRuns(done),
            timeframe="4h",
            current=current,
            step=dt.timedelta(hours=4),
            max_bars=max_bars,
        )

    def test_finds_every_gap_up_to_the_window(self) -> None:
        missed = self._bars(set(), max_bars=3)
        assert missed == [
            dt.datetime(2026, 11, 24, 0, tzinfo=dt.UTC),
            dt.datetime(2026, 11, 24, 4, tzinfo=dt.UTC),
            dt.datetime(2026, 11, 24, 8, tzinfo=dt.UTC),
        ]

    def test_stops_at_the_first_analysed_bar(self) -> None:
        """A completed bar is the boundary; nothing older gets replayed."""
        missed = self._bars({dt.datetime(2026, 11, 24, 4, tzinfo=dt.UTC)})
        assert missed == [dt.datetime(2026, 11, 24, 8, tzinfo=dt.UTC)]

    def test_nothing_to_do_when_the_previous_bar_is_done(self) -> None:
        assert self._bars({dt.datetime(2026, 11, 24, 8, tzinfo=dt.UTC)}) == []

    def test_never_includes_the_current_bar(self) -> None:
        """The current bar is run separately, and is the only one alerted."""
        current = dt.datetime(2026, 11, 24, 12, tzinfo=dt.UTC)
        assert current not in self._bars(set())

    def test_window_bounds_a_long_outage(self) -> None:
        assert len(self._bars(set(), max_bars=2)) == 2

    def test_disabled_window_replays_nothing(self) -> None:
        assert self._bars(set(), max_bars=0) == []
