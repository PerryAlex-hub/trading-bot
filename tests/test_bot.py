"""The Telegram bot: command parsing, access control, and the webhook.

Access control is the important part. The webhook URL is effectively public,
and the bot can read the journal and trigger an analysis, so an unauthenticated
endpoint would hand both to anyone who found it.

Two independent barriers, and both are tested to fail closed when
unconfigured: the shared secret Telegram echoes in a header, and an allow-list
of chat ids.
"""

from __future__ import annotations

from decimal import Decimal

from tradingbot.notify.bot import BotCommand, TelegramBot, _num
from tradingbot.settings import Settings


def update(text: str, chat_id: str = "12345") -> dict:
    """A minimal Telegram update, shaped like the real payload."""
    return {
        "update_id": 1,
        "message": {
            "message_id": 7,
            "chat": {"id": int(chat_id), "type": "private"},
            "text": text,
        },
    }


def configured(**overrides) -> Settings:
    base = {
        "telegram_chat_id": "12345",
        "telegram_bot_token": "token",
        "telegram_webhook_secret": "s3cret",
    }
    base.update(overrides)
    return Settings(**base)


class TestCommandParsing:
    def test_a_plain_command_parses(self) -> None:
        command = BotCommand.parse(update("/status"))
        assert command is not None
        assert command.name == "status"
        assert command.args == []
        assert command.chat_id == "12345"

    def test_arguments_are_captured(self) -> None:
        command = BotCommand.parse(update("/analyse BTCUSDT"))
        assert command is not None
        assert command.name == "analyse"
        assert command.args == ["BTCUSDT"]

    def test_a_group_mention_is_stripped(self) -> None:
        """Group chats deliver '/status@MyBot', which must still dispatch."""
        command = BotCommand.parse(update("/status@MyTradingBot"))
        assert command is not None and command.name == "status"

    def test_case_is_normalised(self) -> None:
        command = BotCommand.parse(update("/STATUS"))
        assert command is not None and command.name == "status"

    def test_a_non_message_update_is_ignored(self) -> None:
        """Telegram sends joins, reactions and edits; none are commands."""
        assert BotCommand.parse({"update_id": 1}) is None
        assert BotCommand.parse({"update_id": 1, "message": {}}) is None

    def test_an_empty_message_is_ignored(self) -> None:
        assert BotCommand.parse(update("")) is None

    def test_an_edited_message_still_parses(self) -> None:
        payload = {
            "update_id": 1,
            "edited_message": {
                "message_id": 7,
                "chat": {"id": 12345},
                "text": "/help",
            },
        }
        command = BotCommand.parse(payload)
        assert command is not None and command.name == "help"


class TestWebhookSecret:
    def test_an_unconfigured_secret_rejects_everything(self) -> None:
        """Failing open here would expose the journal to anyone."""
        bot = TelegramBot(Settings(telegram_chat_id="12345"))
        assert not bot.verify_secret("anything")
        assert not bot.verify_secret(None)
        assert not bot.verify_secret("")

    def test_a_matching_secret_is_accepted(self) -> None:
        bot = TelegramBot(configured())
        assert bot.verify_secret("s3cret")

    def test_a_wrong_secret_is_rejected(self) -> None:
        bot = TelegramBot(configured())
        assert not bot.verify_secret("wrong")
        assert not bot.verify_secret("s3cre")
        assert not bot.verify_secret("s3cret ")


class TestChatAuthorisation:
    def test_the_configured_chat_is_authorised(self) -> None:
        assert TelegramBot(configured()).is_authorised("12345")

    def test_any_other_chat_is_refused(self) -> None:
        bot = TelegramBot(configured())
        assert not bot.is_authorised("99999")
        assert not bot.is_authorised("")

    def test_an_unauthorised_command_produces_no_reply(self) -> None:
        """Silence, not an error: a reply confirms the bot exists."""
        bot = TelegramBot(configured())
        assert bot.handle(update("/status", chat_id="99999")) is None

    def test_an_unconfigured_bot_answers_nobody(self) -> None:
        bot = TelegramBot(Settings())
        assert bot.handle(update("/status")) is None


class TestPublicMode:
    """Public access removes the chat-id friction but not the webhook secret."""

    def _public(self, **overrides):
        base = {
            "telegram_chat_id": "12345",
            "telegram_bot_token": "token",
            "telegram_webhook_secret": "s3cret",
            "telegram_public": True,
        }
        base.update(overrides)
        return Settings(**base)

    def test_a_stranger_is_answered_in_public_mode(self) -> None:
        bot = TelegramBot(self._public())
        assert bot.is_authorised("99999")
        assert bot.handle(update("/help", chat_id="99999")) is not None

    def test_public_mode_still_requires_the_webhook_secret(self) -> None:
        """The secret authenticates Telegram, not the user.

        Without it anyone could POST forged updates straight to the endpoint
        and impersonate any chat id, which is a different thing from letting
        strangers use the bot.
        """
        bot = TelegramBot(self._public(telegram_webhook_secret=None))
        assert not bot.settings.telegram_is_public
        assert not bot.is_authorised("99999")

    def test_private_mode_is_unchanged(self) -> None:
        bot = TelegramBot(self._public(telegram_public=False))
        assert bot.is_authorised("12345")
        assert not bot.is_authorised("99999")


class TestRateLimit:
    """Public mode must not let a stranger exhaust the free compute budget."""

    def _public(self, **overrides):
        base = {
            "telegram_chat_id": "12345",
            "telegram_bot_token": "token",
            "telegram_webhook_secret": "s3cret",
            "telegram_public": True,
        }
        base.update(overrides)
        return Settings(**base)

    def setup_method(self) -> None:
        from tradingbot.notify.bot import _RATE_BUCKETS

        _RATE_BUCKETS.clear()

    def test_general_commands_are_capped(self) -> None:
        bot = TelegramBot(self._public(telegram_rate_limit_per_minute=3))
        for _ in range(3):
            assert bot.rate_limit_exceeded("777", "status") is None
        message = bot.rate_limit_exceeded("777", "status")
        assert message is not None and "Rate limit" in message

    def test_analyse_has_its_own_tighter_budget(self) -> None:
        """It reloads hundreds of candles and recomputes everything."""
        bot = TelegramBot(
            self._public(
                telegram_rate_limit_per_minute=50,
                telegram_analyse_limit_per_minute=2,
            )
        )
        assert bot.rate_limit_exceeded("777", "analyse") is None
        assert bot.rate_limit_exceeded("777", "analyse") is None
        assert bot.rate_limit_exceeded("777", "analyse") is not None
        # The cheap commands are unaffected by the analyse budget.
        assert bot.rate_limit_exceeded("777", "status") is None

    def test_chats_are_limited_independently(self) -> None:
        bot = TelegramBot(self._public(telegram_rate_limit_per_minute=1))
        assert bot.rate_limit_exceeded("aaa", "status") is None
        assert bot.rate_limit_exceeded("aaa", "status") is not None
        assert bot.rate_limit_exceeded("bbb", "status") is None

    def test_a_private_bot_is_never_throttled(self) -> None:
        """Throttling your own single owner would be pure annoyance."""
        bot = TelegramBot(self._public(telegram_public=False))
        for _ in range(100):
            assert bot.rate_limit_exceeded("12345", "analyse") is None

    def test_throttling_is_enforced_through_handle(self) -> None:
        bot = TelegramBot(self._public(telegram_rate_limit_per_minute=2))
        assert bot.handle(update("/help", chat_id="555")) is not None
        assert bot.handle(update("/help", chat_id="555")) is not None
        third = bot.handle(update("/help", chat_id="555"))
        assert third is not None and "Rate limit" in third


class TestDispatch:
    def test_help_is_returned_without_touching_the_database(self) -> None:
        reply = TelegramBot(configured()).handle(update("/help"))
        assert reply is not None
        assert "/status" in reply and "/performance" in reply

    def test_start_is_aliased_to_help(self) -> None:
        bot = TelegramBot(configured())
        assert bot.handle(update("/start")) == bot.handle(update("/help"))

    def test_an_unknown_command_is_reported(self) -> None:
        reply = TelegramBot(configured()).handle(update("/nonsense"))
        assert reply is not None and "Unknown command" in reply

    def test_analyse_without_an_argument_explains_usage(self) -> None:
        reply = TelegramBot(configured()).handle(update("/analyse"))
        assert reply is not None and "Usage" in reply

    def test_analyse_rejects_a_symbol_outside_the_universe(self) -> None:
        reply = TelegramBot(configured()).handle(update("/analyse FAKECOIN"))
        assert reply is not None
        assert "not in the universe" in reply

    def test_a_handler_failure_is_reported_rather_than_swallowed(self, monkeypatch) -> None:
        """A silent bot is indistinguishable from a dead one.

        The failure is injected rather than relying on an absent database: an
        earlier version of this test asserted that /status fails, which passed
        only because the test machine had no DATABASE_URL and broke the moment
        one was configured.
        """
        bot = TelegramBot(configured())

        def boom(_command):
            raise RuntimeError("database unreachable")

        monkeypatch.setattr(bot, "_status", boom)
        reply = bot.handle(update("/status"))
        assert reply is not None
        assert "failed" in reply.lower()
        assert "RuntimeError" in reply

    def test_the_help_text_states_there_is_no_execution(self) -> None:
        reply = TelegramBot(configured()).handle(update("/help"))
        assert reply is not None
        assert "cannot place orders" in reply


class TestSignalNumberFormatting:
    """The alert formatter. A signal you cannot read at a glance is a defect."""

    def test_full_precision_prices_are_trimmed(self) -> None:
        """Observed on a real BTC signal: 60949.98127637765741."""
        from tradingbot.notify.telegram import _fmt

        assert _fmt("60949.98127637765741") == "60949.98"

    def test_tiny_quantities_keep_significant_figures(self) -> None:
        """Observed on a real BTC signal: 0.000176171075538736."""
        from tradingbot.notify.telegram import _fmt

        assert _fmt("0.000176171075538736") == "0.0001762"

    def test_sub_cent_prices_stay_distinguishable(self) -> None:
        from tradingbot.notify.telegram import _fmt

        assert _fmt("0.00004212345") == "0.00004212"

    def test_no_scientific_notation(self) -> None:
        from tradingbot.notify.telegram import _fmt

        for value in ("1E-7", "1E+5", "0.00000001"):
            assert "E" not in _fmt(value)

    def test_whole_numbers_lose_their_decimals(self) -> None:
        from tradingbot.notify.telegram import _fmt

        assert _fmt("92000.00") == "92000"

    def test_zero_and_none(self) -> None:
        from tradingbot.notify.telegram import _fmt

        assert _fmt("0") == "0"
        assert _fmt(None) == "n/a"


class TestNumberFormatting:
    def test_small_numbers_avoid_scientific_notation(self) -> None:
        """`1E-7` is unreadable in a trading alert."""
        assert "E" not in _num("0.00000012")
        assert _num("0.00000012") == "0.00000012"

    def test_trailing_zeros_are_trimmed(self) -> None:
        assert _num("92000.00") == "92000"

    def test_none_renders_as_a_dash(self) -> None:
        assert _num(None) == "-"

    def test_large_numbers_are_plain(self) -> None:
        assert _num("92000.5") == "92000.5"


class TestWebhookEndpoint:
    """The HTTP surface. Every rejection must still return 200.

    Telegram retries non-2xx responses and eventually disables a webhook that
    keeps failing, so returning an error would turn one bad update into an
    outage.
    """

    def _client(self):
        from fastapi.testclient import TestClient

        from tradingbot.api.app import app

        return TestClient(app)

    def test_ping_touches_nothing(self) -> None:
        """The keep-alive must not wake the database or read settings."""
        response = self._client().get("/ping")
        assert response.status_code == 200
        assert response.json() == {"status": "awake"}

    def test_a_missing_secret_returns_200_and_does_nothing(self) -> None:
        response = self._client().post("/telegram/webhook", json=update("/status"))
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_a_wrong_secret_returns_200(self) -> None:
        response = self._client().post(
            "/telegram/webhook",
            json=update("/status"),
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        )
        assert response.status_code == 200

    def test_a_malformed_body_returns_200(self) -> None:
        response = self._client().post(
            "/telegram/webhook",
            content=b"not json",
            headers={
                "X-Telegram-Bot-Api-Secret-Token": "wrong",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200

    def test_health_reports_no_execution_capability(self) -> None:
        payload = self._client().get("/health").json()
        assert payload["live_orders_enabled"] is False
        assert payload["trading_mode"] in {"signal_only", "paper"}


class TestAllMarks:
    """Mark prices for every contract in one request.

    Minimum order sizes are published in the base asset, so converting them to
    USDT needs a price per symbol. Fetching them one at a time would be ~200
    requests to answer one question.
    """

    def _client(self, payload):
        from tradingbot.exchanges.weex_market import WeexMarketData

        client = WeexMarketData()
        client._try_paths = lambda paths, params=None: ("/stub", payload)  # type: ignore[method-assign]
        return client

    def test_parses_a_list_of_tickers(self) -> None:
        client = self._client(
            {
                "data": [
                    {"symbol": "BTC-USDT", "last": "60737.12"},
                    {"symbol": "DOGEUSDT", "markPrice": "0.05533"},
                ]
            }
        )
        marks = client.all_marks()
        # Symbols are normalised, so the hyphenated form still matches config.
        assert marks["BTCUSDT"] == Decimal("60737.12")
        assert marks["DOGEUSDT"] == Decimal("0.05533")

    def test_skips_rows_without_a_usable_price(self) -> None:
        client = self._client(
            {
                "data": [
                    {"symbol": "AAAUSDT", "last": "0"},
                    {"symbol": "BBBUSDT"},
                    {"last": "12"},
                    {"symbol": "CCCUSDT", "last": "1.5"},
                ]
            }
        )
        assert client.all_marks() == {"CCCUSDT": Decimal("1.5")}

    def test_an_unreachable_endpoint_returns_empty_not_an_exception(self) -> None:
        from tradingbot.exchanges.weex import WeexEndpointError
        from tradingbot.exchanges.weex_market import WeexMarketData

        client = WeexMarketData()

        def boom(paths, params=None):
            raise WeexEndpointError("all paths failed")

        client._try_paths = boom  # type: ignore[method-assign]
        assert client.all_marks() == {}
