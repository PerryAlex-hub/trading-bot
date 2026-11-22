"""The Telegram bot: command parsing, access control, and the webhook.

Access control is the important part. The webhook URL is effectively public,
and the bot can read the journal and trigger an analysis, so an unauthenticated
endpoint would hand both to anyone who found it.

Two independent barriers, and both are tested to fail closed when
unconfigured: the shared secret Telegram echoes in a header, and an allow-list
of chat ids.
"""

from __future__ import annotations

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
