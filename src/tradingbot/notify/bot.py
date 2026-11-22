"""The Telegram bot: this is the user interface.

There is no web frontend. Telegram is both the push channel and the dashboard,
which suits the deployment far better than a web app would:

* **Push beats pull here.** The signal matters the moment the 4h bar closes.
  A dashboard requires you to go and look, and on a free host it takes ~60
  seconds to wake up before it can show you anything.
* **No separate deploy, no CORS, no auth to build.** Telegram already knows
  who you are, and the API service is already running.

**Webhook, not polling.** Long-polling needs a process that runs forever, and
there is no free always-on worker. A webhook is a plain HTTP POST into the
existing FastAPI service, which costs nothing extra.

Security matters more than usual, because a webhook URL is effectively public
and this bot can trigger analysis runs. Two independent checks:

1. Telegram echoes a shared secret in `X-Telegram-Bot-Api-Secret-Token`, which
   is verified before the body is even parsed.
2. The sender's chat id must be in the allow-list. Without this, anyone who
   found the bot could read your journal.

Both are required. Failing either returns a silent 200, because a 401 tells an
attacker the endpoint is real.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx

from ..settings import Settings

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
MAX_MESSAGE_LENGTH = 4096

HELP_TEXT = """*trading-bot*

Analysis is automatic: every 4h bar close is scanned and anything approved is
sent here. These commands are for looking at what it has been doing.

*/status* - equity, open paper trades, circuit breaker
*/signals* `[n]` - the last n verdicts (default 10)
*/journal* `[n]` - verdicts with the reason each was approved or rejected
*/performance* `[days]` - results in R (default 90 days)
*/positions* - currently tracked paper trades
*/analyse* `SYMBOL` - analyse one symbol right now
*/universe* - what is traded, and the risk rules
*/runs* - recent scheduled runs, to check the scheduler is firing
*/help* - this message

This bot has no exchange credentials and cannot place orders.
"""


@dataclass
class BotCommand:
    """One parsed incoming command."""

    chat_id: str
    text: str
    name: str
    args: list[str]
    message_id: int | None = None

    @classmethod
    def parse(cls, update: dict[str, Any]) -> BotCommand | None:
        """Pull a command out of a Telegram update.

        Returns None for anything that is not a text message, which covers
        edits, joins, reactions and the other update kinds Telegram sends.
        """
        message = (
            update.get("message")
            or update.get("edited_message")
            or update.get("channel_post")
        )
        if not isinstance(message, dict):
            return None

        text = str(message.get("text") or "").strip()
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id") or "")
        if not text or not chat_id:
            return None

        parts = text.split()
        name = parts[0].lstrip("/").lower()
        # Group chats deliver "/status@MyBotName"; strip the mention.
        name = name.split("@", 1)[0]

        return cls(
            chat_id=chat_id,
            text=text,
            name=name,
            args=parts[1:],
            message_id=message.get("message_id"),
        )


class TelegramBot:
    """Handles incoming commands and renders replies.

    Holds no database session: each handler opens its own short-lived scope,
    because Neon's free compute suspends and a long-lived session would either
    hold it awake or hand out a dead connection.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.token = (
            settings.telegram_bot_token.get_secret_value()
            if settings.telegram_bot_token
            else None
        )
        self.allowed = settings.allowed_chat_ids()

    # -- security ----------------------------------------------------------

    def verify_secret(self, header_value: str | None) -> bool:
        """Check the header Telegram echoes back.

        When no secret is configured this returns False: an unconfigured
        webhook must reject everything rather than accept everything, since the
        URL is guessable and this is the only barrier.
        """
        configured = self.settings.telegram_webhook_secret
        if configured is None:
            log.warning(
                "TELEGRAM_WEBHOOK_SECRET is not set; rejecting webhook traffic"
            )
            return False
        expected = configured.get_secret_value()
        if not header_value or not expected:
            return False
        # Constant-time comparison: a timing oracle on a 32-char secret is a
        # cheap thing to rule out.
        import hmac

        return hmac.compare_digest(header_value, expected)

    def is_authorised(self, chat_id: str) -> bool:
        return chat_id in self.allowed

    # -- dispatch ----------------------------------------------------------

    def handle(self, update: dict[str, Any]) -> str | None:
        """Process one update and return the reply text, if any."""
        command = BotCommand.parse(update)
        if command is None:
            return None

        if not self.is_authorised(command.chat_id):
            log.warning("Rejected a command from unauthorised chat %s", command.chat_id)
            return None

        handlers = {
            "start": self._help,
            "help": self._help,
            "status": self._status,
            "signals": self._signals,
            "journal": self._journal,
            "performance": self._performance,
            "perf": self._performance,
            "positions": self._positions,
            "analyse": self._analyse,
            "analyze": self._analyse,
            "universe": self._universe,
            "runs": self._runs,
        }
        handler = handlers.get(command.name)
        if handler is None:
            return f"Unknown command `/{command.name}`. Try /help."

        try:
            return handler(command)
        except Exception as exc:
            log.exception("Command /%s failed", command.name)
            # Report the failure rather than going silent: a bot that stops
            # answering is indistinguishable from a bot that is down.
            return f"`/{command.name}` failed: {type(exc).__name__}: {exc}"

    # -- handlers ----------------------------------------------------------

    def _help(self, _: BotCommand) -> str:
        return HELP_TEXT

    def _status(self, _: BotCommand) -> str:
        from ..config import load_risk
        from ..db.engine import session_scope
        from ..db.repo import AccountRepo, CircuitBreakerRepo, PositionRepo

        with session_scope() as session:
            account = AccountRepo(session).latest()
            breaker = CircuitBreakerRepo(session).active()
            positions = PositionRepo(session)
            open_positions = positions.open_positions()
            closed = positions.recent_closed(limit=20)
            streak = positions.consecutive_losses()

        equity = account.equity if account else load_risk().assumed_equity
        lines = [
            "*Status*",
            f"mode `{self.settings.trading_mode.value}` | orders: never",
            f"equity `{_num(equity)}` USDT (simulated)",
            f"open `{len(open_positions)}` | losing streak `{streak}`",
        ]

        if breaker is not None:
            remaining = breaker.halted_until - datetime.now(tz=UTC)
            lines += [
                "",
                f"*CIRCUIT BREAKER ACTIVE* - {breaker.reason}",
                f"_{breaker.detail}_",
                f"clears in {remaining.total_seconds()/3600:.1f}h",
            ]

        if open_positions:
            lines += ["", "*Open*"]
            for position in open_positions:
                lines.append(
                    f"`{position.symbol}` {position.side} from `{_num(position.entry_price)}` "
                    f"stop `{_num(position.stop_price)}`"
                )

        if closed:
            wins = sum(1 for p in closed if (p.realised_pnl or 0) > 0)
            total_r = sum((p.r_multiple or Decimal("0")) for p in closed)
            lines += [
                "",
                f"last {len(closed)} closed: {wins}W/{len(closed)-wins}L, "
                f"`{total_r:+.2f}R`",
            ]
        return "\n".join(lines)

    def _signals(self, command: BotCommand) -> str:
        from ..db.engine import session_scope
        from ..db.repo import SignalRepo

        limit = _int_arg(command.args, default=10, maximum=25)
        with session_scope() as session:
            signals = SignalRepo(session).recent(limit=limit)

        if not signals:
            return "No signals recorded yet. Has the scheduler run? Try /runs."

        lines = [f"*Last {len(signals)} verdicts*"]
        for signal in signals:
            mark = {
                "approved": "APPROVED",
                "executed": "TRACKED",
                "rejected_risk": "rejected",
                "rejected_reasoning": "rejected",
                "candidate": "no setup",
                "closed": "closed",
            }.get(signal.status, signal.status)
            score = f"{signal.score:+.2f}" if signal.score is not None else "-"
            lines.append(
                f"`{signal.bar_time:%m-%d %H:%M}` *{signal.symbol}* "
                f"{signal.side} {mark} ({score})"
            )
        return "\n".join(lines)

    def _journal(self, command: BotCommand) -> str:
        from ..db.engine import session_scope
        from ..db.repo import SignalRepo

        limit = _int_arg(command.args, default=6, maximum=12)
        with session_scope() as session:
            signals = SignalRepo(session).recent(limit=limit)

        if not signals:
            return "Nothing journaled yet."

        blocks = [f"*Journal* (last {len(signals)})"]
        for signal in signals:
            header = (
                f"`{signal.bar_time:%m-%d %H:%M}` *{signal.symbol}* "
                f"{signal.side} - {signal.status}"
            )
            detail = _signal_detail(signal)
            blocks.append(header + (f"\n_{detail}_" if detail else ""))
        return "\n\n".join(blocks)

    def _performance(self, command: BotCommand) -> str:
        from ..db.engine import session_scope
        from ..db.repo import PositionRepo

        days = _int_arg(command.args, default=90, maximum=3650)
        since = datetime.now(tz=UTC) - timedelta(days=days)
        with session_scope() as session:
            closed = PositionRepo(session).closed_since(since)

        if not closed:
            return (
                f"No trades resolved in the last {days} days.\n\n"
                "_Paper tracking only records trades when_ `TRADING_MODE=paper`_._"
            )

        wins = [p for p in closed if (p.realised_pnl or 0) > 0]
        losses = [p for p in closed if (p.realised_pnl or 0) <= 0]
        r_values = [p.r_multiple for p in closed if p.r_multiple is not None]
        total_r = sum(r_values) if r_values else Decimal("0")
        gross_win = sum((p.realised_pnl or Decimal("0")) for p in wins)
        gross_loss = abs(sum((p.realised_pnl or Decimal("0")) for p in losses))

        lines = [
            f"*Performance* - last {days} days (paper)",
            f"trades `{len(closed)}` | `{len(wins)}W/{len(losses)}L` "
            f"({len(wins)/len(closed)*100:.0f}%)",
            f"total `{total_r:+.2f}R`",
        ]
        if r_values:
            lines.append(f"average `{total_r/len(r_values):+.2f}R` per trade")
        if gross_loss > 0:
            lines.append(f"profit factor `{gross_win/gross_loss:.2f}`")
        lines.append(f"net P&L `{sum((p.realised_pnl or Decimal('0')) for p in closed):+.2f}` USDT")

        # Expectancy is what says whether the strategy works. Win rate alone
        # is the number that misleads people.
        if r_values:
            expectancy = total_r / len(r_values)
            verdict = (
                "positive expectancy" if expectancy > 0 else "negative expectancy"
            )
            lines += ["", f"_{verdict} over {len(r_values)} resolved trades_"]
            if len(r_values) < 30:
                lines.append(
                    f"_only {len(r_values)} trades: far too few to conclude anything_"
                )
        return "\n".join(lines)

    def _positions(self, _: BotCommand) -> str:
        from ..db.engine import session_scope
        from ..db.repo import PositionRepo

        with session_scope() as session:
            open_positions = PositionRepo(session).open_positions()

        if not open_positions:
            return "No open paper positions."

        blocks = ["*Open paper positions*"]
        for position in open_positions:
            age = datetime.now(tz=UTC) - _aware(position.opened_at)
            blocks.append(
                f"*{position.symbol}* {position.side} `{_num(position.quantity)}`\n"
                f"entry `{_num(position.entry_price)}` | "
                f"stop `{_num(position.stop_price)}` | "
                f"target `{_num(position.target_price)}`\n"
                f"_open {age.total_seconds()/3600:.0f}h_"
            )
        return "\n\n".join(blocks)

    def _analyse(self, command: BotCommand) -> str:
        """Analyse one symbol on demand.

        Reads stored candles rather than fetching, so the reply is fast and
        cannot be made to hammer an exchange by repeated commands.
        """
        if not command.args:
            return "Usage: `/analyse BTCUSDT`"

        symbol = command.args[0].upper()
        from ..analysis import SymbolData, analyse_symbol
        from ..analysis.pipeline import InsufficientHistory
        from ..config import load_universe
        from ..db.engine import session_scope
        from ..db.repo import CandleRepo
        from ..models import Timeframe

        universe = load_universe()
        config = universe.by_symbol(symbol)
        if config is None:
            known = ", ".join(c.symbol for c in universe.symbols)
            return f"`{symbol}` is not in the universe.\nTracked: {known}"

        with session_scope() as session:
            repo = CandleRepo(session)
            frames = {
                timeframe: repo.latest(symbol, timeframe, limit=400)
                for timeframe in universe.all_timeframes
            }
            btc = repo.latest("BTCUSDT", universe.primary_timeframe, limit=400)
            eth = repo.latest("ETHUSDT", universe.primary_timeframe, limit=400)

        if not frames.get(universe.primary_timeframe):
            return f"No stored candles for `{symbol}`. Has ingest run?"

        try:
            snapshot = analyse_symbol(
                SymbolData.from_candles(
                    symbol, config.tier, frames,
                    btc_candles=btc or None, eth_candles=eth or None,
                ),
                primary=Timeframe(universe.primary_timeframe),
            )
        except InsufficientHistory as exc:
            return f"Cannot analyse `{symbol}`: {exc}"

        indicators = snapshot.indicators
        lines = [
            f"*{symbol}* `{_num(snapshot.price)}` ({config.tier.value})",
            f"_bar {snapshot.bar_time:%Y-%m-%d %H:%M} UTC_",
            "",
            f"*score* `{snapshot.score:+.3f}` -> *{snapshot.side.value.upper()}*",
            f"trend `{snapshot.structure.trend.value}` | "
            f"regime `{snapshot.regime.state.value}`",
        ]
        if indicators.rsi_14 is not None:
            lines.append(
                f"RSI `{indicators.rsi_14:.0f}` | "
                f"ADX `{indicators.adx_14:.0f}` | "
                f"ATR `{indicators.atr_pct:.2f}%`"
                if indicators.adx_14 and indicators.atr_pct
                else f"RSI `{indicators.rsi_14:.0f}`"
            )

        support = snapshot.levels.nearest_support(snapshot.price)
        resistance = snapshot.levels.nearest_resistance(snapshot.price)
        if support or resistance:
            lines.append(
                f"S `{_num(support.price) if support else '-'}` | "
                f"R `{_num(resistance.price) if resistance else '-'}`"
            )

        if snapshot.chart_patterns:
            lines.append(
                "patterns: " + ", ".join(p.name for p in snapshot.chart_patterns[:3])
            )

        if snapshot.confluence.blockers:
            lines += ["", "*Blocked:*"] + [
                f"- {blocker}" for blocker in snapshot.confluence.blockers
            ]

        reasons = snapshot.confluence.top_reasons(4)
        if reasons:
            lines += ["", "*Why:*"] + [f"- {reason}" for reason in reasons]

        if snapshot.data_warnings:
            lines += ["", "*Data:*"] + [f"- {w}" for w in snapshot.data_warnings]

        return "\n".join(lines)

    def _universe(self, _: BotCommand) -> str:
        from ..config import load_risk, load_universe

        universe = load_universe()
        risk = load_risk()
        cap = risk.decimal("portfolio", "max_portfolio_btc_beta", default="2.0")

        lines = [
            "*Universe*",
            f"primary timeframe `{universe.primary_timeframe}`",
            "",
        ]
        for config in universe.symbols:
            beta = risk.beta_to_btc(config.symbol)
            blocked = " *[blocked: beta > cap]*" if beta > cap else ""
            lines.append(
                f"`{config.symbol}` {config.tier.value} beta `{beta}`{blocked}"
            )
        lines += [
            "",
            "*Risk*",
            f"risk per trade `{risk.risk_pct}%` | min R:R `{risk.min_risk_reward}`",
            f"max open `{risk.get('portfolio', 'max_open_positions')}` | "
            f"beta cap `{cap}`",
        ]
        return "\n".join(lines)

    def _runs(self, _: BotCommand) -> str:
        from ..db.engine import session_scope
        from ..db.repo import RunRepo

        with session_scope() as session:
            runs = RunRepo(session).recent(limit=8)

        if not runs:
            return "No runs recorded. The scheduler may not be configured."

        lines = ["*Recent runs*"]
        for run in runs:
            status = "ok" if run.ok else "FAILED"
            lines.append(
                f"`{run.started_at:%m-%d %H:%M}` {run.trigger} {status} "
                f"- {run.symbols_scanned} scanned, {run.signals_approved} approved"
            )
            if run.error:
                lines.append(f"  _{run.error[:120]}_")

        newest = _aware(runs[0].started_at)
        age = (datetime.now(tz=UTC) - newest).total_seconds() / 3600
        if age > 8:
            lines += [
                "",
                f"*Warning:* the last run was {age:.0f}h ago. On a 4h schedule "
                "it should be under ~5h. Check GitHub Actions.",
            ]
        return "\n".join(lines)

    # -- sending -----------------------------------------------------------

    def reply(self, chat_id: str, text: str) -> bool:
        """Send a reply. Never raises."""
        if not self.token:
            log.warning("No Telegram token configured; cannot reply")
            return False

        if len(text) > MAX_MESSAGE_LENGTH:
            text = text[: MAX_MESSAGE_LENGTH - 20] + "\n...(truncated)"

        try:
            response = httpx.post(
                f"{API_BASE}/bot{self.token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            log.warning("Telegram reply failed: %s", exc)
            return False

        if response.status_code != 200:
            # Markdown is the usual culprit: an unescaped underscore in a
            # symbol or reason makes Telegram reject the whole message.
            log.warning("Telegram rejected a reply: %s", response.text[:200])
            try:
                httpx.post(
                    f"{API_BASE}/bot{self.token}/sendMessage",
                    json={"chat_id": chat_id, "text": _strip_markdown(text)},
                    timeout=15.0,
                )
                return True
            except httpx.HTTPError:
                return False
        return True


# ---------------------------------------------------------------------------
# Webhook registration
# ---------------------------------------------------------------------------


def register_webhook(settings: Settings, *, drop_pending: bool = True) -> tuple[bool, str]:
    """Point Telegram at the deployed API.

    Called once after deploying, by `tradingbot telegram-setup`. Telegram
    requires HTTPS and will not accept localhost, so this only works against
    the public Render URL.
    """
    if settings.telegram_bot_token is None:
        return False, "TELEGRAM_BOT_TOKEN is not set"
    if settings.telegram_webhook_secret is None:
        return False, (
            "TELEGRAM_WEBHOOK_SECRET is not set. The webhook URL is effectively "
            "public, so the bot refuses to register without a shared secret."
        )
    if not settings.public_base_url:
        return False, "PUBLIC_BASE_URL is not set (e.g. https://your-app.onrender.com)"

    base = settings.public_base_url.rstrip("/")
    if not base.startswith("https://"):
        return False, f"Telegram requires HTTPS; got {base}"

    token = settings.telegram_bot_token.get_secret_value()
    try:
        response = httpx.post(
            f"{API_BASE}/bot{token}/setWebhook",
            json={
                "url": f"{base}/telegram/webhook",
                "secret_token": settings.telegram_webhook_secret.get_secret_value(),
                "drop_pending_updates": drop_pending,
                # Only message updates are handled, so asking for the rest
                # would just burn the free service's wake time.
                "allowed_updates": ["message"],
            },
            timeout=20.0,
        )
    except httpx.HTTPError as exc:
        return False, f"could not reach Telegram: {exc}"

    if response.status_code != 200:
        return False, f"Telegram returned {response.status_code}: {response.text[:300]}"
    payload = response.json()
    if not payload.get("ok"):
        return False, f"Telegram refused: {payload}"
    return True, f"webhook registered at {base}/telegram/webhook"


def webhook_info(settings: Settings) -> dict[str, Any] | None:
    """Ask Telegram what webhook it currently has, for diagnostics."""
    if settings.telegram_bot_token is None:
        return None
    token = settings.telegram_bot_token.get_secret_value()
    try:
        response = httpx.get(f"{API_BASE}/bot{token}/getWebhookInfo", timeout=15.0)
        return response.json().get("result") if response.status_code == 200 else None
    except httpx.HTTPError:
        return None


def set_command_menu(settings: Settings) -> bool:
    """Register the command list so Telegram shows an autocomplete menu."""
    if settings.telegram_bot_token is None:
        return False
    token = settings.telegram_bot_token.get_secret_value()
    commands = [
        {"command": "status", "description": "equity, open trades, breaker"},
        {"command": "signals", "description": "recent verdicts"},
        {"command": "journal", "description": "verdicts with reasons"},
        {"command": "performance", "description": "results in R"},
        {"command": "positions", "description": "open paper trades"},
        {"command": "analyse", "description": "analyse one symbol now"},
        {"command": "universe", "description": "symbols and risk rules"},
        {"command": "runs", "description": "check the scheduler is firing"},
        {"command": "help", "description": "list commands"},
    ]
    try:
        response = httpx.post(
            f"{API_BASE}/bot{token}/setMyCommands",
            json={"commands": commands},
            timeout=15.0,
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _num(value: Any) -> str:
    """Format a number without scientific notation or trailing zeros.

    The universe spans BTC at five figures and tokens at four decimals, and
    `str(Decimal)` will happily produce `1E-7`, which is unreadable in an alert.
    """
    if value is None:
        return "-"
    try:
        decimal_value = Decimal(str(value))
    except Exception:
        return str(value)
    return format(decimal_value.normalize(), "f")


def _int_arg(args: list[str], *, default: int, maximum: int) -> int:
    """Parse a numeric argument, clamped. Bad input silently falls back."""
    if not args:
        return default
    try:
        return max(1, min(maximum, int(args[0])))
    except (TypeError, ValueError):
        return default


def _signal_detail(signal: Any) -> str:
    """The most useful one-line explanation available for a signal."""
    if signal.vetoes:
        rules = signal.vetoes.get("rules") or []
        if rules:
            return "; ".join(
                f"{rule.get('rule')}: {rule.get('detail')}" for rule in rules[:2]
            )[:220]
        if signal.vetoes.get("setup"):
            return str(signal.vetoes["setup"])[:220]
        if signal.vetoes.get("blockers"):
            return "; ".join(signal.vetoes["blockers"])[:220]
    if signal.reasons:
        top = signal.reasons.get("top") or []
        if top:
            return str(top[0])[:220]
    return ""


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _strip_markdown(text: str) -> str:
    for character in ("*", "_", "`", "[", "]"):
        text = text.replace(character, "")
    return text
