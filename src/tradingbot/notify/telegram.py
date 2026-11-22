"""Telegram delivery.

Chosen because it is free and push-based, which matters more than it sounds:
the API runs on a free host that sleeps after 15 minutes idle, so a dashboard
you have to open is a dashboard that takes a minute to wake. A pushed message
arrives whether or not anything else in the stack is awake.

Delivery never raises. A notification failure must not fail the analysis run
that produced it: the signal is already journaled, `notified_at` stays NULL,
and the next run picks it up. Losing a message is recoverable; losing the run
that would have placed the trade is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

import httpx

from ..models import PositionSize, RiskDecision, Side, TradeSetup

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
# Telegram hard-rejects anything longer.
MAX_MESSAGE_LENGTH = 4096


@dataclass
class TelegramNotifier:
    bot_token: str | None
    chat_id: str | None
    timeout: float = 15.0

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, text: str, *, markdown: bool = True) -> bool:
        """Send one message. Returns success, never raises."""
        if not self.enabled:
            log.debug("Telegram is not configured; message not sent")
            return False

        if len(text) > MAX_MESSAGE_LENGTH:
            text = text[: MAX_MESSAGE_LENGTH - 20] + "\n...(truncated)"

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if markdown:
            payload["parse_mode"] = "Markdown"

        try:
            response = httpx.post(
                f"{API_BASE}/bot{self.bot_token}/sendMessage",
                json=payload,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            log.warning("Telegram request failed: %s", exc)
            return False

        if response.status_code != 200:
            # Markdown is the most common cause: an unescaped underscore in a
            # symbol or reason makes Telegram reject the whole message. Retry
            # as plain text so the content still arrives.
            log.warning(
                "Telegram returned %s: %s", response.status_code, response.text[:200]
            )
            if markdown:
                log.info("Retrying the message without Markdown")
                return self.send(_strip_markdown(text), markdown=False)
            return False
        return True

    # -- message builders --------------------------------------------------

    def signal_message(
        self,
        *,
        symbol: str,
        tier: str,
        setup: TradeSetup,
        size: PositionSize | None,
        score: float,
        reasons: tuple[str, ...],
        warnings: tuple[str, ...] = (),
        is_paper: bool = True,
        executed: bool = False,
    ) -> str:
        """Format an approved signal.

        Leads with the mode, because the single most important thing to know at
        a glance is whether real money just moved.
        """
        arrow = "LONG" if setup.side is Side.LONG else "SHORT"
        mode = "PAPER" if is_paper else "*LIVE*"
        header = f"{'EXECUTED' if executed else 'SIGNAL'} [{mode}] {arrow} {symbol}"

        lines = [
            header,
            f"_{tier} tier | score {score:+.3f} | {setup.timeframe.value}_",
            "",
            f"Entry  `{_fmt(setup.entry)}`",
            f"Stop   `{_fmt(setup.stop)}`  ({_pct(setup.stop, setup.entry)})",
        ]
        for index, target in enumerate(setup.targets, start=1):
            lines.append(
                f"T{index}     `{_fmt(target)}`  ({_pct(target, setup.entry)}, "
                f"{setup.risk_reward(target):.2f}R)"
            )

        if size is not None and size.viable:
            lines += [
                "",
                f"Size   `{_fmt(size.quantity)}` ({_fmt(size.notional)} USDT @ {size.leverage}x)",
                f"Margin `{_fmt(size.margin)}` USDT | risking `{_fmt(size.risk_amount)}` USDT",
            ]

        if reasons:
            lines += ["", "*Why:*"] + [f"- {reason}" for reason in reasons[:5]]
        if warnings:
            lines += ["", "*Caution:*"] + [f"- {warning}" for warning in warnings[:4]]

        return "\n".join(lines)

    def rejection_message(
        self, *, symbol: str, side: Side, decision: RiskDecision, score: float
    ) -> str:
        """Format a rejected candidate.

        Worth sending: a rejection tells you the bot saw the setup and why it
        declined, which is the difference between a working filter and a bot
        that has silently stopped finding anything.
        """
        lines = [
            f"REJECTED {side.value.upper()} {symbol}",
            f"_score {score:+.3f}_",
            "",
            "*Failed rules:*",
        ]
        lines += [f"- `{veto.rule}`: {veto.detail}" for veto in decision.vetoes[:6]]
        return "\n".join(lines)

    def position_closed_message(
        self,
        *,
        symbol: str,
        side: str,
        exit_price: Decimal,
        pnl: Decimal | None,
        pnl_pct: Decimal | None,
        r_multiple: Decimal | None,
        reason: str,
        is_paper: bool,
    ) -> str:
        won = pnl is not None and pnl > 0
        lines = [
            f"{'WIN' if won else 'LOSS'} CLOSED [{'PAPER' if is_paper else 'LIVE'}] "
            f"{side.upper()} {symbol}",
            "",
            f"Exit   `{_fmt(exit_price)}`  ({reason})",
        ]
        if pnl is not None:
            lines.append(f"P&L    `{_fmt(pnl)}` USDT" + (f" ({pnl_pct:+.2f}%)" if pnl_pct else ""))
        if r_multiple is not None:
            lines.append(f"Result `{r_multiple:+.2f}R`")
        return "\n".join(lines)

    def alert(self, title: str, detail: str) -> bool:
        """An operational alarm, not a trading signal."""
        return self.send(f"ALERT: *{title}*\n\n{detail}")

    def run_summary(
        self,
        *,
        scanned: int,
        candidates: int,
        approved: int,
        rejected: int,
        errors: tuple[str, ...] = (),
    ) -> str:
        lines = [
            "Run complete",
            f"_scanned {scanned} | candidates {candidates} | "
            f"approved {approved} | rejected {rejected}_",
        ]
        if errors:
            lines += ["", "*Errors:*"] + [f"- {error}" for error in errors[:4]]
        return "\n".join(lines)


def _fmt(value: Decimal | float | None) -> str:
    """Format a price without scientific notation or trailing zeros.

    The universe spans BTC at five figures and satellite tokens at four
    decimals, and `str(Decimal)` will happily produce `1E-7`, which is
    unreadable in a trading alert.
    """
    if value is None:
        return "n/a"
    decimal_value = Decimal(str(value))
    text = format(decimal_value.normalize(), "f")
    return text


def _pct(value: Decimal, reference: Decimal) -> str:
    if reference == 0:
        return "n/a"
    return f"{(value - reference) / reference * 100:+.2f}%"


def _strip_markdown(text: str) -> str:
    for character in ("*", "_", "`", "[", "]"):
        text = text.replace(character, "")
    return text
