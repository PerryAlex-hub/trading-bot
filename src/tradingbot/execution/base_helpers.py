"""Shared execution helpers.

Small, pure functions used by both the paper broker and the live engine. Kept
separate so there is exactly one implementation of the idempotency token and
one of the slippage convention - the two things paper and live must agree on
for paper results to mean anything.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from decimal import Decimal

from ..models import OrderType, Side


def client_order_id(
    *,
    symbol: str,
    side: Side,
    bar_time: datetime,
    purpose: str,
    signal_id: int | None = None,
) -> str:
    """Build a deterministic idempotency token.

    This is the keystone of safe cron-driven execution. The same symbol, bar,
    side and purpose always produce the same id, so:

    * a retried GitHub Actions job re-sends an identical intent, which the
      venue rejects as a duplicate rather than opening a second position;
    * the `orders.client_order_id` unique constraint catches it locally even
      before the request goes out.

    `purpose` separates the entry from its protective orders, which must be
    distinct ids on the same bar. Hashed and truncated because venues cap this
    field (commonly 32-40 characters).
    """
    raw = f"{symbol}|{side.value}|{bar_time.isoformat()}|{purpose}|{signal_id or 0}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:20]
    return f"tb{digest}"


def apply_slippage(
    price: Decimal, *, side: Side, slippage_pct: Decimal, is_exit: bool = False
) -> Decimal:
    """Move a price adversely by `slippage_pct`.

    Always against the trader. On entry a long pays up and a short sells down;
    on exit the direction reverses because closing a long means selling into
    the bid. Modelling slippage as favourable in any case - or as zero - is how
    a paper backtest manufactures an edge that does not exist.
    """
    factor = slippage_pct / Decimal("100")
    adverse_up = (side is Side.LONG) != is_exit
    return price * (Decimal("1") + factor) if adverse_up else price * (Decimal("1") - factor)


def protective_order_type(side: Side, *, is_stop: bool) -> OrderType:
    return OrderType.STOP_MARKET if is_stop else OrderType.TAKE_PROFIT_MARKET
