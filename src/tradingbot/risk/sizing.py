"""Position sizing.

All Decimal, all the way through. This module produces the number that becomes
an order quantity, and float arithmetic here would mean an order rejected for
precision or a position marginally larger than the risk budget approved.

**The sizing identity.** Risk is the distance to the stop, not the position
value:

    quantity = (equity x risk_pct / 100) / |entry - stop|

Leverage does not appear. It changes the *margin* required to hold the
position, never the amount at risk. Conflating the two is the classic way a
small account dies: "10x leverage" is not a risk setting.

**Small accounts hit a floor the math does not know about.** `risk.yaml`
assumes $200 equity at 0.5% risk, which is $1.00 of risk per trade. With a 3x
ATR stop on a volatile satellite, the resulting position can be worth less than
the exchange minimum. That is a legitimate, expected outcome and it is reported
as `PositionSize.unviable` with the arithmetic shown, never silently rounded up
to something that risks more than the budget.
"""

from __future__ import annotations

import logging
from decimal import ROUND_DOWN, Decimal

from ..models import ContractSpec, PositionSize, Tier, TradeSetup

log = logging.getLogger(__name__)


def size_position(
    setup: TradeSetup,
    *,
    equity: Decimal,
    risk_pct: Decimal,
    tier: Tier,
    spec: ContractSpec | None = None,
    max_leverage: int = 10,
    satellite_multiplier: Decimal = Decimal("0.5"),
    regime_multiplier: Decimal = Decimal("1.0"),
    max_margin_fraction: Decimal = Decimal("0.5"),
    fallback_min_notional: Decimal | None = None,
) -> PositionSize:
    """Size a position from the risk budget, then check it against reality."""
    if equity <= 0:
        return PositionSize.unviable(setup.symbol, f"equity is {equity}")
    if risk_pct <= 0:
        return PositionSize.unviable(setup.symbol, f"risk_pct is {risk_pct}")

    risk_per_unit = setup.risk_per_unit
    if risk_per_unit <= 0:
        return PositionSize.unviable(setup.symbol, "stop distance is zero")

    # --- the risk budget -------------------------------------------------
    risk_amount = equity * risk_pct / Decimal("100")
    if tier is Tier.SATELLITE:
        risk_amount *= satellite_multiplier
    risk_amount *= regime_multiplier

    if risk_amount <= 0:
        return PositionSize.unviable(
            setup.symbol,
            f"risk budget reduced to {risk_amount} by tier/regime multipliers",
        )

    raw_quantity = risk_amount / risk_per_unit
    notional = raw_quantity * setup.entry

    # --- leverage --------------------------------------------------------
    # Only as much leverage as the position actually requires, capped. Using
    # the maximum permitted would needlessly raise liquidation risk for no
    # change in the amount at risk.
    leverage_cap = Decimal(max_leverage)
    if spec is not None and spec.max_leverage is not None:
        leverage_cap = min(leverage_cap, Decimal(spec.max_leverage))

    required = notional / equity if equity > 0 else leverage_cap
    leverage = max(Decimal("1"), min(leverage_cap, _ceil_decimal(required)))
    margin = notional / leverage

    # --- venue constraints -----------------------------------------------
    quantity = raw_quantity
    if spec is not None:
        quantity = spec.round_quantity(raw_quantity)
        if quantity <= 0:
            step = spec.size_step
            return PositionSize.unviable(
                setup.symbol,
                f"risk budget {risk_amount:.4f} USDT buys {raw_quantity:.10f} units, "
                f"which rounds to zero at the venue step of {step}. "
                f"Either equity is too small or the {risk_per_unit} stop distance is too wide.",
            )

        notional = quantity * setup.entry
        margin = notional / leverage

        minimum = spec.effective_min_notional(setup.entry)
        if minimum is not None and notional < minimum:
            shortfall_risk = minimum / setup.entry * risk_per_unit
            needed_equity = shortfall_risk / (risk_pct / Decimal("100"))
            return PositionSize.unviable(
                setup.symbol,
                f"position value {notional:.4f} USDT is below the venue minimum "
                f"{minimum:.4f}. Clearing it would risk {shortfall_risk:.4f} USDT "
                f"instead of the budgeted {risk_amount:.4f} "
                f"({(shortfall_risk/risk_amount):.1f}x over budget). "
                f"This trade needs about {needed_equity:.0f} USDT of equity at "
                f"{risk_pct}% risk.",
            )

    # --- floor when the venue spec is unavailable ------------------------
    # A missing spec is normal: WEEX may be unreachable or its cache empty. But
    # with no spec there is no minimum to check, and sizing would otherwise
    # report a $0.80 position as viable. Observed on a real DOGEUSDT signal.
    if spec is None and fallback_min_notional is not None and notional < fallback_min_notional:
        return PositionSize.unviable(
            setup.symbol,
            f"position value {notional:.4f} USDT is below the assumed venue "
            f"minimum {fallback_min_notional} (no live contract spec available, "
            f"so this is a configured floor, not an exchange-reported one). "
            f"Risk budget {risk_amount:.4f} USDT over a {risk_per_unit} stop "
            "cannot buy a tradeable position.",
        )

    # --- margin sanity ---------------------------------------------------
    # A single trade must not consume the account even when its risk is small:
    # margin that large leaves no room for the second concurrent position
    # `risk.yaml` permits, and liquidation would arrive before the stop.
    if margin > equity * max_margin_fraction:
        return PositionSize.unviable(
            setup.symbol,
            f"margin {margin:.4f} USDT exceeds {max_margin_fraction*100:.0f}% of "
            f"{equity} equity at {leverage}x. Risk is within budget but the "
            "position is too large to hold safely alongside another.",
        )

    return PositionSize(
        symbol=setup.symbol,
        quantity=quantity,
        notional=notional,
        margin=margin,
        leverage=leverage,
        risk_amount=risk_amount,
        viable=True,
        reason=(
            f"risking {risk_amount:.4f} USDT over a {risk_per_unit} stop = "
            f"{quantity} units ({notional:.4f} USDT at {leverage}x, "
            f"margin {margin:.4f})"
        ),
    )


def friction_cost(
    *,
    notional: Decimal,
    taker_fee_pct: Decimal,
    spread_pct: Decimal,
    funding_rate_8h_pct: Decimal,
    hold_intervals: int,
) -> Decimal:
    """Total expected round-trip friction in quote currency.

    Fees are doubled for entry and exit, spread is paid once (the limit entry
    avoids one side of it), and funding accrues over the expected hold. On a
    small account with a 1:2 target, friction is not a rounding error: it is
    routinely 10-20% of the expected reward.
    """
    fees = notional * taker_fee_pct / Decimal("100") * Decimal("2")
    spread = notional * spread_pct / Decimal("100")
    funding = notional * abs(funding_rate_8h_pct) / Decimal("100") * Decimal(hold_intervals)
    return fees + spread + funding


def net_risk_reward(
    setup: TradeSetup, size: PositionSize, friction: Decimal
) -> Decimal:
    """Risk:reward after friction.

    The number that should be compared against the `min_risk_reward` floor. As
    the README's own risk config says: a 1:2 setup that loses a third of its
    edge to spread and funding is not a 1:2 setup.
    """
    reward = setup.reward_per_unit() * size.quantity - friction
    risk = setup.risk_per_unit * size.quantity + friction
    if risk <= 0:
        return Decimal("0")
    return reward / risk


def _ceil_decimal(value: Decimal) -> Decimal:
    """Round up to the next whole number.

    Leverage must be an integer at most venues, and rounding down would make
    the required margin unavailable.
    """
    integral = value.quantize(Decimal("1"), rounding=ROUND_DOWN)
    return integral if integral == value else integral + 1
