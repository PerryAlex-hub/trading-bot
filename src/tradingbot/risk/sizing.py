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

**Small accounts hit a floor the math does not know about.** With $50 equity at
0.5% risk the budget is $0.25, and a 3x ATR stop on a volatile satellite puts
the implied position at around $3 - below the venue minimum. Refusing outright
is defensible but produces a bot that emits nothing and therefore learns
nothing, which is how this module originally behaved.

So when `allow_min_size_override` is set, the position is raised to the venue
minimum and the REAL risk is reported on the ticket, as long as it stays under
`max_risk_pct_at_min_size`. Above that it is still refused. The principle is
informed consent: never silently risk more than budgeted, but do not pretend a
tradeable setup is untradeable either.
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
    allow_min_size_override: bool = False,
    max_risk_pct_at_min_size: Decimal = Decimal("2.0"),
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
    budget = equity * risk_pct / Decimal("100")
    if tier is Tier.SATELLITE:
        budget *= satellite_multiplier
    budget *= regime_multiplier

    if budget <= 0:
        return PositionSize.unviable(
            setup.symbol,
            f"risk budget reduced to {budget} by tier/regime multipliers",
        )

    quantity = budget / risk_per_unit
    if spec is not None:
        quantity = spec.round_quantity(quantity)
        if quantity <= 0 and not allow_min_size_override:
            return PositionSize.unviable(
                setup.symbol,
                f"risk budget {budget:.4f} USDT buys "
                f"{(budget / risk_per_unit):.10f} units, which rounds to zero at "
                f"the venue step of {spec.size_step}. Either equity is too small "
                f"or the {risk_per_unit} stop distance is too wide.",
            )

    notional = quantity * setup.entry
    at_minimum = False

    # --- venue minimum ---------------------------------------------------
    # Whichever floor applies: the exchange-reported one, or the configured
    # fallback when WEEX could not be reached. The fallback is clearly labelled
    # as an assumption wherever it is reported.
    minimum, minimum_source = _binding_minimum(spec, setup, fallback_min_notional)

    if minimum is not None and notional < minimum:
        required_quantity = minimum / setup.entry
        if spec is not None:
            required_quantity = spec.round_quantity_up(required_quantity)

        risk_at_minimum = required_quantity * risk_per_unit
        risk_pct_at_minimum = risk_at_minimum / equity * Decimal("100")

        if not allow_min_size_override:
            needed_equity = risk_at_minimum / (risk_pct / Decimal("100"))
            return PositionSize.unviable(
                setup.symbol,
                f"position value {notional:.4f} USDT is below the {minimum_source} "
                f"{minimum:.4f}. Clearing it would risk {risk_at_minimum:.4f} USDT "
                f"instead of the budgeted {budget:.4f} "
                f"({_ratio(risk_at_minimum, budget)} over budget). This trade "
                f"needs about {needed_equity:.0f} USDT of equity at {risk_pct}% risk.",
            )

        if risk_pct_at_minimum > max_risk_pct_at_min_size:
            return PositionSize.unviable(
                setup.symbol,
                f"the smallest tradeable position ({minimum:.4f} USDT, the "
                f"{minimum_source}) would risk {risk_at_minimum:.4f} USDT, which is "
                f"{risk_pct_at_minimum:.2f}% of {equity} equity and above the "
                f"{max_risk_pct_at_min_size}% ceiling for minimum-size trades. "
                f"The {risk_per_unit} stop is too wide for this account size.",
            )

        quantity = required_quantity
        notional = quantity * setup.entry
        budget = risk_at_minimum
        at_minimum = True

    if quantity <= 0:
        return PositionSize.unviable(
            setup.symbol, "sizing resolved to zero units after venue rounding"
        )

    # --- leverage --------------------------------------------------------
    # Only as much leverage as the position actually requires, capped. Using
    # the maximum permitted would needlessly raise liquidation risk for no
    # change in the amount at risk.
    leverage_cap = Decimal(max_leverage)
    if spec is not None and spec.max_leverage is not None:
        leverage_cap = min(leverage_cap, Decimal(spec.max_leverage))

    required_leverage = notional / equity
    leverage = max(Decimal("1"), min(leverage_cap, _ceil_decimal(required_leverage)))
    margin = notional / leverage

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

    effective_risk_pct = budget / equity * Decimal("100")
    if at_minimum:
        reason = (
            f"raised to the {minimum_source} of {minimum:.4f} USDT: "
            f"{quantity} units risking {budget:.4f} USDT "
            f"({effective_risk_pct:.2f}% of equity, above the configured "
            f"{risk_pct}%) at {leverage}x, margin {margin:.4f}"
        )
    else:
        reason = (
            f"risking {budget:.4f} USDT over a {risk_per_unit} stop = "
            f"{quantity} units ({notional:.4f} USDT at {leverage}x, "
            f"margin {margin:.4f})"
        )

    return PositionSize(
        symbol=setup.symbol,
        quantity=quantity,
        notional=notional,
        margin=margin,
        leverage=leverage,
        risk_amount=budget,
        viable=True,
        reason=reason,
        at_venue_minimum=at_minimum,
        effective_risk_pct=effective_risk_pct,
    )


def _binding_minimum(
    spec: ContractSpec | None,
    setup: TradeSetup,
    fallback_min_notional: Decimal | None,
) -> tuple[Decimal | None, str]:
    """The minimum position value that actually applies, and where it came from.

    A missing spec is normal - WEEX may be unreachable or its cache empty - but
    with no floor at all, sizing would report a $0.80 position as viable, which
    no exchange would accept. Observed on a real DOGEUSDT signal.
    """
    if spec is not None:
        reported = spec.effective_min_notional(setup.entry)
        if reported is not None:
            return reported, "venue minimum"
    if fallback_min_notional is not None:
        return fallback_min_notional, "assumed venue minimum (no live contract spec)"
    return None, ""


def _ratio(actual: Decimal, budgeted: Decimal) -> str:
    if budgeted <= 0:
        return "an unbounded multiple"
    return f"{(actual / budgeted):.1f}x"


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
