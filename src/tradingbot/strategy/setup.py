"""Turning a snapshot into a concrete trade proposal.

The confluence engine says *whether* and *which way*. This module decides
*where*: entry, stop and targets. It is deliberately separate, because the
quality of a signal and the quality of its levels are independent problems -
a correct directional call with the stop in the wrong place still loses money.

Three rules, all of which exist because the obvious alternative is worse:

**The stop goes beyond structure, not at it.** `risk.yaml` specifies
`2 x ATR(14)`, widened to 3x for satellites. But a stop at exactly the swing
low is where every other stop sits, which is precisely where price gets wicked.
So the stop is placed at the structural level *plus* an ATR buffer, and the
wider of (ATR rule, structure + buffer) wins.

**Targets come from the chart, not from a ratio.** Taking `entry + 2R` because
the config wants 1:2 is backwards: it assumes the market will deliver a round
multiple. Targets are placed at real obstacles - the next level, the measured
move of a detected pattern - and then the resulting R:R is *checked* against
the config. If the chart does not offer 1:2, there is no trade.

**Entry is limited, not market.** The bot wakes on a cron, so by the time it
acts the trigger bar may be minutes old. A limit order at a defined price
either fills on a pullback or does not fill at all; a market order guarantees
the worst available price on a thin satellite book.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from ..analysis.frame import to_decimal
from ..analysis.snapshot import FeatureSnapshot
from ..models import Side, Tier, TradeSetup

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SetupProposal:
    """A setup, or a recorded reason there is none.

    Carries its own failure reason rather than returning None, because "the
    chart did not offer 1:2" is one of the most informative things the journal
    can record. It is the difference between a strategy that finds no trades
    and one that finds trades it should not take.
    """

    setup: TradeSetup | None
    reason: str
    entry_basis: str = ""
    stop_basis: str = ""
    target_basis: str = ""

    @property
    def ok(self) -> bool:
        return self.setup is not None

    @classmethod
    def rejected(cls, reason: str) -> SetupProposal:
        return cls(setup=None, reason=reason)


def build_setup(
    snapshot: FeatureSnapshot,
    *,
    atr_multiple: Decimal = Decimal("2.0"),
    satellite_atr_multiple: Decimal = Decimal("3.0"),
    min_risk_reward: Decimal = Decimal("2.0"),
    structure_buffer_atr: Decimal = Decimal("0.25"),
    entry_pullback_atr: Decimal = Decimal("0.15"),
) -> SetupProposal:
    """Construct entry, stop and targets from an analysed snapshot."""
    side = snapshot.side
    if side is Side.FLAT:
        return SetupProposal.rejected("no direction: confluence score below threshold")
    if snapshot.confluence.is_blocked:
        return SetupProposal.rejected(
            "blocked by hard rule: " + "; ".join(snapshot.confluence.blockers)
        )

    atr_value = to_decimal(snapshot.indicators.atr_14)
    if atr_value is None or atr_value <= 0:
        return SetupProposal.rejected("ATR unavailable, so no stop distance can be derived")

    price = to_decimal(snapshot.price)
    if price is None or price <= 0:
        return SetupProposal.rejected("invalid price in snapshot")

    is_satellite = snapshot.tier in (Tier.SATELLITE, Tier.PAPER)
    multiple = satellite_atr_multiple if is_satellite else atr_multiple

    entry, entry_basis = _choose_entry(snapshot, price, atr_value, side, entry_pullback_atr)
    stop, stop_basis = _choose_stop(
        snapshot, entry, atr_value, side, multiple, structure_buffer_atr
    )

    # Guard the invariant TradeSetup enforces, but with a readable message:
    # a raised exception from the dataclass would lose the context.
    if side is Side.LONG and stop >= entry:
        return SetupProposal.rejected(
            f"long stop {stop} is not below entry {entry}; structure is too close"
        )
    if side is Side.SHORT and stop <= entry:
        return SetupProposal.rejected(
            f"short stop {stop} is not above entry {entry}; structure is too close"
        )

    targets, target_basis = _choose_targets(snapshot, entry, stop, side, atr_value)
    if not targets:
        return SetupProposal.rejected(
            "no target found: nothing on the chart offers a worthwhile move"
        )

    setup = TradeSetup(
        symbol=snapshot.symbol,
        side=side,
        entry=entry,
        stop=stop,
        targets=tuple(targets),
        timeframe=snapshot.timeframe,
    )

    achieved = setup.risk_reward()
    if achieved < min_risk_reward:
        # This is a rejection worth journaling in detail: the directional call
        # may well have been right, and knowing how often geometry (not
        # analysis) is the binding constraint is what tells you whether the
        # R:R floor is set sensibly.
        return SetupProposal(
            setup=None,
            reason=(
                f"risk:reward {achieved:.2f} is below the {min_risk_reward} floor "
                f"(risk {setup.risk_per_unit}, reward {setup.reward_per_unit()})"
            ),
            entry_basis=entry_basis,
            stop_basis=stop_basis,
            target_basis=target_basis,
        )

    return SetupProposal(
        setup=setup,
        reason=f"risk:reward {achieved:.2f} to first target",
        entry_basis=entry_basis,
        stop_basis=stop_basis,
        target_basis=target_basis,
    )


def _choose_entry(
    snapshot: FeatureSnapshot,
    price: Decimal,
    atr_value: Decimal,
    side: Side,
    pullback_atr: Decimal,
) -> tuple[Decimal, str]:
    """Pick a limit entry slightly better than the current price.

    Asking for a small pullback costs some fills but improves every one it
    gets, and it protects against the cron-delay problem: if price has already
    run, the limit simply does not fill, which is the correct outcome.
    """
    offset = atr_value * pullback_atr
    if side is Side.LONG:
        entry = price - offset
        basis = f"limit {pullback_atr} ATR below {price} to avoid chasing"
        # If support sits between the pullback entry and price, enter at the
        # level instead: that is where the bid actually is.
        support = snapshot.levels.nearest_support(float(price))
        if support is not None:
            level = to_decimal(support.price)
            if level is not None and entry < level < price:
                return level, f"at {support.touches}-touch support {level}"
        return entry, basis

    entry = price + offset
    resistance = snapshot.levels.nearest_resistance(float(price))
    if resistance is not None:
        level = to_decimal(resistance.price)
        if level is not None and price < level < entry:
            return level, f"at {resistance.touches}-touch resistance {level}"
    return entry, f"limit {pullback_atr} ATR above {price} to avoid chasing"


def _choose_stop(
    snapshot: FeatureSnapshot,
    entry: Decimal,
    atr_value: Decimal,
    side: Side,
    multiple: Decimal,
    buffer_atr: Decimal,
) -> tuple[Decimal, str]:
    """Place the stop beyond both the ATR distance and recent structure.

    Takes the WIDER of the two. A stop that is merely ATR-based can sit just
    inside an obvious swing low, which is the single most reliably hunted price
    on any chart. The buffer past structure is what keeps it out of the wick.
    """
    atr_stop = (
        entry - atr_value * multiple if side is Side.LONG
        else entry + atr_value * multiple
    )
    basis = f"{multiple} x ATR from entry"

    buffer = atr_value * buffer_atr
    structural: Decimal | None = None

    if side is Side.LONG:
        swing = snapshot.structure.last_low
        if swing is not None:
            candidate = to_decimal(swing.price)
            if candidate is not None:
                structural = candidate - buffer
        support = snapshot.levels.nearest_support(float(entry))
        if support is not None:
            candidate = to_decimal(support.price)
            if candidate is not None:
                below = candidate - buffer
                structural = below if structural is None else min(structural, below)
        if structural is not None and structural < atr_stop:
            return structural, (
                f"{buffer_atr} ATR below structure (wider than {multiple}x ATR, "
                "so the stop sits outside the wick zone)"
            )
        return atr_stop, basis

    swing = snapshot.structure.last_high
    if swing is not None:
        candidate = to_decimal(swing.price)
        if candidate is not None:
            structural = candidate + buffer
    resistance = snapshot.levels.nearest_resistance(float(entry))
    if resistance is not None:
        candidate = to_decimal(resistance.price)
        if candidate is not None:
            above = candidate + buffer
            structural = above if structural is None else max(structural, above)
    if structural is not None and structural > atr_stop:
        return structural, (
            f"{buffer_atr} ATR above structure (wider than {multiple}x ATR, "
            "so the stop sits outside the wick zone)"
        )
    return atr_stop, basis


def _choose_targets(
    snapshot: FeatureSnapshot,
    entry: Decimal,
    stop: Decimal,
    side: Side,
    atr_value: Decimal,
) -> tuple[list[Decimal], str]:
    """Place targets at real obstacles, in order, nearest first.

    Candidates, in order of preference:
      1. The next opposing level (where price will actually stall).
      2. A detected pattern's measured move.
      3. The volume-profile POC, which attracts price.
      4. An ATR-projected fallback, only if the chart offers nothing.
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return [], "zero risk distance"

    candidates: list[tuple[Decimal, str]] = []

    if side is Side.LONG:
        for level in snapshot.levels.resistances:
            value = to_decimal(level.price)
            if value is not None and value > entry:
                candidates.append((value, f"{level.touches}-touch resistance"))
    else:
        for level in snapshot.levels.supports:
            value = to_decimal(level.price)
            if value is not None and value < entry:
                candidates.append((value, f"{level.touches}-touch support"))

    for pattern in snapshot.chart_patterns:
        if pattern.target is None:
            continue
        value = to_decimal(pattern.target)
        if value is None:
            continue
        correct_way = (value > entry) if side is Side.LONG else (value < entry)
        if correct_way and pattern.direction.sign == side.sign:
            candidates.append((value, f"{pattern.name} measured move"))

    if snapshot.levels.profile is not None:
        poc = to_decimal(snapshot.levels.profile.poc)
        if poc is not None:
            correct_way = (poc > entry) if side is Side.LONG else (poc < entry)
            if correct_way:
                candidates.append((poc, "volume profile POC"))

    if not candidates:
        # Nothing structural ahead. Project 2.5R and say so plainly, rather
        # than pretending a level exists.
        projected = (
            entry + risk * Decimal("2.5") if side is Side.LONG
            else entry - risk * Decimal("2.5")
        )
        return [projected], "2.5R projection (no structural target on the chart)"

    # Nearest obstacle first: that is what price must clear, and it is where a
    # partial exit belongs.
    candidates.sort(key=lambda item: abs(item[0] - entry))

    # Discard targets so close that fees and spread would consume the move.
    minimum = atr_value * Decimal("0.5")
    usable = [item for item in candidates if abs(item[0] - entry) >= minimum]
    if not usable:
        usable = candidates[-1:]

    targets = [usable[0][0]]
    basis = usable[0][1]
    for value, label in usable[1:]:
        if abs(value - entry) > abs(targets[-1] - entry) * Decimal("1.3"):
            targets.append(value)
            basis += f", then {label}"
        if len(targets) >= 2:
            break

    return targets, basis
