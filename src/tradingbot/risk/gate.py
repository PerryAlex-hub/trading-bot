"""The risk gate: deterministic veto power over every proposed trade.

`risk.yaml` states this layer's authority plainly: "if a proposed signal
violates any rule here, it is rejected and journaled as rejected, with the
failing rule recorded. No LLM output can override these." This module is that
sentence in code.

Design decisions that follow from it:

**Every rule is checked, not short-circuited.** A proposal that fails four
rules reports all four. Stopping at the first failure would make the journal
far less useful: knowing a trade was rejected for both correlation and funding
tells you something that "rejected: correlation" does not.

**The gate reads live portfolio state from the database, not from memory.** The
bot is a series of short-lived cron jobs, so "how many positions are open" is
not a fact this process can know on its own.

**Unknown is priced, not assumed away.** Missing market data used to be a hard
veto. That silenced the bot completely whenever the WEEX ticker was
unreachable, which is not a safety property in a project with no execution
path - it just produced an empty journal. Unknown friction is now charged at a
deliberately pessimistic rate, so a setup must survive a bad assumed book to be
reported, and the resulting rejection names the economics rather than the
outage. What is never done is assuming unknown costs are zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from ..analysis.regime import BetaEstimate, MarketRegime, effective_beta
from ..config import RiskConfig
from ..db.schema import Position
from ..models import (
    ContractSpec,
    FundingInfo,
    PositionSize,
    Quote,
    RiskDecision,
    RiskVeto,
    Side,
    Tier,
    TradeSetup,
)
from .sizing import friction_cost, net_risk_reward, size_position

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioState:
    """Live account and position state the gate needs.

    Assembled by the caller from the database so the gate itself stays pure
    and therefore testable without a database.
    """

    equity: Decimal
    open_positions: tuple[Position, ...] = ()
    consecutive_losses: int = 0
    realised_pnl_today: Decimal = Decimal("0")
    equity_at_day_start: Decimal | None = None
    halted_until: datetime | None = None
    halt_reason: str | None = None

    @property
    def open_count(self) -> int:
        return len(self.open_positions)

    def count_direction(self, side: Side) -> int:
        return sum(1 for p in self.open_positions if p.side == side.value)

    def has_base_asset(self, base: str) -> bool:
        """Whether any open position shares this base asset.

        Guards against holding BTCUSDT and BTCUSDC as if they were two
        independent bets.
        """
        return any(
            _base_of(position.symbol) == base for position in self.open_positions
        )

    def satellite_count(self, tiers: dict[str, Tier]) -> int:
        return sum(
            1
            for position in self.open_positions
            if tiers.get(position.symbol) is Tier.SATELLITE
        )

    @property
    def daily_loss_pct(self) -> Decimal | None:
        """Today's loss as a percentage of the day's starting equity."""
        start = self.equity_at_day_start
        if start is None or start <= 0:
            return None
        if self.realised_pnl_today >= 0:
            return Decimal("0")
        return abs(self.realised_pnl_today) / start * Decimal("100")


@dataclass
class RiskGate:
    """Applies every rule in `risk.yaml` to one proposed trade."""

    config: RiskConfig

    def evaluate(
        self,
        setup: TradeSetup,
        *,
        tier: Tier,
        portfolio: PortfolioState,
        spec: ContractSpec | None,
        quote: Quote | None,
        funding: FundingInfo | None,
        regime: MarketRegime | None = None,
        beta: BetaEstimate | None = None,
        tiers_by_symbol: dict[str, Tier] | None = None,
        history_days: float | None = None,
        quote_volume_24h: Decimal | None = None,
        require_market_data: bool = True,
    ) -> RiskDecision:
        """Return an approval or every reason for refusal."""
        vetoes: list[RiskVeto] = []
        warnings: list[str] = []
        tiers_by_symbol = tiers_by_symbol or {}

        # --- 0. Tier eligibility -----------------------------------------
        # PAPER is analysed and journaled but never sized, exactly as
        # universe.yaml specifies for AXISUSDT.
        if not tier.tradeable:
            vetoes.append(
                RiskVeto(
                    "tier.not_tradeable",
                    f"{setup.symbol} is {tier.value} tier: signals are journaled, never sized",
                )
            )

        # --- 1. Circuit breakers -----------------------------------------
        vetoes.extend(self._check_circuit_breakers(portfolio))

        # --- 2. Promotion gate -------------------------------------------
        if tier is Tier.PAPER and history_days is not None:
            required = int(self.config.get("gates", "min_history_days", default=90))
            if history_days < required:
                vetoes.append(
                    RiskVeto(
                        "gates.min_history_days",
                        f"{history_days:.0f} days of stored candles, {required} required "
                        "before promotion out of paper tier",
                    )
                )

        # --- 2b. Liquidity floor -----------------------------------------
        # Enforces gates.min_24h_quote_volume, which went unread for the whole
        # project. A thin book does not produce a wrong signal, it produces a
        # correct one that cannot be traded profitably: spread and slippage
        # take the edge before the setup gets a chance to work. So this is a
        # veto on the symbol, not a warning about it.
        minimum_volume = self.config.decimal(
            "gates", "min_24h_quote_volume", default="0"
        )
        if minimum_volume > 0:
            if quote_volume_24h is None:
                warnings.append(
                    "24h volume unknown (no stored candles covering the window), "
                    "so the liquidity floor could not be checked"
                )
            elif quote_volume_24h < minimum_volume:
                shortfall = minimum_volume / quote_volume_24h if quote_volume_24h > 0 else None
                detail = (
                    f"24h quote volume {quote_volume_24h:,.0f} is below the "
                    f"{minimum_volume:,.0f} floor"
                )
                if shortfall is not None:
                    detail += f" ({shortfall:.0f}x short)"
                vetoes.append(RiskVeto("gates.min_24h_quote_volume", detail))

        # --- 3. Portfolio limits -----------------------------------------
        vetoes.extend(
            self._check_portfolio(setup, tier, portfolio, beta, tiers_by_symbol)
        )

        # --- 4. Market-data-dependent costs ------------------------------
        vetoes.extend(
            self._check_costs(setup, tier, quote, funding, require_market_data, warnings)
        )

        # --- 5. Sizing ----------------------------------------------------
        regime_multiplier = (
            Decimal(str(regime.state.size_multiplier)) if regime is not None
            else Decimal("1.0")
        )
        size = size_position(
            setup,
            equity=portfolio.equity,
            risk_pct=self.config.risk_pct,
            tier=tier,
            spec=spec,
            max_leverage=self.config.max_leverage(tier),
            satellite_multiplier=self.config.decimal(
                "per_trade", "satellite_size_multiplier", default="0.5"
            ),
            regime_multiplier=regime_multiplier,
            fallback_min_notional=self.config.decimal(
                "venue", "fallback_min_notional_usdt", default="5.0"
            ),
            allow_min_size_override=bool(
                self.config.get("per_trade", "allow_min_size_override", default=False)
            ),
            max_risk_pct_at_min_size=self.config.decimal(
                "per_trade", "max_risk_pct_at_min_size", default="2.0"
            ),
        )
        if size.viable and size.at_venue_minimum:
            warnings.append(
                f"sized at the venue minimum, so this risks "
                f"{size.effective_risk_pct:.2f}% of equity rather than the "
                f"configured {self.config.risk_pct}%"
            )
        if not size.viable:
            vetoes.append(RiskVeto("sizing.unviable", size.reason))

        # --- 6. Risk:reward after friction -------------------------------
        # Checked last because it needs the sized quantity, and only when
        # sizing succeeded.
        if size.viable:
            vetoes.extend(
                self._check_net_rr(setup, tier, size, quote, funding, warnings)
            )

        if regime is not None and regime.notes:
            warnings.extend(regime.notes)

        return RiskDecision(
            approved=not vetoes,
            vetoes=tuple(vetoes),
            size=size if size.viable else None,
            warnings=tuple(warnings),
        )

    # -- individual rule groups -------------------------------------------

    def _check_circuit_breakers(
        self, portfolio: PortfolioState
    ) -> list[RiskVeto]:
        vetoes: list[RiskVeto] = []

        if portfolio.halted_until is not None:
            now = datetime.now(tz=UTC)
            if portfolio.halted_until > now:
                remaining = portfolio.halted_until - now
                vetoes.append(
                    RiskVeto(
                        "circuit_breakers.active",
                        f"trading halted for another {remaining.total_seconds()/3600:.1f}h "
                        f"({portfolio.halt_reason or 'reason not recorded'})",
                    )
                )

        max_losses = int(
            self.config.get("circuit_breakers", "max_consecutive_losses", default=4)
        )
        if portfolio.consecutive_losses >= max_losses:
            vetoes.append(
                RiskVeto(
                    "circuit_breakers.max_consecutive_losses",
                    f"{portfolio.consecutive_losses} consecutive losses, limit is {max_losses}",
                )
            )

        max_daily = self.config.decimal(
            "circuit_breakers", "max_daily_loss_pct", default="3.0"
        )
        daily_loss = portfolio.daily_loss_pct
        if daily_loss is not None and daily_loss >= max_daily:
            vetoes.append(
                RiskVeto(
                    "circuit_breakers.max_daily_loss_pct",
                    f"down {daily_loss:.2f}% today, limit is {max_daily}%",
                )
            )

        return vetoes

    def _check_portfolio(
        self,
        setup: TradeSetup,
        tier: Tier,
        portfolio: PortfolioState,
        beta: BetaEstimate | None,
        tiers_by_symbol: dict[str, Tier],
    ) -> list[RiskVeto]:
        vetoes: list[RiskVeto] = []
        get = self.config.get

        max_open = int(get("portfolio", "max_open_positions", default=2))
        if portfolio.open_count >= max_open:
            vetoes.append(
                RiskVeto(
                    "portfolio.max_open_positions",
                    f"{portfolio.open_count} positions already open, limit is {max_open}",
                )
            )

        max_same = int(get("portfolio", "max_same_direction", default=2))
        same = portfolio.count_direction(setup.side)
        if same >= max_same:
            vetoes.append(
                RiskVeto(
                    "portfolio.max_same_direction",
                    f"{same} positions already {setup.side.value}, limit is {max_same}",
                )
            )

        max_per_base = int(get("portfolio", "max_per_base_asset", default=1))
        base = _base_of(setup.symbol)
        if max_per_base <= 1 and portfolio.has_base_asset(base):
            vetoes.append(
                RiskVeto(
                    "portfolio.max_per_base_asset",
                    f"already holding a position in {base}",
                )
            )

        if tier is Tier.SATELLITE:
            max_satellite = int(get("portfolio", "max_satellite_positions", default=1))
            held = portfolio.satellite_count(tiers_by_symbol)
            if held >= max_satellite:
                vetoes.append(
                    RiskVeto(
                        "portfolio.max_satellite_positions",
                        f"{held} satellite positions open, limit is {max_satellite}",
                    )
                )

        # Beta-weighted exposure. This is the rule that stops the book becoming
        # one disguised BTC bet, which risk.yaml calls out explicitly: majors
        # correlate 0.8+ in trends and ~0.95 in a crash.
        max_beta = self.config.decimal("portfolio", "max_portfolio_btc_beta", default="2.0")
        configured = float(self.config.beta_to_btc(setup.symbol))
        resolved, basis = effective_beta(beta, configured)

        existing = sum(
            float(self.config.beta_to_btc(position.symbol))
            for position in portfolio.open_positions
            # Only same-direction exposure adds up; a hedge reduces it.
            if position.side == setup.side.value
        )
        total = Decimal(str(existing + resolved))
        if total > max_beta:
            vetoes.append(
                RiskVeto(
                    "portfolio.max_portfolio_btc_beta",
                    f"beta-weighted {setup.side.value} exposure would reach {total:.2f} "
                    f"(existing {existing:.2f} + {setup.symbol} {resolved:.2f} via {basis}), "
                    f"limit is {max_beta}",
                )
            )

        return vetoes

    def _check_costs(
        self,
        setup: TradeSetup,
        tier: Tier,
        quote: Quote | None,
        funding: FundingInfo | None,
        require_market_data: bool,
        warnings: list[str],
    ) -> list[RiskVeto]:
        """Spread and funding checks.

        A missing quote used to be a hard veto here, on the reasoning that
        assuming a tight spread would eventually put an order into a blown-out
        book. That reasoning does not survive contact with this project: there
        is no execution path, so nothing is ever placed into any book. What the
        veto actually did was suppress every signal whenever the WEEX ticket
        endpoint was unreachable, which is what happened on the first live
        deployment - six symbols, six `costs.spread_unknown` rejections, an
        empty journal and nothing to learn from.

        So an unverifiable spread is now a warning, and the cost of not knowing
        is priced in instead: `_check_net_rr` charges a deliberately pessimistic
        assumed spread for the tier. A setup that still clears the R:R floor
        against a bad assumed book is worth reporting; one that does not is
        rejected for being uneconomic, which is a far more useful reason than
        "the ticker was down".
        """
        vetoes: list[RiskVeto] = []

        max_spread = self.config.decimal("costs", "max_spread_pct", default="0.10")
        if quote is None:
            assumed = self.config.assumed_spread_pct(tier)
            warnings.append(
                f"no venue quote available: spread unverified, so friction is "
                f"priced at the assumed {assumed}% for {tier.value} tier. "
                f"Check the book before entering."
            )
            if require_market_data:
                log.warning(
                    "%s: no WEEX quote; proceeding with an assumed %s%% spread",
                    setup.symbol,
                    assumed,
                )
        else:
            spread = quote.spread_pct
            if spread > max_spread:
                vetoes.append(
                    RiskVeto(
                        "costs.max_spread_pct",
                        f"spread {spread:.4f}% exceeds the {max_spread}% limit",
                    )
                )
            # The signal was computed on Binance candles but will be executed on
            # WEEX. A large divergence means the analysis does not describe the
            # book being traded.
            drift = abs(quote.mark - setup.entry) / setup.entry * Decimal("100")
            if drift > Decimal("1.0"):
                vetoes.append(
                    RiskVeto(
                        "costs.price_drift",
                        f"venue mark {quote.mark} is {drift:.2f}% from the planned "
                        f"entry {setup.entry}: the setup is stale or the venues disagree",
                    )
                )
            elif not setup.is_favourable(quote.mark):
                warnings.append(
                    f"mark {quote.mark} has already passed the entry {setup.entry}; "
                    "the limit order may not fill"
                )

        max_funding = self.config.decimal(
            "costs", "max_abs_funding_rate_8h", default="0.05"
        )
        if funding is None:
            if require_market_data:
                warnings.append("funding rate unavailable; carry cost not verified")
        else:
            if abs(funding.rate_8h_pct) > max_funding:
                vetoes.append(
                    RiskVeto(
                        "costs.max_abs_funding_rate_8h",
                        f"funding {funding.rate_8h_pct:+.4f}%/8h exceeds "
                        f"the +/-{max_funding}% limit",
                    )
                )
            # Funding paid by our side of the trade is a real cost; funding
            # received is a tailwind and is not penalised.
            paying = (
                funding.rate_8h_pct > 0 and setup.side is Side.LONG
            ) or (funding.rate_8h_pct < 0 and setup.side is Side.SHORT)
            if paying:
                intervals = int(
                    self.config.get("costs", "assumed_hold_funding_intervals", default=6)
                )
                warnings.append(
                    f"paying {abs(funding.rate_8h_pct * intervals):.3f}% funding "
                    f"over an assumed {intervals}-interval hold"
                )

        return vetoes

    def _check_net_rr(
        self,
        setup: TradeSetup,
        tier: Tier,
        size: PositionSize,
        quote: Quote | None,
        funding: FundingInfo | None,
        warnings: list[str],
    ) -> list[RiskVeto]:
        """Risk:reward after friction, which is the only R:R that matters.

        Missing market data is charged at a pessimistic assumed rate rather
        than at zero. Treating an unknown spread as free was a real bug: it
        made net R:R look best in exactly the situation where the least was
        known, so an unverifiable setup was easier to approve than a verified
        one with a measured 0.08% spread.
        """
        minimum = self.config.min_risk_reward
        gross = setup.risk_reward()

        spread_assumed = quote is None
        spread_pct = (
            quote.spread_pct if quote is not None
            else self.config.assumed_spread_pct(tier)
        )
        # Funding is directional: unknown funding is charged as a cost rather
        # than assumed to be a tailwind.
        funding_pct = (
            funding.rate_8h_pct if funding is not None
            else self.config.decimal("costs", "max_abs_funding_rate_8h", default="0.05")
        )
        friction = friction_cost(
            notional=size.notional,
            taker_fee_pct=self.config.decimal(
                "costs", "assumed_taker_fee_pct", default="0.06"
            ),
            spread_pct=spread_pct,
            funding_rate_8h_pct=funding_pct,
            hold_intervals=int(
                self.config.get("costs", "assumed_hold_funding_intervals", default=6)
            ),
        )
        net = net_risk_reward(setup, size, friction)

        if net < minimum:
            basis = (
                f"assumed {spread_pct}% spread" if spread_assumed
                else f"measured {spread_pct:.4f}% spread"
            )
            return [
                RiskVeto(
                    "per_trade.min_risk_reward",
                    f"net risk:reward {net:.2f} is below the {minimum} floor "
                    f"(gross {gross:.2f}, friction {friction:.4f} USDT on a "
                    f"{size.notional:.2f} USDT position, {basis})",
                )
            ]

        erosion = (gross - net) / gross * Decimal("100") if gross > 0 else Decimal("0")
        if erosion > Decimal("20"):
            warnings.append(
                f"friction consumes {erosion:.0f}% of this setup's edge "
                f"(gross R:R {gross:.2f} -> net {net:.2f})"
            )
        return []


def _base_of(symbol: str) -> str:
    for quote in ("USDT", "USDC", "USD"):
        if symbol.endswith(quote):
            return symbol[: -len(quote)]
    return symbol


def check_circuit_breakers_after_close(
    config: RiskConfig,
    *,
    consecutive_losses: int,
    daily_loss_pct: Decimal | None,
) -> tuple[str, str, float] | None:
    """Decide whether a breaker should trip, called after a position closes.

    Separate from `RiskGate.evaluate` because tripping is an action with a
    persisted side effect, while evaluation is a pure read. Returns
    `(reason, detail, halt_hours)` or None.
    """
    hours = float(config.get("circuit_breakers", "halt_on_breach_hours", default=24))

    max_losses = int(config.get("circuit_breakers", "max_consecutive_losses", default=4))
    if consecutive_losses >= max_losses:
        return (
            "max_consecutive_losses",
            f"{consecutive_losses} losses in a row (limit {max_losses})",
            hours,
        )

    max_daily = config.decimal("circuit_breakers", "max_daily_loss_pct", default="3.0")
    if daily_loss_pct is not None and daily_loss_pct >= max_daily:
        return (
            "max_daily_loss_pct",
            f"down {daily_loss_pct:.2f}% today (limit {max_daily}%)",
            hours,
        )

    return None
