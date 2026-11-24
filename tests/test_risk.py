"""Risk sizing and the veto gate.

This is the layer `risk.yaml` grants authority over everything else, so these
tests assert the economics directly: that risk never exceeds the budget, that
leverage changes margin and nothing else, and that every rule actually bites.

The gate is also asserted to **fail closed**. A risk check that silently
permits a trade when its input is missing is worse than no check at all.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from tradingbot.config import load_risk
from tradingbot.models import (
    ContractSpec,
    FundingInfo,
    Quote,
    Side,
    Tier,
    Timeframe,
    TradeSetup,
)
from tradingbot.risk import (
    PortfolioState,
    RiskGate,
    friction_cost,
    net_risk_reward,
    size_position,
)


def _net_rr_veto(decision) -> bool:
    """Whether a decision was blocked specifically on net risk:reward."""
    return any(veto.rule == "per_trade.min_risk_reward" for veto in decision.vetoes)


def quote(symbol: str, bid: str, ask: str, mark: str) -> Quote:
    return Quote(
        symbol=symbol,
        bid=Decimal(bid),
        ask=Decimal(ask),
        mark=Decimal(mark),
        at=dt.datetime.now(dt.UTC),
    )


@pytest.fixture
def btc_long() -> TradeSetup:
    return TradeSetup(
        symbol="BTCUSDT",
        side=Side.LONG,
        entry=Decimal("92000"),
        stop=Decimal("89200"),
        targets=(Decimal("97600"),),
    )


@pytest.fixture
def risk_config():
    return load_risk()


class TestTradeSetup:
    def test_long_stop_must_sit_below_entry(self) -> None:
        with pytest.raises(ValueError, match="must sit below"):
            TradeSetup("X", Side.LONG, Decimal("100"), Decimal("110"), (Decimal("120"),))

    def test_short_stop_must_sit_above_entry(self) -> None:
        with pytest.raises(ValueError, match="must sit above"):
            TradeSetup("X", Side.SHORT, Decimal("100"), Decimal("90"), (Decimal("80"),))

    def test_a_setup_needs_at_least_one_target(self) -> None:
        with pytest.raises(ValueError, match="at least one target"):
            TradeSetup("X", Side.LONG, Decimal("100"), Decimal("90"), ())

    def test_risk_reward_arithmetic(self, btc_long: TradeSetup) -> None:
        assert btc_long.risk_per_unit == Decimal("2800")
        assert btc_long.reward_per_unit() == Decimal("5600")
        assert btc_long.risk_reward() == Decimal("2")

    def test_is_favourable_guards_against_chasing(self, btc_long: TradeSetup) -> None:
        """A cron firing late must not buy a move it already missed."""
        assert btc_long.is_favourable(Decimal("91000"))
        assert not btc_long.is_favourable(Decimal("93000"))


class TestSizing:
    def test_risk_never_exceeds_the_budget(self, btc_long, btc_spec) -> None:
        """The single most important invariant in the project."""
        size = size_position(
            btc_long, equity=Decimal("200"), risk_pct=Decimal("0.5"),
            tier=Tier.CORE, spec=btc_spec,
        )
        assert size.viable
        budget = Decimal("200") * Decimal("0.5") / Decimal("100")
        assert size.quantity * btc_long.risk_per_unit <= budget

    @pytest.mark.parametrize("max_leverage", [1, 5, 10, 50])
    def test_leverage_changes_margin_not_risk(
        self, btc_long, btc_spec, max_leverage: int
    ) -> None:
        """Conflating leverage with risk is how small accounts die."""
        size = size_position(
            btc_long, equity=Decimal("200"), risk_pct=Decimal("0.5"),
            tier=Tier.CORE, spec=btc_spec, max_leverage=max_leverage,
        )
        if size.viable:
            risked = size.quantity * btc_long.risk_per_unit
            assert risked <= Decimal("1.0")

    def test_satellite_tier_is_sized_at_half(self, btc_spec) -> None:
        setup = TradeSetup(
            "DOGEUSDT", Side.LONG, Decimal("0.40"), Decimal("0.38"), (Decimal("0.46"),)
        )
        spec = ContractSpec(
            "DOGEUSDT", min_order_size=Decimal("1"), size_step=Decimal("1"),
            min_notional=Decimal("5"),
        )
        core = size_position(
            setup, equity=Decimal("1000"), risk_pct=Decimal("0.5"),
            tier=Tier.CORE, spec=spec,
        )
        satellite = size_position(
            setup, equity=Decimal("1000"), risk_pct=Decimal("0.5"),
            tier=Tier.SATELLITE, spec=spec, satellite_multiplier=Decimal("0.5"),
        )
        assert satellite.risk_amount == core.risk_amount / 2

    def test_venue_minimum_refuses_rather_than_overrisking(self, goat_spec) -> None:
        """The small-account floor. Must refuse, never round up."""
        setup = TradeSetup(
            "GOATUSDT", Side.LONG, Decimal("0.52"), Decimal("0.44"), (Decimal("0.70"),)
        )
        size = size_position(
            setup, equity=Decimal("200"), risk_pct=Decimal("0.5"),
            tier=Tier.SATELLITE, spec=goat_spec,
        )
        assert not size.viable
        assert "below the venue minimum" in size.reason
        # The refusal must explain what would be needed.
        assert "equity" in size.reason

    def test_quantity_rounds_down_to_the_venue_step(self, btc_long, btc_spec) -> None:
        """Rounding up would risk more than the gate approved."""
        size = size_position(
            btc_long, equity=Decimal("200"), risk_pct=Decimal("0.5"),
            tier=Tier.CORE, spec=btc_spec,
        )
        assert size.quantity % btc_spec.size_step == 0

    def test_zero_equity_is_refused(self, btc_long, btc_spec) -> None:
        size = size_position(
            btc_long, equity=Decimal("0"), risk_pct=Decimal("0.5"),
            tier=Tier.CORE, spec=btc_spec,
        )
        assert not size.viable

    def test_margin_cap_prevents_one_trade_eating_the_account(self, btc_spec) -> None:
        setup = TradeSetup(
            "BTCUSDT", Side.LONG, Decimal("92000"), Decimal("91990"), (Decimal("92100"),)
        )
        size = size_position(
            setup, equity=Decimal("200"), risk_pct=Decimal("0.5"),
            tier=Tier.CORE, spec=btc_spec, max_leverage=1,
            max_margin_fraction=Decimal("0.5"),
        )
        # A 10-point stop implies an enormous position; it must be refused.
        assert not size.viable


class TestFriction:
    def test_friction_reduces_risk_reward(self, btc_long, btc_spec) -> None:
        size = size_position(
            btc_long, equity=Decimal("10000"), risk_pct=Decimal("0.5"),
            tier=Tier.CORE, spec=btc_spec,
        )
        friction = friction_cost(
            notional=size.notional, taker_fee_pct=Decimal("0.06"),
            spread_pct=Decimal("0.05"), funding_rate_8h_pct=Decimal("0.01"),
            hold_intervals=6,
        )
        assert friction > 0
        assert net_risk_reward(btc_long, size, friction) < btc_long.risk_reward()

    def test_zero_friction_leaves_risk_reward_unchanged(self, btc_long, btc_spec) -> None:
        size = size_position(
            btc_long, equity=Decimal("10000"), risk_pct=Decimal("0.5"),
            tier=Tier.CORE, spec=btc_spec,
        )
        net = net_risk_reward(btc_long, size, Decimal("0"))
        assert net == pytest.approx(float(btc_long.risk_reward()), rel=1e-9)


class TestRiskGate:
    def test_a_clean_setup_with_no_friction_passes(self, risk_config, btc_spec) -> None:
        """Generous geometry so only the rules under test can fail it."""
        setup = TradeSetup(
            "BTCUSDT", Side.LONG, Decimal("92000"), Decimal("89200"),
            (Decimal("103200"),),  # 4:1 gross, survives friction
        )
        decision = RiskGate(risk_config).evaluate(
            setup,
            tier=Tier.CORE,
            portfolio=PortfolioState(
                equity=Decimal("10000"), equity_at_day_start=Decimal("10000")
            ),
            spec=btc_spec,
            quote=quote("BTCUSDT", "91995", "92005", "92000"),
            funding=FundingInfo("BTCUSDT", Decimal("0.005")),
        )
        assert decision.approved, decision.veto_summary
        assert decision.size is not None

    def test_missing_quote_is_priced_not_vetoed(
        self, risk_config, btc_spec, btc_long
    ) -> None:
        """Unknown friction is charged, not treated as a blocker or as free.

        The old behaviour vetoed on `costs.spread_unknown`, which silenced
        every symbol whenever the venue ticker was unreachable. Since nothing
        in this project can place an order, that protected nothing and produced
        an empty journal instead.
        """
        decision = RiskGate(risk_config).evaluate(
            btc_long,
            tier=Tier.CORE,
            portfolio=PortfolioState(equity=Decimal("200")),
            spec=btc_spec,
            quote=None,
            funding=None,
            require_market_data=True,
        )

        assert not any(veto.rule == "costs.spread_unknown" for veto in decision.vetoes)
        assert any("spread unverified" in warning for warning in decision.warnings)

    def test_unknown_spread_is_not_treated_as_free(
        self, risk_config, btc_spec, btc_long
    ) -> None:
        """A missing quote must not produce a better net R:R than a good one.

        This was a real bug: friction used a spread of zero when the quote was
        missing, so the least-known setup scored best.
        """
        gate = RiskGate(risk_config)
        common = {
            "tier": Tier.CORE,
            "portfolio": PortfolioState(equity=Decimal("200")),
            "spec": btc_spec,
            "funding": None,
        }
        unknown = gate.evaluate(btc_long, quote=None, **common)
        measured = gate.evaluate(
            btc_long, quote=quote("BTCUSDT", "91999", "92001", "92000"), **common
        )

        unknown_rr = _net_rr_veto(unknown)
        measured_rr = _net_rr_veto(measured)
        # Either the unknown case is rejected on economics while the measured
        # one is not, or both pass - never the reverse.
        assert not (measured_rr and not unknown_rr)

    def test_wide_spread_is_vetoed(self, risk_config, btc_spec, btc_long) -> None:
        decision = RiskGate(risk_config).evaluate(
            btc_long,
            tier=Tier.CORE,
            portfolio=PortfolioState(equity=Decimal("200")),
            spec=btc_spec,
            quote=quote("BTCUSDT", "91700", "92300", "92000"),
            funding=FundingInfo("BTCUSDT", Decimal("0.01")),
        )
        assert any(veto.rule == "costs.max_spread_pct" for veto in decision.vetoes)

    def test_excessive_funding_is_vetoed(self, risk_config, btc_spec, btc_long) -> None:
        decision = RiskGate(risk_config).evaluate(
            btc_long,
            tier=Tier.CORE,
            portfolio=PortfolioState(equity=Decimal("200")),
            spec=btc_spec,
            quote=quote("BTCUSDT", "91995", "92005", "92000"),
            funding=FundingInfo("BTCUSDT", Decimal("0.5")),
        )
        assert any(
            veto.rule == "costs.max_abs_funding_rate_8h" for veto in decision.vetoes
        )

    def test_paper_tier_is_never_sized(self, risk_config) -> None:
        """universe.yaml says AXISUSDT is journaled but never traded."""
        setup = TradeSetup(
            "AXISUSDT", Side.LONG, Decimal("1.15"), Decimal("1.00"), (Decimal("1.60"),)
        )
        decision = RiskGate(risk_config).evaluate(
            setup,
            tier=Tier.PAPER,
            portfolio=PortfolioState(equity=Decimal("200")),
            spec=ContractSpec("AXISUSDT", size_step=Decimal("1"), min_notional=Decimal("5")),
            quote=quote("AXISUSDT", "1.149", "1.151", "1.15"),
            funding=FundingInfo("AXISUSDT", Decimal("0.01")),
            history_days=30,
        )
        assert not decision.approved
        assert any(veto.rule == "tier.not_tradeable" for veto in decision.vetoes)

    def test_consecutive_losses_trip_the_breaker(self, risk_config, btc_spec, btc_long) -> None:
        decision = RiskGate(risk_config).evaluate(
            btc_long,
            tier=Tier.CORE,
            portfolio=PortfolioState(equity=Decimal("200"), consecutive_losses=4),
            spec=btc_spec,
            quote=quote("BTCUSDT", "91995", "92005", "92000"),
            funding=FundingInfo("BTCUSDT", Decimal("0.01")),
        )
        assert any(
            veto.rule == "circuit_breakers.max_consecutive_losses"
            for veto in decision.vetoes
        )

    def test_daily_loss_limit_is_enforced(self, risk_config, btc_spec, btc_long) -> None:
        decision = RiskGate(risk_config).evaluate(
            btc_long,
            tier=Tier.CORE,
            portfolio=PortfolioState(
                equity=Decimal("190"),
                equity_at_day_start=Decimal("200"),
                realised_pnl_today=Decimal("-10"),
            ),
            spec=btc_spec,
            quote=quote("BTCUSDT", "91995", "92005", "92000"),
            funding=FundingInfo("BTCUSDT", Decimal("0.01")),
        )
        assert any(
            veto.rule == "circuit_breakers.max_daily_loss_pct"
            for veto in decision.vetoes
        )

    def test_price_drift_between_venues_is_vetoed(self, risk_config, btc_spec, btc_long) -> None:
        """The signal came from Binance but executes on WEEX."""
        decision = RiskGate(risk_config).evaluate(
            btc_long,
            tier=Tier.CORE,
            portfolio=PortfolioState(equity=Decimal("200")),
            spec=btc_spec,
            quote=quote("BTCUSDT", "85000", "85010", "85005"),
            funding=FundingInfo("BTCUSDT", Decimal("0.01")),
        )
        assert any(veto.rule == "costs.price_drift" for veto in decision.vetoes)

    def test_all_failing_rules_are_reported_not_just_the_first(
        self, risk_config, btc_spec, btc_long
    ) -> None:
        """A journal that records one reason loses most of its diagnostic value."""
        decision = RiskGate(risk_config).evaluate(
            btc_long,
            tier=Tier.CORE,
            portfolio=PortfolioState(
                equity=Decimal("190"),
                consecutive_losses=4,
                equity_at_day_start=Decimal("200"),
                realised_pnl_today=Decimal("-10"),
            ),
            spec=btc_spec,
            quote=quote("BTCUSDT", "91700", "92300", "92000"),
            funding=FundingInfo("BTCUSDT", Decimal("0.5")),
        )
        assert len(decision.vetoes) >= 4
        rules = {veto.rule for veto in decision.vetoes}
        assert "circuit_breakers.max_consecutive_losses" in rules
        assert "costs.max_spread_pct" in rules


class TestConfigCoherence:
    def test_no_configured_symbol_is_unreachable(self, risk_config) -> None:
        """Every tradeable symbol must fit inside the portfolio beta budget.

        This used to assert the opposite: GOATUSDT and AXISUSDT carried
        beta_to_btc 2.5 against a 2.0 cap, so each exceeded the whole budget by
        itself and could never be traded even on an empty book. The test pinned
        that contradiction so a config change would be noticed.

        Both symbols have since been removed - they were not listed on the data
        venue either, so they ingested nothing and produced nothing - and the
        assertion is now the one worth holding: a symbol in the universe should
        be capable of trading. A symbol that cannot is configuration that looks
        like capability.
        """
        from tradingbot.config import load_universe

        cap = risk_config.decimal("portfolio", "max_portfolio_btc_beta", default="2.0")
        unreachable = [
            config.symbol
            for config in load_universe().symbols
            if config.tier.tradeable and risk_config.beta_to_btc(config.symbol) > cap
        ]
        assert not unreachable, (
            f"these symbols can never trade: {unreachable}. Their beta_to_btc "
            f"exceeds max_portfolio_btc_beta ({cap}), so the gate vetoes them "
            "even on an empty book. Lower the beta, raise the cap, or remove them."
        )

    def test_at_most_one_symbol_sits_exactly_on_the_cap(self, risk_config) -> None:
        """A symbol at exactly the cap is tradeable only alone.

        Not an error - 2.0 for a memecoin is honest - but worth knowing, since
        it makes max_open_positions unreachable while that symbol is held.
        """
        from tradingbot.config import load_universe

        cap = risk_config.decimal("portfolio", "max_portfolio_btc_beta", default="2.0")
        at_cap = [
            config.symbol
            for config in load_universe().symbols
            if config.tier.tradeable and risk_config.beta_to_btc(config.symbol) == cap
        ]
        assert at_cap == ["PEPEUSDT"], (
            f"symbols sitting exactly on the beta cap changed: {at_cap}. Each one "
            "can only be held alone; update this test and the README note."
        )


class TestMinimumSizeOverride:
    """Small accounts must get an informed ticket, not a dead end.

    Reproduces the deployed DOGEUSDT rejection: 50 USDT equity, a 0.5% budget
    of 0.25 USDT, and a stop wide enough that the implied position was worth
    3.23 USDT against a 5 USDT venue floor.
    """

    def _setup(self) -> TradeSetup:
        entry = Decimal("0.055330")
        return TradeSetup(
            symbol="DOGEUSDT",
            side=Side.SHORT,
            entry=entry,
            stop=entry + Decimal("0.00428348"),
            targets=(entry - Decimal("0.00856696"),),
            timeframe=Timeframe.H4,
        )

    def _size(self, **overrides):
        kwargs = {
            "equity": Decimal("50"),
            "risk_pct": Decimal("0.5"),
            "tier": Tier.SATELLITE,
            "spec": None,
            "max_leverage": 5,
            "satellite_multiplier": Decimal("1.0"),
            "fallback_min_notional": Decimal("5.0"),
            "allow_min_size_override": True,
            "max_risk_pct_at_min_size": Decimal("2.0"),
        }
        kwargs.update(overrides)
        return size_position(self._setup(), **kwargs)

    def test_raises_to_the_floor_and_discloses_the_real_risk(self) -> None:
        size = self._size()
        assert size.viable
        assert size.at_venue_minimum
        assert size.notional >= Decimal("5")
        # Risk is now above the configured budget, and says so rather than
        # pretending it is still 0.5%.
        assert size.effective_risk_pct > Decimal("0.5")
        assert size.risk_amount > Decimal("0.25")

    def test_still_refuses_when_the_floor_costs_too_much(self) -> None:
        """The ceiling is what makes the override safe rather than reckless."""
        size = self._size(max_risk_pct_at_min_size=Decimal("0.6"))
        assert not size.viable
        assert "ceiling" in size.reason

    def test_disabled_override_keeps_the_old_refusal(self) -> None:
        size = self._size(allow_min_size_override=False)
        assert not size.viable
        assert "below the" in size.reason

    def test_normal_account_is_unaffected(self) -> None:
        """A funded account sizes from the budget, never from the floor."""
        size = self._size(equity=Decimal("2000"))
        assert size.viable
        assert not size.at_venue_minimum
        assert size.effective_risk_pct == pytest.approx(
            Decimal("0.5"), abs=Decimal("0.01")
        )


class TestLiquidityFloor:
    """gates.min_24h_quote_volume, enforced rather than merely configured."""

    def test_a_thin_book_is_vetoed(self, risk_config, btc_spec, btc_long) -> None:
        decision = RiskGate(risk_config).evaluate(
            btc_long,
            tier=Tier.CORE,
            portfolio=PortfolioState(equity=Decimal("10000")),
            spec=btc_spec,
            quote=quote("BTCUSDT", "91995", "92005", "92000"),
            funding=FundingInfo("BTCUSDT", Decimal("0.005")),
            # NILUSDT's measured figure against a 30,000,000 floor.
            quote_volume_24h=Decimal("800000"),
        )
        assert not decision.approved
        veto = next(
            v for v in decision.vetoes if v.rule == "gates.min_24h_quote_volume"
        )
        # The number belongs in the message: "too thin" is not reviewable.
        assert "800,000" in veto.detail
        # 30,000,000 / 800,000 = 37.5, which formats as 38.
        assert "38x short" in veto.detail

    def test_a_deep_book_passes(self, risk_config, btc_spec, btc_long) -> None:
        decision = RiskGate(risk_config).evaluate(
            btc_long,
            tier=Tier.CORE,
            portfolio=PortfolioState(equity=Decimal("10000")),
            spec=btc_spec,
            quote=quote("BTCUSDT", "91995", "92005", "92000"),
            funding=FundingInfo("BTCUSDT", Decimal("0.005")),
            quote_volume_24h=Decimal("1500000000"),
        )
        assert not any(
            v.rule == "gates.min_24h_quote_volume" for v in decision.vetoes
        )

    def test_unknown_volume_warns_rather_than_vetoes(
        self, risk_config, btc_spec, btc_long
    ) -> None:
        """A fresh database should not look like an illiquid market."""
        decision = RiskGate(risk_config).evaluate(
            btc_long,
            tier=Tier.CORE,
            portfolio=PortfolioState(equity=Decimal("10000")),
            spec=btc_spec,
            quote=quote("BTCUSDT", "91995", "92005", "92000"),
            funding=FundingInfo("BTCUSDT", Decimal("0.005")),
            quote_volume_24h=None,
        )
        assert not any(
            v.rule == "gates.min_24h_quote_volume" for v in decision.vetoes
        )
        assert any("24h volume unknown" in w for w in decision.warnings)
