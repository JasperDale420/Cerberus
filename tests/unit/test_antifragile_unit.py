"""Unit tests for antifragile strategy classification (convexity-based regime overrides)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.config.models import ConvexityClass, RiskConfig, StrategyConfig
from src.core.domain import (
    LiquidityRegime,
    MarketRegimeSnapshot,
    MarketState,
    Regime,
    RiskRegime,
    SessionRegime,
    TrendRegime,
    VolRegime,
)
from src.engine.risk import RiskManager

# --- Helpers ---


def _make_config(strategies: dict | None = None) -> dict:
    """Build a minimal config dict suitable for RiskManager."""
    strats = strategies or {}
    return {
        "risk": {
            "max_daily_loss": 1000,
            "max_risk_per_trade": 50,
            "strategies": strats,
        }
    }


def _make_risk_manager(strategies: dict | None = None) -> RiskManager:
    return RiskManager(_make_config(strategies), logger=MagicMock())


def _make_snapshot(
    vol: VolRegime = VolRegime.NORMAL,
    risk: RiskRegime = RiskRegime.NEUTRAL,
    liquidity: LiquidityRegime = LiquidityRegime.GOOD,
    vol_symbol: str | None = "VXX",
) -> MarketRegimeSnapshot:
    return MarketRegimeSnapshot(
        time=datetime.now(timezone.utc),
        index_symbol="SPY",
        vol_symbol=vol_symbol,
        trend=TrendRegime.FLAT,
        vol=vol,
        liquidity=liquidity,
        risk=risk,
        session=SessionRegime.MIDDAY,
    )


def _make_market_state(snapshot: MarketRegimeSnapshot) -> MarketState:
    return MarketState(
        time=datetime.now(timezone.utc),
        regime=Regime.CHOP,
        regime_snapshot=snapshot,
    )


# --- Tests ---


class TestConvexityClassEnum:
    def test_enum_values(self):
        assert ConvexityClass.CONVEX.value == "convex"
        assert ConvexityClass.LINEAR.value == "linear"
        assert ConvexityClass.CONCAVE.value == "concave"

    def test_convexity_class_on_strategy_config(self):
        cfg = StrategyConfig(convexity_class="convex")
        assert cfg.convexity_class == "convex"

    def test_convexity_class_defaults_to_linear(self):
        cfg = StrategyConfig()
        assert cfg.convexity_class == "linear"


class TestConvexStrategyOverrides:
    """Convex strategies should get HIGHER multipliers in stress regimes."""

    def test_convex_higher_in_shock_vol(self):
        rm = _make_risk_manager({"trend_rider_pro": {"convexity_class": "convex"}})
        snapshot = _make_snapshot(vol=VolRegime.SHOCK)
        ms = _make_market_state(snapshot)

        mult = rm._get_regime_multiplier(ms, strategy_name="trend_rider_pro")
        # Convex override for shock vol = 1.50, global default = 0.00
        assert mult >= 1.5, f"Convex strategy should benefit from SHOCK vol, got {mult}"

    def test_convex_higher_in_risk_off(self):
        rm = _make_risk_manager({"trend_rider_pro": {"convexity_class": "convex"}})
        snapshot = _make_snapshot(vol=VolRegime.NORMAL, risk=RiskRegime.RISK_OFF)
        ms = _make_market_state(snapshot)

        mult = rm._get_regime_multiplier(ms, strategy_name="trend_rider_pro")
        # Convex override: vol.normal=1.0 * risk.risk_off=1.20 = 1.20
        # Global default would be: vol.normal=1.0 * risk.risk_off=0.50 = 0.50
        assert mult >= 1.0, f"Convex strategy should not shrink in RISK_OFF, got {mult}"


class TestConcaveStrategyOverrides:
    """Concave strategies should get LOWER multipliers in stress regimes."""

    def test_concave_lower_in_high_vol(self):
        rm = _make_risk_manager({"mean_reversion_pro": {"convexity_class": "concave"}})
        snapshot = _make_snapshot(vol=VolRegime.HIGH)
        ms = _make_market_state(snapshot)

        mult = rm._get_regime_multiplier(ms, strategy_name="mean_reversion_pro")
        # Concave override for high vol = 0.40, global default = 0.60
        assert mult <= 0.40, f"Concave strategy should reduce more in HIGH vol, got {mult}"

    def test_concave_zero_in_shock_vol(self):
        rm = _make_risk_manager({"mean_reversion_pro": {"convexity_class": "concave"}})
        snapshot = _make_snapshot(vol=VolRegime.SHOCK)
        ms = _make_market_state(snapshot)

        mult = rm._get_regime_multiplier(ms, strategy_name="mean_reversion_pro")
        assert mult == 0.0, f"Concave strategy should be zero in SHOCK vol, got {mult}"


class TestLinearStrategyUnchanged:
    """Linear strategies should use global defaults (no override applied)."""

    def test_linear_uses_global_defaults(self):
        rm = _make_risk_manager({"orb": {"convexity_class": "linear"}})
        snapshot = _make_snapshot(vol=VolRegime.SHOCK)
        ms = _make_market_state(snapshot)

        mult_with_name = rm._get_regime_multiplier(ms, strategy_name="orb")
        mult_without_name = rm._get_regime_multiplier(ms, strategy_name=None)
        assert mult_with_name == mult_without_name, (
            f"Linear strategy should match global defaults: {mult_with_name} != {mult_without_name}"
        )

    def test_unknown_strategy_defaults_to_linear(self):
        rm = _make_risk_manager({})
        snapshot = _make_snapshot(vol=VolRegime.SHOCK)
        ms = _make_market_state(snapshot)

        mult_unknown = rm._get_regime_multiplier(ms, strategy_name="nonexistent_strategy")
        mult_none = rm._get_regime_multiplier(ms, strategy_name=None)
        assert mult_unknown == mult_none, f"Unknown strategy should behave as linear: {mult_unknown} != {mult_none}"


class TestAntifragileConfigParsing:
    """Verify antifragile_overrides field on RiskConfig parses correctly."""

    def test_default_overrides_present(self):
        cfg = RiskConfig()
        assert "convex" in cfg.antifragile_overrides
        assert "concave" in cfg.antifragile_overrides
        assert "linear" not in cfg.antifragile_overrides

    def test_convex_defaults_are_correct(self):
        cfg = RiskConfig()
        convex = cfg.antifragile_overrides["convex"]
        assert convex["vol"]["shock"] == 1.50
        assert convex["vol"]["high"] == 1.30
        assert convex["risk"]["risk_off"] == 1.20

    def test_concave_defaults_are_correct(self):
        cfg = RiskConfig()
        concave = cfg.antifragile_overrides["concave"]
        assert concave["vol"]["shock"] == 0.00
        assert concave["vol"]["high"] == 0.40
        assert concave["risk"]["risk_off"] == 0.30
