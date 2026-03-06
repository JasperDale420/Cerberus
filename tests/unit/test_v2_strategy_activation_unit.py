from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.config import ConfigLoader
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
from src.engine.strategy_engine import StrategyEngine, StrategyRouting
from src.strategies.config_models import build_activation_policies_from_config


def _make_snapshot(
    *,
    trend: TrendRegime,
    vol: VolRegime = VolRegime.NORMAL,
    risk: RiskRegime = RiskRegime.NEUTRAL,
    session: SessionRegime = SessionRegime.MIDDAY,
    liquidity: LiquidityRegime = LiquidityRegime.GOOD,
) -> MarketRegimeSnapshot:
    return MarketRegimeSnapshot(
        time=MagicMock(),
        index_symbol="SPY",
        vol_symbol="VXX",
        trend=trend,
        vol=vol,
        liquidity=liquidity,
        risk=risk,
        session=session,
        confidence={
            "trend": 1.0,
            "vol": 1.0,
            "liquidity": 1.0,
            "risk": 1.0,
            "session": 1.0,
        },
    )


@pytest.mark.unit
def test_backtest_v2_builds_activation_policies_for_all_v2_strategies() -> None:
    config = ConfigLoader().load_config("config/backtest_v2.yaml")

    policies = build_activation_policies_from_config(config["strategies"])

    assert {
        "trend_rider_pro",
        "mean_reversion_pro",
        "flow_alpha",
        "orb_v2",
        "rsi_bounce",
        "momentum_fade",
    }.issubset(set(policies))


@pytest.mark.unit
def test_trend_rider_pro_uses_multi_axis_activation_instead_of_legacy_bull_mapping() -> None:
    config = ConfigLoader().load_config("config/backtest_v2.yaml")
    policies = build_activation_policies_from_config(config["strategies"])

    engine = StrategyEngine(
        strategies_by_name={"trend_rider_pro": MagicMock()},
        routing=StrategyRouting(
            strategies_by_regime={Regime.BULL: ["trend_rider_pro"]},
            activation_policies=policies,
        ),
        logger=MagicMock(),
    )

    symbol_state = MagicMock()
    symbol_state.allowed_strategies = ["trend_rider_pro"]
    symbol_state.meta = {}

    market_state = MarketState(
        time=MagicMock(),
        regime=Regime.BULL,
        regime_snapshot=_make_snapshot(trend=TrendRegime.FLAT),
    )

    assert engine._get_active_strategies(symbol_state, market_state) == []


@pytest.mark.unit
def test_mean_reversion_pro_uses_multi_axis_activation_instead_of_legacy_chop_mapping() -> None:
    config = ConfigLoader().load_config("config/backtest_v2.yaml")
    policies = build_activation_policies_from_config(config["strategies"])

    engine = StrategyEngine(
        strategies_by_name={"mean_reversion_pro": MagicMock()},
        routing=StrategyRouting(
            strategies_by_regime={Regime.CHOP: ["mean_reversion_pro"]},
            activation_policies=policies,
        ),
        logger=MagicMock(),
    )

    symbol_state = MagicMock()
    symbol_state.allowed_strategies = ["mean_reversion_pro"]
    symbol_state.meta = {}

    market_state = MarketState(
        time=MagicMock(),
        regime=Regime.CHOP,
        regime_snapshot=_make_snapshot(
            trend=TrendRegime.FLAT,
            risk=RiskRegime.RISK_ON,
        ),
    )

    assert engine._get_active_strategies(symbol_state, market_state) == []
