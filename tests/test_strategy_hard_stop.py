from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.core.domain import MarketState, Regime, SymbolState
from src.engine.strategy_engine import StrategyEngine, StrategyRouting


@pytest.fixture
def mock_strategy():
    s = MagicMock()
    s.name = "test_strat"
    s.on_bar.return_value = None
    return s


@pytest.mark.unit
def test_strategy_hard_stop_enforcement():
    # Setup strategy with hard stop at 11:00
    strat = MagicMock()
    strat.name = "momentum"
    # is_past_hard_stop logic normally lives in BaseStrategy, but we mock it here
    strat.is_past_hard_stop = lambda t: t.hour >= 11

    routing = StrategyRouting(strategies_by_regime={Regime.BULL: ["momentum"]})
    engine = StrategyEngine({"momentum": strat}, routing, MagicMock())

    symbol_state = MagicMock(spec=SymbolState)
    symbol_state.allowed_strategies = ["momentum"]
    symbol_state.meta = {}

    # 1. Market at 10:30 (Should be ACTIVE)
    market_tz = ZoneInfo("America/New_York")
    time_1030 = datetime(2023, 1, 3, 10, 30, tzinfo=market_tz)
    market_state = MarketState(time=time_1030, regime=Regime.BULL, regime_snapshot=None)

    active = engine._get_active_strategies(symbol_state, market_state)
    assert "momentum" in active

    # 2. Market at 11:00 (Should be FILTERED OUT)
    time_1100 = datetime(2023, 1, 3, 11, 0, tzinfo=market_tz)
    market_state.time = time_1100

    active = engine._get_active_strategies(symbol_state, market_state)
    assert "momentum" not in active
