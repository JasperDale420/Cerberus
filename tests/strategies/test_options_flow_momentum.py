from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from src.core.domain import Bar, MarketState, OrderSide, Regime, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.options_flow_momentum import OptionsFlowMomentumStrategy


@pytest.fixture
def logger():
    return StructuredLogger("TestLogger")


@pytest.fixture
def strategy(logger):
    return OptionsFlowMomentumStrategy({}, logger)


@pytest.fixture
def bar():
    return Bar(
        symbol="SPY",
        time=datetime(2023, 1, 1, 10, 0, tzinfo=UTC),
        open=400.0,
        high=402.0,
        low=398.0,
        close=400.0,
        volume=1000,
    )


@pytest.fixture
def market_state():
    return MarketState(
        time=datetime(2023, 1, 1, 10, 0, tzinfo=UTC),
        index_price=400.0,
        regime=Regime.BULL,
        meta={},
    )


def create_symbol_state(features: Mock):
    from collections import deque

    return SymbolState(
        symbol="SPY",
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["options_flow_momentum"],
        meta={"features": features},
    )


def test_options_flow_no_signal_low_zscore(strategy, bar, market_state):
    features = Mock()
    features.symbol = "SPY"
    features.price = 400.0
    features.flow_zscore = 1.5
    features.dof_score = 0.8
    features.flow_bias = 0.4
    features.call_put_ratio = 1.5
    signal = strategy.on_bar("SPY", bar, create_symbol_state(features), market_state)
    assert signal is None


def test_options_flow_no_signal_low_dof(strategy, bar, market_state):
    features = Mock()
    features.symbol = "SPY"
    features.price = 400.0
    features.flow_zscore = 3.0
    features.dof_score = 0.5
    features.flow_bias = 0.4
    features.call_put_ratio = 1.5
    signal = strategy.on_bar("SPY", bar, create_symbol_state(features), market_state)
    assert signal is None


def test_options_flow_long_signal(strategy, bar, market_state):
    features = Mock()
    features.symbol = "SPY"
    features.price = 400.0
    features.flow_zscore = 2.5
    features.dof_score = 0.8
    features.flow_bias = 0.5
    features.call_put_ratio = 1.5
    signal = strategy.on_bar("SPY", bar, create_symbol_state(features), market_state)
    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.target_price == bar.close * 1.03
    assert signal.stop_price == bar.close * 0.985
    assert "dof_score" in signal.meta
    assert signal.size_hint > 0.5


def test_options_flow_short_signal(strategy, bar, market_state):
    features = Mock()
    features.symbol = "SPY"
    features.price = 400.0
    features.flow_zscore = 3.1
    features.dof_score = 0.9
    features.flow_bias = -0.4
    features.call_put_ratio = 0.5
    signal = strategy.on_bar("SPY", bar, create_symbol_state(features), market_state)
    assert signal is not None
    assert signal.side == OrderSide.SELL
    assert signal.target_price == bar.close * 0.97
    assert signal.stop_price == bar.close * 1.015
    assert signal.size_hint > 0.5


def test_options_flow_no_signal_mixed_direction(strategy, bar, market_state):
    features = Mock()
    features.symbol = "SPY"
    features.price = 400.0
    features.flow_zscore = 2.5
    features.dof_score = 0.8
    features.flow_bias = 0.5
    features.call_put_ratio = 0.5
    signal = strategy.on_bar("SPY", bar, create_symbol_state(features), market_state)
    assert signal is None
