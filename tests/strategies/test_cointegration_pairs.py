from datetime import datetime

import pytest

from src.core.domain import Bar, MarketState, OrderSide, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.cointegration_pairs import CointegrationPairsStrategy


class MockFeatures:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def logger():
    return StructuredLogger("TestCointegrationPairs")

@pytest.fixture
def strategy(logger):
    config = {
        "entry_z_threshold": 2.0,
        "stop_z_threshold": 3.5,
    }
    return CointegrationPairsStrategy(config, logger)


@pytest.fixture
def sample_bar():
    return Bar(
        symbol="XOM",
        time=datetime(2025, 4, 4, 10, 0),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000000,
    )


def test_cointegration_short_outperformer(strategy, sample_bar):
    market_state = MarketState(time=sample_bar.time, regime=None)

    # Positive cluster_residual > 2.0 means it's outperforming
    mock_features = MockFeatures(cluster_residual=2.5)

    symbol_state = SymbolState("XOM")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 2.0,
    }

    signal = strategy.on_bar("XOM", sample_bar, symbol_state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.SELL
    # stop_dist = max((3.5 - 2.0) * (2.0/2), 2.0) = max(1.5, 2.0) = 2.0
    assert signal.stop_price == 100.5 + 2.0
    # target_dist = 2.5 * (2.0/2) = 2.5
    assert signal.target_price == 100.5 - 2.5

def test_cointegration_long_underperformer(strategy, sample_bar):
    market_state = MarketState(time=sample_bar.time, regime=None)

    # Negative cluster_residual < -2.0 means it's underperforming
    mock_features = MockFeatures(cluster_residual=-2.5)

    symbol_state = SymbolState("XOM")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 2.0,
    }

    signal = strategy.on_bar("XOM", sample_bar, symbol_state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.stop_price == 100.5 - 2.0
    assert signal.target_price == 100.5 + 2.5

def test_cointegration_insufficient_deviation(strategy, sample_bar):
    market_state = MarketState(time=sample_bar.time, regime=None)

    mock_features = MockFeatures(cluster_residual=1.5) # Below 2.0

    symbol_state = SymbolState("XOM")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 2.0,
    }

    signal = strategy.on_bar("XOM", sample_bar, symbol_state, market_state)
    assert signal is None

    # Negative case below threshold
    symbol_state.meta["features"] = MockFeatures(cluster_residual=-1.9)
    signal = strategy.on_bar("XOM", sample_bar, symbol_state, market_state)
    assert signal is None
