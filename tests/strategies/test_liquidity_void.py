from datetime import datetime

import pytest

from src.core.domain import Bar, MarketState, OrderSide, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.liquidity_void import LiquidityVoidStrategy


class MockFeatures:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def logger():
    return StructuredLogger("TestLiquidityVoid")

@pytest.fixture
def strategy(logger):
    config = {
        "volume_spike_multiplier": 3.0,
        "deviation_threshold": 0.02, # 2% threshold
        "stop_atr_multiplier": 1.0,
        "target_atr_multiplier": 2.0,
    }
    return LiquidityVoidStrategy(config, logger)


@pytest.fixture
def sample_bar():
    return Bar(
        symbol="TSLA",
        time=datetime(2025, 3, 3, 10, 0),
        open=200.0,
        high=210.0, # Massive spike up
        low=200.0,
        close=208.0,
        volume=10000000, # Large volume
    )


def test_liquidity_void_fade_short(strategy, sample_bar):
    market_state = MarketState(time=sample_bar.time, regime=None)

    mock_features = MockFeatures(
        avg_volume=2000000, # Actual volume is 5x avg
        distance_from_ema20=0.03 # 3% above EMA20
    )

    symbol_state = SymbolState("TSLA")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 4.0,
    }

    signal = strategy.on_bar("TSLA", sample_bar, symbol_state, market_state)

    # We fade the spike upwards by shorting
    assert signal is not None
    assert signal.side == OrderSide.SELL
    assert signal.stop_price == 208.0 + (1.0 * 4.0)
    assert signal.target_price == 208.0 - (2.0 * 4.0)

def test_liquidity_void_fade_long(strategy):
    sample_bar_down = Bar(
        symbol="TSLA",
        time=datetime(2025, 3, 3, 10, 0),
        open=200.0,
        high=200.0,
        low=190.0, # Massive spike down
        close=192.0,
        volume=10000000,
    )
    market_state = MarketState(time=sample_bar_down.time, regime=None)

    mock_features = MockFeatures(
        avg_volume=2000000, # 5x volume
        distance_from_ema20=-0.04 # 4% below EMA20
    )

    symbol_state = SymbolState("TSLA")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 4.0,
    }

    signal = strategy.on_bar("TSLA", sample_bar_down, symbol_state, market_state)

    # We fade the crash by going long
    assert signal is not None
    assert signal.side == OrderSide.BUY

def test_liquidity_void_insufficient_volume(strategy, sample_bar):
    market_state = MarketState(time=sample_bar.time, regime=None)

    mock_features = MockFeatures(
        avg_volume=4000000, # volume is only 2.5x avg (below 3x threshold)
        distance_from_ema20=0.03
    )

    symbol_state = SymbolState("TSLA")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 4.0,
    }

    signal = strategy.on_bar("TSLA", sample_bar, symbol_state, market_state)
    assert signal is None

def test_liquidity_void_insufficient_deviation(strategy, sample_bar):
    market_state = MarketState(time=sample_bar.time, regime=None)

    mock_features = MockFeatures(
        avg_volume=2000000, # 5x volume
        distance_from_ema20=0.01 # Only 1% above EMA20 (below 2% threshold)
    )

    symbol_state = SymbolState("TSLA")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 4.0,
    }

    signal = strategy.on_bar("TSLA", sample_bar, symbol_state, market_state)
    assert signal is None
