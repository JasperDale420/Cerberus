from datetime import datetime

import pytest

from src.core.domain import Bar, MarketState, OrderSide, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.pead_drift import PEADStrategy


class MockFeatures:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def logger():
    return StructuredLogger("TestPEAD")


@pytest.fixture
def strategy(logger):
    config = {
        "surprise_threshold": 0.15,
        "min_days_post_earnings": 1,
        "max_days_post_earnings": 5,
        "stop_atr_multiplier": 2.0,
        "target_atr_multiplier": 4.0,
    }
    return PEADStrategy(config, logger)


@pytest.fixture
def sample_bar():
    return Bar(
        symbol="META",
        time=datetime(2025, 2, 2, 10, 0),
        open=200.0,
        high=205.0,
        low=199.0,
        close=204.0,
        volume=5000000,
    )


def test_pead_bullish_drift(strategy, sample_bar):
    market_state = MarketState(time=sample_bar.time, regime=None)

    mock_features = MockFeatures(
        days_since_earnings=2,
        earnings_surprise=0.20, # 20% beat
        ema20_slope=0.01 # Upward momentum
    )

    symbol_state = SymbolState("META")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 5.0,
    }

    signal = strategy.on_bar("META", sample_bar, symbol_state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.stop_price == 204.0 - (2.0 * 5.0)
    assert signal.target_price == 204.0 + (4.0 * 5.0)

def test_pead_bearish_drift(strategy, sample_bar):
    market_state = MarketState(time=sample_bar.time, regime=None)

    mock_features = MockFeatures(
        days_since_earnings=3,
        earnings_surprise=-0.25, # 25% miss
        ema20_slope=-0.01 # Downward momentum
    )

    symbol_state = SymbolState("META")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 5.0,
    }

    signal = strategy.on_bar("META", sample_bar, symbol_state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.SELL

def test_pead_insufficient_surprise(strategy, sample_bar):
    market_state = MarketState(time=sample_bar.time, regime=None)

    mock_features = MockFeatures(
        days_since_earnings=2,
        earnings_surprise=0.10, # 10% beat (too low)
        ema20_slope=0.01
    )

    symbol_state = SymbolState("META")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 5.0,
    }

    signal = strategy.on_bar("META", sample_bar, symbol_state, market_state)
    assert signal is None

def test_pead_outside_days_window(strategy, sample_bar):
    market_state = MarketState(time=sample_bar.time, regime=None)

    mock_features = MockFeatures(
        days_since_earnings=10, # Out of 1-5 day window
        earnings_surprise=0.50,
        ema20_slope=0.01
    )

    symbol_state = SymbolState("META")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 5.0,
    }

    signal = strategy.on_bar("META", sample_bar, symbol_state, market_state)
    assert signal is None
