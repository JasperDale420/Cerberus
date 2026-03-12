from datetime import datetime

import pytest

from src.core.domain import Bar, MarketState, OrderSide, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.lead_lag import LeadLagStrategy


# A robust dataclass/mock for structural features testing.
class MockFeatures:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def logger():
    return StructuredLogger("TestLeadLag")


@pytest.fixture
def strategy(logger):
    config = {
        "min_correlation": 0.8,
        "index_spike_threshold": 0.002,
        "stop_atr_multiplier": 1.5,
        "target_atr_multiplier": 3.0,
    }
    return LeadLagStrategy(config, logger)


@pytest.fixture
def sample_bar():
    return Bar(
        symbol="AAPL",
        time=datetime(2025, 1, 1, 10, 0),
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.0,
        volume=1000000,
    )


def test_lead_lag_bullish_spike(strategy, sample_bar):
    market_state = MarketState(
        time=sample_bar.time,
        regime=None,
        index_symbol="SPY",
        index_price=500.0,
        index_return=0.003,  # +0.3% impulse
    )

    mock_features = MockFeatures(correlation_to_index=0.85)

    symbol_state = SymbolState("AAPL")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 2.0,
    }

    signal = strategy.on_bar("AAPL", sample_bar, symbol_state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.stop_price == 151.0 - (1.5 * 2.0)
    assert signal.target_price == 151.0 + (3.0 * 2.0)
    assert signal.symbol == "AAPL"


def test_lead_lag_bearish_spike(strategy, sample_bar):
    market_state = MarketState(
        time=sample_bar.time,
        regime=None,
        index_symbol="SPY",
        index_price=500.0,
        index_return=-0.004,  # -0.4% impulse
    )

    mock_features = MockFeatures(correlation_to_index=0.85)

    symbol_state = SymbolState("AAPL")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 2.0,
    }

    signal = strategy.on_bar("AAPL", sample_bar, symbol_state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.SELL
    assert signal.stop_price == 151.0 + (1.5 * 2.0)
    assert signal.target_price == 151.0 - (3.0 * 2.0)


def test_lead_lag_insufficient_correlation(strategy, sample_bar):
    market_state = MarketState(
        time=sample_bar.time,
        regime=None,
        index_symbol="SPY",
        index_price=500.0,
        index_return=0.005,
    )

    mock_features = MockFeatures(correlation_to_index=0.50)  # Too low

    symbol_state = SymbolState("AAPL")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 2.0,
    }

    signal = strategy.on_bar("AAPL", sample_bar, symbol_state, market_state)
    assert signal is None


def test_lead_lag_insufficient_spike(strategy, sample_bar):
    market_state = MarketState(
        time=sample_bar.time,
        regime=None,
        index_symbol="SPY",
        index_price=500.0,
        index_return=0.001,  # Too low
    )

    mock_features = MockFeatures(correlation_to_index=0.90)

    symbol_state = SymbolState("AAPL")
    symbol_state.meta = {
        "features": mock_features,
        "atr": 2.0,
    }

    signal = strategy.on_bar("AAPL", sample_bar, symbol_state, market_state)
    assert signal is None
