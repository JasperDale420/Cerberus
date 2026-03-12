from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from src.core.domain import Bar, MarketState, OrderSide, Regime, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.gamma_pinning import GammaPinningStrategy


@pytest.fixture
def logger():
    return StructuredLogger("TestLogger")


@pytest.fixture
def strategy(logger):
    return GammaPinningStrategy({}, logger)


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
        allowed_strategies=["gamma_pinning"],
        meta={"features": features},
    )


def test_gamma_pinning_no_signal_low_gex(strategy, bar, market_state):
    features = Mock()
    features.symbol = "SPY"
    features.price = 400.0
    features.net_gex = 500_000.0  # Below threshold of 1_000_000
    features.gex_flip_dist = 0.02

    signal = strategy.on_bar("SPY", bar, create_symbol_state(features), market_state)
    assert signal is None


def test_gamma_pinning_short_signal(strategy, bar, market_state):
    features = Mock()
    features.symbol = "SPY"
    features.price = 400.0
    features.net_gex = 2_000_000.0
    features.gex_flip_dist = -8.0

    signal = strategy.on_bar("SPY", bar, create_symbol_state(features), market_state)
    assert signal is not None
    assert signal.side == OrderSide.SELL
    assert signal.target_price == 392.0
    assert signal.stop_price == bar.close * 1.01
    assert "net_gex" in signal.meta
    assert "deviation" in signal.meta


def test_gamma_pinning_long_signal(strategy, bar, market_state):
    features = Mock()
    features.symbol = "SPY"
    features.price = 400.0
    features.net_gex = 5_000_000.0
    features.gex_flip_dist = 10.0
    signal = strategy.on_bar("SPY", bar, create_symbol_state(features), market_state)
    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.target_price == 410.0
    assert signal.stop_price == bar.close * 0.99


def test_gamma_pinning_no_signal_small_deviation(strategy, bar, market_state):
    features = Mock()
    features.symbol = "SPY"
    features.price = 400.0
    features.net_gex = 2_000_000.0
    features.gex_flip_dist = 2.0
    signal = strategy.on_bar("SPY", bar, create_symbol_state(features), market_state)
    assert signal is None
