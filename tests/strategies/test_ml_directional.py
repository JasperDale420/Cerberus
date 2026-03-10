from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from src.core.domain import Bar, MarketState, OrderSide, Regime, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.ml_directional import MLDirectionalStrategy


@dataclass
class MockFeatures:
    ml_prediction: float = 0.0
    ema_trend_strength: float = 0.0
    atr: float = 2.0



@pytest.fixture
def strategy():
    config = {
        "ml_threshold": 0.6,
        "trend_alignment_required": True,
        "stop_atr_mult": 1.5,
        "target_atr_mult": 3.0,
    }
    logger = StructuredLogger("TestLogger")
    return MLDirectionalStrategy(config, logger)


@pytest.fixture
def bar():
    return Bar(
        symbol="AAPL",
        time=datetime(2023, 1, 1, 10, 0, tzinfo=UTC),
        open=150.0,
        high=151.0,
        low=149.0,
        close=150.5,
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


def test_ml_directional_long_signal(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={"ema_close_5m:20": 150.0, "ema_close_5m:50": 140.0},
        position=None,
        open_orders={},
        allowed_strategies=["ml_directional"],
        meta={"features": MockFeatures(ml_prediction=0.8, ema_trend_strength=2.0, atr=2.0)},
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.stop_price == 150.5 - (1.5 * 2.0)
    assert signal.target_price == 150.5 + (3.0 * 2.0)
    assert signal.meta["ml_prediction"] == 0.8


def test_ml_directional_short_signal(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={"ema_close_5m:20": 140.0, "ema_close_5m:50": 150.0},
        position=None,
        open_orders={},
        allowed_strategies=["ml_directional"],
        meta={"features": MockFeatures(ml_prediction=-0.7, ema_trend_strength=-2.0, atr=2.0)},
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is not None
    assert signal.side == OrderSide.SELL
    assert signal.stop_price == 150.5 + (1.5 * 2.0)
    assert signal.target_price == 150.5 - (3.0 * 2.0)
    assert signal.meta["ml_prediction"] == -0.7


def test_ml_directional_no_signal_weak_prediction(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={"ema_close_5m:20": 150.0, "ema_close_5m:50": 140.0},
        position=None,
        open_orders={},
        allowed_strategies=["ml_directional"],
        meta={"features": MockFeatures(ml_prediction=0.5, ema_trend_strength=2.0, atr=2.0)},  # Below 0.6 threshold
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is None


def test_ml_directional_no_signal_trend_mismatch(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={"ema_close_5m:20": 140.0, "ema_close_5m:50": 150.0},
        position=None,
        open_orders={},
        allowed_strategies=["ml_directional"],
        meta={"features": MockFeatures(ml_prediction=0.8, ema_trend_strength=-1.0, atr=2.0)},  # ML says long, trend says short
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is None
