from datetime import datetime, timezone

import pytest

from src.core.domain import Bar, MarketState, OrderSide, Regime, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.order_flow_imbalance import OrderFlowImbalanceStrategy


@pytest.fixture
def strategy():
    config = {"tfi_threshold": 1.5, "min_flow_bias": 0.2, "stop_atr_mult": 1.0, "target_atr_mult": 2.0}
    logger = StructuredLogger("TestLogger")
    return OrderFlowImbalanceStrategy(config, logger)


@pytest.fixture
def bar():
    return Bar(
        symbol="AAPL",
        time=datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc),
        open=150.0,
        high=151.0,
        low=149.0,
        close=150.5,
        volume=1000,
    )


@pytest.fixture
def market_state():
    return MarketState(
        time=datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc),
        index_price=400.0,
        regime=Regime.BULL,
        meta={},
    )


def test_ofi_long_signal(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["order_flow_imbalance"],
        meta={"tfi": 2.0, "flow_bias": 0.3, "atr": 2.0},
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.stop_price == 150.5 - (1.0 * 2.0)
    assert signal.target_price == 150.5 + (2.0 * 2.0)
    assert signal.meta["tfi"] == 2.0
    assert signal.meta["flow_bias"] == 0.3


def test_ofi_short_signal(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["order_flow_imbalance"],
        meta={"tfi": -2.0, "flow_bias": -0.3, "atr": 2.0},
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is not None
    assert signal.side == OrderSide.SELL
    assert signal.stop_price == 150.5 + (1.0 * 2.0)
    assert signal.target_price == 150.5 - (2.0 * 2.0)
    assert signal.meta["tfi"] == -2.0
    assert signal.meta["flow_bias"] == -0.3


def test_ofi_no_signal_weak_tfi(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["order_flow_imbalance"],
        meta={"tfi": 1.0, "flow_bias": 0.3, "atr": 2.0},
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is None
