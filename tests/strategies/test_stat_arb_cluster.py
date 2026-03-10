from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from src.core.domain import Bar, MarketState, OrderSide, Regime, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.stat_arb_cluster import StatArbClusterStrategy


@dataclass
class MockFeatures:
    cluster_residual: float = 0.0
    cluster_id: int = 0
    atr: float = 2.0



@pytest.fixture
def strategy():
    config = {
        "residual_threshold": 2.0,
        "stop_atr_mult": 1.0,
        "target_atr_mult": 2.0,
    }
    logger = StructuredLogger("TestLogger")
    return StatArbClusterStrategy(config, logger)


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


def test_stat_arb_short_signal_overperforming(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["stat_arb_cluster"],
        meta={"features": MockFeatures(cluster_residual=2.5, atr=2.0)},  # Over threshold, should mean revert (short)
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is not None
    assert signal.side == OrderSide.SELL
    assert signal.stop_price == 150.5 + (1.0 * 2.0)
    assert signal.target_price == 150.5 - (1.0 * 2.0 * 1.5)
    assert signal.meta["cluster_residual"] == 2.5


def test_stat_arb_long_signal_underperforming(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["stat_arb_cluster"],
        meta={"features": MockFeatures(cluster_residual=-2.5, atr=2.0)},  # Under threshold, should mean revert (long)
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.stop_price == 150.5 - (1.0 * 2.0)
    assert signal.target_price == 150.5 + (1.0 * 2.0 * 1.5)
    assert signal.meta["cluster_residual"] == -2.5


def test_stat_arb_no_signal_within_threshold(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["stat_arb_cluster"],
        meta={"features": MockFeatures(cluster_residual=1.5, atr=2.0)},  # Within threshold 2.0
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is None


def test_stat_arb_no_signal_missing_residual(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["stat_arb_cluster"],
        meta={"features": MockFeatures(atr=2.0)},  # Missing cluster_residual
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is None
