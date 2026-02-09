from datetime import datetime, timezone

import pytest

from src.core.domain import (
    Bar,
    MarketState,
    OrderSide,
    Position,
    Regime,
    Side,
    SymbolState,
)
from src.core.logger import StructuredLogger
from src.strategies.pair_trading import PairTradingStrategy


@pytest.fixture
def strategy():
    config = {"entry_zscore": 2.0, "exit_zscore": 0.5}
    logger = StructuredLogger("TestLogger")
    return PairTradingStrategy(config, logger)


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
        regime=Regime.CHOP,
        meta={},
    )


def test_pair_trading_exit_long(strategy, bar, market_state):
    symbol_state = SymbolState(
        symbol="AAPL",
        bars=[],
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["pair_trading"],
        meta={},
    )
    symbol_state.position = Position(
        symbol="AAPL",
        side=Side.LONG,
        avg_price=148.0,
        unrealized_pnl=2.5,
        realized_pnl=0.0,
        qty=100,
        entry_time=datetime(2023, 1, 1, 9, 30, tzinfo=timezone.utc),
        strategy="pair_trading",
    )
    symbol_state.meta["pair_trade"] = {"z_score": 0.6, "side": "buy"}

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.SELL
    assert signal.meta.get("exit_reason") == "mean_reversion"


def test_pair_trading_exit_short(strategy, bar, market_state):
    symbol_state = SymbolState(
        symbol="AAPL",
        bars=[],
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["pair_trading"],
        meta={},
    )
    symbol_state.position = Position(
        symbol="AAPL",
        side=Side.SHORT,
        avg_price=152.0,
        unrealized_pnl=1.5,
        realized_pnl=0.0,
        qty=100,
        entry_time=datetime(2023, 1, 1, 9, 30, tzinfo=timezone.utc),
        strategy="pair_trading",
    )
    symbol_state.meta["pair_trade"] = {"z_score": -0.6, "side": "sell"}

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.meta.get("exit_reason") == "mean_reversion"


def test_pair_trading_no_exit_yet(strategy, bar, market_state):
    symbol_state = SymbolState(
        symbol="AAPL",
        bars=[],
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["pair_trading"],
        meta={},
    )
    symbol_state.position = Position(
        symbol="AAPL",
        side=Side.LONG,
        avg_price=148.0,
        unrealized_pnl=2.5,
        realized_pnl=0.0,
        qty=100,
        entry_time=datetime(2023, 1, 1, 9, 30, tzinfo=timezone.utc),
        strategy="pair_trading",
    )
    symbol_state.meta["pair_trade"] = {"z_score": 0.2, "side": "buy"}

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)

    assert signal is None
