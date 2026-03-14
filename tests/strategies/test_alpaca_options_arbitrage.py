from collections import deque
from datetime import UTC, datetime

import pytest

from src.core.domain import Bar, MarketState, OrderSide, Regime, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.alpaca_options_arbitrage import AlpacaOptionsArbitrage


@pytest.fixture
def logger():
    return StructuredLogger("TestLogger")


@pytest.fixture
def market_state():
    return MarketState(
        time=datetime(2023, 1, 1, 10, 0, tzinfo=UTC),
        index_price=400.0,
        regime=Regime.BULL,
        meta={},
    )


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


def create_symbol_state():
    return SymbolState(
        symbol="SPY",
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["alpaca_options_arbitrage"],
        meta={},
    )


def test_options_arbitrage_no_signal_low_vol(logger, bar, market_state):
    strategy = AlpacaOptionsArbitrage({}, logger)
    market_state.realized_vol = 0.10  # Low vol, below 0.15 threshold

    signal = strategy.on_bar("SPY", bar, create_symbol_state(), market_state)
    assert signal is None


def test_options_arbitrage_creates_signal_high_vol(logger, bar, market_state):
    strategy = AlpacaOptionsArbitrage({}, logger)
    market_state.realized_vol = 0.20  # High vol, above 0.15 threshold

    signal = strategy.on_bar("SPY", bar, create_symbol_state(), market_state)

    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.target_price == bar.close * 1.05
    assert signal.stop_price == bar.close * 0.98
    assert signal.meta["arb_type"] == "volatility_dispersion"
