from datetime import UTC, datetime
from unittest.mock import Mock, patch

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
    from collections import deque
    return SymbolState(
        symbol="SPY",
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["alpaca_options_arbitrage"],
        meta={},
    )

@patch("src.strategies.alpaca_options_arbitrage.AlpacaClient")
def test_options_arbitrage_no_signal_low_vol(mock_alpaca_client_class, logger, bar, market_state):
    # Setup mock AlpacaClient and chain data
    mock_client_instance = mock_alpaca_client_class.return_value
    mock_client_instance.get_historical_option_chain.return_value = {"SPY230120C00400000": Mock()}

    strategy = AlpacaOptionsArbitrage({}, logger)
    market_state.realized_vol = 0.10 # Low vol, below 0.15 threshold

    signal = strategy.on_bar("SPY", bar, create_symbol_state(), market_state)
    assert signal is None
    mock_client_instance.get_historical_option_chain.assert_called_once_with("SPY")


@patch("src.strategies.alpaca_options_arbitrage.AlpacaClient")
def test_options_arbitrage_creates_signal_high_vol(mock_alpaca_client_class, logger, bar, market_state):
    # Setup mock AlpacaClient and chain data
    mock_client_instance = mock_alpaca_client_class.return_value
    mock_client_instance.get_historical_option_chain.return_value = {"SPY230120C00400000": Mock()}

    strategy = AlpacaOptionsArbitrage({}, logger)
    market_state.realized_vol = 0.20 # High vol, above 0.15 threshold

    signal = strategy.on_bar("SPY", bar, create_symbol_state(), market_state)

    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.target_price == bar.close * 1.05
    assert signal.stop_price == bar.close * 0.98
    assert signal.meta["arb_type"] == "volatility_dispersion"
    mock_client_instance.get_historical_option_chain.assert_called_once_with("SPY")


@patch("src.strategies.alpaca_options_arbitrage.AlpacaClient")
def test_options_arbitrage_no_chain_data(mock_alpaca_client_class, logger, bar, market_state):
    # Setup mock AlpacaClient with empty chain data
    mock_client_instance = mock_alpaca_client_class.return_value
    mock_client_instance.get_historical_option_chain.return_value = {}

    strategy = AlpacaOptionsArbitrage({}, logger)
    market_state.realized_vol = 0.20 # High vol, but no chain data

    signal = strategy.on_bar("SPY", bar, create_symbol_state(), market_state)
    assert signal is None
