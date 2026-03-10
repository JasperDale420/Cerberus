from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.core.domain import Bar, MarketState, OrderSide, Regime, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.alpaca_options_trend import AlpacaOptionsTrend


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
        allowed_strategies=["alpaca_options_trend"],
        meta={},
    )

@patch("src.strategies.alpaca_options_trend.AlpacaClient")
def test_options_trend_no_signal_few_trades(mock_alpaca_client_class, logger, bar, market_state):
    # Setup mock AlpacaClient with chain and few trades
    mock_client_instance = mock_alpaca_client_class.return_value
    mock_client_instance.get_historical_option_chain.return_value = {"SPY230120C00400000": Mock()}
    mock_client_instance.get_historical_option_trades.return_value = [Mock() for _ in range(10)] # Only 10 trades, threshold is 50

    strategy = AlpacaOptionsTrend({}, logger)

    signal = strategy.on_bar("SPY", bar, create_symbol_state(), market_state)
    assert signal is None
    mock_client_instance.get_historical_option_chain.assert_called_once_with("SPY")
    mock_client_instance.get_historical_option_trades.assert_called_once()

@patch("src.strategies.alpaca_options_trend.AlpacaClient")
def test_options_trend_creates_signal_many_trades(mock_alpaca_client_class, logger, bar, market_state):
    # Setup mock AlpacaClient with chain and many trades
    mock_client_instance = mock_alpaca_client_class.return_value
    mock_client_instance.get_historical_option_chain.return_value = {"SPY230120C00400000": Mock()}
    mock_client_instance.get_historical_option_trades.return_value = [Mock() for _ in range(60)] # 60 trades, above 50 threshold

    strategy = AlpacaOptionsTrend({}, logger)

    signal = strategy.on_bar("SPY", bar, create_symbol_state(), market_state)

    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.target_price == bar.close * 1.05
    assert signal.stop_price == bar.close * 0.95
    assert signal.meta["tape_signal"] == "bull_sweeps"
    mock_client_instance.get_historical_option_chain.assert_called_once_with("SPY")
    mock_client_instance.get_historical_option_trades.assert_called_once()

@patch("src.strategies.alpaca_options_trend.AlpacaClient")
def test_options_trend_no_chain_data(mock_alpaca_client_class, logger, bar, market_state):
    # Setup mock AlpacaClient with empty chain
    mock_client_instance = mock_alpaca_client_class.return_value
    mock_client_instance.get_historical_option_chain.return_value = {}

    strategy = AlpacaOptionsTrend({}, logger)

    signal = strategy.on_bar("SPY", bar, create_symbol_state(), market_state)
    assert signal is None
    # get_historical_option_trades should not be called if chain is empty
    mock_client_instance.get_historical_option_trades.assert_not_called()
