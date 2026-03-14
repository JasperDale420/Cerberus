from collections import deque
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.domain import Bar, OrderSide, Regime, Signal, SymbolState
from src.engine.execution import ExecutionEngine
from src.strategies.base import BaseStrategy


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_alpaca():
    client = MagicMock()
    # Mock trading client methods used in _fetch_broker_data
    client.trading_client.get_account = MagicMock()
    client.trading_client.get_all_positions = MagicMock()
    client.trading_client.get_orders = MagicMock()
    return client


@pytest.fixture
def execution_engine(mock_logger, mock_alpaca):
    config = {
        "max_daily_loss": 1000,
        "max_risk_per_trade": 50,
        "max_notional_per_order": 50000,
        "risk": {"max_daily_loss": 1000},
    }
    return ExecutionEngine(config, mock_logger)


@pytest.mark.unit
def test_on_bar_updates_state_and_runs_strategy(execution_engine):
    # Setup
    symbol = "SPY"
    bar = Bar(symbol, datetime.now(timezone.utc), 100.0, 101.0, 99.0, 100.0, 1000.0)

    # Mock strategy setup
    strat = MagicMock(spec=BaseStrategy)
    # on_bar returns list of signals or empty list
    strat.on_bar.return_value = []

    execution_engine.strategies = {"MyStrat": strat}

    # Mock StrategyEngine
    execution_engine.strategy_engine = MagicMock()
    execution_engine.strategy_engine.on_bar.return_value = []

    # Pre-populate symbol state
    execution_engine.symbol_states[symbol] = SymbolState(
        symbol=symbol,
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["MyStrat"],
        meta={},
    )

    # Act - Sync call
    execution_engine.on_bar(bar)

    # Assert
    # 1. Bar should be in history
    assert len(execution_engine.symbol_states[symbol].bars) == 1
    assert execution_engine.symbol_states[symbol].bars[0] == bar

    # 2. StrategyEngine should be called
    execution_engine.strategy_engine.on_bar.assert_called_once()


@pytest.mark.unit
def test_on_bar_processes_signal(execution_engine):
    symbol = "SPY"
    bar = Bar(symbol, datetime.now(timezone.utc), 100.0, 101.0, 99.0, 100.0, 1000.0)

    # Signal
    sig = Signal(
        symbol=symbol,
        side=OrderSide.BUY,
        size_hint=0.0,
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        strategy="MyStrat",
        regime=Regime.BULL,
        generated_at=datetime.now(timezone.utc),
        meta={},
    )

    # Mock StrategyEngine to return this signal
    execution_engine.strategy_engine = MagicMock()
    execution_engine.strategy_engine.on_bar.return_value = [sig]

    execution_engine.symbol_states[symbol] = SymbolState(
        symbol=symbol,
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["MyStrat"],
        meta={},
    )

    # Mock processing
    execution_engine._process_signal = MagicMock()

    execution_engine.on_bar(bar)

    execution_engine._process_signal.assert_called_once_with(sig)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconcile_broker_state_runs(execution_engine, mock_alpaca):
    # Mock fetch outputs
    mock_acct = MagicMock()
    mock_acct.equity = "100000"

    mock_positions: list[MagicMock] = []
    mock_orders: list[MagicMock] = []
    mock_closed: list[MagicMock] = []

    # Since _fetch_broker_data uses asyncio.to_thread, we need to handle that or let it run if simple.
    # But usually MagicMocks are not thread-safe or async friendly if default.
    # execution_engine._fetch_broker_data calls asyncio.to_thread on mocks.
    # Let's mock _fetch_broker_data directly to avoid complexity.

    with patch.object(execution_engine, "_fetch_broker_data", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = (mock_acct, mock_positions, mock_orders, mock_closed)

        await execution_engine.reconcile_broker_state()

        mock_fetch.assert_awaited_once()


@pytest.mark.unit
def test_flatten_all_calls_broker(execution_engine, mock_alpaca):
    # Setup
    execution_engine.config["position_mismatch_mode"] = "log"
    # Set broker_client so flatten_all actually runs broker operations
    execution_engine.broker_client = mock_alpaca

    # Mock return values for verification steps to prevent errors
    mock_alpaca.trading_client.get_all_positions.return_value = []
    mock_alpaca.trading_client.get_orders.return_value = []

    execution_engine.flatten_all(reason="Test")

    # Verify calls
    mock_alpaca.trading_client.cancel_orders.assert_called_once()
    mock_alpaca.trading_client.close_all_positions.assert_called_once_with(cancel_orders=True)


@pytest.mark.unit
def test_flatten_all_resets_local_state(execution_engine, mock_alpaca):
    symbol = "SPY"
    execution_engine.symbol_states[symbol] = SymbolState(
        symbol=symbol,
        bars=deque(),
        indicators={},
        position=MagicMock(),  # Placeholder
        open_orders={"ord1": {}},
        allowed_strategies=[],
        meta={},
    )

    mock_alpaca.trading_client.get_all_positions.return_value = []
    mock_alpaca.trading_client.get_orders.return_value = []

    execution_engine.flatten_all()

    state = execution_engine.symbol_states[symbol]
    assert state.position is None
    assert state.open_orders == {}
