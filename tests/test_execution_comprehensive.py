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
def test_flatten_all_uses_scoped_broker_calls(execution_engine, mock_alpaca):
    """Flatten must NOT call the broad close_all_positions/cancel_orders APIs —
    those would liquidate every position in the shared paper account. It must
    use scoped per-order/per-position calls instead.

    Scenario: one Cerberus-owned position (AAPL) and one foreign-owned position
    (NVDA, opened by Orbit). Only AAPL should be closed.
    """
    execution_engine.config["position_mismatch_mode"] = "log"
    execution_engine.broker_client = mock_alpaca

    aapl_pos = MagicMock()
    aapl_pos.symbol = "AAPL"
    nvda_pos = MagicMock()
    nvda_pos.symbol = "NVDA"

    cerb_open_order = MagicMock()
    cerb_open_order.id = "cerb_open_1"
    cerb_open_order.symbol = "AAPL"
    cerb_open_order.client_order_id = "cerberus_s-AAPL-1-a"

    orbit_open_order = MagicMock()
    orbit_open_order.id = "orbit_open_1"
    orbit_open_order.symbol = "NVDA"
    orbit_open_order.client_order_id = "orbit-NVDA-1-b"

    cerb_filled = MagicMock()
    cerb_filled.symbol = "AAPL"
    cerb_filled.client_order_id = "cerberus_s-AAPL-0-a"
    cerb_filled.status = "filled"
    cerb_filled.filled_at = datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc)
    cerb_filled.updated_at = cerb_filled.filled_at

    orbit_filled = MagicMock()
    orbit_filled.symbol = "NVDA"
    orbit_filled.client_order_id = "orbit-NVDA-0-b"
    orbit_filled.status = "filled"
    orbit_filled.filled_at = datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc)
    orbit_filled.updated_at = orbit_filled.filled_at

    def _get_orders(req=None, *_a, **_kw):
        status = getattr(req, "status", None)
        status_str = str(getattr(status, "value", status) or "").lower()
        if "closed" in status_str:
            return [cerb_filled, orbit_filled]
        return [cerb_open_order, orbit_open_order]

    mock_alpaca.trading_client.get_orders.side_effect = _get_orders
    mock_alpaca.trading_client.get_all_positions.return_value = [aapl_pos, nvda_pos]

    execution_engine.flatten_all(reason="Test")

    # Legacy broad methods must NOT be called — they'd hit foreign positions.
    mock_alpaca.trading_client.cancel_orders.assert_not_called()
    mock_alpaca.trading_client.close_all_positions.assert_not_called()

    # Only the Cerberus-owned open order is cancelled.
    mock_alpaca.trading_client.cancel_order_by_id.assert_called_once_with("cerb_open_1")
    # Only the Cerberus-owned position is closed.
    mock_alpaca.trading_client.close_position.assert_called_once_with("AAPL")


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
