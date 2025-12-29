from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.analysis.regime import Regime
from src.core.domain import OrderSide
from src.engine.execution import ExecutionEngine
from src.strategies.base import Signal, SymbolState


@pytest.mark.unit
def test_process_signal_flow():
    # Mocks
    mock_config = {
        "max_daily_loss": 1000,
        "max_risk_per_trade": 50,
        "max_notional_per_order": 10000,
    }
    mock_logger = MagicMock()
    mock_alpaca = MagicMock()

    engine = ExecutionEngine(mock_config, mock_logger, alpaca_client=mock_alpaca)

    # Mock RiskManager and OrderExecutor behaviors via engine's internal instances
    # But better to mock the classes or just test the logic flow if we trust the components.
    # Since we are testing integration, let's let them run but mock the external Alpaca call.

    # Setup Signal
    signal = Signal(
        symbol="AAPL",
        side=OrderSide.BUY,
        size_hint=0.0,
        entry_price=150.0,
        stop_price=149.0,  # Risk = 1.0 per share
        target_price=152.0,
        strategy="test_strat",
        regime=Regime.CHOP,
        generated_at=datetime.now(timezone.utc),
        meta={},
    )

    # Setup State
    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=deque(),
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    # Run
    engine._process_signal(signal)

    # Verify RiskManager approved (qty = 50 / 1 = 50)
    # Verify OrderExecutor called Alpaca
    # Debug failure
    if mock_alpaca.trading_client.submit_order.call_count == 0:
        print("\nLogger warnings:", mock_logger.warning.call_args_list)
        print("Logger errors:", mock_logger.error.call_args_list)

    # Check args
    mock_alpaca.trading_client.submit_order.assert_called_once()
    call_args = mock_alpaca.trading_client.submit_order.call_args[0][0]
    assert call_args.symbol == "AAPL"
    assert call_args.qty == 50
    assert call_args.side.value == "buy"


@pytest.mark.unit
def test_process_signal_risk_rejection():
    mock_config = {"max_daily_loss": 1000, "max_risk_per_trade": 50}
    mock_logger = MagicMock()
    mock_alpaca = MagicMock()

    engine = ExecutionEngine(mock_config, mock_logger, mock_alpaca)

    # Simulate max loss exceeded
    engine.risk_manager.current_daily_pnl = -1100

    signal = Signal(
        symbol="AAPL",
        side=OrderSide.BUY,
        size_hint=1.0,
        entry_price=150.0,
        stop_price=149.0,
        target_price=152.0,
        strategy="test_strat",
        regime=Regime.CHOP,
        generated_at=datetime.now(timezone.utc),
        meta={},
    )

    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=deque(),
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    engine._process_signal(signal)

    # Verify NO order submitted
    mock_alpaca.trading_client.submit_order.assert_not_called()
