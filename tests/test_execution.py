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

    engine = ExecutionEngine(mock_config, mock_logger)
    # Set up a mock order executor so signal processing can submit orders
    mock_order_executor = MagicMock()
    engine.order_executor = mock_order_executor

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
    # Verify OrderExecutor was called
    mock_order_executor.submit.assert_called_once()
    call_args = mock_order_executor.submit.call_args
    intent = call_args[0][0]
    assert intent.symbol == "AAPL"
    assert intent.qty == 50


@pytest.mark.unit
def test_process_signal_risk_rejection():
    mock_config = {"max_daily_loss": 1000, "max_risk_per_trade": 50}
    mock_logger = MagicMock()

    engine = ExecutionEngine(mock_config, mock_logger)
    mock_order_executor = MagicMock()
    engine.order_executor = mock_order_executor

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
    mock_order_executor.submit.assert_not_called()
