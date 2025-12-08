import pytest
from unittest.mock import MagicMock
from datetime import datetime
from src.engine.execution import ExecutionEngine
from src.strategies.base import Signal, SymbolState, MarketState
from src.analysis.regime import Regime
from src.engine.risk import OrderIntent

def test_process_signal_flow():
    # Mocks
    mock_config = {"max_daily_loss": 1000, "max_risk_per_trade": 50}
    mock_logger = MagicMock()
    mock_alpaca = MagicMock()
    
    engine = ExecutionEngine(mock_config, mock_logger, mock_alpaca)
    
    # Mock RiskManager and OrderExecutor behaviors via engine's internal instances
    # But better to mock the classes or just test the logic flow if we trust the components.
    # Since we are testing integration, let's let them run but mock the external Alpaca call.
    
    # Setup Signal
    signal = Signal(
        symbol="AAPL",
        side="buy",
        size_hint=1.0,
        entry_price=150.0,
        stop_price=149.0, # Risk = 1.0 per share
        target_price=152.0,
        strategy="test_strat",
        regime=Regime.CHOP,
        generated_at=datetime.utcnow(),
        meta={}
    )
    
    # Setup State
    engine.symbol_states["AAPL"] = SymbolState(symbol="AAPL", bars=[], position=None)
    
    # Run
    engine._process_signal(signal)
    
    # Verify RiskManager approved (qty = 50 / 1 = 50)
    # Verify OrderExecutor called Alpaca
    mock_alpaca.trading_client.submit_order.assert_called_once()
    
    # Check args
    call_args = mock_alpaca.trading_client.submit_order.call_args[0][0]
    assert call_args.symbol == "AAPL"
    assert call_args.qty == 50
    assert call_args.side.value == "buy"

def test_process_signal_risk_rejection():
    mock_config = {"max_daily_loss": 1000, "max_risk_per_trade": 50}
    mock_logger = MagicMock()
    mock_alpaca = MagicMock()
    
    engine = ExecutionEngine(mock_config, mock_logger, mock_alpaca)
    
    # Simulate max loss exceeded
    engine.risk_manager.current_daily_pnl = -1100
    
    signal = Signal(
        symbol="AAPL",
        side="buy",
        size_hint=1.0,
        entry_price=150.0,
        stop_price=149.0,
        target_price=152.0,
        strategy="test_strat",
        regime=Regime.CHOP,
        generated_at=datetime.utcnow(),
        meta={}
    )
    
    engine.symbol_states["AAPL"] = SymbolState(symbol="AAPL", bars=[], position=None)
    
    engine._process_signal(signal)
    
    # Verify NO order submitted
    mock_alpaca.trading_client.submit_order.assert_not_called()
