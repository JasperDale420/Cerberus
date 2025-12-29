from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.analysis.schema import Trade as DbTrade
from src.engine.execution import ExecutionEngine


@pytest.mark.unit
def test_on_fill_persistence():
    # Setup
    config: dict = {}
    logger = MagicMock()
    db = MagicMock()
    mock_session = MagicMock()
    db.write.side_effect = lambda _kind, fn: (fn(mock_session), True)[1]

    engine = ExecutionEngine(config, logger, db=db)

    # 1. Open Position
    # Buy 10 SPY @ 100
    fill_open = {
        "symbol": "SPY",
        "side": "buy",
        "qty": 10,
        "price": 100.0,
        "timestamp": datetime.now(timezone.utc),
        "strategy": "StratA",
        "correlation_id": "corr1",
    }

    # Mock symbol state existence (usually created in on_bar, but here we force it or rely on engine creating it?
    # Engine creates it on_bar. safe to pre-populate)
    from collections import deque

    from src.core.domain import SymbolState

    engine.symbol_states["SPY"] = SymbolState("SPY", deque(), {}, None, {}, [], {})

    engine.on_fill(fill_open)

    state = engine.symbol_states["SPY"]
    assert state.position is not None
    assert state.position.qty == 10
    assert state.position is not None
    assert state.position.qty == 10
    assert state.position.avg_price == pytest.approx(100.0)

    # 2. Close Position
    # Sell 10 SPY @ 110 (Profit 100)
    fill_close = {
        "symbol": "SPY",
        "side": "sell",
        "qty": 10,
        "price": 110.0,
        "timestamp": datetime.now(timezone.utc),
        "strategy": "StratA",
        "correlation_id": "corr1",
    }

    engine.on_fill(fill_close)

    # Assertions
    assert state.position is None  # Closed

    # Check DB Persist
    # Should have added a DbTrade object via db.write(...)
    assert db.write.called
    assert mock_session.add.called
    args, _ = mock_session.add.call_args
    trade_obj = args[0]

    assert isinstance(trade_obj, DbTrade)
    assert trade_obj.pnl_gross == pytest.approx(100.0)  # (110-100)*10
    assert trade_obj.symbol == "SPY"
