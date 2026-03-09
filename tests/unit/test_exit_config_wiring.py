"""Test that _store_pending_entry reads exit_config from signal metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.core.domain import OrderSide
from src.strategies.base import Signal


@pytest.mark.unit
def test_store_pending_entry_reads_signal_exit_config():
    """exit_config in signal.meta should populate max_hold_seconds and trailing fields."""
    from collections import deque

    from src.core.domain import SymbolState
    from src.engine.execution import ExecutionEngine

    config = {
        "strategies": {
            "trend_rider_pro": {"enabled": True},
        },
        "risk": {},
    }
    logger = MagicMock()
    logger.bind = MagicMock(return_value=logger)

    engine = ExecutionEngine(config=config, logger=logger)
    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=100),
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=["trend_rider_pro"],
        meta={},
    )

    signal = Signal(
        symbol="AAPL",
        side=OrderSide.BUY,
        strategy="trend_rider_pro",
        entry_price=150.0,
        stop_price=148.0,
        target_price=155.0,
        size_hint=10,
        generated_at=datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc),
        meta={
            "exit_config": {
                "trailing_enabled": True,
                "trail_pct": 0.03,
                "max_hold_minutes": 120,
            }
        },
    )

    intent = MagicMock()
    intent.qty = 10

    engine._store_pending_entry(signal, [intent])

    pending = engine.symbol_states["AAPL"].meta["pending_entries"]
    entry = pending[signal.correlation_id]

    # max_hold_seconds = 120 min * 60 = 7200s
    assert entry["max_hold_seconds"] == 7200
    assert entry["trailing_stop_enabled"] is True
    assert entry["trailing_stop_pct"] == pytest.approx(0.03)


@pytest.mark.unit
def test_store_pending_entry_falls_back_to_yaml_config():
    """When signal has no exit_config, should read from YAML strategy config."""
    from collections import deque

    from src.core.domain import SymbolState
    from src.engine.execution import ExecutionEngine

    config = {
        "strategies": {
            "trend_rider_pro": {
                "enabled": True,
                "max_hold_minutes": 90,  # YAML provides this
            },
        },
        "risk": {
            "advanced_exits": {
                "trailing_stop": {"enabled": True, "trail_pct": 0.02},
            },
        },
    }
    logger = MagicMock()
    logger.bind = MagicMock(return_value=logger)

    engine = ExecutionEngine(config=config, logger=logger)
    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=100),
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=["trend_rider_pro"],
        meta={},
    )

    signal = Signal(
        symbol="AAPL",
        side=OrderSide.BUY,
        strategy="trend_rider_pro",
        entry_price=150.0,
        stop_price=148.0,
        target_price=155.0,
        size_hint=10,
        generated_at=datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc),
        meta={},  # No exit_config in signal
    )

    intent = MagicMock()
    intent.qty = 10

    engine._store_pending_entry(signal, [intent])

    pending = engine.symbol_states["AAPL"].meta["pending_entries"]
    entry = pending[signal.correlation_id]

    # Should fall back to YAML: 90min * 60 = 5400s
    assert entry["max_hold_seconds"] == 5400
    # Should fall back to global risk config
    assert entry["trailing_stop_enabled"] is True
    assert entry["trailing_stop_pct"] == pytest.approx(0.02)
