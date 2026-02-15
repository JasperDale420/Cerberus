from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.backtest.mock_executor import BacktestOrderExecutor
from src.core.domain import Bar, OrderIntent, OrderSide, OrderType, SymbolState
from src.engine.execution import ExecutionEngine


def _build_engine_with_executor(logger: MagicMock, executor: BacktestOrderExecutor) -> ExecutionEngine:
    engine = ExecutionEngine({"risk": {}}, logger, alpaca_client=None)
    engine.order_executor = executor  # type: ignore[assignment]
    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=__import__("collections").deque(maxlen=100),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["test_strategy"],
        meta={},
    )
    return engine


@pytest.mark.unit
def test_backtest_executor_invalid_partial_fill_mode_defaults_to_none() -> None:
    logger = MagicMock()
    executor = BacktestOrderExecutor(logger, initial_cash=10000)

    executor.set_backtest_config({"partial_fill_mode": "banana"})

    assert executor._partial_fill_mode == "none"
    assert logger.warning.called
    assert logger.warning.call_args[0][0] == "Invalid partial fill mode; defaulting to none"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backtest_min_volume_fill_improves_entry_price() -> None:
    t0 = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)

    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.MARKET,
        limit_price=None,
        time_in_force="day",
        correlation_id="test_id",
        strategy="test_strategy",
        stop_loss=None,
        take_profit=None,
        meta={"created_at": t0.isoformat()},
    )

    bar_zero_volume = Bar(
        symbol="AAPL",
        time=t0 + timedelta(minutes=1),
        open=110.0,
        high=111.0,
        low=109.0,
        close=110.5,
        volume=0,
    )
    bar_liquid = Bar(
        symbol="AAPL",
        time=t0 + timedelta(minutes=2),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=500,
    )

    baseline_logger = MagicMock()
    baseline_executor = BacktestOrderExecutor(baseline_logger, initial_cash=10000)
    baseline_executor.set_backtest_config({"min_bar_volume_for_fill": 0})
    baseline_engine = _build_engine_with_executor(baseline_logger, baseline_executor)
    baseline_executor.submit(intent)
    baseline_executor.fill_pending_for_bar(baseline_engine, "AAPL", bar_zero_volume)

    improved_logger = MagicMock()
    improved_executor = BacktestOrderExecutor(improved_logger, initial_cash=10000)
    improved_executor.set_backtest_config({"min_bar_volume_for_fill": 1})
    improved_engine = _build_engine_with_executor(improved_logger, improved_executor)
    improved_executor.submit(intent)
    improved_executor.fill_pending_for_bar(improved_engine, "AAPL", bar_zero_volume)
    improved_executor.fill_pending_for_bar(improved_engine, "AAPL", bar_liquid)

    assert baseline_executor.fills[0]["fill_price"] == pytest.approx(110.0)
    assert improved_executor.fills[0]["fill_price"] == pytest.approx(100.0)
