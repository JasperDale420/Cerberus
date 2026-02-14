from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.backtest.mock_executor import BacktestOrderExecutor
from src.core.domain import Bar, OrderIntent, OrderSide, OrderType, Regime, SymbolState
from src.engine.execution import ExecutionEngine


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backtest_order_executor_fills_market_order_on_next_bar_open():
    logger = MagicMock()
    executor = BacktestOrderExecutor(logger, initial_cash=10000)
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
    order = executor.submit(intent)

    assert order is not None
    assert order["status"] == "new"
    assert executor.cash == 10000

    # Next bar should fill at open
    bar = Bar(
        symbol="AAPL",
        time=t0 + timedelta(minutes=1),
        open=150.0,
        high=151.0,
        low=149.0,
        close=150.5,
        volume=1,
    )
    executor.fill_pending_for_bar(engine, "AAPL", bar)

    assert executor.cash == pytest.approx(10000 - (10 * 150.0))
    assert len(executor.fills) == 1
    assert engine.symbol_states["AAPL"].position is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backtest_order_executor_bracket_exit_stop_priority():
    logger = MagicMock()
    executor = BacktestOrderExecutor(logger, initial_cash=100000)
    engine = ExecutionEngine({"risk": {}}, logger, alpaca_client=None)
    engine.order_executor = executor  # type: ignore[assignment]

    # Seed a position with stop/target (as if opened via engine.on_fill).
    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=__import__("collections").deque(maxlen=100),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["test_strategy"],
        meta={},
    )
    engine.market_state.regime = Regime.CHOP
    t0 = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
    engine.on_fill(
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
            "price": 100.0,
            "timestamp": t0,
            "strategy": "test_strategy",
            "correlation_id": "cid",
        }
    )
    pos = engine.symbol_states["AAPL"].position
    assert pos is not None
    pos.stop_price = 99.0
    pos.target_price = 102.0

    # Bar crosses both stop and target; stop should win.
    bar = Bar(
        symbol="AAPL",
        time=t0 + timedelta(minutes=1),
        open=100.0,
        high=103.0,
        low=98.0,
        close=101.0,
        volume=1,
    )
    executor.maybe_trigger_bracket_exit(engine, "AAPL", bar)

    assert engine.symbol_states["AAPL"].position is None
    assert len(executor.fills) == 1
    assert executor.fills[0]["kind"] == "bracket"


@pytest.mark.unit
def test_backtest_executor_invalid_modes_default_and_warn():
    logger = MagicMock()
    executor = BacktestOrderExecutor(logger, initial_cash=100000)

    executor.set_backtest_config(
        {
            "partial_fill_mode": "nonsense",
            "slippage_mode": "warp",
            "spread_mode": "bananas",
        }
    )

    assert executor._partial_fill_mode == "none"
    assert executor._slippage_mode == "fixed"
    assert executor._spread_mode == "fixed"
    assert logger.warning.call_count >= 1
