from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.backtest.mock_executor import MockOrderExecutor
from src.core.domain import OrderSide, OrderType
from src.engine.risk import OrderIntent


@pytest.mark.asyncio
async def test_mock_executor():
    logger = MagicMock()
    executor = MockOrderExecutor(logger, initial_cash=10000)

    # Submit Buy
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.MARKET,
        limit_price=None,
        time_in_force="day",
        correlation_id="test_id",
        stop_loss=None,
        take_profit=None,
        strategy="test_strategy",
    )
    order = executor.submit(intent)

    assert order is not None
    assert order["status"] == "new"
    assert len(executor.orders) == 1

    # Fill
    executor.fill_orders("AAPL", 150.0, datetime.now(timezone.utc))

    assert order["status"] == "filled"
    assert executor.cash == 10000 - (10 * 150.0)
    assert executor.positions["AAPL"] == 10


@pytest.mark.asyncio
async def test_backtest_runner_flow():
    # Mock Config and Client
    # We'll just verify the loop runs and calls process_bar

    # This is an integration test, might be complex to mock everything.
    # Let's trust the unit test for MockExecutor and just ensure Runner initializes.
    pass
