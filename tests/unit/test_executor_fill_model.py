from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.backtest.fill_models.fixed import FixedSlippageFillModel


@pytest.mark.unit
def test_executor_uses_injected_fill_model():
    from src.backtest.executor import SimulatedOrderExecutor

    mock_model = MagicMock()
    mock_model.compute_fill.return_value = SimpleNamespace(
        fill_price=150.03,
        filled_qty=100,
        commission=0.10,
        slippage_bps=2.0,
        market_impact=0.0,
    )
    executor = SimulatedOrderExecutor(
        logger=MagicMock(),
        db=None,
        risk_cfg={"slippage_bps": 2.0, "commission_per_share": 0.001},
        fill_model=mock_model,
    )
    assert executor.fill_model is mock_model


@pytest.mark.unit
def test_executor_defaults_to_fixed_model_when_none():
    from src.backtest.executor import SimulatedOrderExecutor

    executor = SimulatedOrderExecutor(
        logger=MagicMock(),
        db=None,
        risk_cfg={"slippage_bps": 2.0, "commission_per_share": 0.001},
    )
    assert isinstance(executor.fill_model, FixedSlippageFillModel)


@pytest.mark.unit
def test_executor_default_model_inherits_risk_cfg():
    from src.backtest.executor import SimulatedOrderExecutor

    executor = SimulatedOrderExecutor(
        logger=MagicMock(),
        db=None,
        risk_cfg={"slippage_bps": 5.0, "commission_per_share": 0.005, "min_commission": 1.0},
    )
    model = executor.fill_model
    assert isinstance(model, FixedSlippageFillModel)
    assert model.slippage_bps == 5.0
    assert model.commission_per_share == 0.005
    assert model.min_commission == 1.0


@pytest.mark.unit
def test_process_bar_calls_fill_model():
    from src.backtest.executor import SimulatedOrderExecutor

    mock_model = MagicMock()
    mock_model.compute_fill.return_value = SimpleNamespace(
        fill_price=100.02,
        filled_qty=50,
        commission=0.05,
        slippage_bps=2.0,
        market_impact=0.0,
    )

    account = SimpleNamespace(cash=100_000.0, positions_qty={})
    executor = SimulatedOrderExecutor(
        logger=MagicMock(),
        db=None,
        risk_cfg={},
        fill_model=mock_model,
        account=account,
    )

    # Submit a market order
    from src.core.domain import OrderIntent, OrderSide, OrderType

    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=50,
        order_type=OrderType.MARKET,
        limit_price=None,
        time_in_force="day",
        stop_loss=None,
        take_profit=None,
        strategy="test",
        correlation_id="corr-1",
    )
    executor.submit(intent)

    # Create a bar
    bar = SimpleNamespace(symbol="AAPL", open=100.0, high=101.0, low=99.0, close=100.5, volume=10000)
    executor.process_bar(bar)

    # fill_model.compute_fill should have been called
    mock_model.compute_fill.assert_called_once_with(
        order_side="buy",
        order_qty=50.0,
        order_price=100.0,
        order_type="market",
        bar=bar,
    )

    # Commission should be deducted from account
    assert account.cash == 100_000.0 - (50 * 100.02) - 0.05
