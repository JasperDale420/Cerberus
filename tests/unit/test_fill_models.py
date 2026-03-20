import pytest

from src.backtest.fill_models.fixed import FixedSlippageFillModel
from src.backtest.fill_models.protocol import FillModel, FillResult


@pytest.mark.unit
def test_fill_result_has_required_fields():
    result = FillResult(
        fill_price=150.25,
        filled_qty=100,
        commission=0.10,
        slippage_bps=2.0,
        market_impact=0.05,
    )
    assert result.fill_price == 150.25
    assert result.filled_qty == 100
    assert result.commission == 0.10
    assert result.slippage_bps == 2.0
    assert result.market_impact == 0.05


@pytest.mark.unit
def test_fill_model_is_protocol():
    """FillModel should be a runtime-checkable Protocol."""
    assert hasattr(FillModel, "compute_fill")


# --- FixedSlippageFillModel tests ---


@pytest.mark.unit
def test_fixed_fill_model_buy_slippage():
    model = FixedSlippageFillModel(slippage_bps=2.0, commission_per_share=0.001)
    result = model.compute_fill(
        order_side="buy",
        order_qty=100,
        order_price=150.00,
        order_type="market",
        bar=None,
    )
    expected_price = 150.00 * (1 + 2.0 / 10_000)
    assert result.fill_price == pytest.approx(expected_price)
    assert result.filled_qty == 100
    assert result.commission == pytest.approx(0.10)
    assert result.slippage_bps == 2.0
    assert result.market_impact == 0.0


@pytest.mark.unit
def test_fixed_fill_model_sell_slippage():
    model = FixedSlippageFillModel(slippage_bps=2.0, commission_per_share=0.001)
    result = model.compute_fill(
        order_side="sell",
        order_qty=50,
        order_price=200.00,
        order_type="market",
        bar=None,
    )
    expected_price = 200.00 * (1 - 2.0 / 10_000)
    assert result.fill_price == pytest.approx(expected_price)
    assert result.filled_qty == 50
    assert result.commission == pytest.approx(0.05)


@pytest.mark.unit
def test_fixed_fill_model_min_commission():
    model = FixedSlippageFillModel(slippage_bps=2.0, commission_per_share=0.001, min_commission=1.0)
    result = model.compute_fill(
        order_side="buy",
        order_qty=10,
        order_price=150.00,
        order_type="market",
        bar=None,
    )
    # 10 * 0.001 = 0.01, but min_commission = 1.0
    assert result.commission == pytest.approx(1.0)


@pytest.mark.unit
def test_fixed_fill_model_satisfies_protocol():
    model = FixedSlippageFillModel(slippage_bps=2.0, commission_per_share=0.001)
    assert isinstance(model, FillModel)
