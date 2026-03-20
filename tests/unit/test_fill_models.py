import pytest

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
