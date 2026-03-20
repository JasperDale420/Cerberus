from types import SimpleNamespace

import pytest

from src.backtest.fill_models.fixed import FixedSlippageFillModel
from src.backtest.fill_models.protocol import FillModel, FillResult
from src.backtest.fill_models.volume_aware import VolumeAwareFillModel


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


# --- VolumeAwareFillModel tests ---


@pytest.mark.unit
def test_volume_aware_low_participation():
    """Small order relative to volume — slippage close to base."""
    model = VolumeAwareFillModel(base_slippage_bps=2.0, impact_coefficient=200.0, commission_per_share=0.001)
    bar = SimpleNamespace(volume=100_000)
    result = model.compute_fill(
        order_side="buy",
        order_qty=100,
        order_price=150.00,
        order_type="market",
        bar=bar,
    )
    # participation = 100/100_000 = 0.001
    # effective_slip = 2.0 + (0.001 * 200) = 2.2 bps
    expected_price = 150.00 * (1 + 2.2 / 10_000)
    assert result.fill_price == pytest.approx(expected_price, rel=1e-6)
    assert result.slippage_bps == pytest.approx(2.2)
    assert result.market_impact == pytest.approx(0.2)


@pytest.mark.unit
def test_volume_aware_high_participation():
    """Large order relative to volume — significant additional slippage."""
    model = VolumeAwareFillModel(base_slippage_bps=2.0, impact_coefficient=200.0, commission_per_share=0.001)
    bar = SimpleNamespace(volume=1_000)
    result = model.compute_fill(
        order_side="buy",
        order_qty=100,
        order_price=150.00,
        order_type="market",
        bar=bar,
    )
    # participation = 100/1_000 = 0.10
    # effective_slip = 2.0 + (0.10 * 200) = 22.0 bps
    assert result.slippage_bps == pytest.approx(22.0)
    assert result.market_impact == pytest.approx(20.0)


@pytest.mark.unit
def test_volume_aware_zero_volume_bar_uses_max_slippage():
    """Zero volume bar — cap slippage at max_slippage_bps."""
    model = VolumeAwareFillModel(
        base_slippage_bps=2.0,
        impact_coefficient=200.0,
        commission_per_share=0.001,
        max_slippage_bps=50.0,
    )
    bar = SimpleNamespace(volume=0)
    result = model.compute_fill(
        order_side="buy",
        order_qty=100,
        order_price=150.00,
        order_type="market",
        bar=bar,
    )
    assert result.slippage_bps == pytest.approx(50.0)


@pytest.mark.unit
def test_volume_aware_sell_slippage_direction():
    model = VolumeAwareFillModel(base_slippage_bps=2.0, impact_coefficient=200.0, commission_per_share=0.001)
    bar = SimpleNamespace(volume=100_000)
    result = model.compute_fill(
        order_side="sell",
        order_qty=100,
        order_price=150.00,
        order_type="market",
        bar=bar,
    )
    assert result.fill_price < 150.00  # Sell gets worse price


@pytest.mark.unit
def test_volume_aware_satisfies_protocol():
    model = VolumeAwareFillModel()
    assert isinstance(model, FillModel)
