from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.core.domain import SymbolFeatures
from src.scanner.validation import DataValidator


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def validator(mock_logger):
    return DataValidator(mock_logger)


@pytest.fixture
def valid_features():
    return SymbolFeatures(
        symbol="AAPL",
        last_updated=datetime.now(timezone.utc),
        price=150.0,
        avg_volume=1000000.0,
        atr_pct=0.015,
        intraday_range_pct=0.02,
        gap_pct=0.005,
        ema20_slope=0.1,
        ema_trend_strength=0.1,
        distance_from_vwap=0.01,
        premarket_volume=10000.0,
        adx=25.0,
        distance_from_ema20=0.01,
        prior_day_high=149.0,
        prior_day_low=148.0,
        bb_upper=155.0,
        bb_lower=145.0,
        price_zscore=0.5,
        flow_zscore=1.0,
        call_put_ratio=1.5,
        large_sweeps_count=5,
        aggressive_flow_share=0.6,
        extra={"flow_raw_count": 10},
    )


@pytest.mark.unit
def test_validate_technicals_success(validator, valid_features):
    # Should pass with defaults
    assert validator.validate_technicals(valid_features) is True


@pytest.mark.unit
def test_validate_technicals_price_range(validator, valid_features):
    # Too low
    valid_features.price = 9.0
    assert validator.validate_technicals(valid_features, min_price=10.0) is False

    # Too high
    valid_features.price = 1000.0
    assert validator.validate_technicals(valid_features, max_price=500.0) is False


@pytest.mark.unit
def test_validate_technicals_invalid_price(validator, valid_features):
    valid_features.price = -10.0
    assert validator.validate_technicals(valid_features) is False
    validator.logger.warning.assert_called()


@pytest.mark.unit
def test_validate_technicals_volume_filter(validator, valid_features):
    valid_features.avg_volume = 1000.0
    assert validator.validate_technicals(valid_features, min_volume=5000.0) is False


@pytest.mark.unit
def test_validate_technicals_atr_filter(validator, valid_features):
    valid_features.atr_pct = 0.05
    assert validator.validate_technicals(valid_features, max_atr_pct=0.04) is False

    valid_features.atr_pct = 0.001
    assert validator.validate_technicals(valid_features, min_atr_pct=0.01) is False


@pytest.mark.unit
def test_validate_technicals_exception_handling(validator):
    # Passing None should raise AttributeError internally which is caught
    assert validator.validate_technicals(None) is False


@pytest.mark.unit
def test_validate_flow(validator, valid_features):
    assert validator.validate_flow(valid_features) is True

    valid_features.extra["flow_raw_count"] = -1
    assert validator.validate_flow(valid_features) is False
