from datetime import datetime, timezone

from src.core.domain import SymbolFeatures
from src.scanner.validation import DataValidator


def create_features(symbol="AAPL", price=100.0, volume=100000.0, atr_pct=2.0):
    return SymbolFeatures(
        symbol=symbol,
        price=price,
        avg_volume=volume,
        atr_pct=atr_pct,
        # dummies
        intraday_range_pct=0.0,
        gap_pct=0.0,
        ema20_slope=0.0,
        ema_trend_strength=0.0,
        distance_from_vwap=0.0,
        premarket_volume=0.0,
        adx=0.0,
        distance_from_ema20=0.0,
        prior_day_high=0.0,
        prior_day_low=0.0,
        bb_upper=0.0,
        bb_lower=0.0,
        price_zscore=0.0,
        flow_zscore=0.0,
        call_put_ratio=0.0,
        large_sweeps_count=0,
        aggressive_flow_share=0.0,
        last_updated=datetime.now(timezone.utc),
        extra={},
    )


def test_validate_technicals_basic():
    validator = DataValidator()
    feat = create_features(price=100.0)
    assert validator.validate_technicals(feat) is True


def test_validate_technicals_min_price():
    validator = DataValidator()
    # default min_price is 0
    assert (
        validator.validate_technicals(create_features(price=10.0), min_price=20.0)
        is False
    )
    assert (
        validator.validate_technicals(create_features(price=25.0), min_price=20.0)
        is True
    )


def test_validate_technicals_volume():
    validator = DataValidator()
    assert (
        validator.validate_technicals(create_features(volume=500), min_volume=1000)
        is False
    )
    assert (
        validator.validate_technicals(create_features(volume=1500), min_volume=1000)
        is True
    )


def test_validate_technicals_bad_data():
    validator = DataValidator()
    # Zero price or negative is invalid
    assert validator.validate_technicals(create_features(price=0.0)) is False
    assert validator.validate_technicals(create_features(price=-10.0)) is False


def test_validate_technicals_atr_filter():
    validator = DataValidator()
    # Min ATR
    assert (
        validator.validate_technicals(create_features(atr_pct=1.0), min_atr_pct=2.0)
        is False
    )
    # Max ATR
    assert (
        validator.validate_technicals(create_features(atr_pct=10.0), max_atr_pct=5.0)
        is False
    )
    # Good
    assert (
        validator.validate_technicals(
            create_features(atr_pct=3.0), min_atr_pct=2.0, max_atr_pct=5.0
        )
        is True
    )


def test_validate_flow_placeholders():
    validator = DataValidator()
    feat = create_features()
    # Default is valid
    assert validator.validate_flow(feat) is True

    # Check extra validation if implemented
    feat.extra["flow_raw_count"] = -1
    assert validator.validate_flow(feat) is False
