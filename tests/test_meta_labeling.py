from datetime import datetime, timezone

from src.core.domain import Signal, TechnicalFeatures
from src.engine.execution import ExecutionEngine


class MockLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def test_meta_labeler_vetting():
    logger = MockLogger()
    engine = ExecutionEngine(config={}, logger=logger)

    # Create a feature snapshot that is bearish (negative TFI, trending down-ish)
    bearish_features = TechnicalFeatures(
        price=100.0,
        volume=1000,
        timestamp=datetime.now(timezone.utc),
        atr=1.0,
        atr_pct=0.01,
        intraday_range_pct=0.02,
        gap_pct=0.0,
        ema20_slope=-0.1,
        distance_from_vwap=-0.01,
        adx=30.0,
        distance_from_ema20=-0.01,
        prior_day_high=105.0,
        prior_day_low=95.0,
        bb_upper=102.0,
        bb_lower=98.0,
        price_zscore=-1.0,
        premarket_volume=500,
        last_updated=datetime.now(timezone.utc),
        tfi=-0.5,  # Heavy selling
        hurst_exponent=0.6,  # Trending
        net_gex=-2000000,  # Negative GEX pressure
    )

    # Signal: Long (buy)
    signal = Signal(
        symbol="AAPL",
        side="buy",
        size_hint=100,
        entry_price=100.0,
        stop_price=98.0,
        target_price=105.0,
        strategy="test_strat",
        generated_at=datetime.now(timezone.utc),
        feature_snapshot=bearish_features,
    )

    # Meta-labeler should reject a LONG in a bearish context
    assert engine.meta_labeler.vet_signal(signal) is False


def test_meta_labeler_allowance():
    logger = MockLogger()
    engine = ExecutionEngine(config={}, logger=logger)

    # Create a feature snapshot that is bullish
    bullish_features = TechnicalFeatures(
        price=100.0,
        volume=1000,
        timestamp=datetime.now(timezone.utc),
        atr=1.0,
        atr_pct=0.01,
        intraday_range_pct=0.02,
        gap_pct=0.0,
        ema20_slope=0.1,
        distance_from_vwap=0.01,
        adx=30.0,
        distance_from_ema20=0.01,
        prior_day_high=105.0,
        prior_day_low=95.0,
        bb_upper=102.0,
        bb_lower=98.0,
        price_zscore=1.0,
        premarket_volume=500,
        last_updated=datetime.now(timezone.utc),
        tfi=0.5,  # Heavy buying
        hurst_exponent=0.6,  # Trending
        net_gex=1000000,  # Positive GEX
    )

    # Signal: Long (buy)
    signal = Signal(
        symbol="AAPL",
        side="buy",
        size_hint=100,
        entry_price=100.0,
        stop_price=98.0,
        target_price=105.0,
        strategy="test_strat",
        generated_at=datetime.now(timezone.utc),
        feature_snapshot=bullish_features,
    )

    # Meta-labeler should allow
    assert engine.meta_labeler.vet_signal(signal) is True
