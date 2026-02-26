import numpy as np
import pytest

from src.analysis.regime import MarketContextService, Regime
from src.core.domain import Bar, TrendRegime


def _bar(close: float, i: int = 0) -> Bar:
    from datetime import datetime, timedelta, timezone

    ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)
    return Bar(
        symbol="SPY",
        time=ts,
        open=float(close),
        high=float(close),
        low=float(close),
        close=float(close),
        volume=1.0,
    )


@pytest.mark.unit
def test_regime_initialization():
    svc = MarketContextService(min_bars=10, smooth_k=1)
    # No data yet; the service hasn't produced a snapshot.
    assert svc.get_snapshot() is None


@pytest.mark.unit
def test_insufficient_data():
    svc = MarketContextService(min_bars=10, smooth_k=1)
    for i in range(5):
        snap = svc.update(_bar(100 + i, i=i))
    assert snap.legacy_regime == Regime.CHOP
    assert snap.confidence.get("trend") == 0.0


@pytest.mark.unit
def test_bull_regime():
    svc = MarketContextService(window=20, min_bars=10, smooth_k=1)
    # Generate a strong upward trend
    prices = np.linspace(100, 110, 20)
    for i, p in enumerate(prices):
        snap = svc.update(_bar(float(p), i=i))

    # With a strong linear trend, trend_score should be high and cum_ret positive
    assert snap.legacy_regime == Regime.BULL


@pytest.mark.unit
def test_bear_regime():
    svc = MarketContextService(window=20, min_bars=10, smooth_k=1)
    # Generate a strong downward trend
    prices = np.linspace(100, 90, 20)
    for i, p in enumerate(prices):
        snap = svc.update(_bar(float(p), i=i))

    assert snap.legacy_regime == Regime.BEAR


@pytest.mark.unit
def test_chop_regime():
    svc = MarketContextService(window=20, min_bars=10, smooth_k=1)
    # Generate alternating chop
    prices = []
    base = 100
    for i in range(20):
        prices.append(base + (1 if i % 2 == 0 else -1))

    for i, p in enumerate(prices):
        snap = svc.update(_bar(float(p), i=i))

    # Should be chop due to low trend score (high vol, low cum_ret)
    assert snap.legacy_regime == Regime.CHOP


@pytest.mark.unit
def test_smoothing():
    svc = MarketContextService(window=20, min_bars=5, smooth_k=3)
    # Force a sequence of classifications: BULL, BULL, BEAR -> should be BULL
    # We can't easily force the internal logic without mocking, but we can test the smoothing logic if we exposed it or via careful input.
    # Instead, let's trust the logic and just verify it doesn't flip flop instantly on one data point if we had a stable history.

    # Establish BULL
    prices = np.linspace(100, 110, 20)
    for i, p in enumerate(prices):
        svc.update(_bar(float(p), i=i))
    assert svc.get_legacy_regime() == Regime.BULL

    # Add one drop
    snap = svc.update(_bar(100.0, i=999))
    # Trend axis should not flip instantly on one datapoint when smoothing is enabled.
    assert snap.trend == TrendRegime.UP


@pytest.mark.unit
def test_regime_determinism_golden_data():
    """
    PRD 10.1: Verify deterministic behavior against known fixed input.
    """
    # Fixed input sequence (random walk with seed for reproducibility conceptual, but here explicit)
    # 20 bars
    prices = [
        100.0,
        100.5,
        101.0,
        100.2,
        100.8,
        101.5,
        102.0,
        101.8,
        102.5,
        103.0,  # Uptrend
        102.8,
        102.5,
        102.2,
        102.0,
        101.5,  # Pullback
        101.0,
        100.5,
        100.0,
        99.5,
        99.0,  # Downtrend
    ]

    svc = MarketContextService(window=10, min_bars=5, smooth_k=1)

    results = []
    for i, p in enumerate(prices):
        snap = svc.update(_bar(p, i=i))
        results.append((snap.legacy_regime, snap.regime_tags))

    # Expected sequence logic:
    # First 4 bars: < min_bars (5) -> CHOP
    # Bar 5 (idx 4): 100.8. Window [100.0...100.8]. CumRet ~0.8%. Trend?
    # Validating exact expected output for regression.

    # We define the "Golden" expectation. If logic changes, this test should purposely break.
    expected_final_regime = Regime.CHOP

    assert results[-1][0] == expected_final_regime
    assert results[0][0] == Regime.CHOP

    # Run again, must match exactly
    svc2 = MarketContextService(window=10, min_bars=5, smooth_k=1)
    results2 = []
    for i, p in enumerate(prices):
        snap = svc2.update(_bar(p, i=i))
        results2.append((snap.legacy_regime, snap.regime_tags))

    assert results == results2, "MarketContextService must be deterministic"
