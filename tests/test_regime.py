import numpy as np

from src.analysis.regime import Regime, RegimeDetector


def test_regime_initialization():
    detector = RegimeDetector()
    assert detector.get_regime() == Regime.CHOP


def test_insufficient_data():
    detector = RegimeDetector(min_bars=10)
    for i in range(5):
        detector.update(100 + i)
    assert detector.get_regime() == Regime.CHOP


def test_bull_regime():
    detector = RegimeDetector(window=20, min_bars=10, smooth_k=1)
    # Generate a strong upward trend
    prices = np.linspace(100, 110, 20)
    for p in prices:
        regime = detector.update(p)

    # With a strong linear trend, trend_score should be high and cum_ret positive
    assert regime == Regime.BULL


def test_bear_regime():
    detector = RegimeDetector(window=20, min_bars=10, smooth_k=1)
    # Generate a strong downward trend
    prices = np.linspace(100, 90, 20)
    for p in prices:
        regime = detector.update(p)

    assert regime == Regime.BEAR


def test_chop_regime():
    detector = RegimeDetector(window=20, min_bars=10, smooth_k=1)
    # Generate alternating chop
    prices = []
    base = 100
    for i in range(20):
        prices.append(base + (1 if i % 2 == 0 else -1))

    for p in prices:
        regime = detector.update(p)

    # Should be chop due to low trend score (high vol, low cum_ret)
    assert regime == Regime.CHOP


def test_smoothing():
    detector = RegimeDetector(window=20, min_bars=5, smooth_k=3)
    # Force a sequence of classifications: BULL, BULL, BEAR -> should be BULL
    # We can't easily force the internal logic without mocking, but we can test the smoothing logic if we exposed it or via careful input.
    # Instead, let's trust the logic and just verify it doesn't flip flop instantly on one data point if we had a stable history.

    # Establish BULL
    prices = np.linspace(100, 110, 20)
    for p in prices:
        detector.update(p)
    assert detector.get_regime() == Regime.BULL

    # Add one drop
    detector.update(100)
    # Should still be BULL due to smoothing (majority vote of last 3)
    assert detector.get_regime() == Regime.BULL
