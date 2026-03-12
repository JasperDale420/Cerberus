"""Unit tests for Permutation Entropy / Lempel-Ziv Complexity analyzer."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.entropy import ComplexityRegime, EntropyAnalyzer, EntropyResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer() -> EntropyAnalyzer:
    """Default analyzer with window=60."""
    return EntropyAnalyzer()


@pytest.fixture
def small_analyzer() -> EntropyAnalyzer:
    """Analyzer with a small window for faster regime testing."""
    return EntropyAnalyzer(pe_order=3, pe_delay=1, window=30, zscore_window=50)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_entropy_initialization(analyzer: EntropyAnalyzer) -> None:
    """Verify default parameter values are set correctly."""
    assert analyzer.pe_order == 5
    assert analyzer.pe_delay == 1
    assert analyzer.window == 60
    assert analyzer.zscore_window == 1200


@pytest.mark.unit
def test_entropy_returns_none_until_window_full(analyzer: EntropyAnalyzer) -> None:
    """Feeding fewer than `window` values must return None."""
    np.random.seed(42)
    for _ in range(analyzer.window - 1):
        result = analyzer.update(np.random.randn())
    assert result is None

    # The window-th value should produce a result
    result = analyzer.update(np.random.randn())
    assert result is not None
    assert isinstance(result, EntropyResult)


@pytest.mark.unit
def test_pe_random_series() -> None:
    """A Gaussian random series should have high permutation entropy (>0.9)."""
    np.random.seed(42)
    analyzer = EntropyAnalyzer(pe_order=5, pe_delay=1, window=200)

    result = None
    for _ in range(200):
        result = analyzer.update(np.random.randn())

    assert result is not None
    assert result.pe_normalized > 0.9, f"PE_norm for random series was {result.pe_normalized:.4f}, expected > 0.9"


@pytest.mark.unit
def test_pe_trending_series() -> None:
    """A monotonically increasing series should have low permutation entropy (<0.3)."""
    analyzer = EntropyAnalyzer(pe_order=5, pe_delay=1, window=200)

    result = None
    for i in range(200):
        result = analyzer.update(float(i))

    assert result is not None
    assert result.pe_normalized < 0.3, f"PE_norm for trending series was {result.pe_normalized:.4f}, expected < 0.3"


@pytest.mark.unit
def test_lzc_random_series() -> None:
    """A random binary series should have high LZC (>0.7)."""
    np.random.seed(42)
    analyzer = EntropyAnalyzer(pe_order=3, pe_delay=1, window=500)

    result = None
    for _ in range(500):
        result = analyzer.update(np.random.randn())

    assert result is not None
    assert result.lzc_normalized > 0.7, f"LZC_norm for random series was {result.lzc_normalized:.4f}, expected > 0.7"


@pytest.mark.unit
def test_lzc_periodic_series() -> None:
    """A repeating 0/1 pattern (alternating +1/-1) should have low LZC (<0.3)."""
    analyzer = EntropyAnalyzer(pe_order=3, pe_delay=1, window=500)

    result = None
    for i in range(500):
        # Alternating positive/negative → binary 10101010...
        value = 1.0 if i % 2 == 0 else -1.0
        result = analyzer.update(value)

    assert result is not None
    assert result.lzc_normalized < 0.3, f"LZC_norm for periodic series was {result.lzc_normalized:.4f}, expected < 0.3"


@pytest.mark.unit
def test_complexity_regime_structured(small_analyzer: EntropyAnalyzer) -> None:
    """A regime shift from random to strongly trending should detect STRUCTURED.

    Strategy: first fill the z-score history with random data (high PE),
    then inject a strongly trending series so PE drops sharply below the mean.
    """
    np.random.seed(42)

    # Phase 1 — fill z-score history with random (high PE) values
    for _ in range(50):
        small_analyzer.update(np.random.randn())

    # Phase 2 — inject monotonic trend to produce very low PE
    result = None
    for i in range(30):
        result = small_analyzer.update(float(i) * 0.01)

    assert result is not None
    assert result.regime == ComplexityRegime.STRUCTURED, (
        f"Expected STRUCTURED regime, got {result.regime.value} (zscore={result.complexity_zscore:.2f})"
    )


@pytest.mark.unit
def test_complexity_regime_random() -> None:
    """A regime shift from trending to random noise should detect RANDOM.

    Strategy: use a larger zscore window filled entirely with trending data
    (low PE), then inject strong noise so PE jumps well above the mean.
    """
    np.random.seed(42)
    analyzer = EntropyAnalyzer(pe_order=3, pe_delay=1, window=30, zscore_window=120)

    # Phase 1 — fill z-score history with trending data (low PE)
    for i in range(120):
        analyzer.update(float(i) * 0.001)

    # Phase 2 — inject strong random noise to spike PE
    result = None
    for _ in range(30):
        result = analyzer.update(np.random.randn() * 100)

    assert result is not None
    assert result.regime == ComplexityRegime.RANDOM, (
        f"Expected RANDOM regime, got {result.regime.value} (zscore={result.complexity_zscore:.2f})"
    )


@pytest.mark.unit
def test_reset_clears_state(analyzer: EntropyAnalyzer) -> None:
    """After reset, the analyzer should require a full window before producing results."""
    np.random.seed(42)

    # Fill the window
    for _ in range(analyzer.window):
        analyzer.update(np.random.randn())

    analyzer.reset()

    # After reset, should return None until window fills again
    result = analyzer.update(0.5)
    assert result is None

    # Internal state should be empty
    assert len(analyzer._buffer) == 1
    assert len(analyzer._pe_history) == 0
