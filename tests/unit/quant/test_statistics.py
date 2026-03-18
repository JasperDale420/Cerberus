"""Tests for quant statistical hypothesis testing primitives."""

import numpy as np

from src.quant.statistics import (
    ADFResult,
    ADFTest,
    CUSUMDetector,
    GrangerCausalityTest,
    GrangerResult,
    HurstExponent,
    HurstResult,
)

# ---------------------------------------------------------------------------
# HurstExponent
# ---------------------------------------------------------------------------


class TestHurstExponent:
    def test_returns_none_before_min_observations(self):
        h = HurstExponent(min_observations=100)
        for i in range(99):
            result = h.update(float(i))
        assert result is None

    def test_random_walk_hurst_approx_half(self):
        """A random walk should have H ~ 0.5."""
        rng = np.random.default_rng(42)
        prices = np.cumsum(rng.standard_normal(2000)) + 100.0

        h = HurstExponent(min_observations=100, lookback=500)
        result = None
        for p in prices:
            result = h.update(float(p))

        assert result is not None
        assert isinstance(result, HurstResult)
        assert 0.3 <= result.H <= 0.7, f"Random walk H={result.H}, expected ~0.5"
        assert result.n_observations == 500

    def test_mean_reverting_series_hurst_below_half(self):
        """An AR(1) with strong mean reversion should have H < 0.5."""
        rng = np.random.default_rng(123)
        series = [100.0]
        mean = 100.0
        for _ in range(2000):
            series.append(mean + 0.05 * (series[-1] - mean) + rng.standard_normal() * 0.5)

        h = HurstExponent(min_observations=100, lookback=500)
        result = None
        for p in series:
            result = h.update(float(p))

        assert result is not None
        assert result.H < 0.5
        assert result.is_mean_reverting is True
        assert result.is_trending is False

    def test_hurst_clamped_to_zero_one(self):
        """H should always be in [0, 1]."""
        rng = np.random.default_rng(99)
        prices = np.cumsum(rng.standard_normal(500)) + 100.0
        h = HurstExponent(min_observations=100, lookback=500)
        result = None
        for p in prices:
            result = h.update(float(p))
        assert result is not None
        assert 0.0 <= result.H <= 1.0


# ---------------------------------------------------------------------------
# CUSUMDetector
# ---------------------------------------------------------------------------


class TestCUSUMDetector:
    def test_returns_none_before_warmup(self):
        det = CUSUMDetector(threshold=4.0, drift=0.5)
        for i in range(9):
            result = det.update(float(i))
        assert result is None

    def test_no_breakout_in_range_bound(self):
        """Stable series should not trigger breakout."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal(200) * 0.1 + 50.0

        det = CUSUMDetector(threshold=5.0, drift=0.5)
        breakouts = []
        for v in values:
            r = det.update(float(v))
            if r is not None and r.is_breakout:
                breakouts.append(r)
        assert len(breakouts) == 0

    def test_detects_upward_breakout(self):
        """A large upward shift should trigger a positive breakout."""
        rng = np.random.default_rng(42)
        stable = rng.standard_normal(100) * 0.1 + 50.0
        shift = rng.standard_normal(50) * 0.1 + 55.0
        values = np.concatenate([stable, shift])

        det = CUSUMDetector(threshold=4.0, drift=0.5)
        breakouts = []
        for v in values:
            r = det.update(float(v))
            if r is not None and r.is_breakout:
                breakouts.append(r)

        assert len(breakouts) > 0
        assert any(b.direction == 1.0 for b in breakouts)

    def test_detects_downward_breakout(self):
        """A large downward shift should trigger a negative breakout."""
        rng = np.random.default_rng(42)
        stable = rng.standard_normal(100) * 0.1 + 50.0
        shift = rng.standard_normal(50) * 0.1 + 45.0
        values = np.concatenate([stable, shift])

        det = CUSUMDetector(threshold=4.0, drift=0.5)
        breakouts = []
        for v in values:
            r = det.update(float(v))
            if r is not None and r.is_breakout:
                breakouts.append(r)

        assert len(breakouts) > 0
        assert any(b.direction == -1.0 for b in breakouts)


# ---------------------------------------------------------------------------
# GrangerCausalityTest
# ---------------------------------------------------------------------------


class TestGrangerCausalityTest:
    def test_detects_causal_relationship(self):
        """When y is a lagged function of x, Granger test should detect causality."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for i in range(2, n):
            y[i] = 0.8 * x[i - 2] + rng.standard_normal() * 0.3

        result = GrangerCausalityTest.test(x, y, max_lag=5, significance=0.05)
        assert isinstance(result, GrangerResult)
        assert result.is_causal is True
        assert result.p_value < 0.05
        assert result.best_lag >= 1

    def test_no_causality_for_independent_series(self):
        """Two independent random series should not show Granger causality."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        y = rng.standard_normal(500)

        result = GrangerCausalityTest.test(x, y, max_lag=5, significance=0.05)
        assert result.is_causal is False
        assert result.p_value >= 0.05


# ---------------------------------------------------------------------------
# ADFTest
# ---------------------------------------------------------------------------


class TestADFTest:
    def test_stationary_series_passes(self):
        """White noise is stationary; ADF should reject unit root."""
        rng = np.random.default_rng(42)
        series = rng.standard_normal(500)

        result = ADFTest.test(series, significance=0.05)
        assert isinstance(result, ADFResult)
        assert result.is_stationary is True
        assert result.p_value < 0.05

    def test_random_walk_fails(self):
        """A random walk has a unit root; ADF should not reject."""
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.standard_normal(500))

        result = ADFTest.test(series, significance=0.05)
        assert result.is_stationary is False
        assert result.p_value >= 0.05

    def test_critical_values_present(self):
        rng = np.random.default_rng(42)
        series = rng.standard_normal(200)
        result = ADFTest.test(series)
        assert isinstance(result.critical_values, dict)
        assert len(result.critical_values) > 0
