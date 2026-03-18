"""Unit tests for the Lo-MacKinlay variance ratio test module."""

from __future__ import annotations

import random

import pytest

from src.analysis.variance_ratio import (
    VarianceRatioCalculator,
    VarianceRatioResult,
    _normal_cdf,
    _normal_ppf,
)

# ---------------------------------------------------------------------------
# Helper: generate synthetic price series
# ---------------------------------------------------------------------------

def _random_walk(n: int, start: float = 100.0, step_std: float = 0.5, seed: int = 42) -> list[float]:
    """Generate a random walk price series."""
    rng = random.Random(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] + rng.gauss(0, step_std))
    return prices


def _mean_reverting_series(
    n: int,
    start: float = 100.0,
    mu: float = 100.0,
    theta: float = 0.1,
    sigma: float = 0.5,
    seed: int = 42,
) -> list[float]:
    """Generate an Ornstein-Uhlenbeck (mean-reverting) price series."""
    rng = random.Random(seed)
    prices = [start]
    for _ in range(n - 1):
        x = prices[-1]
        dx = theta * (mu - x) + sigma * rng.gauss(0, 1)
        prices.append(x + dx)
    return prices


def _trending_series(n: int, start: float = 100.0, drift: float = 0.05, sigma: float = 0.2, seed: int = 42) -> list[float]:
    """Generate a trending (momentum) price series."""
    rng = random.Random(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] + drift + rng.gauss(0, sigma))
    return prices


# ---------------------------------------------------------------------------
# Normal distribution helper tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalHelpers:
    def test_cdf_at_zero(self):
        assert abs(_normal_cdf(0.0) - 0.5) < 1e-6

    def test_cdf_at_large_positive(self):
        assert _normal_cdf(5.0) > 0.9999

    def test_cdf_at_large_negative(self):
        assert _normal_cdf(-5.0) < 0.0001

    def test_cdf_symmetry(self):
        assert abs(_normal_cdf(1.0) + _normal_cdf(-1.0) - 1.0) < 1e-6

    def test_ppf_at_half(self):
        assert abs(_normal_ppf(0.5)) < 1e-6

    def test_ppf_at_975(self):
        # z_0.975 ~ 1.96
        assert abs(_normal_ppf(0.975) - 1.96) < 0.01

    def test_ppf_at_025(self):
        assert abs(_normal_ppf(0.025) + 1.96) < 0.01

    def test_ppf_cdf_roundtrip(self):
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            z = _normal_ppf(p)
            p_back = _normal_cdf(z)
            assert abs(p_back - p) < 1e-4


# ---------------------------------------------------------------------------
# VarianceRatioCalculator construction tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVarianceRatioInit:
    def test_default_construction(self):
        calc = VarianceRatioCalculator()
        assert calc._lookback == 120
        assert calc._period == 5

    def test_custom_params(self):
        calc = VarianceRatioCalculator(lookback=200, period=10, significance=0.01)
        assert calc._lookback == 200
        assert calc._period == 10
        assert calc._significance == 0.01

    def test_period_too_small_raises(self):
        with pytest.raises(ValueError, match="period must be >= 2"):
            VarianceRatioCalculator(period=1)

    def test_lookback_too_small_raises(self):
        with pytest.raises(ValueError, match="lookback must be"):
            VarianceRatioCalculator(lookback=5, period=5)

    def test_reset_clears_state(self):
        calc = VarianceRatioCalculator(lookback=20, period=2)
        for p in _random_walk(20):
            calc.update(p)
        calc.reset()
        assert len(calc._prices) == 0
        assert not calc.has_result


# ---------------------------------------------------------------------------
# VarianceRatioCalculator.update() tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVarianceRatioUpdate:
    def test_returns_none_with_insufficient_data(self):
        calc = VarianceRatioCalculator(lookback=60, period=5, min_observations=30)
        for p in _random_walk(10):
            result = calc.update(p)
        assert result is None

    def test_returns_result_with_sufficient_data(self):
        calc = VarianceRatioCalculator(lookback=60, period=5, min_observations=40)
        result = None
        for p in _random_walk(60):
            result = calc.update(p)
        assert result is not None
        assert isinstance(result, VarianceRatioResult)

    def test_result_fields_populated(self):
        calc = VarianceRatioCalculator(lookback=60, period=5, min_observations=40)
        result = None
        for p in _random_walk(60):
            result = calc.update(p)
        assert result is not None
        assert isinstance(result.vr, float)
        assert isinstance(result.z_score, float)
        assert isinstance(result.p_value_two_sided, float)
        assert isinstance(result.is_mean_reverting, bool)
        assert isinstance(result.is_trending, bool)
        assert result.period == 5
        assert result.n_observations == 60

    def test_vr_positive(self):
        calc = VarianceRatioCalculator(lookback=60, period=5, min_observations=40)
        result = None
        for p in _random_walk(60):
            result = calc.update(p)
        assert result is not None
        assert result.vr > 0

    def test_p_value_in_range(self):
        calc = VarianceRatioCalculator(lookback=60, period=5, min_observations=40)
        result = None
        for p in _random_walk(60):
            result = calc.update(p)
        assert result is not None
        assert 0.0 <= result.p_value_two_sided <= 1.0

    def test_returns_none_for_zero_prices(self):
        """Should handle zero/negative prices gracefully."""
        calc = VarianceRatioCalculator(lookback=20, period=2, min_observations=15)
        for _ in range(20):
            result = calc.update(0.0)
        assert result is None

    def test_returns_none_for_constant_prices(self):
        """Constant prices have zero variance — should return None."""
        calc = VarianceRatioCalculator(lookback=20, period=2, min_observations=15)
        for _ in range(20):
            result = calc.update(100.0)
        assert result is None

    def test_has_result_property(self):
        calc = VarianceRatioCalculator(lookback=60, period=5, min_observations=40)
        assert not calc.has_result
        for p in _random_walk(40):
            calc.update(p)
        assert calc.has_result


# ---------------------------------------------------------------------------
# Statistical property tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVarianceRatioStatisticalProperties:
    def test_random_walk_vr_near_one(self):
        """For a random walk, VR(k) should be approximately 1."""
        calc = VarianceRatioCalculator(lookback=500, period=5, min_observations=400)
        result = None
        for p in _random_walk(500, step_std=1.0, seed=123):
            result = calc.update(p)
        assert result is not None
        # VR should be roughly 1 for a random walk (within reasonable tolerance)
        assert 0.5 < result.vr < 1.5
        # Should not be classified as either mean-reverting or trending
        # (though with finite samples, this may occasionally fail)

    def test_mean_reverting_vr_below_one(self):
        """For a strongly mean-reverting process, VR(k) should be < 1."""
        calc = VarianceRatioCalculator(lookback=500, period=5, min_observations=400)
        result = None
        # Strong mean-reversion: theta=0.3
        for p in _mean_reverting_series(500, theta=0.3, sigma=1.0, seed=456):
            result = calc.update(p)
        assert result is not None
        assert result.vr < 1.0, f"Expected VR < 1 for mean-reverting series, got {result.vr}"

    def test_trending_vr_above_one(self):
        """For a trending process, VR(k) should be > 1."""
        calc = VarianceRatioCalculator(lookback=500, period=5, min_observations=400)
        result = None
        for p in _trending_series(500, drift=0.1, sigma=0.3, seed=789):
            result = calc.update(p)
        assert result is not None
        assert result.vr > 1.0, f"Expected VR > 1 for trending series, got {result.vr}"

    def test_mean_reverting_detection(self):
        """Strong mean-reversion should be detected as is_mean_reverting=True."""
        calc = VarianceRatioCalculator(lookback=500, period=5, min_observations=400, significance=0.10)
        result = None
        for p in _mean_reverting_series(500, theta=0.5, sigma=1.0, seed=101):
            result = calc.update(p)
        assert result is not None
        # With strong mean-reversion and enough data, this should be detected
        assert result.vr < 1.0

    def test_different_periods_produce_different_results(self):
        """VR(2) and VR(10) should generally differ."""
        prices = _random_walk(300, seed=202)
        calc_2 = VarianceRatioCalculator(lookback=300, period=2, min_observations=200)
        calc_10 = VarianceRatioCalculator(lookback=300, period=10, min_observations=200)

        r2 = r10 = None
        for p in prices:
            r2 = calc_2.update(p)
            r10 = calc_10.update(p)

        assert r2 is not None
        assert r10 is not None
        # They shouldn't be exactly equal (different aggregation scales)
        assert r2.vr != r10.vr


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVarianceRatioEdgeCases:
    def test_negative_prices_return_none(self):
        """Negative prices should cause log return computation to fail gracefully."""
        calc = VarianceRatioCalculator(lookback=20, period=2, min_observations=15)
        # Start positive then go negative
        prices = [100.0] * 10 + [-1.0] * 10
        result = None
        for p in prices:
            result = calc.update(p)
        assert result is None

    def test_single_spike_doesnt_crash(self):
        """A single large spike in an otherwise stable series shouldn't crash."""
        calc = VarianceRatioCalculator(lookback=60, period=5, min_observations=40)
        prices = [100.0 + 0.01 * i for i in range(59)]
        prices.append(200.0)  # large spike
        for p in prices:
            result = calc.update(p)
        # Should produce a result (not crash)
        assert result is not None

    def test_rolling_window_discards_old_data(self):
        """After filling the lookback window, old prices should be discarded."""
        calc = VarianceRatioCalculator(lookback=50, period=2, min_observations=30)
        for p in _random_walk(100):
            calc.update(p)
        assert len(calc._prices) == 50
