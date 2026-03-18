"""Tests for GARCH forecaster and realized volatility estimators."""

from __future__ import annotations

import math
import random

import pytest

from src.quant.volatility import GARCHForecaster, GARCHResult, RealizedVolEstimator

# ---------------------------------------------------------------------------
# GARCHForecaster tests
# ---------------------------------------------------------------------------


class TestGARCHForecaster:
    """Tests for the GARCHForecaster class."""

    def test_returns_none_before_min_observations(self):
        """GARCHForecaster must return None until min_observations prices received."""
        forecaster = GARCHForecaster(min_observations=50)
        # Feed 49 prices — should always return None
        for i in range(49):
            result = forecaster.update(100.0 + i * 0.1)
        assert result is None

    def test_returns_positive_vol_after_enough_data(self):
        """After min_observations, update should return a GARCHResult with positive vol."""
        forecaster = GARCHForecaster(min_observations=50, refit_interval=1)
        random.seed(42)
        result = None
        for _i in range(60):
            price = 100.0 + random.gauss(0, 1)
            result = forecaster.update(price)
        assert result is not None
        assert isinstance(result, GARCHResult)
        assert result.conditional_vol > 0
        assert result.annualized_vol > 0
        assert result.n_observations >= 50

    def test_conditional_zscore_returns_finite_float(self):
        """conditional_zscore should return a finite float after fitting."""
        forecaster = GARCHForecaster(min_observations=50, refit_interval=1)
        random.seed(42)
        for _i in range(60):
            price = 100.0 + random.gauss(0, 1)
            forecaster.update(price)
        z = forecaster.conditional_zscore(0.02)
        assert isinstance(z, float)
        assert math.isfinite(z)

    def test_higher_vol_period_produces_higher_forecast(self):
        """A volatile price series should produce a higher forecast than a calm one."""
        random.seed(42)

        # Calm series
        calm = GARCHForecaster(min_observations=50, refit_interval=1)
        for _i in range(100):
            price = 100.0 + random.gauss(0, 0.1)
            calm.update(price)
        calm_result = calm.update(100.0)

        # Volatile series
        random.seed(99)
        volatile = GARCHForecaster(min_observations=50, refit_interval=1)
        for _i in range(100):
            price = 100.0 + random.gauss(0, 5.0)
            volatile.update(price)
        vol_result = volatile.update(100.0)

        assert calm_result is not None
        assert vol_result is not None
        assert vol_result.annualized_vol > calm_result.annualized_vol

    def test_garch_result_has_persistence(self):
        """GARCHResult.persistence should be between 0 and 1."""
        forecaster = GARCHForecaster(min_observations=50, refit_interval=1)
        random.seed(42)
        result = None
        for _i in range(80):
            price = 100.0 + random.gauss(0, 1)
            result = forecaster.update(price)
        assert result is not None
        assert 0 <= result.persistence <= 1.0


# ---------------------------------------------------------------------------
# RealizedVolEstimator tests
# ---------------------------------------------------------------------------


class TestRealizedVolEstimator:
    """Tests for the RealizedVolEstimator class."""

    def test_returns_none_before_min_observations(self):
        """RealizedVolEstimator must return None until lookback bars received."""
        est = RealizedVolEstimator(method="parkinson", lookback=20)
        for _i in range(19):
            result = est.update(high=101.0, low=99.0, open=100.0, close=100.5)
        assert result is None

    def test_parkinson_method_works(self):
        """Parkinson estimator should return positive annualized vol after lookback bars."""
        est = RealizedVolEstimator(method="parkinson", lookback=20)
        random.seed(42)
        result = None
        for _i in range(25):
            base = 100.0 + random.gauss(0, 1)
            h = base + abs(random.gauss(0, 0.5))
            lo = base - abs(random.gauss(0, 0.5))
            o = base + random.gauss(0, 0.2)
            c = base + random.gauss(0, 0.2)
            result = est.update(high=h, low=lo, open=o, close=c)
        assert result is not None
        assert result > 0
        assert math.isfinite(result)

    def test_garman_klass_method_works(self):
        """Garman-Klass estimator should return positive annualized vol."""
        est = RealizedVolEstimator(method="garman_klass", lookback=20)
        random.seed(42)
        result = None
        for _i in range(25):
            base = 100.0 + random.gauss(0, 1)
            h = base + abs(random.gauss(0, 0.5))
            lo = base - abs(random.gauss(0, 0.5))
            o = base + random.gauss(0, 0.2)
            c = base + random.gauss(0, 0.2)
            result = est.update(high=h, low=lo, open=o, close=c)
        assert result is not None
        assert result > 0
        assert math.isfinite(result)

    def test_yang_zhang_method_works(self):
        """Yang-Zhang estimator should return positive annualized vol."""
        est = RealizedVolEstimator(method="yang_zhang", lookback=20)
        random.seed(42)
        result = None
        for _i in range(25):
            base = 100.0 + random.gauss(0, 1)
            h = base + abs(random.gauss(0, 0.5))
            lo = base - abs(random.gauss(0, 0.5))
            o = base + random.gauss(0, 0.2)
            c = base + random.gauss(0, 0.2)
            result = est.update(high=h, low=lo, open=o, close=c)
        assert result is not None
        assert result > 0
        assert math.isfinite(result)

    def test_invalid_method_raises(self):
        """An invalid method name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            RealizedVolEstimator(method="invalid_method")
