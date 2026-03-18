"""Tests for Kalman filter wrappers and regime-aware EWMA."""

from src.quant.filters import KalmanHedgeRatio, KalmanMeanTracker, RegimeAwareEWMA


class TestKalmanMeanTracker:
    """Tests for KalmanMeanTracker."""

    def test_tracks_constant_mean(self):
        """After many observations of the same value, estimate converges."""
        tracker = KalmanMeanTracker(process_noise=0.01, measurement_noise=1.0)
        for _ in range(100):
            estimate = tracker.update(5.0)
        assert abs(estimate - 5.0) < 0.1
        assert abs(tracker.state - 5.0) < 0.1

    def test_adapts_to_mean_shift(self):
        """After a mean shift, tracker adapts to the new mean."""
        tracker = KalmanMeanTracker(process_noise=0.01, measurement_noise=1.0)
        for _ in range(50):
            tracker.update(5.0)
        # Shift mean to 10.0
        for _ in range(100):
            estimate = tracker.update(10.0)
        assert abs(estimate - 10.0) < 0.5

    def test_uncertainty_positive(self):
        """Uncertainty is always positive after updates."""
        tracker = KalmanMeanTracker()
        tracker.update(1.0)
        assert tracker.uncertainty > 0
        tracker.update(2.0)
        assert tracker.uncertainty > 0

    def test_first_observation_initializes_state(self):
        """First observation sets state directly."""
        tracker = KalmanMeanTracker()
        result = tracker.update(42.0)
        assert abs(tracker.state - 42.0) < 1e-6
        assert abs(result - 42.0) < 1e-6


class TestKalmanHedgeRatio:
    """Tests for KalmanHedgeRatio."""

    def test_tracks_linear_relationship(self):
        """Tracks y = 1.5 * x relationship."""
        khr = KalmanHedgeRatio(process_noise=0.001, measurement_noise=1.0)
        for i in range(1, 201):
            x = float(i)
            y = 1.5 * x
            khr.update(x, y)
        assert abs(khr.hedge_ratio - 1.5) < 0.1
        assert abs(khr.intercept) < 1.0

    def test_adapts_to_changing_ratio(self):
        """Adapts when the hedge ratio changes from 1.5 to 3.0."""
        khr = KalmanHedgeRatio(process_noise=0.001, measurement_noise=1.0)
        for i in range(1, 101):
            x = float(i)
            y = 1.5 * x
            khr.update(x, y)
        # Shift ratio to 3.0
        for i in range(101, 301):
            x = float(i)
            y = 3.0 * x
            khr.update(x, y)
        assert abs(khr.hedge_ratio - 3.0) < 0.3

    def test_hedge_ratio_uncertainty_positive(self):
        """Uncertainty on hedge ratio is positive."""
        khr = KalmanHedgeRatio()
        khr.update(1.0, 2.0)
        assert khr.hedge_ratio_uncertainty > 0

    def test_update_returns_spread(self):
        """update() returns the residual (spread)."""
        khr = KalmanHedgeRatio()
        spread = khr.update(1.0, 2.0)
        assert isinstance(spread, float)


class TestRegimeAwareEWMA:
    """Tests for RegimeAwareEWMA."""

    def test_shock_reacts_faster_than_low(self):
        """SHOCK regime reacts faster to a jump than LOW regime."""
        ewma_shock = RegimeAwareEWMA(base_span=20)
        ewma_low = RegimeAwareEWMA(base_span=20)

        # Feed same baseline
        for _ in range(50):
            ewma_shock.update(10.0, vol_regime="NORMAL")
            ewma_low.update(10.0, vol_regime="NORMAL")

        # Now feed a jump with different regimes
        for _ in range(10):
            val_shock = ewma_shock.update(20.0, vol_regime="SHOCK")
            val_low = ewma_low.update(20.0, vol_regime="LOW")

        # SHOCK should be closer to 20 than LOW
        assert val_shock > val_low

    def test_value_property_none_before_update(self):
        """value is None before any observations."""
        ewma = RegimeAwareEWMA()
        assert ewma.value is None

    def test_value_property_set_after_update(self):
        """value is set after first update."""
        ewma = RegimeAwareEWMA()
        ewma.update(5.0)
        assert ewma.value is not None
        assert abs(ewma.value - 5.0) < 1e-6

    def test_normal_regime_default(self):
        """Default regime is NORMAL."""
        ewma = RegimeAwareEWMA(base_span=20)
        ewma.update(10.0)
        val = ewma.update(20.0)
        # alpha = 2/(20+1) ≈ 0.0952
        expected_alpha = 2.0 / 21.0
        expected = 10.0 + expected_alpha * (20.0 - 10.0)
        assert abs(val - expected) < 1e-6
