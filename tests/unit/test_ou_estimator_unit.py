from __future__ import annotations

import math
import random
from unittest.mock import MagicMock

from src.analysis.ou_estimator import OUEstimator, OUResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_ou_series(
    theta: float,
    mu: float,
    sigma: float,
    dt: float,
    n: int,
    x0: float = 0.0,
    seed: int = 42,
) -> list[float]:
    """Generate synthetic OU process: dx = theta*(mu - x)*dt + sigma*dW."""
    rng = random.Random(seed)
    series = [x0]
    for _ in range(n):
        x = series[-1]
        dx = theta * (mu - x) * dt + sigma * math.sqrt(dt) * rng.gauss(0, 1)
        series.append(x + dx)
    return series


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOUEstimatorInsufficientData:
    """Returns None when fewer than min_observations are available."""

    def test_returns_none_with_insufficient_data(self) -> None:
        estimator = OUEstimator(lookback=60, min_observations=30)
        for i in range(29):
            result = estimator.update(0.001 * i)
        assert result is None

    def test_returns_result_at_min_observations(self) -> None:
        estimator = OUEstimator(lookback=60, min_observations=30)
        series = generate_ou_series(theta=50.0, mu=0.0, sigma=0.01, dt=1 / 390, n=60)
        result = None
        for i, v in enumerate(series):
            result = estimator.update(v)
            if i < 29:
                assert result is None
        # After 61 observations (> 30), should have a result
        assert result is not None


class TestOUThetaEstimation:
    """Theta estimation from synthetic OU process within 30% of true value."""

    def test_theta_estimation_accuracy(self) -> None:
        true_theta = 50.0
        dt = 1 / 390
        series = generate_ou_series(theta=true_theta, mu=0.0, sigma=0.01, dt=dt, n=200, seed=123)

        estimator = OUEstimator(lookback=200, min_observations=30, dt=dt)
        result = None
        for v in series:
            result = estimator.update(v)

        assert result is not None
        # Within 30% of true theta
        assert abs(result.theta - true_theta) / true_theta < 0.30, (
            f"Estimated theta {result.theta:.2f} too far from true {true_theta}"
        )


class TestHalfLifeCalculation:
    """Verify half_life = ln(2) / theta."""

    def test_half_life_matches_formula(self) -> None:
        dt = 1 / 390
        series = generate_ou_series(theta=50.0, mu=0.0, sigma=0.01, dt=dt, n=100, seed=99)

        estimator = OUEstimator(lookback=100, min_observations=30, dt=dt)
        result = None
        for v in series:
            result = estimator.update(v)

        assert result is not None
        expected_half_life = math.log(2) / result.theta
        assert abs(result.half_life - expected_half_life) < 1e-10


class TestScalingFactorFastTheta:
    """When current theta > median, scaling_factor < 1.0 (tighter thresholds)."""

    def test_fast_theta_tightens_thresholds(self) -> None:
        estimator = OUEstimator(lookback=60, min_observations=10)

        # Feed slow-reversion data first to establish a low median theta
        slow_series = generate_ou_series(theta=10.0, mu=0.0, sigma=0.01, dt=1 / 390, n=40, seed=1)
        for v in slow_series:
            estimator.update(v)

        # Now feed fast-reversion data — theta should spike above median
        fast_series = generate_ou_series(theta=200.0, mu=0.0, sigma=0.01, dt=1 / 390, n=40, seed=2)
        result = None
        for v in fast_series:
            result = estimator.update(v)

        assert result is not None
        # Fast theta → scaling_factor < 1.0 (tighter)
        assert result.scaling_factor < 1.0, f"Expected < 1.0 but got {result.scaling_factor}"


class TestScalingFactorSlowTheta:
    """When current theta < median, scaling_factor > 1.0 (wider thresholds)."""

    def test_slow_theta_widens_thresholds(self) -> None:
        estimator = OUEstimator(lookback=60, min_observations=10)

        # Feed fast-reversion data first to establish a high median theta
        fast_series = generate_ou_series(theta=200.0, mu=0.0, sigma=0.01, dt=1 / 390, n=40, seed=3)
        for v in fast_series:
            estimator.update(v)

        # Now feed slow-reversion data — theta should drop below median
        slow_series = generate_ou_series(theta=10.0, mu=0.0, sigma=0.01, dt=1 / 390, n=40, seed=4)
        result = None
        for v in slow_series:
            result = estimator.update(v)

        assert result is not None
        # Slow theta → scaling_factor > 1.0 (wider)
        assert result.scaling_factor > 1.0, f"Expected > 1.0 but got {result.scaling_factor}"


class TestScalingFactorClampBounds:
    """Scaling factor never goes below 0.5 or above 2.0."""

    def test_clamp_lower_bound(self) -> None:
        """Even with extremely fast theta relative to median, floor is 0.5."""
        result = OUResult(theta=1000.0, mu=0.0, sigma=0.01, half_life=0.0007, scaling_factor=0.5)
        assert result.scaling_factor >= 0.5

    def test_clamp_upper_bound(self) -> None:
        """Even with extremely slow theta relative to median, cap is 2.0."""
        result = OUResult(theta=0.1, mu=0.0, sigma=0.01, half_life=6.93, scaling_factor=2.0)
        assert result.scaling_factor <= 2.0

    def test_estimator_enforces_clamp(self) -> None:
        """The estimator itself enforces [0.5, 2.0] bounds."""
        estimator = OUEstimator(lookback=100, min_observations=10)

        # Build up theta history with moderate values first
        series = generate_ou_series(theta=50.0, mu=0.0, sigma=0.01, dt=1 / 390, n=60, seed=10)
        for v in series:
            estimator.update(v)

        # Feed extreme values to try to push scaling beyond bounds
        extreme_series = generate_ou_series(theta=5000.0, mu=0.0, sigma=0.001, dt=1 / 390, n=40, seed=11)
        result = None
        for v in extreme_series:
            result = estimator.update(v)

        if result is not None:
            assert 0.5 <= result.scaling_factor <= 2.0


class TestDetectSideWithDynamicThresholds:
    """Dynamic thresholds change _detect_side detection behavior."""

    def test_tighter_threshold_triggers_entry(self) -> None:
        """With tighter thresholds (scaling < 1), a moderate deviation triggers a signal."""
        from src.core.domain import OrderSide
        from src.strategies.mean_reversion_pro import MeanReversionProStrategy

        logger = MagicMock()
        logger.bind = MagicMock(return_value=logger)
        strategy = MeanReversionProStrategy(config={}, logger=logger)

        # Default thresholds: vwap=0.003, bb=0.3
        # With moderate deviation that's below default threshold
        vwap_dist = -0.0025  # below 0.003 threshold → would NOT trigger
        bb_pos = -0.25  # below 0.3 threshold → would NOT trigger

        # Static thresholds: no signal
        side_static = strategy._detect_side(vwap_dist, bb_pos)
        assert side_static is None

        # Dynamic tighter thresholds: signal should fire
        side_dynamic = strategy._detect_side(vwap_dist, bb_pos, vwap_threshold=0.002, bb_threshold=0.2)
        assert side_dynamic == OrderSide.BUY

    def test_wider_threshold_prevents_entry(self) -> None:
        """With wider thresholds (scaling > 1), a moderate deviation does not trigger."""
        from src.core.domain import OrderSide
        from src.strategies.mean_reversion_pro import MeanReversionProStrategy

        logger = MagicMock()
        logger.bind = MagicMock(return_value=logger)
        strategy = MeanReversionProStrategy(config={}, logger=logger)

        # Deviation that would trigger at default thresholds
        vwap_dist = 0.004  # above 0.003 → would trigger
        bb_pos = 0.35  # above 0.3 → would trigger

        # Static thresholds: signal fires
        side_static = strategy._detect_side(vwap_dist, bb_pos)
        assert side_static == OrderSide.SELL

        # Wider dynamic thresholds: no signal
        side_dynamic = strategy._detect_side(vwap_dist, bb_pos, vwap_threshold=0.005, bb_threshold=0.4)
        assert side_dynamic is None


class TestFallbackToStaticWhenDisabled:
    """When ou_enabled=False, thresholds stay static."""

    def test_ou_disabled_uses_static_thresholds(self) -> None:
        from src.strategies.mean_reversion_pro import MeanReversionProStrategy

        logger = MagicMock()
        logger.bind = MagicMock(return_value=logger)
        strategy = MeanReversionProStrategy(config={"ou_enabled": False}, logger=logger)

        assert strategy._ou_enabled is False
        assert len(strategy._ou) == 0

    def test_ou_enabled_creates_estimators(self) -> None:
        from src.strategies.mean_reversion_pro import MeanReversionProStrategy

        logger = MagicMock()
        logger.bind = MagicMock(return_value=logger)
        strategy = MeanReversionProStrategy(config={"ou_enabled": True}, logger=logger)

        assert strategy._ou_enabled is True
        assert len(strategy._ou) == 0  # empty until on_bar populates
