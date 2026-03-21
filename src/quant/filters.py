"""Kalman filter wrappers and regime-aware EWMA for signal processing."""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanMeanTracker:
    """1-D Kalman filter for tracking a scalar mean (random walk model).

    Uses filterpy with dim_x=1, dim_z=1. State transition F=[[1.0]].
    First observation initializes state directly.
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0) -> None:
        self._kf = KalmanFilter(dim_x=1, dim_z=1)
        self._kf.F = np.array([[1.0]])  # state transition
        self._kf.H = np.array([[1.0]])  # observation
        self._kf.Q = np.array([[process_noise]])
        self._kf.R = np.array([[measurement_noise]])
        self._kf.P = np.array([[1000.0]])  # large initial uncertainty
        self._initialized = False

    def update(self, measurement: float) -> float:
        """Update with a new measurement and return filtered state estimate."""
        if not self._initialized:
            self._kf.x = np.array([[measurement]])
            self._initialized = True
            return float(self._kf.x[0, 0])

        self._kf.predict()
        self._kf.update(np.array([[measurement]]))
        return float(self._kf.x[0, 0])

    @property
    def state(self) -> float:
        """Current state estimate."""
        return float(self._kf.x[0, 0])

    @property
    def uncertainty(self) -> float:
        """Current uncertainty (P[0,0])."""
        return float(self._kf.P[0, 0])


class KalmanHedgeRatio:
    """Kalman filter for tracking a dynamic hedge ratio (linear regression).

    Models y = intercept + slope * x.
    State: [intercept, slope], H = [[1.0, x]] (dynamic observation matrix).
    """

    def __init__(self, process_noise: float = 0.001, measurement_noise: float = 1.0) -> None:
        self._kf = KalmanFilter(dim_x=2, dim_z=1)
        self._kf.F = np.eye(2)  # state transition (random walk for both params)
        self._kf.H = np.array([[1.0, 0.0]])  # updated dynamically
        self._kf.Q = np.eye(2) * process_noise
        self._kf.R = np.array([[measurement_noise]])
        self._kf.P = np.eye(2) * 1000.0
        self._kf.x = np.array([[0.0], [0.0]])  # [intercept, slope]

    def update(self, x: float, y: float) -> float:
        """Update with new (x, y) pair. Returns innovation residual (prior prediction error)."""
        self._kf.predict()
        # Dynamic observation matrix: y = intercept + slope * x
        self._kf.H = np.array([[1.0, x]])
        # Compute innovation BEFORE update (prior prediction vs observation)
        predicted_y = float(self._kf.x[0, 0]) + float(self._kf.x[1, 0]) * x
        innovation = float(y - predicted_y)
        self._kf.update(np.array([[y]]))
        return innovation

    @property
    def hedge_ratio(self) -> float:
        """Current slope (hedge ratio)."""
        return float(self._kf.x[1, 0])

    @property
    def intercept(self) -> float:
        """Current intercept."""
        return float(self._kf.x[0, 0])

    @property
    def hedge_ratio_uncertainty(self) -> float:
        """Uncertainty on the hedge ratio (P[1,1])."""
        return float(self._kf.P[1, 1])


_REGIME_MULTIPLIERS: dict[str, float] = {
    "LOW": 1.5,
    "NORMAL": 1.0,
    "HIGH": 0.6,
    "SHOCK": 0.3,
}


class RegimeAwareEWMA:
    """Exponentially weighted moving average with regime-dependent smoothing.

    Regime multipliers adjust the effective span:
        LOW=1.5 (slow), NORMAL=1.0, HIGH=0.6 (fast), SHOCK=0.3 (fastest).
    Effective span = base_span * regime_multiplier.
    Alpha = 2 / (effective_span + 1).
    """

    def __init__(self, base_span: int = 20) -> None:
        self._base_span = base_span
        self._value: float | None = None

    def update(self, observation: float, vol_regime: str = "NORMAL") -> float:
        """Update EWMA with a new observation under the given volatility regime."""
        if self._value is None:
            self._value = observation
            return self._value

        multiplier = _REGIME_MULTIPLIERS.get(vol_regime, 1.0)
        effective_span = self._base_span * multiplier
        alpha = 2.0 / (effective_span + 1.0)
        self._value = self._value + alpha * (observation - self._value)
        return self._value

    @property
    def value(self) -> float | None:
        """Current EWMA value, or None if no observations yet."""
        return self._value
