from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class OUResult:
    """Result of Ornstein-Uhlenbeck parameter estimation.

    Based on Leung & Li (2015) discrete MLE for OU processes.
    """

    theta: float  # mean-reversion speed
    mu: float  # long-run mean
    sigma: float  # volatility
    half_life: float  # ln(2) / theta
    scaling_factor: float  # median_theta / current_theta, clamped [0.5, 2.0]


class OUEstimator:
    """Discrete MLE estimator for Ornstein-Uhlenbeck process parameters.

    Feeds on VWAP distance observations and estimates mean-reversion speed
    (theta). When theta is high (fast reversion), the scaling_factor < 1
    tightens entry thresholds. When theta is low (slow reversion), the
    scaling_factor > 1 widens them.

    Reference: Tang (2018) regime-switching OU.
    """

    def __init__(self, lookback: int = 60, min_observations: int = 30, dt: float = 1 / 390) -> None:
        self._lookback = lookback
        self._min_observations = min_observations
        self._dt = dt
        self._values: deque[float] = deque(maxlen=lookback)
        self._theta_history: deque[float] = deque(maxlen=1200)  # for median reference

    def update(self, vwap_distance: float) -> Optional[OUResult]:
        """Feed a new VWAP distance observation. Returns OUResult or None if insufficient data."""
        self._values.append(vwap_distance)

        if len(self._values) < self._min_observations:
            return None

        # Discrete MLE for theta
        values = list(self._values)
        n = len(values)

        x_bar = sum(values) / n

        # Compute autocovariance terms
        num = sum((values[t] - x_bar) * (values[t - 1] - x_bar) for t in range(1, n))
        den = sum((values[t - 1] - x_bar) ** 2 for t in range(1, n))

        if abs(den) < 1e-12:
            return None

        a = num / den  # autocorrelation

        # Clamp a to prevent log of non-positive
        a = max(a, 1e-6)
        a = min(a, 1.0 - 1e-6)

        theta = -math.log(a) / self._dt

        if theta <= 0:
            return None

        mu = x_bar  # long-run mean (should be ~0 for VWAP distance)

        # Sigma estimation
        residuals = [values[t] - a * values[t - 1] for t in range(1, n)]
        var_resid = sum(r**2 for r in residuals) / len(residuals)
        sigma = math.sqrt(var_resid * 2 * theta / (1 - a**2)) if (1 - a**2) > 1e-12 else 0.0

        half_life = math.log(2) / theta

        # Track theta history for median reference
        self._theta_history.append(theta)

        # Compute scaling factor
        if len(self._theta_history) >= self._min_observations:
            median_theta = sorted(self._theta_history)[len(self._theta_history) // 2]
            theta_ratio = median_theta / theta
            scaling_factor = max(0.5, min(2.0, theta_ratio))
        else:
            scaling_factor = 1.0

        return OUResult(
            theta=theta,
            mu=mu,
            sigma=sigma,
            half_life=half_life,
            scaling_factor=scaling_factor,
        )
