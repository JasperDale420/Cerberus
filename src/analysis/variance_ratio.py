"""Lo-MacKinlay (1988) Variance Ratio Test for mean-reversion detection.

The variance ratio test compares the variance of k-period returns to the
variance of 1-period returns. Under a random walk, VR(k) = 1. Values
significantly below 1 indicate mean-reversion; values above 1 indicate
momentum/trending behavior.

Reference:
    Lo, A.W. & MacKinlay, A.C. (1988). "Stock Market Prices Do Not Follow
    Random Walks: Evidence from a Simple Specification Test." Review of
    Financial Studies, 1(1), 41-66.

Usage::

    from src.analysis.variance_ratio import VarianceRatioCalculator, VarianceRatioResult

    vr_calc = VarianceRatioCalculator(lookback=60, period=5)
    for price in price_stream:
        result = vr_calc.update(price)
        if result is not None and result.is_mean_reverting:
            # price exhibits mean-reversion behavior
            ...
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class VarianceRatioResult:
    """Result of a Lo-MacKinlay variance ratio test.

    Attributes:
        vr: Variance ratio VR(k). < 1 = mean-reverting, > 1 = trending.
        z_score: Heteroscedasticity-robust z-statistic for VR(k) != 1.
        p_value_two_sided: Two-sided p-value for the null VR(k) = 1.
        is_mean_reverting: True if VR(k) < 1 at the configured significance level.
        is_trending: True if VR(k) > 1 at the configured significance level.
        period: The aggregation period k used for this test.
        n_observations: Number of price observations in the window.
    """

    vr: float
    z_score: float
    p_value_two_sided: float
    is_mean_reverting: bool
    is_trending: bool
    period: int
    n_observations: int


class VarianceRatioCalculator:
    """Rolling Lo-MacKinlay variance ratio test on streaming price data.

    Feeds on price observations and computes VR(k) using a rolling window.
    Uses the heteroscedasticity-robust test statistic (Lo-MacKinlay 1988,
    Equation 11) which is valid under conditional heteroscedasticity
    (GARCH-like behavior common in financial time series).

    Args:
        lookback: Number of price observations to maintain in the rolling window.
            Must be >= 2 * period + 1. Default 120 (2 hours of 1-min bars).
        period: Aggregation period k for the variance ratio. Default 5.
            VR(k) = Var(k-period returns) / (k * Var(1-period returns)).
        significance: Significance level for mean-reversion / trending
            classification. Default 0.05 (95% confidence).
        min_observations: Minimum observations before producing a result.
            Default is 2 * lookback // 3.

    Example::

        calc = VarianceRatioCalculator(lookback=120, period=5)
        result = calc.update(150.25)
        if result and result.is_mean_reverting:
            print(f"Mean-reverting: VR={result.vr:.3f}, z={result.z_score:.2f}")
    """

    def __init__(
        self,
        lookback: int = 120,
        period: int = 5,
        significance: float = 0.05,
        min_observations: int | None = None,
    ) -> None:
        if period < 2:
            raise ValueError(f"period must be >= 2, got {period}")
        if lookback < 2 * period + 1:
            raise ValueError(f"lookback must be >= 2*period+1 ({2 * period + 1}), got {lookback}")

        self._lookback = lookback
        self._period = period
        self._significance = significance
        self._min_obs = min_observations if min_observations is not None else max(2 * lookback // 3, 2 * period + 1)

        self._prices: deque[float] = deque(maxlen=lookback)

    def update(self, price: float) -> Optional[VarianceRatioResult]:
        """Feed a new price observation. Returns VarianceRatioResult or None if insufficient data."""
        self._prices.append(price)

        n = len(self._prices)
        if n < self._min_obs:
            return None

        return self._compute(list(self._prices))

    def _compute(self, prices: list[float]) -> Optional[VarianceRatioResult]:
        """Compute the Lo-MacKinlay heteroscedasticity-robust variance ratio test."""
        n = len(prices)
        k = self._period

        # Compute log returns
        log_returns = []
        for i in range(1, n):
            if prices[i] <= 0 or prices[i - 1] <= 0:
                return None
            log_returns.append(math.log(prices[i] / prices[i - 1]))

        nq = len(log_returns)
        if nq < 2 * k:
            return None

        # --- 1-period return variance ---
        mu = sum(log_returns) / nq
        sigma_1_sq = sum((r - mu) ** 2 for r in log_returns) / (nq - 1)

        if sigma_1_sq < 1e-20:
            return None

        # --- k-period return variance ---
        k_returns = []
        for i in range(k, nq + 1):
            k_ret = sum(log_returns[i - k : i])
            k_returns.append(k_ret)

        if len(k_returns) < 2:
            return None

        mu_k = sum(k_returns) / len(k_returns)
        sigma_k_sq = sum((r - mu_k) ** 2 for r in k_returns) / (len(k_returns) - 1)

        # --- Variance ratio ---
        vr = sigma_k_sq / (k * sigma_1_sq)

        # --- Heteroscedasticity-robust z-statistic (Lo-MacKinlay 1988) ---
        # Compute the asymptotic variance under heteroscedasticity
        theta_sum = 0.0
        for j in range(1, k):
            # Compute delta_j (heteroscedasticity-robust weight)
            delta_j = 0.0
            for t in range(j + 1, nq + 1):
                r_t = log_returns[t - 1] - mu
                r_t_j = log_returns[t - j - 1] - mu
                delta_j += (r_t * r_t) * (r_t_j * r_t_j)

            denom = (sum((log_returns[t - 1] - mu) ** 2 for t in range(1, nq + 1))) ** 2
            if denom < 1e-30:
                return None

            delta_j = nq * delta_j / denom

            weight = 2.0 * (k - j) / k
            theta_sum += weight * weight * delta_j

        if theta_sum <= 0:
            return None

        z_score = (vr - 1.0) / math.sqrt(theta_sum)

        # --- Two-sided p-value (standard normal approximation) ---
        p_value = 2.0 * _normal_cdf(-abs(z_score))

        # --- Classification ---
        z_critical = _normal_ppf(1.0 - self._significance / 2.0)
        is_mean_reverting = vr < 1.0 and z_score < -z_critical
        is_trending = vr > 1.0 and z_score > z_critical

        return VarianceRatioResult(
            vr=vr,
            z_score=z_score,
            p_value_two_sided=p_value,
            is_mean_reverting=is_mean_reverting,
            is_trending=is_trending,
            period=k,
            n_observations=len(self._prices),
        )

    def reset(self) -> None:
        """Clear all internal state."""
        self._prices.clear()

    @property
    def has_result(self) -> bool:
        """True if enough observations have been accumulated for a result."""
        return len(self._prices) >= self._min_obs


# ---------------------------------------------------------------------------
# Lightweight normal distribution helpers (avoid scipy dependency)
# ---------------------------------------------------------------------------


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the Abramowitz & Stegun approximation.

    Accurate to ~1e-7. Avoids scipy import for this lightweight module.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Standard normal quantile function (inverse CDF).

    Uses the Beasley-Springer-Moro algorithm for rational approximation.
    Accurate to ~1e-9 for 0.0001 < p < 0.9999.
    """
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0

    # Rational approximation coefficients
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
