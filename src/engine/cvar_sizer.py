"""CVaR (Conditional Value-at-Risk) position sizer with EVT tail estimation.

Sizes positions so that the expected loss in the worst (1-alpha)% of cases
doesn't exceed a configurable threshold. Uses Generalized Pareto Distribution
for accurate tail estimation when EVT is enabled.

Papers:
  Rockafellar & Uryasev (2000) -- "Optimization of Conditional Value-at-Risk"
  Almeida et al. (2023) -- EVT/GPD tail modeling for financial series
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.logger import StructuredLogger


@dataclass(frozen=True)
class CVaRResult:
    """Result of a CVaR computation."""

    cvar: float
    var: float
    cvar_multiplier: float
    method: str  # "historical" | "evt_gpd"
    tail_index: Optional[float]  # GPD shape parameter xi, if EVT used
    n_tail_obs: int


class CVaRSizer:
    """Position sizer based on Conditional Value-at-Risk with optional EVT/GPD tail estimation.

    Maintains a rolling window of log returns and computes CVaR at a given
    confidence level. The sizing multiplier scales positions so that the
    expected tail loss stays within ``max_acceptable_cvar``.
    """

    _MULTIPLIER_MIN = 0.1
    _MULTIPLIER_MAX = 2.0
    _RECOMPUTE_INTERVAL = 50

    def __init__(self, config: object, logger: Optional[StructuredLogger] = None) -> None:
        self._confidence_level: float = getattr(config, "confidence_level", 0.95)
        self._lookback_bars: int = getattr(config, "lookback_bars", 500)
        self._min_bars: int = getattr(config, "min_bars", 100)
        self._max_acceptable_cvar: float = getattr(config, "max_acceptable_cvar", 0.03)
        self._use_evt: bool = getattr(config, "use_evt", True)
        self._tail_threshold_pct: float = getattr(config, "tail_threshold_pct", 10.0)
        self._enabled: bool = getattr(config, "enabled", True)

        self._returns: deque[float] = deque(maxlen=self._lookback_bars)
        self._logger = logger or StructuredLogger("cvar_sizer")

        # Caching state
        self._cached_multiplier: Optional[float] = None
        self._obs_since_last_compute: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_return(self, bar_return: float) -> None:
        """Append a log return to the rolling window."""
        self._returns.append(bar_return)
        self._obs_since_last_compute += 1

    def compute_cvar(self) -> Optional[CVaRResult]:
        """Compute CVaR from the current return window.

        Returns ``None`` when insufficient data is available (fewer than
        ``min_bars`` observations).
        """
        if len(self._returns) < self._min_bars:
            return None

        returns = np.array(self._returns, dtype=np.float64)

        # Try EVT/GPD first when enabled
        if self._use_evt:
            result = self._compute_evt_gpd(returns)
            if result is not None:
                return result

        return self._compute_historical(returns)

    def get_cvar_multiplier(self) -> float:
        """Return the cached CVaR sizing multiplier, recomputing periodically.

        Returns 1.0 (neutral) when CVaR has not been computed yet.
        """
        if not self._enabled:
            return 1.0

        if self._cached_multiplier is None or self._obs_since_last_compute >= self._RECOMPUTE_INTERVAL:
            result = self.compute_cvar()
            if result is not None:
                self._cached_multiplier = result.cvar_multiplier
                self._obs_since_last_compute = 0

        return self._cached_multiplier if self._cached_multiplier is not None else 1.0

    # ------------------------------------------------------------------
    # Historical CVaR
    # ------------------------------------------------------------------

    def _compute_historical(self, returns: np.ndarray) -> CVaRResult:
        sorted_returns = np.sort(returns)  # ascending, worst first
        alpha = 1.0 - self._confidence_level
        var_idx = int(np.floor(alpha * len(sorted_returns)))
        var_idx = max(var_idx, 1)  # at least one observation
        var = float(sorted_returns[var_idx - 1])
        tail = sorted_returns[:var_idx]

        if len(tail) == 0:
            # All returns identical — no tail
            cvar = 0.0
        else:
            cvar = float(np.mean(tail))

        multiplier = self._calc_multiplier(cvar)

        return CVaRResult(
            cvar=cvar,
            var=var,
            cvar_multiplier=multiplier,
            method="historical",
            tail_index=None,
            n_tail_obs=len(tail),
        )

    # ------------------------------------------------------------------
    # EVT / GPD CVaR
    # ------------------------------------------------------------------

    def _compute_evt_gpd(self, returns: np.ndarray) -> Optional[CVaRResult]:
        """Fit a Generalized Pareto Distribution to the loss tail via method of moments."""
        n = len(returns)
        alpha = 1.0 - self._confidence_level

        # Threshold: percentile of losses (left tail)
        threshold_quantile = self._tail_threshold_pct / 100.0
        threshold = float(np.percentile(returns, threshold_quantile * 100.0))

        # Extract tail observations (returns worse than threshold)
        tail_mask = returns <= threshold
        tail_obs = returns[tail_mask]
        n_tail = len(tail_obs)

        if n_tail < 10:
            self._logger.debug("evt_gpd_insufficient_tail", n_tail=n_tail)
            return None

        # Excesses above the threshold (positive values representing magnitude of loss beyond threshold)
        excess = np.abs(tail_obs - threshold)

        mean_excess = float(np.mean(excess))
        if mean_excess <= 0.0:
            self._logger.debug("evt_gpd_zero_excess")
            return None

        var_excess = float(np.var(excess, ddof=1)) if n_tail > 1 else 0.0
        if var_excess <= 0.0:
            # Constant excesses — GPD cannot be fit
            self._logger.debug("evt_gpd_zero_variance")
            return None

        # Method-of-moments estimators for GPD(xi, sigma)
        # From GPD mean/var: mean^2/var = (1-2*xi), so xi = 0.5*(1 - ratio)
        ratio = mean_excess**2 / var_excess
        xi_hat = 0.5 * (1.0 - ratio)
        sigma_hat = 0.5 * mean_excess * (ratio + 1.0)

        if xi_hat >= 1.0:
            self._logger.warning("evt_gpd_infinite_mean", xi=xi_hat)
            return None  # infinite mean — fall back to historical

        if xi_hat > 0.5:
            self._logger.warning("evt_gpd_heavy_tail", xi=xi_hat)

        if sigma_hat <= 0.0:
            self._logger.debug("evt_gpd_bad_scale", sigma=sigma_hat)
            return None

        # GPD-based VaR and CVaR
        # P(X > u) ~ n_tail / n
        tail_fraction = n_tail / n
        if tail_fraction <= 0.0:
            return None

        # VaR at level alpha
        try:
            if abs(xi_hat) < 1e-10:
                # Exponential case (xi -> 0)
                var_gpd = threshold - sigma_hat * np.log(tail_fraction / alpha)
            else:
                var_gpd = threshold - (sigma_hat / xi_hat) * ((tail_fraction / alpha) ** xi_hat - 1.0)
        except (OverflowError, FloatingPointError):
            self._logger.debug("evt_gpd_var_overflow")
            return None

        # CVaR (Expected Shortfall) from GPD — McNeil/Frey/Embrechts formulation
        # In the returns space (losses negative): ES = VaR/(1-xi) - (sigma - xi*|u|)/(1-xi)
        # The subtraction converts from the loss-space formula to return-space.
        # Formula valid only for xi < 1 (GPD mean must exist)
        if xi_hat >= 1.0:
            self._logger.warning("evt_gpd_xi_ge_1", xi=xi_hat)
            return None
        denom = 1.0 - xi_hat
        cvar_gpd = var_gpd / denom - (sigma_hat - xi_hat * abs(threshold)) / denom

        if not np.isfinite(cvar_gpd):
            self._logger.debug("evt_gpd_nonfinite_cvar", cvar=cvar_gpd)
            return None

        multiplier = self._calc_multiplier(cvar_gpd)

        return CVaRResult(
            cvar=float(cvar_gpd),
            var=float(var_gpd),
            cvar_multiplier=multiplier,
            method="evt_gpd",
            tail_index=float(xi_hat),
            n_tail_obs=n_tail,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_multiplier(self, cvar: float) -> float:
        abs_cvar = abs(cvar)
        if abs_cvar < 1e-12:
            # Effectively zero CVaR — no tail risk detected
            mult = self._MULTIPLIER_MAX
        elif self._max_acceptable_cvar < 1e-12:
            # Zero threshold configured — treat as maximally conservative
            mult = self._MULTIPLIER_MIN
        else:
            mult = self._max_acceptable_cvar / abs_cvar

        mult = max(self._MULTIPLIER_MIN, min(self._MULTIPLIER_MAX, mult))

        if self._max_acceptable_cvar > 1e-12 and abs_cvar > 2.0 * self._max_acceptable_cvar:
            self._logger.critical(
                "cvar_exceeds_threshold",
                cvar=cvar,
                max_acceptable=self._max_acceptable_cvar,
                ratio=abs_cvar / self._max_acceptable_cvar,
            )

        return mult
