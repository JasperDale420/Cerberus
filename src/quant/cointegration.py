"""Cointegration testing and monitoring for pairs/mean-reversion strategies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CointegrationResult:
    """Result of a pairwise cointegration test."""

    is_cointegrated: bool
    p_value: float
    test_statistic: float
    hedge_ratio: float
    intercept: float
    half_life: float | None


# ---------------------------------------------------------------------------
# Engle-Granger Two-Step Test
# ---------------------------------------------------------------------------


class EngleGrangerTest:
    """Stateless Engle-Granger cointegration test with half-life estimation."""

    @staticmethod
    def test(x: np.ndarray, y: np.ndarray, significance: float = 0.05) -> CointegrationResult:
        """Run the Engle-Granger cointegration test on two price series.

        Parameters
        ----------
        x, y : np.ndarray
            Price series (same length).
        significance : float
            P-value threshold for cointegration.

        Returns
        -------
        CointegrationResult
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Statsmodels cointegration test (AEG)
        test_stat, p_value, _ = coint(x, y)

        # OLS hedge ratio: y = hedge_ratio * x + intercept
        A = np.column_stack([x, np.ones(len(x))])
        result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        hedge_ratio = float(result[0])
        intercept = float(result[1])

        # Spread and OU half-life estimation
        spread = y - hedge_ratio * x - intercept
        half_life = _estimate_half_life(spread)

        return CointegrationResult(
            is_cointegrated=float(p_value) < significance,
            p_value=float(p_value),
            test_statistic=float(test_stat),
            hedge_ratio=hedge_ratio,
            intercept=intercept,
            half_life=half_life,
        )


# ---------------------------------------------------------------------------
# Johansen Test
# ---------------------------------------------------------------------------


class JohansenTest:
    """Stateless Johansen cointegration test for multivariate series."""

    @staticmethod
    def test(series_matrix: np.ndarray, significance: float = 0.05) -> dict:
        """Run the Johansen cointegration test.

        Parameters
        ----------
        series_matrix : np.ndarray
            (T, n) matrix of n price series over T observations.
        significance : float
            Unused (95% critical values always used from Johansen output).

        Returns
        -------
        dict with keys: n_cointegrating_vectors, is_cointegrated,
             eigenvectors, eigenvalues, trace_statistics, critical_values_95
        """
        series_matrix = np.asarray(series_matrix, dtype=np.float64)

        # det_order=-1 => no deterministic terms; k_ar_diff=1 => 1 lag in differences
        result = coint_johansen(series_matrix, det_order=-1, k_ar_diff=1)

        trace_stats = result.lr1  # trace statistics
        cvt_95 = result.cvt[:, 1]  # 95% critical values (column index 1)

        n_vectors = int(np.sum(trace_stats > cvt_95))

        return {
            "n_cointegrating_vectors": n_vectors,
            "is_cointegrated": n_vectors > 0,
            "eigenvectors": result.evec,
            "eigenvalues": result.eig,
            "trace_statistics": trace_stats,
            "critical_values_95": cvt_95,
        }


# ---------------------------------------------------------------------------
# Rolling Cointegration Monitor
# ---------------------------------------------------------------------------


class RollingCointegrationMonitor:
    """Tracks a pair of series and periodically re-tests cointegration."""

    def __init__(self, lookback: int = 100, retest_interval: int = 20) -> None:
        self._lookback = lookback
        self._retest_interval = retest_interval
        self._x: deque[float] = deque(maxlen=lookback)
        self._y: deque[float] = deque(maxlen=lookback)
        self._bar_count = 0
        self._last_result: CointegrationResult | None = None

    @property
    def is_valid(self) -> bool:
        """Whether cointegration still holds (assumes valid until first test)."""
        if self._last_result is None:
            return True
        return self._last_result.is_cointegrated

    def update(self, x: float, y: float) -> CointegrationResult | None:
        """Append a new observation and optionally re-test.

        Returns the latest CointegrationResult when a re-test fires,
        otherwise None.
        """
        self._x.append(x)
        self._y.append(y)
        self._bar_count += 1

        if len(self._x) < self._lookback:
            return None

        if self._bar_count % self._retest_interval == 0:
            result = EngleGrangerTest.test(np.array(self._x), np.array(self._y))
            self._last_result = result
            return result

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_half_life(spread: np.ndarray) -> float | None:
    """Estimate OU half-life via lag regression on the spread.

    Regresses diff(spread) on spread[:-1].  theta = -slope.
    half_life = ln(2) / theta.  Returns None if theta <= 0.
    """
    spread = np.asarray(spread, dtype=np.float64)
    if len(spread) < 3:
        return None

    delta = np.diff(spread)
    lagged = spread[:-1]

    A = np.column_stack([lagged, np.ones(len(lagged))])
    result, _, _, _ = np.linalg.lstsq(A, delta, rcond=None)
    slope = result[0]

    theta = -slope
    if theta <= 0:
        return None
    return float(np.log(2) / theta)
