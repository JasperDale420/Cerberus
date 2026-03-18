"""Statistical hypothesis testing primitives for quantitative trading."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# ---------------------------------------------------------------------------
# Hurst Exponent (R/S method)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HurstResult:
    """Result of Hurst exponent estimation."""

    H: float
    is_mean_reverting: bool
    is_trending: bool
    n_observations: int


class HurstExponent:
    """Streaming Hurst exponent estimator using rescaled range (R/S) analysis."""

    def __init__(self, min_observations: int = 100, lookback: int = 500) -> None:
        self._min_observations = min_observations
        self._lookback = lookback
        self._prices: deque[float] = deque(maxlen=lookback)

    def update(self, price: float) -> HurstResult | None:
        self._prices.append(price)
        if len(self._prices) < self._min_observations:
            return None
        return self._compute()

    def _compute(self) -> HurstResult:
        prices = np.array(self._prices, dtype=np.float64)
        n = len(prices)

        # R/S analysis operates on returns (differences), not raw prices.
        series = np.diff(prices)
        m = len(series)

        # Sub-period sizes for R/S analysis
        sizes = []
        s = 10
        while s <= m // 2:
            sizes.append(s)
            s = int(s * 1.5)
            if s == sizes[-1]:
                s += 1

        if len(sizes) < 2:
            return HurstResult(H=0.5, is_mean_reverting=False, is_trending=False, n_observations=n)

        log_sizes = []
        log_rs = []

        for size in sizes:
            n_chunks = m // size
            if n_chunks < 1:
                continue
            rs_values = []
            for i in range(n_chunks):
                chunk = series[i * size : (i + 1) * size]
                mean = np.mean(chunk)
                deviations = chunk - mean
                cumulative = np.cumsum(deviations)
                r = np.max(cumulative) - np.min(cumulative)
                s = np.std(chunk, ddof=1)
                if s > 1e-12:
                    rs_values.append(r / s)
            if rs_values:
                log_sizes.append(np.log(size))
                log_rs.append(np.log(np.mean(rs_values)))

        if len(log_sizes) < 2:
            return HurstResult(H=0.5, is_mean_reverting=False, is_trending=False, n_observations=n)

        # Log-log regression: log(R/S) = H * log(n) + c
        coeffs = np.polyfit(log_sizes, log_rs, 1)
        h = float(np.clip(coeffs[0], 0.0, 1.0))

        return HurstResult(
            H=h,
            is_mean_reverting=h < 0.5,
            is_trending=h > 0.5,
            n_observations=n,
        )


# ---------------------------------------------------------------------------
# CUSUM Breakout Detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CUSUMResult:
    """Result of CUSUM change-point detection."""

    is_breakout: bool
    direction: float
    cusum_pos: float
    cusum_neg: float
    threshold: float


class CUSUMDetector:
    """Online CUSUM change-point detector with rolling normalisation."""

    _WARMUP = 10
    _REFERENCE_WINDOW = 50

    def __init__(self, threshold: float = 4.0, drift: float = 0.5) -> None:
        self._threshold = threshold
        self._drift = drift
        self._values: list[float] = []
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0

    def update(self, value: float) -> CUSUMResult | None:
        self._values.append(value)
        if len(self._values) < self._WARMUP:
            return None

        # Use a fixed reference window to avoid the mean tracking the shift.
        ref = np.array(self._values[: self._REFERENCE_WINDOW], dtype=np.float64)
        mean = float(np.mean(ref))
        std = float(np.std(ref, ddof=1))
        if std < 1e-12:
            return CUSUMResult(
                is_breakout=False,
                direction=0.0,
                cusum_pos=0.0,
                cusum_neg=0.0,
                threshold=self._threshold,
            )

        z = (value - mean) / std

        self._cusum_pos = max(0.0, self._cusum_pos + z - self._drift)
        self._cusum_neg = max(0.0, self._cusum_neg - z - self._drift)

        is_breakout = False
        direction = 0.0

        if self._cusum_pos > self._threshold:
            is_breakout = True
            direction = 1.0
            self._cusum_pos = 0.0
        elif self._cusum_neg > self._threshold:
            is_breakout = True
            direction = -1.0
            self._cusum_neg = 0.0

        return CUSUMResult(
            is_breakout=is_breakout,
            direction=direction,
            cusum_pos=self._cusum_pos,
            cusum_neg=self._cusum_neg,
            threshold=self._threshold,
        )


# ---------------------------------------------------------------------------
# Granger Causality Test
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GrangerResult:
    """Result of Granger causality test."""

    is_causal: bool
    p_value: float
    best_lag: int
    f_statistic: float


class GrangerCausalityTest:
    """Stateless Granger causality wrapper around statsmodels."""

    @staticmethod
    def test(
        x: ArrayLike,
        y: ArrayLike,
        max_lag: int = 5,
        significance: float = 0.05,
    ) -> GrangerResult:
        """Test whether *x* Granger-causes *y*.

        Parameters
        ----------
        x, y : array-like of shape (n,)
        max_lag : maximum lag order to test
        significance : p-value threshold

        Returns
        -------
        GrangerResult
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        # statsmodels expects a 2-column array [y, x] — column 0 is the
        # "effect" and column 1 is the potential "cause".
        data = np.column_stack([y_arr, x_arr])

        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        best_lag = 1
        best_p = 1.0
        best_f = 0.0

        for lag in range(1, max_lag + 1):
            test_result = results[lag][0]
            p = test_result["ssr_ftest"][1]
            f = test_result["ssr_ftest"][0]
            if p < best_p:
                best_p = p
                best_f = f
                best_lag = lag

        return GrangerResult(
            is_causal=bool(best_p < significance),
            p_value=float(best_p),
            best_lag=best_lag,
            f_statistic=float(best_f),
        )


# ---------------------------------------------------------------------------
# Augmented Dickey-Fuller Test
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ADFResult:
    """Result of an Augmented Dickey-Fuller stationarity test."""

    is_stationary: bool
    p_value: float
    test_statistic: float
    critical_values: dict[str, float]


class ADFTest:
    """Stateless ADF stationarity test wrapper around statsmodels."""

    @staticmethod
    def test(series: ArrayLike, significance: float = 0.05) -> ADFResult:
        """Run the ADF test on *series*.

        Returns
        -------
        ADFResult
        """
        arr = np.asarray(series, dtype=np.float64)
        stat, p_value, _usedlag, _nobs, crit, _icbest = adfuller(arr, autolag="AIC")

        return ADFResult(
            is_stationary=bool(p_value < significance),
            p_value=float(p_value),
            test_statistic=float(stat),
            critical_values={k: float(v) for k, v in crit.items()},
        )
