"""GARCH conditional volatility and realized volatility estimators.

Provides two core volatility primitives used across Cerberus strategies:

1. **GARCHForecaster** — fits a GARCH(1,1) model on streaming prices and
   produces 1-step-ahead conditional volatility forecasts.  Refits every
   ``refit_interval`` bars to amortise the cost of MLE.

2. **RealizedVolEstimator** — computes annualized realized volatility from
   OHLC bars using Parkinson, Garman-Klass, or Yang-Zhang estimators.

Usage::

    from src.quant.volatility import GARCHForecaster, RealizedVolEstimator

    garch = GARCHForecaster(min_observations=50)
    for price in price_stream:
        result = garch.update(price)
        if result is not None:
            print(result.annualized_vol)

    rv = RealizedVolEstimator(method="garman_klass", lookback=20)
    for bar in ohlc_bars:
        vol = rv.update(bar.high, bar.low, bar.open, bar.close)
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

ANNUALIZATION_FACTOR = math.sqrt(252)

_VALID_RV_METHODS = frozenset({"parkinson", "garman_klass", "yang_zhang"})


# ---------------------------------------------------------------------------
# GARCH forecaster
# ---------------------------------------------------------------------------


@dataclass
class GARCHResult:
    """Result of a GARCH(1,1) conditional volatility forecast.

    Attributes:
        conditional_vol: 1-step-ahead sigma (decimal, not %).
        annualized_vol: conditional_vol * sqrt(252).
        persistence: alpha + beta of the fitted GARCH model.
        n_observations: Number of return observations used for fitting.
    """

    conditional_vol: float
    annualized_vol: float
    persistence: float
    n_observations: int


class GARCHForecaster:
    """Streaming GARCH(1,1) conditional volatility forecaster.

    Collects prices, computes log-returns, and periodically refits a
    GARCH(1,1) model via the ``arch`` library.  Between refits the most
    recent forecast is reused.

    Parameters:
        min_observations: Minimum number of prices before first fit.
        lookback: Maximum number of returns kept in the rolling window.
        refit_interval: Number of new bars between GARCH refits.
    """

    def __init__(
        self,
        min_observations: int = 50,
        lookback: int = 500,
        refit_interval: int = 20,
    ) -> None:
        self._min_obs = min_observations
        self._lookback = lookback
        self._refit_interval = refit_interval

        self._prices: deque[float] = deque(maxlen=lookback + 1)
        self._bars_since_fit: int = 0
        self._last_result: Optional[GARCHResult] = None

    # -- public API ---------------------------------------------------------

    def update(self, price: float) -> Optional[GARCHResult]:
        """Add a price observation and return a GARCH forecast (or None)."""
        self._prices.append(price)

        n_returns = len(self._prices) - 1
        if n_returns < self._min_obs:
            return None

        self._bars_since_fit += 1

        if self._last_result is None or self._bars_since_fit >= self._refit_interval:
            self._last_result = self._fit()
            self._bars_since_fit = 0

        return self._last_result

    def conditional_zscore(self, deviation: float) -> float:
        """Normalise *deviation* by the current conditional vol estimate.

        Returns 0.0 if no model has been fitted yet.
        """
        if self._last_result is None or self._last_result.conditional_vol == 0:
            return 0.0
        return deviation / self._last_result.conditional_vol

    # -- internals ----------------------------------------------------------

    def _fit(self) -> GARCHResult:
        """Fit GARCH(1,1) on current return series and return forecast."""
        prices = np.array(self._prices, dtype=np.float64)
        returns = np.diff(np.log(prices)) * 100  # percentage log-returns

        try:
            return self._fit_garch(returns)
        except Exception:
            # Fallback to rolling std when GARCH fails on degenerate data
            return self._fallback_rolling_std(returns)

    def _fit_garch(self, returns: np.ndarray) -> GARCHResult:
        """Run arch_model MLE and extract 1-step forecast."""
        from arch import arch_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(returns, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
            res = am.fit(disp="off", show_warning=False)

        # 1-step-ahead variance forecast
        forecast = res.forecast(horizon=1)
        cond_var = forecast.variance.iloc[-1, 0]
        cond_vol = math.sqrt(max(cond_var, 0.0)) / 100  # back to decimal

        alpha = float(res.params.get("alpha[1]", 0.0))
        beta = float(res.params.get("beta[1]", 0.0))
        persistence = alpha + beta

        return GARCHResult(
            conditional_vol=cond_vol,
            annualized_vol=cond_vol * ANNUALIZATION_FACTOR,
            persistence=min(persistence, 1.0),
            n_observations=len(returns),
        )

    def _fallback_rolling_std(self, returns: np.ndarray) -> GARCHResult:
        """Simple rolling-std fallback when GARCH MLE fails."""
        std = float(np.std(returns[-self._min_obs :]))
        cond_vol = std / 100  # percentage → decimal

        return GARCHResult(
            conditional_vol=cond_vol,
            annualized_vol=cond_vol * ANNUALIZATION_FACTOR,
            persistence=0.0,
            n_observations=len(returns),
        )


# ---------------------------------------------------------------------------
# Realized volatility estimator
# ---------------------------------------------------------------------------


class RealizedVolEstimator:
    """Rolling realized volatility from OHLC bars.

    Supports three estimators that are more efficient than close-to-close:

    - **parkinson** — uses high-low range (Parkinson 1980).
    - **garman_klass** — uses OHLC (Garman & Klass 1980).
    - **yang_zhang** — drift-adjusted, uses overnight + OHLC (Yang & Zhang 2000).

    All methods return annualized volatility (multiplied by sqrt(252)).

    Parameters:
        method: One of ``"parkinson"``, ``"garman_klass"``, ``"yang_zhang"``.
        lookback: Number of bars in the rolling window.
    """

    def __init__(self, method: str = "garman_klass", lookback: int = 20) -> None:
        if method not in _VALID_RV_METHODS:
            raise ValueError(f"Unknown method {method!r}. Choose from {sorted(_VALID_RV_METHODS)}.")
        self._method = method
        self._lookback = lookback

        self._highs: deque[float] = deque(maxlen=lookback)
        self._lows: deque[float] = deque(maxlen=lookback)
        self._opens: deque[float] = deque(maxlen=lookback)
        self._closes: deque[float] = deque(maxlen=lookback)

    def update(
        self,
        high: float,
        low: float,
        open: float,  # noqa: A002
        close: float,
    ) -> Optional[float]:
        """Add an OHLC bar and return annualized realized vol (or None)."""
        self._highs.append(high)
        self._lows.append(low)
        self._opens.append(open)
        self._closes.append(close)

        if len(self._highs) < self._lookback:
            return None

        if self._method == "parkinson":
            return self._parkinson()
        elif self._method == "garman_klass":
            return self._garman_klass()
        else:
            return self._yang_zhang()

    # -- estimators ---------------------------------------------------------

    def _parkinson(self) -> float:
        """Parkinson (1980) high-low range estimator."""
        n = len(self._highs)
        s = 0.0
        for i in range(n):
            hl = math.log(self._highs[i] / self._lows[i])
            s += hl * hl
        variance = s / (4.0 * n * math.log(2))
        return math.sqrt(variance) * ANNUALIZATION_FACTOR

    def _garman_klass(self) -> float:
        """Garman-Klass (1980) OHLC estimator."""
        n = len(self._highs)
        s = 0.0
        for i in range(n):
            hl = math.log(self._highs[i] / self._lows[i])
            co = math.log(self._closes[i] / self._opens[i])
            s += 0.5 * hl * hl - (2.0 * math.log(2) - 1.0) * co * co
        variance = s / n
        return math.sqrt(max(variance, 0.0)) * ANNUALIZATION_FACTOR

    def _yang_zhang(self) -> float:
        """Yang-Zhang (2000) drift-adjusted OHLC estimator."""
        n = len(self._highs)
        highs = list(self._highs)
        lows = list(self._lows)
        opens = list(self._opens)
        closes = list(self._closes)

        # Overnight returns (use previous close as proxy; first bar uses open)
        log_oc = [math.log(opens[i] / closes[i - 1]) if i > 0 else 0.0 for i in range(n)]
        # Close-to-open of same bar
        log_co = [math.log(closes[i] / opens[i]) for i in range(n)]

        mean_oc = sum(log_oc) / n
        mean_co = sum(log_co) / n

        # Overnight variance
        sigma_o_sq = sum((r - mean_oc) ** 2 for r in log_oc) / (n - 1) if n > 1 else 0.0
        # Close variance
        sigma_c_sq = sum((r - mean_co) ** 2 for r in log_co) / (n - 1) if n > 1 else 0.0

        # Rogers-Satchell variance
        rs_sum = 0.0
        for i in range(n):
            hi = math.log(highs[i] / opens[i])
            lo = math.log(lows[i] / opens[i])
            co = math.log(closes[i] / opens[i])
            rs_sum += hi * (hi - co) + lo * (lo - co)
        sigma_rs_sq = rs_sum / n

        k = 0.34 / (1.34 + (n + 1) / (n - 1)) if n > 1 else 0.34

        variance = sigma_o_sq + k * sigma_c_sq + (1.0 - k) * sigma_rs_sq
        return math.sqrt(max(variance, 0.0)) * ANNUALIZATION_FACTOR
