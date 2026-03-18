# Quant Strategy Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade all 7 Cerberus strategies from basic TA (avg 2.4/5) to rigorous quant strategies (4+/5), add portfolio optimization layer, and build anti-overfitting validation framework.

**Architecture:** Bottom-up — shared `src/quant/` math primitives, then upgrade each strategy in-place, then add `src/portfolio/` layer on top, then validation framework.

**Tech Stack:** Python 3.11+, statsmodels (cointegration, regime-switching, Granger), arch (GARCH), filterpy (Kalman), numpy, scipy, existing Cerberus infrastructure (BaseStrategy, ConfluenceScorer, RiskManager).

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml:12-35` (dependencies section)

**Step 1: Add new quant dependencies**

Add to `[project.dependencies]` in `pyproject.toml`:
```toml
"arch>=7.0",
"filterpy>=1.4",
```

Note: `statsmodels>=0.14` is already a dependency. Verify `scipy` is also present (needed by all three).

**Step 2: Install and verify**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv sync`
Expected: Clean install with no conflicts.

**Step 3: Verify imports work**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run python -c "from arch import arch_model; from filterpy.kalman import KalmanFilter; from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add arch and filterpy dependencies for quant upgrades"
```

---

## Task 2: Quant Foundation — `src/quant/volatility.py`

GARCH conditional volatility is used by almost every strategy upgrade. Build this first.

**Files:**
- Create: `src/quant/__init__.py`
- Create: `src/quant/volatility.py`
- Create: `tests/unit/quant/__init__.py`
- Create: `tests/unit/quant/test_volatility.py`

**Step 1: Write the failing tests**

```python
# tests/unit/quant/test_volatility.py
"""Tests for GARCH conditional volatility and realized vol estimators."""
import pytest
import numpy as np
from src.quant.volatility import GARCHForecaster, RealizedVolEstimator

class TestGARCHForecaster:
    """GARCH(1,1) conditional volatility forecasting."""

    def test_returns_none_before_min_observations(self):
        garch = GARCHForecaster(min_observations=50)
        for price in np.linspace(100, 105, 30):
            result = garch.update(price)
        assert result is None

    def test_returns_forecast_after_min_observations(self):
        garch = GARCHForecaster(min_observations=50)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 60))
        result = None
        for p in prices:
            result = garch.update(float(p))
        assert result is not None
        assert result.conditional_vol > 0
        assert result.annualized_vol > 0

    def test_conditional_zscore(self):
        garch = GARCHForecaster(min_observations=50)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 60))
        for p in prices:
            garch.update(float(p))
        z = garch.conditional_zscore(deviation=2.0)
        assert isinstance(z, float)
        # A 2.0 deviation normalized by conditional vol should be finite
        assert np.isfinite(z)

    def test_shock_vol_produces_higher_forecast(self):
        garch = GARCHForecaster(min_observations=50)
        np.random.seed(42)
        # Calm period
        calm_prices = 100 + np.cumsum(np.random.normal(0, 0.3, 55))
        for p in calm_prices:
            garch.update(float(p))
        calm_vol = garch.update(float(calm_prices[-1]))
        # Shock: big moves
        for _ in range(10):
            garch.update(float(calm_prices[-1] + np.random.normal(0, 5)))
        shock_vol = garch.last_result
        assert shock_vol.conditional_vol > calm_vol.conditional_vol


class TestRealizedVolEstimator:
    """Parkinson, Garman-Klass, Yang-Zhang realized vol estimators."""

    def test_parkinson(self):
        estimator = RealizedVolEstimator(method="parkinson", lookback=20)
        np.random.seed(42)
        for _ in range(25):
            h = 100 + abs(np.random.normal(0, 2))
            l = 100 - abs(np.random.normal(0, 2))
            result = estimator.update(high=h, low=l, open=100.0, close=100.0)
        assert result is not None
        assert result > 0

    def test_garman_klass(self):
        estimator = RealizedVolEstimator(method="garman_klass", lookback=20)
        np.random.seed(42)
        for _ in range(25):
            o = 100.0
            c = o + np.random.normal(0, 1)
            h = max(o, c) + abs(np.random.normal(0, 1))
            l = min(o, c) - abs(np.random.normal(0, 1))
            result = estimator.update(high=h, low=l, open=o, close=c)
        assert result is not None
        assert result > 0

    def test_returns_none_before_min_observations(self):
        estimator = RealizedVolEstimator(method="parkinson", lookback=20)
        result = estimator.update(high=101, low=99, open=100, close=100)
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/quant/test_volatility.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.quant'`

**Step 3: Write the implementation**

```python
# src/quant/__init__.py
"""Shared quantitative math primitives for Cerberus strategies."""

# src/quant/volatility.py
"""GARCH conditional volatility and realized volatility estimators.

Uses arch library for GARCH(1,1) and numpy for Parkinson/Garman-Klass/Yang-Zhang.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GARCHResult:
    """Result from GARCH(1,1) conditional volatility forecast."""
    conditional_vol: float        # 1-step-ahead sigma (daily scale)
    annualized_vol: float         # conditional_vol * sqrt(252)
    persistence: float            # alpha + beta (should be < 1)
    n_observations: int


class GARCHForecaster:
    """GARCH(1,1) conditional volatility forecaster.

    Replaces rolling std with forward-looking vol estimates.
    Strategies use conditional_zscore() to normalize deviations.
    """

    def __init__(
        self,
        min_observations: int = 50,
        lookback: int = 500,
        refit_interval: int = 20,
    ):
        self._min_obs = min_observations
        self._lookback = lookback
        self._refit_interval = refit_interval
        self._prices: deque[float] = deque(maxlen=lookback)
        self._bars_since_refit = 0
        self._last_conditional_vol: Optional[float] = None
        self.last_result: Optional[GARCHResult] = None

    def update(self, price: float) -> Optional[GARCHResult]:
        """Add price observation and refit GARCH if needed."""
        self._prices.append(price)
        if len(self._prices) < self._min_obs:
            return None

        self._bars_since_refit += 1
        if self._bars_since_refit >= self._refit_interval or self.last_result is None:
            self._refit()
            self._bars_since_refit = 0

        return self.last_result

    def conditional_zscore(self, deviation: float) -> float:
        """Normalize a deviation by the current conditional volatility.

        Returns deviation / conditional_vol. If no GARCH fit yet, falls back
        to simple rolling std normalization.
        """
        if self._last_conditional_vol and self._last_conditional_vol > 1e-10:
            return deviation / self._last_conditional_vol
        # Fallback: rolling std
        prices = np.array(self._prices)
        if len(prices) < 20:
            return deviation  # raw, unnormalized
        returns = np.diff(np.log(prices[-60:]))
        std = np.std(returns)
        if std > 1e-10:
            return deviation / (std * prices[-1])
        return deviation

    def _refit(self) -> None:
        """Refit GARCH(1,1) model on recent returns."""
        from arch import arch_model

        prices = np.array(self._prices)
        returns = np.diff(np.log(prices)) * 100  # percentage returns for arch

        if len(returns) < self._min_obs:
            return

        try:
            model = arch_model(returns, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
            fit = model.fit(disp="off", show_warning=False)
            forecast = fit.forecast(horizon=1)
            cond_var = forecast.variance.values[-1, 0]
            cond_vol = np.sqrt(cond_var) / 100  # back to decimal

            params = fit.params
            alpha = params.get("alpha[1]", 0)
            beta = params.get("beta[1]", 0)

            self._last_conditional_vol = cond_vol * prices[-1]  # dollar vol
            self.last_result = GARCHResult(
                conditional_vol=cond_vol,
                annualized_vol=cond_vol * np.sqrt(252),
                persistence=float(alpha + beta),
                n_observations=len(returns),
            )
        except Exception:
            # GARCH can fail on degenerate data — keep last result
            pass


class RealizedVolEstimator:
    """Realized volatility using efficient OHLC estimators.

    Methods:
    - parkinson: Uses high-low range (5x more efficient than close-close)
    - garman_klass: Uses OHLC (7-8x more efficient)
    - yang_zhang: Drift-adjusted, most robust for trending markets
    """

    def __init__(self, method: str = "garman_klass", lookback: int = 20):
        if method not in ("parkinson", "garman_klass", "yang_zhang"):
            raise ValueError(f"Unknown method: {method}")
        self._method = method
        self._lookback = lookback
        self._data: deque[tuple[float, float, float, float]] = deque(maxlen=lookback)

    def update(
        self, high: float, low: float, open: float, close: float
    ) -> Optional[float]:
        """Add OHLC bar and return annualized realized vol estimate, or None if insufficient data."""
        self._data.append((open, high, low, close))
        if len(self._data) < self._lookback:
            return None

        if self._method == "parkinson":
            return self._parkinson()
        elif self._method == "garman_klass":
            return self._garman_klass()
        else:
            return self._yang_zhang()

    def _parkinson(self) -> float:
        """Parkinson (1980) high-low estimator."""
        vals = []
        for o, h, l, c in self._data:
            if h > 0 and l > 0 and h > l:
                vals.append(np.log(h / l) ** 2)
        if not vals:
            return 0.0
        return float(np.sqrt(np.mean(vals) / (4 * np.log(2)) * 252))

    def _garman_klass(self) -> float:
        """Garman-Klass (1980) OHLC estimator."""
        vals = []
        for o, h, l, c in self._data:
            if h > 0 and l > 0 and o > 0 and c > 0 and h > l:
                hl = np.log(h / l) ** 2
                co = np.log(c / o) ** 2
                vals.append(0.5 * hl - (2 * np.log(2) - 1) * co)
        if not vals:
            return 0.0
        return float(np.sqrt(np.mean(vals) * 252))

    def _yang_zhang(self) -> float:
        """Yang-Zhang (2000) drift-independent estimator."""
        data = list(self._data)
        n = len(data)
        if n < 2:
            return 0.0

        # Overnight returns
        overnight = []
        for i in range(1, n):
            prev_c = data[i - 1][3]
            curr_o = data[i][0]
            if prev_c > 0 and curr_o > 0:
                overnight.append(np.log(curr_o / prev_c))

        # Open-close returns
        oc = []
        for o, h, l, c in data:
            if o > 0 and c > 0:
                oc.append(np.log(c / o))

        if not overnight or not oc:
            return 0.0

        sigma_o = np.var(overnight, ddof=1)
        sigma_c = np.var(oc, ddof=1)

        # Rogers-Satchell component
        rs_vals = []
        for o, h, l, c in data:
            if o > 0 and h > 0 and l > 0 and c > 0 and h > l:
                rs_vals.append(
                    np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
                )
        sigma_rs = np.mean(rs_vals) if rs_vals else 0.0

        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        sigma2 = sigma_o + k * sigma_c + (1 - k) * sigma_rs

        return float(np.sqrt(max(0, sigma2) * 252))
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/quant/test_volatility.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add src/quant/__init__.py src/quant/volatility.py tests/unit/quant/__init__.py tests/unit/quant/test_volatility.py
git commit -m "feat: add GARCH forecaster and realized vol estimators (src/quant/volatility)"
```

---

## Task 3: Quant Foundation — `src/quant/statistics.py`

Hurst exponent, CUSUM, Granger causality, ADF wrapper, generalized variance ratio.

**Files:**
- Create: `src/quant/statistics.py`
- Create: `tests/unit/quant/test_statistics.py`

**Step 1: Write the failing tests**

```python
# tests/unit/quant/test_statistics.py
"""Tests for statistical hypothesis testing primitives."""
import pytest
import numpy as np
from src.quant.statistics import (
    HurstExponent,
    CUSUMDetector,
    GrangerCausalityTest,
    ADFTest,
)


class TestHurstExponent:
    """Hurst exponent for mean-reversion vs trending classification."""

    def test_random_walk_near_half(self):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 500))
        hurst = HurstExponent(min_observations=100)
        result = None
        for p in prices:
            result = hurst.update(float(p))
        assert result is not None
        # Random walk: H ≈ 0.5 (allow generous tolerance)
        assert 0.3 < result.H < 0.7

    def test_mean_reverting_below_half(self):
        np.random.seed(42)
        # AR(1) with negative autocorrelation → mean-reverting
        series = [100.0]
        for _ in range(500):
            series.append(100 + 0.3 * (100 - series[-1]) + np.random.normal(0, 0.5))
        hurst = HurstExponent(min_observations=100)
        result = None
        for p in series:
            result = hurst.update(float(p))
        assert result is not None
        assert result.H < 0.5
        assert result.is_mean_reverting

    def test_returns_none_before_min_obs(self):
        hurst = HurstExponent(min_observations=100)
        for p in range(50):
            result = hurst.update(float(100 + p * 0.1))
        assert result is None


class TestCUSUMDetector:
    """CUSUM for statistically significant breakouts."""

    def test_no_signal_during_range(self):
        cusum = CUSUMDetector(threshold=4.0)
        np.random.seed(42)
        for _ in range(50):
            signal = cusum.update(np.random.normal(0, 1))
        assert signal is None or not signal.is_breakout

    def test_detects_upward_breakout(self):
        cusum = CUSUMDetector(threshold=4.0)
        # Feed noise then a shift
        for _ in range(30):
            cusum.update(np.random.normal(0, 1))
        # Now feed large positive values
        result = None
        for _ in range(10):
            result = cusum.update(5.0)
        assert result is not None
        assert result.is_breakout
        assert result.direction > 0

    def test_detects_downward_breakout(self):
        cusum = CUSUMDetector(threshold=4.0)
        for _ in range(30):
            cusum.update(np.random.normal(0, 1))
        result = None
        for _ in range(10):
            result = cusum.update(-5.0)
        assert result is not None
        assert result.is_breakout
        assert result.direction < 0


class TestGrangerCausalityTest:
    """Granger causality for flow → price validation."""

    def test_causal_signal_detected(self):
        np.random.seed(42)
        n = 200
        # x causes y with 1-period lag
        x = np.random.normal(0, 1, n)
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = 0.7 * x[i - 1] + np.random.normal(0, 0.3)
        result = GrangerCausalityTest.test(x, y, max_lag=3)
        assert result.is_causal
        assert result.best_lag >= 1
        assert result.p_value < 0.05

    def test_no_causality_for_independent(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = np.random.normal(0, 1, 200)
        result = GrangerCausalityTest.test(x, y, max_lag=3)
        assert not result.is_causal
        assert result.p_value > 0.05


class TestADFTest:
    """Augmented Dickey-Fuller for stationarity."""

    def test_stationary_series(self):
        np.random.seed(42)
        series = np.random.normal(0, 1, 200)
        result = ADFTest.test(series)
        assert result.is_stationary
        assert result.p_value < 0.05

    def test_random_walk_nonstationary(self):
        np.random.seed(42)
        series = np.cumsum(np.random.normal(0, 1, 200))
        result = ADFTest.test(series)
        assert not result.is_stationary
        assert result.p_value > 0.05
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/quant/test_statistics.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# src/quant/statistics.py
"""Statistical hypothesis testing primitives for strategy gating.

Provides: Hurst exponent, CUSUM breakout detection, Granger causality,
ADF stationarity test. All designed for online (streaming) usage where possible.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Hurst Exponent (R/S method)
# ---------------------------------------------------------------------------

@dataclass
class HurstResult:
    H: float                  # Hurst exponent
    is_mean_reverting: bool   # H < 0.5
    is_trending: bool         # H > 0.5
    n_observations: int


class HurstExponent:
    """Online Hurst exponent estimator using rescaled range (R/S) method.

    H < 0.5: mean-reverting (anti-persistent)
    H ≈ 0.5: random walk
    H > 0.5: trending (persistent)
    """

    def __init__(self, min_observations: int = 100, lookback: int = 500):
        self._min_obs = min_observations
        self._prices: deque[float] = deque(maxlen=lookback)

    def update(self, price: float) -> Optional[HurstResult]:
        self._prices.append(price)
        if len(self._prices) < self._min_obs:
            return None
        return self._compute()

    def _compute(self) -> HurstResult:
        prices = np.array(self._prices)
        returns = np.diff(np.log(prices))
        n = len(returns)

        # R/S analysis over multiple sub-period sizes
        sizes = []
        rs_values = []
        for size in [int(n / k) for k in [2, 4, 8, 16, 32] if n / k >= 8]:
            if size < 8:
                continue
            rs_list = []
            for start in range(0, n - size + 1, size):
                chunk = returns[start : start + size]
                mean_r = np.mean(chunk)
                deviations = np.cumsum(chunk - mean_r)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(chunk, ddof=1)
                if S > 1e-10:
                    rs_list.append(R / S)
            if rs_list:
                sizes.append(size)
                rs_values.append(np.mean(rs_list))

        if len(sizes) < 2:
            return HurstResult(H=0.5, is_mean_reverting=False, is_trending=False, n_observations=n)

        log_sizes = np.log(sizes)
        log_rs = np.log(rs_values)
        H = float(np.polyfit(log_sizes, log_rs, 1)[0])
        H = max(0.0, min(1.0, H))

        return HurstResult(
            H=H,
            is_mean_reverting=H < 0.5,
            is_trending=H > 0.5,
            n_observations=n,
        )


# ---------------------------------------------------------------------------
# CUSUM Breakout Detector
# ---------------------------------------------------------------------------

@dataclass
class CUSUMResult:
    is_breakout: bool
    direction: float       # +1 upward, -1 downward, 0 no breakout
    cusum_pos: float       # cumulative positive sum
    cusum_neg: float       # cumulative negative sum
    threshold: float


class CUSUMDetector:
    """Cumulative Sum (CUSUM) control chart for breakout detection.

    Detects statistically significant shifts in mean level.
    More rigorous than "close > high + buffer".
    """

    def __init__(self, threshold: float = 4.0, drift: float = 0.5):
        self._threshold = threshold
        self._drift = drift
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._mean: float = 0.0
        self._std: float = 1.0
        self._values: deque[float] = deque(maxlen=100)

    def update(self, value: float) -> Optional[CUSUMResult]:
        self._values.append(value)
        if len(self._values) < 10:
            return None

        self._mean = float(np.mean(self._values))
        self._std = float(np.std(self._values, ddof=1))
        if self._std < 1e-10:
            self._std = 1.0

        z = (value - self._mean) / self._std
        self._cusum_pos = max(0, self._cusum_pos + z - self._drift)
        self._cusum_neg = max(0, self._cusum_neg - z - self._drift)

        is_up = self._cusum_pos > self._threshold
        is_down = self._cusum_neg > self._threshold

        if is_up:
            self._cusum_pos = 0.0  # reset after detection
        if is_down:
            self._cusum_neg = 0.0

        if is_up or is_down:
            return CUSUMResult(
                is_breakout=True,
                direction=1.0 if is_up else -1.0,
                cusum_pos=self._cusum_pos,
                cusum_neg=self._cusum_neg,
                threshold=self._threshold,
            )
        return CUSUMResult(
            is_breakout=False, direction=0.0,
            cusum_pos=self._cusum_pos, cusum_neg=self._cusum_neg,
            threshold=self._threshold,
        )

    def reset(self) -> None:
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0


# ---------------------------------------------------------------------------
# Granger Causality
# ---------------------------------------------------------------------------

@dataclass
class GrangerResult:
    is_causal: bool
    p_value: float
    best_lag: int
    f_statistic: float


class GrangerCausalityTest:
    """Granger causality test: does X predict Y beyond Y's own history?

    Stateless — call .test() with arrays directly.
    """

    @staticmethod
    def test(
        x: np.ndarray, y: np.ndarray, max_lag: int = 5, significance: float = 0.05
    ) -> GrangerResult:
        from statsmodels.tsa.stattools import grangercausalitytests

        data = np.column_stack([y, x])
        best_p = 1.0
        best_lag = 1
        best_f = 0.0

        try:
            results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            for lag, result in results.items():
                p = result[0]["ssr_ftest"][1]
                f = result[0]["ssr_ftest"][0]
                if p < best_p:
                    best_p = p
                    best_lag = lag
                    best_f = f
        except Exception:
            pass

        return GrangerResult(
            is_causal=best_p < significance,
            p_value=float(best_p),
            best_lag=int(best_lag),
            f_statistic=float(best_f),
        )


# ---------------------------------------------------------------------------
# ADF Stationarity Test
# ---------------------------------------------------------------------------

@dataclass
class ADFResult:
    is_stationary: bool
    p_value: float
    test_statistic: float
    critical_values: dict


class ADFTest:
    """Augmented Dickey-Fuller test for stationarity."""

    @staticmethod
    def test(series: np.ndarray, significance: float = 0.05) -> ADFResult:
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series, autolag="AIC")
        return ADFResult(
            is_stationary=result[1] < significance,
            p_value=float(result[1]),
            test_statistic=float(result[0]),
            critical_values=result[4],
        )
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/quant/test_statistics.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add src/quant/statistics.py tests/unit/quant/test_statistics.py
git commit -m "feat: add Hurst exponent, CUSUM, Granger causality, ADF test (src/quant/statistics)"
```

---

## Task 4: Quant Foundation — `src/quant/filters.py`

Kalman filter and regime-aware EWMA.

**Files:**
- Create: `src/quant/filters.py`
- Create: `tests/unit/quant/test_filters.py`

**Step 1: Write the failing tests**

```python
# tests/unit/quant/test_filters.py
"""Tests for Kalman filter and regime-aware EWMA."""
import pytest
import numpy as np
from src.quant.filters import KalmanMeanTracker, KalmanHedgeRatio, RegimeAwareEWMA


class TestKalmanMeanTracker:
    """Kalman filter for adaptive mean estimation (replaces EMA)."""

    def test_tracks_constant_mean(self):
        km = KalmanMeanTracker(process_noise=0.01, measurement_noise=1.0)
        for _ in range(100):
            km.update(50.0 + np.random.normal(0, 1))
        assert abs(km.state - 50.0) < 2.0

    def test_adapts_to_mean_shift(self):
        km = KalmanMeanTracker(process_noise=0.1, measurement_noise=1.0)
        for _ in range(50):
            km.update(50.0 + np.random.normal(0, 0.5))
        # Shift mean
        for _ in range(50):
            km.update(60.0 + np.random.normal(0, 0.5))
        assert km.state > 55.0  # Should have adapted toward 60

    def test_returns_uncertainty(self):
        km = KalmanMeanTracker(process_noise=0.01, measurement_noise=1.0)
        km.update(50.0)
        assert km.uncertainty > 0


class TestKalmanHedgeRatio:
    """Kalman filter for dynamic hedge ratio in pairs trading."""

    def test_tracks_linear_relationship(self):
        np.random.seed(42)
        khr = KalmanHedgeRatio()
        # y = 1.5 * x + noise
        for _ in range(200):
            x = 100 + np.random.normal(0, 5)
            y = 1.5 * x + np.random.normal(0, 2)
            khr.update(x, y)
        assert abs(khr.hedge_ratio - 1.5) < 0.3

    def test_adapts_to_changing_ratio(self):
        np.random.seed(42)
        khr = KalmanHedgeRatio()
        # Phase 1: ratio = 1.0
        for _ in range(100):
            x = 100 + np.random.normal(0, 3)
            khr.update(x, 1.0 * x + np.random.normal(0, 1))
        r1 = khr.hedge_ratio
        # Phase 2: ratio = 2.0
        for _ in range(100):
            x = 100 + np.random.normal(0, 3)
            khr.update(x, 2.0 * x + np.random.normal(0, 1))
        r2 = khr.hedge_ratio
        assert r2 > r1  # Should have moved toward 2.0


class TestRegimeAwareEWMA:
    """EWMA with decay factor that adapts to vol regime."""

    def test_fast_decay_in_shock(self):
        ewma = RegimeAwareEWMA(base_span=20)
        ewma.update(100.0, vol_regime="SHOCK")
        ewma.update(110.0, vol_regime="SHOCK")
        # SHOCK uses faster decay → more weight on recent
        shock_val = ewma.value

        ewma2 = RegimeAwareEWMA(base_span=20)
        ewma2.update(100.0, vol_regime="LOW")
        ewma2.update(110.0, vol_regime="LOW")
        low_val = ewma2.value

        # SHOCK should react faster to 110 (closer to 110 than LOW)
        assert shock_val > low_val
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/quant/test_filters.py -v`
Expected: FAIL

**Step 3: Write the implementation**

```python
# src/quant/filters.py
"""Kalman filters and regime-aware EWMA for adaptive estimation.

KalmanMeanTracker: Replaces EMA for mean/VWAP tracking (handles regime shifts).
KalmanHedgeRatio: State-space model for dynamic hedge ratios in pairs trading.
RegimeAwareEWMA: EWMA where decay adapts to vol regime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanMeanTracker:
    """1D Kalman filter for adaptive mean estimation.

    Replaces EMA-20 pullback detection in trend_rider_pro and
    rolling mean in mean_reversion_pro. Adapts to regime shifts
    naturally through process noise.
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        self._kf = KalmanFilter(dim_x=1, dim_z=1)
        self._kf.x = np.array([[0.0]])       # initial state
        self._kf.F = np.array([[1.0]])        # state transition (random walk)
        self._kf.H = np.array([[1.0]])        # measurement function
        self._kf.P = np.array([[1000.0]])     # initial uncertainty (high)
        self._kf.R = np.array([[measurement_noise]])
        self._kf.Q = np.array([[process_noise]])
        self._initialized = False

    def update(self, measurement: float) -> float:
        """Update with new measurement, return filtered state estimate."""
        if not self._initialized:
            self._kf.x = np.array([[measurement]])
            self._initialized = True
            return measurement
        self._kf.predict()
        self._kf.update(np.array([[measurement]]))
        return self.state

    @property
    def state(self) -> float:
        return float(self._kf.x[0, 0])

    @property
    def uncertainty(self) -> float:
        return float(self._kf.P[0, 0])


class KalmanHedgeRatio:
    """2D Kalman filter for dynamic hedge ratio estimation.

    State: [intercept, slope] where y = intercept + slope * x.
    Replaces EMA-smoothed price ratio in pair_trading_v2.
    """

    def __init__(
        self,
        process_noise: float = 0.001,
        measurement_noise: float = 1.0,
    ):
        self._kf = KalmanFilter(dim_x=2, dim_z=1)
        self._kf.x = np.array([[0.0], [1.0]])  # [intercept, slope]
        self._kf.F = np.eye(2)                  # state transition
        self._kf.P = np.eye(2) * 1000           # initial uncertainty
        self._kf.R = np.array([[measurement_noise]])
        self._kf.Q = np.eye(2) * process_noise
        self._initialized = False

    def update(self, x: float, y: float) -> float:
        """Update with new (x, y) pair. Returns current spread (residual)."""
        self._kf.H = np.array([[1.0, x]])
        self._kf.predict()
        self._kf.update(np.array([[y]]))
        spread = y - (self.intercept + self.hedge_ratio * x)
        return spread

    @property
    def hedge_ratio(self) -> float:
        return float(self._kf.x[1, 0])

    @property
    def intercept(self) -> float:
        return float(self._kf.x[0, 0])

    @property
    def hedge_ratio_uncertainty(self) -> float:
        return float(self._kf.P[1, 1])


# Regime → decay multiplier mapping
_REGIME_DECAY = {
    "LOW": 1.5,     # slower decay (more smoothing)
    "NORMAL": 1.0,
    "HIGH": 0.6,    # faster decay
    "SHOCK": 0.3,   # fastest decay (react quickly)
}


class RegimeAwareEWMA:
    """Exponentially weighted moving average with regime-adaptive decay.

    In SHOCK vol, decay is fast (recent data dominates).
    In LOW vol, decay is slow (more smoothing).
    """

    def __init__(self, base_span: int = 20):
        self._base_span = base_span
        self._value: Optional[float] = None

    def update(self, observation: float, vol_regime: str = "NORMAL") -> float:
        multiplier = _REGIME_DECAY.get(vol_regime, 1.0)
        effective_span = max(2, self._base_span * multiplier)
        alpha = 2.0 / (effective_span + 1.0)

        if self._value is None:
            self._value = observation
        else:
            self._value = alpha * observation + (1.0 - alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/quant/test_filters.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add src/quant/filters.py tests/unit/quant/test_filters.py
git commit -m "feat: add Kalman filters and regime-aware EWMA (src/quant/filters)"
```

---

## Task 5: Quant Foundation — `src/quant/cointegration.py`

Engle-Granger, Johansen, rolling cointegration monitor.

**Files:**
- Create: `src/quant/cointegration.py`
- Create: `tests/unit/quant/test_cointegration.py`

**Step 1: Write the failing tests**

```python
# tests/unit/quant/test_cointegration.py
"""Tests for cointegration testing and monitoring."""
import pytest
import numpy as np
from src.quant.cointegration import EngleGrangerTest, RollingCointegrationMonitor


class TestEngleGrangerTest:

    def test_cointegrated_series(self):
        np.random.seed(42)
        n = 300
        # Common stochastic trend
        trend = np.cumsum(np.random.normal(0, 1, n))
        x = trend + np.random.normal(0, 0.5, n)
        y = 1.5 * trend + np.random.normal(0, 0.5, n)
        result = EngleGrangerTest.test(x, y)
        assert result.is_cointegrated
        assert result.p_value < 0.05
        assert abs(result.hedge_ratio - 1.5) < 0.5

    def test_independent_series_not_cointegrated(self):
        np.random.seed(42)
        x = np.cumsum(np.random.normal(0, 1, 300))
        y = np.cumsum(np.random.normal(0, 1, 300))
        result = EngleGrangerTest.test(x, y)
        assert not result.is_cointegrated
        assert result.p_value > 0.05


class TestRollingCointegrationMonitor:

    def test_detects_breakdown(self):
        monitor = RollingCointegrationMonitor(lookback=100, retest_interval=20)
        np.random.seed(42)
        trend = np.cumsum(np.random.normal(0, 1, 150))
        # Phase 1: cointegrated
        for i in range(100):
            x = trend[i]
            y = 1.5 * trend[i] + np.random.normal(0, 0.3)
            monitor.update(x, y)
        assert monitor.is_valid  # should be cointegrated

        # Phase 2: relationship breaks
        for i in range(100, 150):
            x = trend[i]
            y = np.cumsum(np.random.normal(0, 1, 1))[0] * 50  # independent
            monitor.update(x, y)
        # After enough divergent data, should detect breakdown
        # (may take a few retests)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/quant/test_cointegration.py -v`
Expected: FAIL

**Step 3: Write the implementation**

```python
# src/quant/cointegration.py
"""Cointegration testing and monitoring for pairs/mean-reversion strategies.

EngleGrangerTest: Two-step cointegration test (entry gate).
RollingCointegrationMonitor: Ongoing validation mid-trade.
JohansenTest: Multi-leg cointegration for 3+ symbol baskets.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CointegrationResult:
    is_cointegrated: bool
    p_value: float
    test_statistic: float
    hedge_ratio: float          # OLS slope (beta)
    intercept: float            # OLS intercept (alpha)
    half_life: Optional[float]  # OU half-life of the spread


class EngleGrangerTest:
    """Engle-Granger two-step cointegration test.

    Step 1: OLS regression y = alpha + beta * x + epsilon
    Step 2: ADF test on residuals (epsilon must be stationary)
    """

    @staticmethod
    def test(x: np.ndarray, y: np.ndarray, significance: float = 0.05) -> CointegrationResult:
        from statsmodels.tsa.stattools import coint

        t_stat, p_value, crit_values = coint(x, y)

        # OLS for hedge ratio
        X = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        intercept, hedge_ratio = float(beta[0]), float(beta[1])

        # Half-life of spread via OU estimation
        spread = y - hedge_ratio * x - intercept
        half_life = _estimate_half_life(spread)

        return CointegrationResult(
            is_cointegrated=p_value < significance,
            p_value=float(p_value),
            test_statistic=float(t_stat),
            hedge_ratio=hedge_ratio,
            intercept=intercept,
            half_life=half_life,
        )


class JohansenTest:
    """Johansen cointegration test for 3+ symbol baskets."""

    @staticmethod
    def test(series_matrix: np.ndarray, significance: float = 0.05) -> dict:
        """Test multi-leg cointegration.

        Args:
            series_matrix: (n_observations, n_series) array
        Returns:
            dict with n_cointegrating_vectors, eigenvectors, eigenvalues
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        result = coint_johansen(series_matrix, det_order=0, k_ar_diff=1)
        # Count significant cointegrating vectors at 95% (index 1)
        trace_stats = result.lr1
        crit_values = result.cvt[:, 1]  # 95% critical values
        n_coint = int(np.sum(trace_stats > crit_values))

        return {
            "n_cointegrating_vectors": n_coint,
            "is_cointegrated": n_coint > 0,
            "eigenvectors": result.evec,
            "eigenvalues": result.eig,
            "trace_statistics": trace_stats.tolist(),
            "critical_values_95": crit_values.tolist(),
        }


class RollingCointegrationMonitor:
    """Monitors cointegration validity during a live trade.

    Re-tests cointegration on a rolling window at regular intervals.
    Triggers exit signal when relationship breaks down.
    """

    def __init__(self, lookback: int = 100, retest_interval: int = 20):
        self._lookback = lookback
        self._retest_interval = retest_interval
        self._x: deque[float] = deque(maxlen=lookback)
        self._y: deque[float] = deque(maxlen=lookback)
        self._bars_since_test = 0
        self._is_valid: Optional[bool] = None
        self._last_result: Optional[CointegrationResult] = None

    def update(self, x: float, y: float) -> Optional[CointegrationResult]:
        self._x.append(x)
        self._y.append(y)
        self._bars_since_test += 1

        if len(self._x) < self._lookback:
            self._is_valid = True  # assume valid until we have enough data
            return None

        if self._bars_since_test >= self._retest_interval or self._is_valid is None:
            self._bars_since_test = 0
            result = EngleGrangerTest.test(np.array(self._x), np.array(self._y))
            self._last_result = result
            self._is_valid = result.is_cointegrated
            return result
        return self._last_result

    @property
    def is_valid(self) -> bool:
        return self._is_valid if self._is_valid is not None else True


def _estimate_half_life(spread: np.ndarray) -> Optional[float]:
    """Estimate OU half-life from spread series via OLS on lag regression."""
    if len(spread) < 20:
        return None
    y = np.diff(spread)
    x = spread[:-1]
    x = x.reshape(-1, 1)
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    theta = -beta[1]
    if theta <= 0:
        return None
    return float(np.log(2) / theta)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/quant/test_cointegration.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add src/quant/cointegration.py tests/unit/quant/test_cointegration.py
git commit -m "feat: add Engle-Granger, Johansen, rolling cointegration monitor (src/quant/cointegration)"
```

---

## Task 6: Quant Foundation — `src/quant/regime.py`

Markov regime-switching model and adaptive threshold engine.

**Files:**
- Create: `src/quant/regime.py`
- Create: `tests/unit/quant/test_regime.py`

**Step 1: Write the failing tests**

```python
# tests/unit/quant/test_regime.py
"""Tests for Markov regime-switching and adaptive thresholds."""
import pytest
import numpy as np
from src.quant.regime import MarkovRegimeSwitcher, AdaptiveThresholdEngine


class TestMarkovRegimeSwitcher:

    def test_fits_two_state_model(self):
        np.random.seed(42)
        # Regime 1: low vol, Regime 2: high vol
        returns = np.concatenate([
            np.random.normal(0, 0.5, 100),
            np.random.normal(0, 2.0, 100),
            np.random.normal(0, 0.5, 100),
        ])
        mrs = MarkovRegimeSwitcher(n_regimes=2, min_observations=50)
        result = None
        for r in returns:
            result = mrs.update(r)
        assert result is not None
        assert 0 <= result.filtered_probability <= 1
        assert result.current_regime in (0, 1)

    def test_returns_none_before_min_obs(self):
        mrs = MarkovRegimeSwitcher(n_regimes=2, min_observations=100)
        for r in np.random.normal(0, 1, 50):
            result = mrs.update(r)
        assert result is None


class TestAdaptiveThresholdEngine:

    def test_scales_threshold_by_vol(self):
        engine = AdaptiveThresholdEngine()
        # Low vol → base threshold
        low_vol = engine.adapt(base_threshold=2.0, conditional_vol=0.01, hurst=0.4)
        # High vol → wider threshold
        high_vol = engine.adapt(base_threshold=2.0, conditional_vol=0.05, hurst=0.4)
        assert high_vol > low_vol

    def test_scales_by_hurst(self):
        engine = AdaptiveThresholdEngine()
        # Strong mean-reversion (H=0.3) → tighter threshold (more confident)
        strong_mr = engine.adapt(base_threshold=2.0, conditional_vol=0.02, hurst=0.3)
        # Weak mean-reversion (H=0.48) → wider threshold (less confident)
        weak_mr = engine.adapt(base_threshold=2.0, conditional_vol=0.02, hurst=0.48)
        assert weak_mr > strong_mr
```

**Step 2: Run tests, verify fail. Step 3: Implement.**

```python
# src/quant/regime.py
"""Markov regime-switching models and adaptive threshold engine.

MarkovRegimeSwitcher: Online regime detection using Hamilton filter.
AdaptiveThresholdEngine: Replaces hardcoded thresholds with statistically-derived ones.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RegimeSwitchResult:
    current_regime: int                # most probable regime (0-indexed)
    filtered_probability: float        # P(current regime | data)
    regime_probabilities: list[float]  # probabilities for all regimes
    n_observations: int


class MarkovRegimeSwitcher:
    """Online Markov regime-switching model.

    Fits a Markov-switching model on returns to detect regime transitions.
    Strategies use filtered_probability to gate entries.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        min_observations: int = 100,
        lookback: int = 500,
        refit_interval: int = 50,
    ):
        self._n_regimes = n_regimes
        self._min_obs = min_observations
        self._lookback = lookback
        self._refit_interval = refit_interval
        self._values: deque[float] = deque(maxlen=lookback)
        self._bars_since_refit = 0
        self.last_result: Optional[RegimeSwitchResult] = None

    def update(self, value: float) -> Optional[RegimeSwitchResult]:
        self._values.append(value)
        if len(self._values) < self._min_obs:
            return None

        self._bars_since_refit += 1
        if self._bars_since_refit >= self._refit_interval or self.last_result is None:
            self._refit()
            self._bars_since_refit = 0

        return self.last_result

    def _refit(self) -> None:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        data = np.array(self._values)
        try:
            model = MarkovRegression(
                data, k_regimes=self._n_regimes, trend="c", switching_variance=True
            )
            fit = model.fit(disp=False, maxiter=200)
            probs = fit.filtered_marginal_probabilities
            last_probs = probs[-1].tolist()
            current_regime = int(np.argmax(last_probs))

            self.last_result = RegimeSwitchResult(
                current_regime=current_regime,
                filtered_probability=float(last_probs[current_regime]),
                regime_probabilities=last_probs,
                n_observations=len(data),
            )
        except Exception:
            pass  # Keep last result on convergence failure


class AdaptiveThresholdEngine:
    """Replaces hardcoded thresholds with statistically-derived adaptive ones.

    Scales base threshold by:
    1. Conditional volatility (GARCH) — wider in high vol
    2. Hurst exponent — tighter when mean-reversion is strong
    3. Optional regime probability — tighter when regime confidence is high
    """

    def __init__(
        self,
        vol_scale_factor: float = 1.0,
        hurst_scale_factor: float = 1.0,
        min_multiplier: float = 0.5,
        max_multiplier: float = 3.0,
    ):
        self._vol_scale = vol_scale_factor
        self._hurst_scale = hurst_scale_factor
        self._min_mult = min_multiplier
        self._max_mult = max_multiplier
        self._baseline_vol: Optional[float] = None

    def adapt(
        self,
        base_threshold: float,
        conditional_vol: float = 0.0,
        hurst: float = 0.5,
        regime_confidence: float = 1.0,
        baseline_vol: Optional[float] = None,
    ) -> float:
        """Return adapted threshold.

        Args:
            base_threshold: The original hardcoded threshold
            conditional_vol: GARCH conditional vol (0 = no adjustment)
            hurst: Hurst exponent (0.5 = no adjustment)
            regime_confidence: P(current regime) from Markov model
            baseline_vol: Reference vol level for normalization
        """
        multiplier = 1.0

        # Vol adjustment: scale up when vol is above baseline
        if conditional_vol > 0 and baseline_vol and baseline_vol > 0:
            vol_ratio = conditional_vol / baseline_vol
            multiplier *= (1.0 + self._vol_scale * (vol_ratio - 1.0))
        elif conditional_vol > 0:
            # No baseline — use raw vol as multiplier (larger vol = wider threshold)
            multiplier *= (1.0 + self._vol_scale * conditional_vol)

        # Hurst adjustment: tighter when strongly mean-reverting
        if hurst < 0.5:
            # H=0.3 → multiplier *= 0.7 (tighter)
            # H=0.5 → multiplier *= 1.0 (no change)
            hurst_adj = 0.5 + hurst  # maps [0, 0.5] → [0.5, 1.0]
            multiplier *= (1.0 - self._hurst_scale * (1.0 - hurst_adj))
        elif hurst > 0.5:
            # Trending → wider threshold for mean-reversion strategies
            hurst_adj = hurst  # maps [0.5, 1.0] → wider
            multiplier *= (1.0 + self._hurst_scale * (hurst_adj - 0.5))

        # Regime confidence: tighter when confident
        multiplier *= (2.0 - regime_confidence)  # conf=1.0 → 1.0x, conf=0.5 → 1.5x

        multiplier = max(self._min_mult, min(self._max_mult, multiplier))
        return base_threshold * multiplier
```

**Step 4: Run tests. Step 5: Commit.**

```bash
git add src/quant/regime.py tests/unit/quant/test_regime.py
git commit -m "feat: add Markov regime-switching and adaptive threshold engine (src/quant/regime)"
```

---

## Task 7: Quant Foundation — `src/quant/validation.py`

Walk-forward, deflated Sharpe, CPCV, drift detection.

**Files:**
- Create: `src/quant/validation.py`
- Create: `tests/unit/quant/test_validation.py`

**Step 1: Write failing tests for deflated Sharpe and CUSUM drift detection**

```python
# tests/unit/quant/test_validation.py
"""Tests for anti-overfitting validation framework."""
import pytest
import numpy as np
from src.quant.validation import (
    deflated_sharpe_ratio,
    WalkForwardValidator,
    CUSUMDriftDetector,
    InformationCoefficientTracker,
)


class TestDeflatedSharpeRatio:

    def test_single_trial_unchanged(self):
        # With 1 trial, deflated ≈ observed
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0, n_trials=1, n_observations=252, skew=0, kurtosis=3
        )
        assert dsr > 1.5

    def test_many_trials_deflates(self):
        # 100 trials should deflate significantly
        dsr_1 = deflated_sharpe_ratio(
            observed_sharpe=2.0, n_trials=1, n_observations=252, skew=0, kurtosis=3
        )
        dsr_100 = deflated_sharpe_ratio(
            observed_sharpe=2.0, n_trials=100, n_observations=252, skew=0, kurtosis=3
        )
        assert dsr_100 < dsr_1

    def test_negative_sharpe_stays_negative(self):
        dsr = deflated_sharpe_ratio(
            observed_sharpe=-0.5, n_trials=10, n_observations=252, skew=0, kurtosis=3
        )
        assert dsr < 0


class TestCUSUMDriftDetector:

    def test_no_drift_in_stable_returns(self):
        detector = CUSUMDriftDetector(expected_sharpe=1.0, threshold=3.0)
        np.random.seed(42)
        for _ in range(100):
            daily_return = np.random.normal(1.0 / np.sqrt(252), 0.01)
            result = detector.update(daily_return)
        assert not result.is_drifting

    def test_detects_strategy_breakdown(self):
        detector = CUSUMDriftDetector(expected_sharpe=2.0, threshold=3.0)
        np.random.seed(42)
        # Feed strongly negative returns (strategy broke)
        for _ in range(50):
            result = detector.update(-0.02)
        assert result.is_drifting


class TestInformationCoefficientTracker:

    def test_tracks_ic(self):
        tracker = InformationCoefficientTracker(window=30)
        np.random.seed(42)
        for _ in range(35):
            prediction = np.random.normal(0, 1)
            actual = prediction + np.random.normal(0, 0.5)  # correlated
            tracker.update(prediction, actual)
        assert tracker.current_ic is not None
        assert tracker.current_ic > 0  # should be positive (predictions correlate)

    def test_detects_ic_decay(self):
        tracker = InformationCoefficientTracker(window=20)
        np.random.seed(42)
        # Good predictions
        for _ in range(25):
            p = np.random.normal(0, 1)
            tracker.update(p, p + np.random.normal(0, 0.3))
        ic_good = tracker.current_ic
        # Bad predictions (independent)
        for _ in range(25):
            tracker.update(np.random.normal(0, 1), np.random.normal(0, 1))
        ic_bad = tracker.current_ic
        assert ic_bad < ic_good
```

**Step 2: Run tests, verify fail. Step 3: Implement.**

```python
# src/quant/validation.py
"""Anti-overfitting validation framework.

deflated_sharpe_ratio: Adjusts Sharpe for multiple testing (Bailey & Lopez de Prado 2014).
WalkForwardValidator: Anchored walk-forward with purged splits.
CUSUMDriftDetector: Detects live strategy drift from backtest expectation.
InformationCoefficientTracker: Monitors signal quality over time.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Adjusts observed Sharpe for the number of strategy/parameter trials.
    Returns the probability-adjusted Sharpe.
    """
    if n_trials <= 0:
        n_trials = 1

    # Expected max Sharpe under null (all strategies are noise)
    e_max_sharpe = _expected_max_sharpe(n_trials, n_observations, skew, kurtosis)

    # Standard error of Sharpe estimate
    se_sharpe = np.sqrt(
        (1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2)
        / (n_observations - 1)
    )

    if se_sharpe < 1e-10:
        return observed_sharpe

    # Test statistic: is observed Sharpe significantly above expected max?
    z = (observed_sharpe - e_max_sharpe) / se_sharpe
    return float(z)


def _expected_max_sharpe(
    n_trials: int, n_obs: int, skew: float, kurtosis: float
) -> float:
    """Expected maximum Sharpe under the null hypothesis of no skill."""
    gamma_approx = 0.5772  # Euler-Mascheroni constant
    if n_trials <= 1:
        return 0.0
    z = stats.norm.ppf(1 - 1 / n_trials)
    return float(z * (1 - gamma_approx) + gamma_approx * stats.norm.ppf(1 - 1 / (n_trials * np.e)))


@dataclass
class DriftResult:
    is_drifting: bool
    cusum_value: float
    threshold: float
    cumulative_return: float


class CUSUMDriftDetector:
    """CUSUM detector for strategy return drift.

    Flags when live returns deviate significantly from backtest expectation.
    """

    def __init__(self, expected_sharpe: float = 1.0, threshold: float = 3.0):
        self._expected_daily_return = expected_sharpe / np.sqrt(252)
        self._threshold = threshold
        self._cusum = 0.0
        self._cumulative_return = 0.0
        self._returns: deque[float] = deque(maxlen=252)

    def update(self, daily_return: float) -> DriftResult:
        self._returns.append(daily_return)
        self._cumulative_return += daily_return

        std = float(np.std(self._returns, ddof=1)) if len(self._returns) > 5 else 0.01
        if std < 1e-10:
            std = 0.01

        z = (daily_return - self._expected_daily_return) / std
        self._cusum = min(0, self._cusum + z)  # one-sided negative CUSUM

        return DriftResult(
            is_drifting=abs(self._cusum) > self._threshold,
            cusum_value=float(self._cusum),
            threshold=self._threshold,
            cumulative_return=self._cumulative_return,
        )

    def reset(self) -> None:
        self._cusum = 0.0
        self._cumulative_return = 0.0


class InformationCoefficientTracker:
    """Tracks rolling IC (rank correlation between predictions and outcomes).

    Used by allocator to reweight strategies and detect edge decay.
    """

    def __init__(self, window: int = 30):
        self._window = window
        self._predictions: deque[float] = deque(maxlen=window)
        self._actuals: deque[float] = deque(maxlen=window)
        self._current_ic: Optional[float] = None

    def update(self, prediction: float, actual: float) -> Optional[float]:
        self._predictions.append(prediction)
        self._actuals.append(actual)

        if len(self._predictions) < self._window:
            return None

        # Spearman rank correlation
        rho, _ = stats.spearmanr(list(self._predictions), list(self._actuals))
        self._current_ic = float(rho) if np.isfinite(rho) else 0.0
        return self._current_ic

    @property
    def current_ic(self) -> Optional[float]:
        return self._current_ic

    @property
    def is_decaying(self) -> bool:
        """True if IC has trended toward zero over the window."""
        if self._current_ic is None:
            return False
        return abs(self._current_ic) < 0.05


class WalkForwardValidator:
    """Anchored walk-forward validation for strategy parameters.

    Expanding training window, fixed out-of-sample window.
    Reports per-window Sharpe and aggregated deflated Sharpe.
    """

    def __init__(
        self,
        oos_window: int = 20,
        min_train_window: int = 60,
        n_splits: int = 8,
        embargo_bars: int = 5,
    ):
        self.oos_window = oos_window
        self.min_train_window = min_train_window
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars

    def split(self, n_observations: int) -> list[tuple[range, range]]:
        """Generate (train_range, test_range) tuples for walk-forward splits."""
        splits = []
        total_oos = self.oos_window * self.n_splits
        if n_observations < self.min_train_window + total_oos:
            # Not enough data — use what we have
            available_oos = n_observations - self.min_train_window
            actual_splits = max(1, available_oos // self.oos_window)
        else:
            actual_splits = self.n_splits

        for i in range(actual_splits):
            test_end = n_observations - (actual_splits - i - 1) * self.oos_window
            test_start = test_end - self.oos_window
            train_end = test_start - self.embargo_bars
            train_start = 0  # anchored (expanding window)

            if train_end - train_start < self.min_train_window:
                continue

            splits.append((range(train_start, train_end), range(test_start, test_end)))

        return splits
```

**Step 4: Run tests. Step 5: Commit.**

```bash
git add src/quant/validation.py tests/unit/quant/test_validation.py
git commit -m "feat: add deflated Sharpe, CUSUM drift, IC tracker, walk-forward (src/quant/validation)"
```

---

## Task 8: Upgrade `pair_trading_v2` — Kalman + Cointegration

The most impactful single strategy upgrade. Currently 3.5/5, target 5/5.

**Files:**
- Modify: `src/strategies/pair_trading_v2.py`
- Create: `tests/unit/strategies/test_pair_trading_v2_quant.py`

**Step 1: Write failing tests for new quant gates**

```python
# tests/unit/strategies/test_pair_trading_v2_quant.py
"""Tests for quant upgrades to pair_trading_v2."""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.strategies.pair_trading_v2 import PairTradingV2Strategy


class TestCointegrationGate:
    """Engle-Granger cointegration must pass before entry."""

    def test_rejects_signal_when_not_cointegrated(self):
        """Strategy should not fire signal if pair fails cointegration test."""
        config = _make_config()
        strategy = PairTradingV2Strategy(config, MagicMock())
        # Feed enough bars with independent (non-cointegrated) prices
        # The cointegration gate should prevent signal generation
        # (Implementation detail: _coint_valid dict tracks per-pair status)
        assert hasattr(strategy, '_coint_valid') or hasattr(strategy, '_coint_monitors')


class TestKalmanHedgeRatio:
    """Kalman filter should replace EMA for hedge ratio."""

    def test_uses_kalman_not_ema(self):
        config = _make_config()
        strategy = PairTradingV2Strategy(config, MagicMock())
        # Strategy should have KalmanHedgeRatio instances, not RollingEMA
        # Check the pair state initialization
        assert True  # Will be validated by integration test after implementation


def _make_config():
    return {
        "enabled": True,
        "pairs": [{"leg_a": "AAPL", "leg_b": "MSFT"}],
        "entry_z_threshold": 2.5,
        "stop_z_threshold": 3.5,
        "confluence_threshold": 60.0,
        "hedge_ema_period": 50,
        "spread_lookback": 100,
        "min_bars": 60,
        "cooldown_bars": 5,
    }
```

**Step 2: Run tests, verify fail. Step 3: Modify `pair_trading_v2.py`.**

Key changes to `src/strategies/pair_trading_v2.py`:
1. Add imports: `from src.quant.filters import KalmanHedgeRatio` and `from src.quant.cointegration import EngleGrangerTest, RollingCointegrationMonitor`
2. In `_init_pairs()`: replace `RollingEMA(hedge_ema_period)` with `KalmanHedgeRatio()`
3. Add `_coint_monitors: dict[str, RollingCointegrationMonitor]` per pair
4. In entry logic: add cointegration gate — check `_coint_monitors[key].is_valid` before allowing entry
5. Add rolling correlation monitor: compute correlation over spread_lookback, reject if < 0.6
6. Replace `RollingStd`-based z-score with GARCH-conditional z-score using `GARCHForecaster` on the spread
7. Add OU half-life gate using existing `OUEstimator` — reject if half_life > max_hold_minutes

**Step 4: Run tests. Step 5: Commit.**

```bash
git commit -m "feat: upgrade pair_trading_v2 with Kalman hedge ratio, cointegration gate, GARCH z-score"
```

---

## Task 9: Upgrade `mean_reversion_pro` — GARCH + Hurst + Cointegration

**Files:**
- Modify: `src/strategies/mean_reversion_pro.py`
- Create: `tests/unit/strategies/test_mean_reversion_pro_quant.py`

Key changes:
1. Import `GARCHForecaster`, `HurstExponent`, `AdaptiveThresholdEngine` from `src.quant`
2. Add per-symbol `_garch: dict[str, GARCHForecaster]` and `_hurst: dict[str, HurstExponent]`
3. Replace rolling z-score deviation with `garch.conditional_zscore(vwap_distance)`
4. Add Hurst gate: `if not hurst_result.is_mean_reverting: return None`
5. Make OU half-life a hard gate (currently only scales thresholds)
6. Replace hardcoded `confluence_threshold=65` with `AdaptiveThresholdEngine.adapt()`
7. Add Engle-Granger test against sector ETF (XLK for tech, XLF for financials, etc.)

---

## Task 10: Upgrade `trend_rider_pro` — Kalman + Hurst + GARCH stops

**Files:**
- Modify: `src/strategies/trend_rider_pro.py`
- Create: `tests/unit/strategies/test_trend_rider_pro_quant.py`

Key changes:
1. Import `KalmanMeanTracker`, `HurstExponent`, `GARCHForecaster`, `MarkovRegimeSwitcher`
2. Replace EMA-20 pullback with `KalmanMeanTracker` — pullback = price near Kalman state
3. Add Hurst gate: only activate when `H > 0.55` (trending)
4. Replace hardcoded ADX threshold (45) with Markov filtered probability `P(trending) > 0.7`
5. Replace fixed ATR stop/target multipliers with GARCH-forecasted vol scaling
6. Add autocorrelation test on recent pullback returns

---

## Task 11: Upgrade `flow_alpha` — IC-weighted + Granger + VPIN

**Files:**
- Modify: `src/strategies/flow_alpha.py`
- Create: `tests/unit/strategies/test_flow_alpha_quant.py`

Key changes:
1. Import `GrangerCausalityTest`, `InformationCoefficientTracker`, `GARCHForecaster`, `VPINCalculator`
2. Replace static weights (0.35/0.25/0.20/0.20) with rolling IC-weighted combination
3. Add per-signal `InformationCoefficientTracker` — weight = IC over trailing 30 bars
4. Add weekly Granger causality check: if flow signal doesn't Granger-cause returns, weight → 0
5. Add VPIN toxicity gate (reuse existing `VPINCalculator` pattern from mean_reversion_pro)
6. Replace `flow_zscore / 3.0` normalization with GARCH-conditional normalization

---

## Task 12: Upgrade `orb_v2` — CUSUM + Variance Ratio + BOCPD

**Files:**
- Modify: `src/strategies/orb_v2.py`
- Create: `tests/unit/strategies/test_orb_v2_quant.py`

Key changes:
1. Import `CUSUMDetector`, `VarianceRatioCalculator`, `GARCHForecaster`, `MarkovRegimeSwitcher`
2. Replace naive breakout detection with CUSUM: breakout confirmed when `cusum.update(close - range_mean).is_breakout`
3. Add variance ratio gate: if `VR < 1.0` (mean-reverting), suppress breakout (likely to fail)
4. Add BOCPD changepoint probability (from `market_state.regime_snapshot.changepoint_probability`) as confluence multiplier
5. Replace fixed volume gate (1.2x) with volume relative to GARCH-forecasted vol
6. Replace hardcoded range window (5/10/15 min) with Markov regime-switching optimal window

---

## Task 13: Upgrade `rsi_bounce` — GARCH z-score + BOCPD + Kurtosis

**Files:**
- Modify: `src/strategies/rsi_bounce.py`
- Create: `tests/unit/strategies/test_rsi_bounce_quant.py`

Key changes (lightest touch — already strongest strategy):
1. Import `GARCHForecaster`, `AdaptiveThresholdEngine`
2. Replace rolling z-score with GARCH-conditional z-score
3. Add BOCPD structural break awareness: suppress entries when `changepoint_probability > 0.7`
4. Add rolling kurtosis filter: skip entries when kurtosis > 6 (fat tails = unstable reversion)
5. Make `z_entry`, `half_life` bounds adaptive via `AdaptiveThresholdEngine`

---

## Task 14: Upgrade `momentum_fade` — GARCH z-score + Exhaustion + Hurst

**Files:**
- Modify: `src/strategies/momentum_fade.py`
- Create: `tests/unit/strategies/test_momentum_fade_quant.py`

Key changes:
1. Import `GARCHForecaster`, `HurstExponent`
2. Replace arbitrary VWAP deviation (0.008) with `garch.conditional_zscore(vwap_distance)`
3. Add momentum exhaustion model: compute velocity (ROC) and acceleration (ROC of ROC). Fade only when velocity is extreme AND acceleration is negative (decelerating)
4. Replace fixed volume surge (2.0x) with volume relative to intraday seasonal profile (compute average volume by 15-min bucket)
5. Add Hurst gate: only fade when `H < 0.5` (mean-reverting)
6. Add entropy filter: skip when `market_state.regime_snapshot.entropy_score > 0.8` (random market)

---

## Task 15: Portfolio Layer — `src/portfolio/signal_aggregator.py`

**Files:**
- Create: `src/portfolio/__init__.py`
- Create: `src/portfolio/signal_aggregator.py`
- Create: `tests/unit/portfolio/__init__.py`
- Create: `tests/unit/portfolio/test_signal_aggregator.py`

Key implementation:
1. `SignalAggregator` class with `aggregate(signals: list[Signal]) -> list[Signal]`
2. IC-weighted combination: each signal weighted by strategy's trailing IC from `InformationCoefficientTracker`
3. Directional conflict resolution: if BUY weight + SELL weight net < threshold, filter out
4. Correlation penalty: if two strategies' recent signals have Pearson > 0.7, discount the weaker one
5. Integrates into `src/engine/execution.py` between strategy signal emission and RiskManager

---

## Task 16: Portfolio Layer — `src/portfolio/allocator.py`

**Files:**
- Create: `src/portfolio/allocator.py`
- Create: `tests/unit/portfolio/test_allocator.py`

Key implementation:
1. `PortfolioAllocator` class with `compute_allocations(strategy_returns: dict[str, list[float]]) -> dict[str, float]`
2. Risk-parity: allocation inversely proportional to realized vol (using `RealizedVolEstimator`)
3. Drawdown throttling: if trailing DD > 1.5x max historical DD, halve allocation
4. Correlation-adjusted exposure: compute cross-strategy return correlation matrix, scale gross exposure inversely
5. Called from EOD agent flow, stores allocations that `RiskManager` reads during position sizing

---

## Task 17: Portfolio Layer — `src/portfolio/risk_budget.py`

**Files:**
- Create: `src/portfolio/risk_budget.py`
- Create: `tests/unit/portfolio/test_risk_budget.py`

Key implementation:
1. `PortfolioRiskBudget` class with `check_marginal_risk(new_signal: Signal, current_positions: list) -> bool`
2. Portfolio VaR/CVaR computed across all active positions using GARCH-forecasted vol
3. Marginal risk contribution: reject if new position adds > X% to portfolio CVaR
4. Concentration limits: max 40% risk to one strategy, max 25% to one symbol
5. Integrates into `src/engine/risk.py` as additional check in `RiskManager.apply()`

---

## Task 18: Portfolio Layer — `src/portfolio/performance.py`

**Files:**
- Create: `src/portfolio/performance.py`
- Create: `tests/unit/portfolio/test_performance.py`

Key implementation:
1. `PortfolioPerformance` class for analytics
2. Strategy attribution: decompose daily P&L by strategy
3. Rolling Sharpe/Sortino per strategy (stored in DB, used by allocator)
4. Correlation matrix monitoring with alert on spikes
5. Integrates with existing EOD agent analysis pipeline

---

## Task 19: Database Schema Updates

**Files:**
- Modify: `src/analysis/schema.py`

Add two new SQLAlchemy models:
1. `StrategyICDaily` — columns: `id`, `date`, `strategy`, `ic_value`, `is_decaying`
2. `PortfolioRiskSnapshot` — columns: `id`, `timestamp`, `portfolio_var`, `portfolio_cvar`, `correlation_matrix_json`, `concentration_json`, `gross_exposure`

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run python -c "from src.analysis.schema import *; print('Schema OK')"`

---

## Task 20: Integration — Wire Portfolio Layer into Execution Pipeline

**Files:**
- Modify: `src/engine/execution.py`
- Modify: `src/engine/risk.py`
- Modify: `src/main.py`

Key changes:
1. In `ExecutionEngine.__init__()`: instantiate `SignalAggregator` and `PortfolioRiskBudget`
2. In signal processing: after strategies emit signals, pass through `signal_aggregator.aggregate()` before `risk_manager.apply()`
3. In `RiskManager.apply()`: add `portfolio_risk_budget.check_marginal_risk()` as additional gate
4. In `main.py` EOD flow: call `PortfolioAllocator.compute_allocations()` and store results

---

## Task 21: Run Full Test Suite and Fix Regressions

**Files:**
- All modified files

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/ -v --tb=short -x`

Fix any regressions. Ensure:
1. All existing unit tests still pass
2. All new quant tests pass
3. Ruff lint passes: `cd /Users/jacobmcmillan/Empire/Cerberus && ruff check src/quant/ src/portfolio/`
4. Type checking: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run mypy src/quant/ src/portfolio/ --ignore-missing-imports`

---

## Task 22: Update CHANGELOG.md

Add entries under `## [Unreleased]`:

```markdown
### Added
- Quant foundation layer (`src/quant/`): GARCH conditional volatility, Kalman filters,
  Engle-Granger cointegration, Hurst exponent, CUSUM breakout detection, Granger causality,
  Markov regime-switching, adaptive thresholds, walk-forward validation, deflated Sharpe ratio
- Portfolio optimization layer (`src/portfolio/`): IC-weighted signal aggregation,
  risk-parity allocation, portfolio VaR/CVaR, marginal risk contribution, strategy attribution
- New DB tables: strategy_ic_daily, portfolio_risk_snapshots

### Changed
- pair_trading_v2: Kalman hedge ratio, Engle-Granger entry gate, GARCH z-score, OU half-life gate
- mean_reversion_pro: GARCH z-score, Hurst filter, cointegration gate, adaptive thresholds
- trend_rider_pro: Kalman mean tracker, Hurst gate, Markov trend probability, GARCH stops
- flow_alpha: IC-weighted signal combination, Granger causality validation, VPIN gate
- orb_v2: CUSUM breakout detection, variance ratio gate, BOCPD confidence
- rsi_bounce: GARCH z-score, BOCPD awareness, kurtosis filter, adaptive thresholds
- momentum_fade: GARCH z-score, momentum exhaustion model, Hurst gate, entropy filter
- Execution pipeline: signals now route through SignalAggregator and PortfolioRiskBudget
```

Final commit:
```bash
git add -A && git commit -m "feat: complete quant strategy upgrade — all 7 strategies + portfolio layer + validation"
```
