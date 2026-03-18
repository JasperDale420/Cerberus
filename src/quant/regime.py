"""Markov regime-switching model and adaptive threshold engine.

Provides two primitives for regime-aware trading logic:

1. **MarkovRegimeSwitcher** — fits a Markov-switching regression on streaming
   values and identifies the most probable regime (e.g. low-vol vs high-vol).
   Refits every ``refit_interval`` bars to amortise MLE cost.

2. **AdaptiveThresholdEngine** — scales a base threshold by volatility, Hurst
   exponent, and regime confidence so that entry/exit signals adapt to current
   market conditions.

Usage::

    from src.quant.regime import MarkovRegimeSwitcher, AdaptiveThresholdEngine

    switcher = MarkovRegimeSwitcher(n_regimes=2)
    engine = AdaptiveThresholdEngine()

    for value in data_stream:
        result = switcher.update(value)
        if result is not None:
            threshold = engine.adapt(
                base_threshold=1.5,
                conditional_vol=current_vol,
                hurst=current_hurst,
                regime_confidence=result.filtered_probability,
            )
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RegimeSwitchResult:
    """Result of a Markov regime-switching model fit.

    Attributes:
        current_regime: Most probable regime (0-indexed).
        filtered_probability: P(current regime | data) at the last observation.
        regime_probabilities: Probabilities for all regimes at the last observation.
        n_observations: Number of observations used in the fit.
    """

    current_regime: int
    filtered_probability: float
    regime_probabilities: list[float]
    n_observations: int


class MarkovRegimeSwitcher:
    """Streaming Markov regime-switching model.

    Accumulates values in a rolling buffer and periodically fits a
    ``MarkovRegression`` model from statsmodels to identify the current
    regime.

    Parameters:
        n_regimes: Number of regimes to detect.
        min_observations: Minimum values before first fit attempt.
        lookback: Rolling window size for the model.
        refit_interval: Refit the model every N new observations.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        min_observations: int = 100,
        lookback: int = 500,
        refit_interval: int = 50,
    ) -> None:
        self._n_regimes = n_regimes
        self._min_observations = min_observations
        self._lookback = lookback
        self._refit_interval = refit_interval

        self._buffer: deque[float] = deque(maxlen=lookback)
        self._count: int = 0
        self._bars_since_fit: int = 0
        self.last_result: Optional[RegimeSwitchResult] = None

    def update(self, value: float) -> Optional[RegimeSwitchResult]:
        """Ingest a new value and optionally refit the model.

        Returns:
            A ``RegimeSwitchResult`` if a fit has been performed, or the last
            cached result if available.  ``None`` before ``min_observations``.
        """
        self._buffer.append(value)
        self._count += 1
        self._bars_since_fit += 1

        if self._count < self._min_observations:
            return None

        if self._bars_since_fit >= self._refit_interval or self.last_result is None:
            self._bars_since_fit = 0
            result = self._fit()
            if result is not None:
                self.last_result = result

        return self.last_result

    def _fit(self) -> Optional[RegimeSwitchResult]:
        """Fit the Markov regression model on the current buffer."""
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        data = np.array(self._buffer, dtype=np.float64)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = MarkovRegression(
                    data,
                    k_regimes=self._n_regimes,
                    trend="c",
                    switching_variance=True,
                )
                fitted = model.fit(disp=False, maxiter=200)
        except Exception:  # noqa: BLE001 — convergence / singular matrix
            return None

        # filtered_marginal_probabilities: (n_obs, n_regimes)
        probs = fitted.filtered_marginal_probabilities
        last_probs = probs[-1]
        current_regime = int(np.argmax(last_probs))
        regime_probabilities = [float(p) for p in last_probs]

        return RegimeSwitchResult(
            current_regime=current_regime,
            filtered_probability=float(last_probs[current_regime]),
            regime_probabilities=regime_probabilities,
            n_observations=self._count,
        )


class AdaptiveThresholdEngine:
    """Scales a base threshold by market conditions.

    Adjusts thresholds dynamically based on:

    1. **Volatility**: wider when conditional vol exceeds baseline vol,
       tighter when vol is suppressed.
    2. **Hurst exponent**: tighter when H < 0.5 (mean-reverting), wider
       when H > 0.5 (trending).
    3. **Regime confidence**: tighter when the regime model is confident
       (multiplier = 2.0 - confidence).

    The final combined multiplier is clamped to ``[min_multiplier, max_multiplier]``.

    Parameters:
        vol_scale_factor: How aggressively vol ratio scales the threshold.
        hurst_scale_factor: How aggressively Hurst deviations scale.
        min_multiplier: Floor for the combined multiplier.
        max_multiplier: Ceiling for the combined multiplier.
    """

    def __init__(
        self,
        vol_scale_factor: float = 1.0,
        hurst_scale_factor: float = 1.0,
        min_multiplier: float = 0.5,
        max_multiplier: float = 3.0,
    ) -> None:
        self._vol_scale = vol_scale_factor
        self._hurst_scale = hurst_scale_factor
        self._min_mult = min_multiplier
        self._max_mult = max_multiplier

    def adapt(
        self,
        base_threshold: float,
        conditional_vol: float = 0.0,
        hurst: float = 0.5,
        regime_confidence: float = 1.0,
        baseline_vol: Optional[float] = None,
    ) -> float:
        """Scale *base_threshold* by current market conditions.

        Args:
            base_threshold: The unadjusted threshold value.
            conditional_vol: Current conditional volatility estimate.
            hurst: Hurst exponent (0-1). <0.5 = mean-reverting, >0.5 = trending.
            regime_confidence: Confidence in the current regime (0-1).
            baseline_vol: Long-run volatility baseline. If ``None``, vol
                adjustment is skipped.

        Returns:
            The adjusted threshold.
        """
        multiplier = 1.0

        # --- Volatility adjustment ---
        if baseline_vol is not None and baseline_vol > 0:
            vol_ratio = conditional_vol / baseline_vol
            # Scale deviation from 1.0 by vol_scale_factor
            vol_adjustment = 1.0 + self._vol_scale * (vol_ratio - 1.0)
            multiplier *= vol_adjustment

        # --- Hurst adjustment ---
        # hurst=0.5 → neutral (factor=1.0)
        # hurst<0.5 → mean-reverting → tighter (factor<1.0)
        # hurst>0.5 → trending → wider (factor>1.0)
        hurst_deviation = hurst - 0.5
        hurst_adjustment = 1.0 + self._hurst_scale * (2.0 * hurst_deviation)
        multiplier *= hurst_adjustment

        # --- Regime confidence adjustment ---
        # confidence=1.0 → multiplier=1.0 (tight, certain)
        # confidence=0.5 → multiplier=1.5 (wider, uncertain)
        confidence_adjustment = 2.0 - regime_confidence
        multiplier *= confidence_adjustment

        # Clamp
        multiplier = max(self._min_mult, min(self._max_mult, multiplier))

        return base_threshold * multiplier
