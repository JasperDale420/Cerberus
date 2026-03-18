"""Anti-overfitting validation framework.

Provides deflated Sharpe ratio, walk-forward validation, CUSUM drift detection,
and information coefficient tracking for robust strategy evaluation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, spearmanr

# ---------------------------------------------------------------------------
# Euler-Mascheroni constant
# ---------------------------------------------------------------------------
EULER_MASCHERONI = 0.5772156649015329

# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
# ---------------------------------------------------------------------------


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio.

    Tests whether an observed Sharpe ratio is statistically significant after
    accounting for multiple testing (strategy selection bias).

    Returns a z-score: positive means the observed Sharpe exceeds what you'd
    expect from the best of n_trials pure-noise strategies.
    """
    # Expected maximum Sharpe under the null (all strategies are noise)
    # E[max] ≈ (1 - γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N*e))
    # Simplified approximation using Euler-Mascheroni constant
    if n_trials <= 1:
        expected_max_sharpe = 0.0
    else:
        expected_max_sharpe = norm.ppf(1.0 - 1.0 / n_trials) * (1.0 - EULER_MASCHERONI) + EULER_MASCHERONI * norm.ppf(
            1.0 - 1.0 / (n_trials * math.e)
        )

    # Standard error of the Sharpe ratio (Lo, 2002 — with skew/kurtosis adjustment)
    sr = observed_sharpe
    se_sharpe = math.sqrt((1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2) / (n_observations - 1))

    # Z-score: how many SE the observed Sharpe exceeds the expected max
    if se_sharpe == 0:
        return 0.0
    return (observed_sharpe - expected_max_sharpe) / se_sharpe


# ---------------------------------------------------------------------------
# Walk-Forward Validator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Anchored walk-forward cross-validator with embargo.

    Train window always starts at index 0 (expanding window). Test windows
    are sequential and non-overlapping, laid out from the end of the series.
    An embargo gap separates train and test to prevent information leakage.
    """

    def __init__(
        self,
        oos_window: int = 20,
        min_train_window: int = 60,
        n_splits: int = 8,
        embargo_bars: int = 5,
    ) -> None:
        self._oos_window = oos_window
        self._min_train_window = min_train_window
        self._n_splits = n_splits
        self._embargo_bars = embargo_bars

    def split(self, n_observations: int) -> list[tuple[range, range]]:
        """Generate (train_range, test_range) tuples.

        Test windows are placed from the end of the series backwards,
        ensuring non-overlapping sequential blocks.
        """
        splits: list[tuple[range, range]] = []

        for i in range(self._n_splits):
            # Test windows laid out from end, backwards
            test_end = n_observations - i * self._oos_window
            test_start = test_end - self._oos_window

            if test_start < 0:
                continue

            # Train ends before embargo
            train_end = test_start - self._embargo_bars

            if train_end <= 0:
                continue

            train = range(0, train_end)
            test = range(test_start, test_end)

            # Skip if train window is too small
            if len(train) < self._min_train_window:
                continue

            splits.append((train, test))

        # Reverse so splits are chronological (earliest test first)
        splits.reverse()
        return splits


# ---------------------------------------------------------------------------
# CUSUM Drift Detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DriftResult:
    """Result of a single CUSUM drift check."""

    is_drifting: bool
    cusum_value: float
    threshold: float
    cumulative_return: float


class CUSUMDriftDetector:
    """One-sided negative CUSUM detector for strategy underperformance.

    Monitors daily returns against an expected level derived from a backtest
    Sharpe ratio. Triggers when cumulative shortfall exceeds a threshold.
    """

    def __init__(self, expected_sharpe: float = 1.0, threshold: float = 3.0) -> None:
        self._expected_daily = expected_sharpe / math.sqrt(252)
        self._threshold = threshold
        self._cusum: float = 0.0
        self._cumulative_return: float = 0.0
        self._returns: deque[float] = deque(maxlen=60)

    def update(self, daily_return: float) -> DriftResult:
        """Ingest a daily return and check for drift."""
        self._returns.append(daily_return)
        self._cumulative_return += daily_return

        # Need minimum observations to estimate volatility
        if len(self._returns) < 5:
            return DriftResult(
                is_drifting=False,
                cusum_value=0.0,
                threshold=self._threshold,
                cumulative_return=self._cumulative_return,
            )

        # Rolling std of returns for z-normalization
        rolling_std = float(np.std(self._returns, ddof=1))
        # Floor std to avoid division by zero for constant returns
        rolling_std = max(rolling_std, abs(self._expected_daily) * 0.01 + 1e-10)

        # Z-score of shortfall: how many stds below expected is current return
        # Use Page's CUSUM: accumulate only when significantly underperforming
        z_shortfall = (self._expected_daily - daily_return) / rolling_std

        # Subtract 0.5 as slack (standard CUSUM allowance) to avoid false positives
        # from normal noise
        self._cusum = max(0.0, self._cusum + z_shortfall - 0.5)

        is_drifting = self._cusum >= self._threshold

        return DriftResult(
            is_drifting=is_drifting,
            cusum_value=self._cusum,
            threshold=self._threshold,
            cumulative_return=self._cumulative_return,
        )

    def reset(self) -> None:
        """Reset the detector to initial state."""
        self._cusum = 0.0
        self._cumulative_return = 0.0
        self._returns.clear()


# ---------------------------------------------------------------------------
# Information Coefficient Tracker
# ---------------------------------------------------------------------------


class InformationCoefficientTracker:
    """Rolling Spearman rank IC between predictions and actuals."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._predictions: deque[float] = deque(maxlen=window)
        self._actuals: deque[float] = deque(maxlen=window)
        self._current_ic: float | None = None

    def update(self, prediction: float, actual: float) -> float | None:
        """Add a prediction/actual pair and return IC if window is filled."""
        self._predictions.append(prediction)
        self._actuals.append(actual)

        if len(self._predictions) < self._window:
            self._current_ic = None
            return None

        corr, _ = spearmanr(list(self._predictions), list(self._actuals))
        self._current_ic = float(corr)
        return self._current_ic

    @property
    def current_ic(self) -> float | None:
        """Most recent IC value, or None if window not yet filled."""
        return self._current_ic

    @property
    def is_decaying(self) -> bool:
        """True if |IC| < 0.05, indicating signal decay."""
        if self._current_ic is None:
            return False
        return abs(self._current_ic) < 0.05
