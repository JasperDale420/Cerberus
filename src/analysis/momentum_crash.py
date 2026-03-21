"""Momentum Crash Hedging module based on Daniel & Moskowitz (2016).

Monitors momentum factor spread and market conditions to estimate the probability
of a momentum crash. When P(crash) is elevated, momentum strategies should scale
exposure by (1 - P(crash)) to reduce drawdown risk.

Academic basis: Daniel, K. and Moskowitz, T.J. (2016), "Momentum Crashes",
Journal of Financial Economics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.logger import StructuredLogger


@dataclass
class MomentumCrashConfig:
    """Configuration for the momentum crash probability estimator."""

    enabled: bool = False

    # Logistic regression coefficients (calibrated from literature)
    beta_intercept: float = -3.0  # Base rate ~5%
    beta_vol: float = 8.0  # Higher vol -> higher crash prob
    beta_bear: float = 1.5  # Bear market raises crash prob
    beta_spread: float = 2.0  # Wider momentum spread -> higher crash prob

    # Bear market detection
    bear_threshold: float = -0.10  # Cumulative return threshold for bear market

    # Rolling spread z-score
    spread_lookback: int = 60  # Bars of momentum spread history

    # Exposure limits
    min_exposure: float = 0.2  # Never go below 20% exposure
    crash_prob_threshold: float = 0.3  # Start scaling when P(crash) > 30%


@dataclass(frozen=True)
class MomentumCrashResult:
    """Output of the momentum crash probability estimator."""

    crash_prob: float  # 0.0 to 1.0
    exposure_scalar: float  # max(min_exposure, 1.0 - crash_prob)
    bear_market: bool
    momentum_spread_zscore: float
    realized_vol: float


def _sigmoid(x: float) -> float:
    """Logistic sigmoid function, numerically stable."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


class MomentumCrashDetector:
    """Crash probability estimator for momentum strategies.

    Uses a logistic model to estimate P(crash) based on:
    - Realized volatility
    - Bear market indicator (cumulative market return below threshold)
    - Momentum spread z-score (crowding proxy)

    When P(crash) is elevated, the exposure_scalar in the result indicates
    how much to scale momentum strategy exposure: (1 - P(crash)), clamped
    to a configurable minimum.
    """

    def __init__(
        self,
        config: Optional[MomentumCrashConfig] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self._config = config or MomentumCrashConfig()
        self._logger = logger

        # Rolling momentum spread history for z-score computation
        self._spread_history: deque[float] = deque(maxlen=self._config.spread_lookback)

        # Latest state
        self._last_result: Optional[MomentumCrashResult] = None
        self._last_realized_vol: float = 0.0
        self._last_bear_market: bool = False
        self._last_spread_zscore: float = 0.0

    @property
    def config(self) -> MomentumCrashConfig:
        """Return current configuration."""
        return self._config

    def update(
        self,
        realized_vol: float,
        market_return_cumulative: float,
        momentum_spread: float,
    ) -> MomentumCrashResult:
        """Update the crash probability estimate with new market data.

        Args:
            realized_vol: Current realized volatility (e.g., 20-day annualized).
            market_return_cumulative: Cumulative market return over lookback period.
                Negative values indicate a bear market.
            momentum_spread: Spread between top-decile and bottom-decile momentum
                returns. Wider spread indicates more crowded momentum, higher crash risk.

        Returns:
            MomentumCrashResult with crash probability and exposure scalar.
        """
        cfg = self._config

        # Bear market indicator
        bear_indicator = 1.0 if market_return_cumulative < cfg.bear_threshold else 0.0
        self._last_bear_market = bear_indicator == 1.0

        # Track momentum spread for z-score (append AFTER computing to avoid self-inclusion)
        spread_zscore = self._compute_spread_zscore(momentum_spread)
        self._spread_history.append(momentum_spread)
        self._last_spread_zscore = spread_zscore

        # Store realized vol
        self._last_realized_vol = realized_vol

        # Logistic model: P(crash) = sigmoid(beta @ x)
        logit = (
            cfg.beta_intercept
            + cfg.beta_vol * realized_vol
            + cfg.beta_bear * bear_indicator
            + cfg.beta_spread * spread_zscore
        )
        crash_prob = _sigmoid(logit)

        # Exposure scalar: scale down when crash prob exceeds threshold
        if crash_prob > cfg.crash_prob_threshold:
            exposure_scalar = max(cfg.min_exposure, 1.0 - crash_prob)
        else:
            exposure_scalar = 1.0

        result = MomentumCrashResult(
            crash_prob=crash_prob,
            exposure_scalar=exposure_scalar,
            bear_market=self._last_bear_market,
            momentum_spread_zscore=spread_zscore,
            realized_vol=realized_vol,
        )
        self._last_result = result

        if self._logger:
            self._logger.debug(
                "Momentum crash update",
                crash_prob=round(crash_prob, 4),
                exposure_scalar=round(exposure_scalar, 4),
                bear_market=self._last_bear_market,
                spread_zscore=round(spread_zscore, 4),
                realized_vol=round(realized_vol, 4),
            )

        return result

    def get_crash_probability(self) -> Optional[MomentumCrashResult]:
        """Return the most recent crash probability result, or None if never updated."""
        return self._last_result

    def _compute_spread_zscore(self, current_spread: float) -> float:
        """Compute z-score of current momentum spread vs rolling history.

        Returns 0.0 if insufficient history or zero standard deviation.
        """
        if len(self._spread_history) < 2:
            return 0.0

        spread_array = np.array(self._spread_history)
        mean = float(np.mean(spread_array))
        std = float(np.std(spread_array, ddof=1))

        if std < 1e-12:
            return 0.0

        return (current_spread - mean) / std
