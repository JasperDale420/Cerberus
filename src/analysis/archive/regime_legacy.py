from __future__ import annotations

"""
Legacy regime classifier (BULL/BEAR/CHOP).

Archived for backwards compatibility and historical reference.
New code should prefer multi-axis classification via:
  - src.analysis.regime.MarketContextService
  - src.core.domain.MarketRegimeSnapshot
"""

from collections import Counter, deque
from typing import Callable, Optional, Tuple

import numpy as np

from src.core.domain import Bar, Regime
from src.core.logger import StructuredLogger


class RegimeDetector:
    """
    Detects market regime (BULL, BEAR, CHOP) based on price history.
    """

    def __init__(
        self,
        window: int = 60,
        min_bars: int = 20,
        smooth_k: int = 10,
        logger: Optional[StructuredLogger] = None,
        on_error: Optional[Callable[[], None]] = None,
    ):
        self.window = window
        self.min_bars = min_bars
        self.smooth_k = smooth_k
        self.logger = logger
        self.on_error = on_error

        self.prices: deque[float] = deque(maxlen=window)
        self.last_classifications: deque[Regime] = deque(maxlen=smooth_k)
        self.current_regime = Regime.CHOP

        # Expose latest computed metrics for regime_history persistence.
        self.last_cum_ret: Optional[float] = None
        self.last_trend_score: Optional[float] = None
        self.last_vol: Optional[float] = None

    def update(self, bar: Bar) -> Regime:
        """
        Updates the detector with a new bar and returns the current regime.
        """
        self.prices.append(float(bar.close))

        if len(self.prices) < self.min_bars:
            return self.current_regime

        try:
            cum_ret, trend_score, vol = self._compute_metrics()
            self.last_cum_ret = cum_ret
            self.last_trend_score = trend_score
            self.last_vol = vol
            new_regime = self._classify(cum_ret, trend_score)
        except Exception as e:
            if self.on_error:
                try:
                    self.on_error()
                except Exception:
                    pass
            if self.logger:
                from src.core.errors import ErrorCode

                self.logger.error(
                    "Regime calculation failed",
                    error_code=ErrorCode.REGIME_CALC_FAILED.value,
                    symbol=getattr(bar, "symbol", None),
                    error=str(e),
                )
            return self.current_regime

        self.last_classifications.append(new_regime)
        prev = self.current_regime
        self.current_regime = self._smooth_regime()

        if self.logger:
            if self.current_regime != prev:
                self.logger.info(
                    "Regime changed",
                    from_regime=prev.value,
                    to_regime=self.current_regime.value,
                    cum_ret=cum_ret,
                    trend_score=trend_score,
                    vol=vol,
                )
            else:
                self.logger.debug(
                    "Regime updated",
                    regime=self.current_regime.value,
                    cum_ret=cum_ret,
                    trend_score=trend_score,
                    vol=vol,
                )

        return self.current_regime

    def _compute_metrics(self) -> Tuple[float, float, float]:
        """
        Computes cumulative return and trend score.
        """
        prices = np.array(self.prices)
        returns = np.diff(np.log(prices))

        cum_ret = np.sum(returns)
        vol = float(np.std(returns)) + 1e-9  # Avoid division by zero

        trend_score = abs(cum_ret) / vol
        return float(cum_ret), float(trend_score), vol

    def _classify(
        self,
        cum_ret: float,
        trend_score: float,
        up_thresh: float = 1.5,
        down_thresh: float = 1.5,
    ) -> Regime:
        """
        Classifies regime based on metrics.
        """
        if trend_score < 1.0:
            return Regime.CHOP
        if cum_ret > 0 and trend_score >= up_thresh:
            return Regime.BULL
        if cum_ret < 0 and trend_score >= down_thresh:
            return Regime.BEAR
        return Regime.CHOP

    def _smooth_regime(self) -> Regime:
        """
        Returns the majority vote regime from the last K classifications.
        """
        if not self.last_classifications:
            return Regime.CHOP

        counts = Counter(self.last_classifications)
        top_regime, _ = counts.most_common(1)[0]
        return top_regime

    def get_regime(self) -> Regime:
        return self.current_regime
