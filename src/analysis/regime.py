from enum import Enum
from collections import deque, Counter
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from src.core.logger import StructuredLogger

class Regime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    CHOP = "chop"

from src.data.models import Bar

class RegimeDetector:
    """
    Detects market regime (BULL, BEAR, CHOP) based on price history.
    """
    def __init__(self, window: int = 60, min_bars: int = 20, smooth_k: int = 10, logger: Optional[StructuredLogger] = None):
        self.window = window
        self.min_bars = min_bars
        self.smooth_k = smooth_k
        self.logger = logger
        
        self.prices = deque(maxlen=window)
        self.last_classifications = deque(maxlen=smooth_k)
        self.current_regime = Regime.CHOP

    def update(self, price: float) -> Regime:
        """
        Updates the detector with a new price and returns the current regime.
        """
        self.prices.append(price)
        
        if len(self.prices) < self.min_bars:
            return self.current_regime

        try:
            cum_ret, trend_score = self._compute_metrics()
            new_regime = self._classify(cum_ret, trend_score)
        except Exception as e:
            if self.logger:
                self.logger.error("Regime calculation failed", error=str(e))
            return self.current_regime

        self.last_classifications.append(new_regime)
        self.current_regime = self._smooth_regime()
        
        if self.logger:
            self.logger.debug("Regime updated", regime=self.current_regime, cum_ret=cum_ret, trend_score=trend_score)
            
        return self.current_regime

    def _compute_metrics(self) -> Tuple[float, float]:
        """
        Computes cumulative return and trend score.
        """
        prices = np.array(self.prices)
        returns = np.diff(np.log(prices))
        
        cum_ret = np.sum(returns)
        vol = np.std(returns) + 1e-9 # Avoid division by zero
        
        trend_score = abs(cum_ret) / vol
        return cum_ret, trend_score

    def _classify(self, cum_ret: float, trend_score: float, up_thresh: float = 1.5, down_thresh: float = 1.5) -> Regime:
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
