"""V8a: Multi-Factor Oversold Confluence.

Scores oversold conditions across 3 independent factors:
  1. Consecutive down closes (momentum exhaustion)
  2. Z-score from SMA (statistical oversold)
  3. IBS (intraday selling exhaustion)

Entry requires any 2 of 3 factors confirming. This avoids over-filtering
while maintaining signal quality through confluence.
Target: SMA midline (mean reversion to the mean).
Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v8a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        # Factor thresholds
        self.consec_down_min = int(config.get("consec_down_min", 2))
        self.sma_period = int(config.get("sma_period", 20))
        self.zscore_threshold = float(config.get("zscore_threshold", -0.8))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.25))
        # Require N of 3 factors
        self.min_factors = int(config.get("min_factors", 2))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 50))
        self.drawdown_max = float(config.get("drawdown_max", 0.12))
        # ATR for stop
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        subset = values[-period:]
        mean = sum(subset) / period
        variance = sum((v - mean) ** 2 for v in subset) / period
        return variance**0.5

    @staticmethod
    def _atr(bars: list[Bar], period: int) -> Optional[float]:
        if len(bars) < period + 1:
            return None
        trs = []
        for i in range(-period, 0):
            b = bars[i]
            prev_close = bars[i - 1].close
            tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
            trs.append(tr)
        return sum(trs) / period

    @staticmethod
    def _consecutive_down(closes: list[float]) -> int:
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
        return count

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # --- Compute indicators ---
        sma = self._sma(closes, self.sma_period)
        std = self._std(closes, self.sma_period)
        if sma is None or std is None or std < 1e-9:
            return None

        zscore = (bar.close - sma) / std
        consec = self._consecutive_down(closes)

        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range

        # --- Confluence scoring: count confirming factors ---
        factors = 0
        if consec >= self.consec_down_min:
            factors += 1
        if zscore <= self.zscore_threshold:
            factors += 1
        if ibs <= self.ibs_threshold:
            factors += 1

        if factors < self.min_factors:
            return None

        # --- Drawdown guard ---
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.drawdown_max:
            return None

        # --- Regime filter (skip DOWN trend and HIGH/SHOCK vol) ---
        labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = labels.get("regime_trend", "FLAT")
        regime_vol = labels.get("regime_vol", "NORMAL")
        if regime_trend == "DOWN":
            return None
        if regime_vol in ("HIGH", "SHOCK"):
            return None

        # --- ATR + SMA target ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        target = sma if sma > bar.close else bar.close + self.target_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "factors": factors,
                "consec_down": consec,
                "zscore": round(zscore, 2),
                "ibs": round(ibs, 3),
                "atr": round(atr, 4),
                "regime": f"{regime_trend}+{regime_vol}",
            },
        )
