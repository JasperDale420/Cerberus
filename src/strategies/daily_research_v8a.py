"""V8a: Consecutive-Down + Z-Score + IBS Mean Reversion.

Multi-factor mean reversion using consecutive down closes, distance from
moving average (z-score), and Internal Bar Strength. No RSI dependency.
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
        # Consecutive down days
        self.consec_down_min = int(config.get("consec_down_min", 3))
        # Z-score (distance from SMA)
        self.sma_period = int(config.get("sma_period", 20))
        self.zscore_threshold = float(config.get("zscore_threshold", -1.0))
        # IBS filter
        self.ibs_threshold = float(config.get("ibs_threshold", 0.2))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 50))
        self.drawdown_max = float(config.get("drawdown_max", 0.12))
        # Volume filter
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_mult = float(config.get("vol_min_mult", 0.5))
        # ATR for stop/target
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 7))

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
        """Count consecutive down closes from the most recent bar."""
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

        # --- Event filter (skip earnings/FOMC unpredictability) ---
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False):
            return None
        if labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # --- Filter 1: Consecutive down days ---
        consec = self._consecutive_down(closes)
        if consec < self.consec_down_min:
            return None

        # --- Filter 2: Z-score below threshold ---
        sma = self._sma(closes, self.sma_period)
        std = self._std(closes, self.sma_period)
        if sma is None or std is None or std < 1e-9:
            return None
        zscore = (bar.close - sma) / std
        if zscore > self.zscore_threshold:
            return None

        # --- Filter 3: IBS (close near low = selling exhaustion) ---
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None

        # --- Filter 4: Drawdown guard (don't buy into crashes) ---
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.drawdown_max:
            return None

        # --- Filter 5: Regime filter (skip DOWN trend entirely) ---
        regime_trend = labels.get("regime_trend", "FLAT")
        regime_vol = labels.get("regime_vol", "NORMAL")
        if regime_trend == "DOWN":
            return None
        if regime_vol in ("HIGH", "SHOCK"):
            return None

        # --- Filter 6: Volume filter (minimum participation) ---
        volumes = [b.volume for b in bars]
        if len(volumes) >= self.vol_avg_period:
            avg_vol = sum(volumes[-self.vol_avg_period :]) / self.vol_avg_period
            if avg_vol > 0 and bar.volume < avg_vol * self.vol_min_mult:
                return None

        # --- ATR for stop/target ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        target = bar.close + self.target_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "consec_down": consec,
                "zscore": round(zscore, 2),
                "ibs": round(ibs, 3),
                "atr": round(atr, 4),
                "regime": f"{regime_trend}+{regime_vol}",
                "drawdown_from_peak": round((peak - bar.close) / peak, 4),
            },
        )
