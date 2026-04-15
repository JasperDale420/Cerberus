"""Multi-Factor Confluence Strategy.

Computes a composite score from multiple weak signals:
  - ATR expansion (volatility breakout component)
  - Price vs Bollinger Bands (mean-reversion component)
  - Volume surge
  - SMA trend alignment
  - Recent return momentum

Only enters when composite score exceeds threshold.
Different factors dominate in different regimes, providing natural adaptation.
Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v7c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.atr_period = int(config.get("atr_period", 14))

        # Optimizable params
        self.atr_expansion = float(config.get("atr_expansion", 1.3))
        self.vol_surge_mult = float(config.get("vol_surge_mult", 1.3))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))

        # Fixed structural params
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std_mult = float(config.get("bb_std_mult", 2.0))
        self.sma_fast = int(config.get("sma_fast", 10))
        self.sma_slow = int(config.get("sma_slow", 30))
        self.breakout_lookback = int(config.get("breakout_lookback", 10))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        self.min_score = float(config.get("min_score", 3.0))

    # --- Indicator helpers ---

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
    def _sma_vals(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        data = values[-period:]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return variance**0.5

    def _atr_series(self, bars: list[Bar], period: int, count: int) -> list[float]:
        result = []
        for i in range(count):
            end_idx = len(bars) - count + i + 1
            if end_idx < period + 1:
                continue
            sub = bars[:end_idx]
            val = self._atr(sub, period)
            if val is not None:
                result.append(val)
        return result

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

        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings"):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        price = bar.close

        # --- Compute factor scores (each 0 or 1) ---
        score = 0.0
        mode_hints = []

        # Factor 1: ATR expansion (vol breakout signal)
        atr_values = self._atr_series(bars, self.atr_period, 20)
        atr_ratio = 1.0
        if len(atr_values) >= 10:
            atr_avg = sum(atr_values) / len(atr_values)
            if atr_avg > 1e-9:
                atr_ratio = atr / atr_avg
                if atr_ratio >= self.atr_expansion:
                    score += 1.0
                    mode_hints.append("atr_exp")

        # Factor 2: Price breakout (new N-day high)
        if len(bars) >= self.breakout_lookback + 1:
            lookback_highs = [b.high for b in bars[-(self.breakout_lookback + 1) : -1]]
            if price > max(lookback_highs):
                score += 1.0
                mode_hints.append("breakout")

        # Factor 3: Volume surge
        avg_vol = self._sma_vals(volumes, 20)
        vol_ratio = 0.0
        if avg_vol is not None and avg_vol > 1e-9:
            vol_ratio = bar.volume / avg_vol
            if vol_ratio >= self.vol_surge_mult:
                score += 1.0
                mode_hints.append("vol_surge")

        # Factor 4: SMA trend alignment (fast > slow, price > fast)
        sma_f = self._sma_vals(closes, self.sma_fast)
        sma_s = self._sma_vals(closes, self.sma_slow)
        if sma_f is not None and sma_s is not None:
            if sma_f > sma_s and price > sma_f:
                score += 1.0
                mode_hints.append("trend_up")
            elif sma_f > sma_s and price >= sma_s:
                score += 0.5
                mode_hints.append("trend_mild")

        # Factor 5: BB position (oversold bounce or above midline)
        bb_mean = self._sma_vals(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_mean is not None and bb_std is not None and bb_std > 0.01:
            lower_band = bb_mean - self.bb_std_mult * bb_std
            if price < lower_band:
                # Oversold — mean reversion opportunity
                score += 1.5
                mode_hints.append("oversold")
            elif price < bb_mean and price > lower_band:
                # Below mean but not oversold — mild signal
                score += 0.5
                mode_hints.append("below_mean")

        # Factor 6: Recent positive momentum (3-day return > 0 after pullback)
        if len(closes) >= 5:
            ret_3d = (price - closes[-4]) / closes[-4] if closes[-4] > 0 else 0
            ret_5d = (price - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0
            # Positive 3d return after negative 5d = bounce
            if ret_3d > 0.005 and ret_5d < -0.01:
                score += 1.0
                mode_hints.append("bounce")

        # --- Regime adjustment ---
        trend = regime_labels.get("regime_trend", "FLAT")
        vol_regime = regime_labels.get("regime_vol", "NORMAL")

        # Skip SHOCK
        if vol_regime == "SHOCK":
            return None

        # DOWN trend penalty — require stronger confluence
        if trend == "DOWN":
            score -= 0.5

        # HIGH vol bonus for oversold bounces
        if vol_regime == "HIGH" and "oversold" in mode_hints:
            score += 0.5

        # --- Entry decision ---
        if score < self.min_score:
            return None

        # Determine stop/target based on dominant mode
        if "oversold" in mode_hints:
            # Mean-reversion: target the BB mean
            target = bb_mean if bb_mean is not None else price + self.target_atr_mult * atr
        else:
            target = price + self.target_atr_mult * atr

        stop = price - self.stop_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "mode": "+".join(mode_hints[:3]),
                "score": round(score, 1),
                "atr_ratio": round(atr_ratio, 2),
                "trend": trend,
            },
        )
