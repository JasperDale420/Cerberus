"""daily_research_v7a — Multi-factor composite score mean reversion.

Archetype: Soft-scoring mean reversion (no hard trend gate).
Entry: composite score from z-score + IBS + volume + consecutive downs + regime.
Regime adapts the entry threshold rather than blocking trades entirely.
Target: BB midline or ATR-based. Stop: 2x ATR. Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v7a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.zscore_entry = float(config.get("zscore_entry", 1.5))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        self.vol_avg_mult = float(config.get("vol_avg_mult", 0.5))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.drawdown_max = float(config.get("drawdown_max", 0.15))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
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
    def _count_down_days(closes: list[float], max_look: int = 5) -> int:
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
            if count >= max_look:
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

        # Bollinger Band z-score
        sma = self._sma(closes, self.bb_period)
        std = self._std(closes, self.bb_period)
        if sma is None or std is None or std < 1e-9:
            return None

        zscore = (bar.close - sma) / std

        # ATR for stop/target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # IBS (Internal Bar Strength)
        bar_range = bar.high - bar.low
        ibs = (bar.close - bar.low) / bar_range if bar_range > 1e-9 else 0.5

        # Volume ratio
        volumes = [b.volume for b in bars[-20:]]
        avg_vol = sum(volumes) / len(volumes) if volumes else 1
        vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 1.0

        # Hard drawdown filter
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        drawdown = (peak - bar.close) / peak if peak > 0 else 0
        if drawdown > self.drawdown_max:
            return None

        # Regime labels
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "FLAT")
        vol_regime = labels.get("regime_vol", "NORMAL")

        # Skip SHOCK volatility
        if vol_regime == "SHOCK":
            return None

        # Skip near earnings
        if labels.get("near_earnings", False):
            return None

        # Consecutive down days
        down_days = self._count_down_days(closes)

        # --- Multi-factor composite score (0-100) ---
        score = 0.0

        # Factor 1: Z-score oversold (0-30 pts)
        if zscore < -self.zscore_entry:
            score += min(30.0, 10.0 * abs(zscore))
        elif zscore < -0.5:
            score += 10.0 * (abs(zscore) - 0.5) / max(self.zscore_entry - 0.5, 0.01)

        # Factor 2: Low IBS (0-25 pts)
        if ibs < self.ibs_threshold:
            score += 20.0 * (1.0 - ibs / self.ibs_threshold)
            if ibs < 0.15:
                score += 5.0
        elif ibs < 0.5:
            score += 5.0

        # Factor 3: Volume confirmation (0-15 pts)
        if vol_ratio >= self.vol_avg_mult:
            score += min(15.0, 5.0 * vol_ratio)

        # Factor 4: Drawdown safety (0-15 pts)
        safety = 1.0 - drawdown / self.drawdown_max
        score += 15.0 * safety

        # Factor 5: Consecutive down days (0-20 pts)
        score += min(20.0, 10.0 * down_days)

        # Regime-adaptive threshold
        if trend == "UP":
            threshold = 40.0
        elif trend == "FLAT":
            threshold = 45.0
        else:  # DOWN
            threshold = 60.0

        if score < threshold:
            return None

        # Target: BB midline (SMA) or ATR-based, whichever is higher
        target = max(sma, bar.close + 1.5 * atr)
        if target <= bar.close:
            return None

        stop = bar.close - self.stop_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "zscore": round(zscore, 2),
                "ibs": round(ibs, 3),
                "score": round(score, 1),
                "down_days": down_days,
                "trend": trend,
                "threshold": threshold,
            },
        )
