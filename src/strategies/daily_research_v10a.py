"""Daily Research v10a — Multi-Factor Oversold Score.

Regime-adaptive mean reversion using a composite oversold score from:
- Consecutive down days
- IBS (Internal Bar Strength)
- Distance from 20-day high (pullback depth)
- Volume confirmation

Works across all regimes with adaptive stops.
Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v10a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        # Oversold scoring thresholds
        self.consec_down_min = int(config.get("consec_down_min", 2))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        self.pullback_pct_min = float(config.get("pullback_pct_min", 0.02))
        self.min_score = float(config.get("min_score", 2.5))
        self.min_score_down = float(config.get("min_score_down", 4.0))
        # Risk management
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 4))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 50))
        self.drawdown_max = float(config.get("drawdown_max", 0.15))
        # Trend SMA for regime adaptation
        self.trend_sma_period = int(config.get("trend_sma_period", 40))
        # Volume confirmation
        self.vol_sma_period = int(config.get("vol_sma_period", 20))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

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
    def _count_consecutive_down(closes: list[float]) -> int:
        """Count consecutive down closes from the end."""
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

        # Composite oversold score
        score = 0.0

        # Factor 1: Consecutive down days
        consec_down = self._count_consecutive_down(closes)
        if consec_down >= self.consec_down_min:
            score += min(consec_down - self.consec_down_min + 1, 3)  # Cap at 3 points

        # Factor 2: IBS (close near low)
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs < self.ibs_threshold:
            score += 1.0
            if ibs < 0.15:
                score += 0.5  # Extra for very low IBS

        # Factor 3: Pullback from recent high
        recent_high = max(closes[-20:])
        pullback = (recent_high - bar.close) / recent_high if recent_high > 0 else 0
        if pullback >= self.pullback_pct_min:
            score += 1.0
            if pullback >= 0.05:
                score += 0.5  # Extra for deeper pullback

        # Regime-aware minimum score
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "FLAT")

        # In DOWN regimes, require much stronger oversold signal
        required_score = self.min_score_down if trend == "DOWN" else self.min_score
        if score < required_score:
            return None

        # Volume confirmation: today's volume should be above average
        volumes = [b.volume for b in bars[-self.vol_sma_period :]]
        if len(volumes) >= self.vol_sma_period:
            avg_vol = sum(volumes) / len(volumes)
            if bar.volume < avg_vol * 0.7:
                return None

        # Drawdown filter: avoid catching falling knives
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.drawdown_max:
            return None

        # ATR for stops and targets
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Regime-adaptive targets
        if trend == "DOWN":
            target_mult = self.target_atr_mult * 0.6
            stop_mult = self.stop_atr_mult * 0.7
        elif trend == "UP":
            target_mult = self.target_atr_mult * 1.2
            stop_mult = self.stop_atr_mult
        else:
            target_mult = self.target_atr_mult
            stop_mult = self.stop_atr_mult

        stop = bar.close - stop_mult * atr
        target = bar.close + target_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "score": round(score, 1),
                "consec_down": consec_down,
                "pullback_pct": round(pullback, 4),
                "ibs": round(ibs, 3),
                "atr": round(atr, 4),
                "trend": trend,
                "seed": "mean_reversion",
            },
        )
