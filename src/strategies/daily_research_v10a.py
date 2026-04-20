"""Daily Research v10a — Consecutive Down Days Trend Pullback.

Buy dips in confirmed uptrends using consecutive down close detection,
trend confirmation via moving average alignment, and ATR-based risk management.
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
        # Trend filter
        self.trend_sma_period = int(config.get("trend_sma_period", 50))
        # Pullback detection
        self.consec_down_days = int(config.get("consec_down_days", 3))
        self.pullback_pct = float(config.get("pullback_pct", 0.03))
        # IBS filter (close near low = more oversold)
        self.ibs_threshold = float(config.get("ibs_threshold", 0.4))
        # Risk management
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.drawdown_max = float(config.get("drawdown_max", 0.15))

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
    def _consecutive_down(closes: list[float], n: int) -> bool:
        """Check if last n closes are each lower than their predecessor."""
        if len(closes) < n + 1:
            return False
        for i in range(-n, 0):
            if closes[i] >= closes[i - 1]:
                return False
        return True

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

        # Trend filter: price above rising SMA
        sma = self._sma(closes, self.trend_sma_period)
        if sma is None:
            return None
        if bar.close < sma:
            return None
        # SMA must be rising (current > 5 bars ago)
        sma_prev = self._sma(closes[:-5], self.trend_sma_period)
        if sma_prev is not None and sma <= sma_prev:
            return None

        # Consecutive down days
        if not self._consecutive_down(closes, self.consec_down_days):
            return None

        # Pullback magnitude: price must have pulled back at least pullback_pct from recent high
        lookback_high = max(closes[-20:])
        pullback = (lookback_high - bar.close) / lookback_high
        if pullback < self.pullback_pct:
            return None

        # IBS filter: close near low of day
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None

        # Drawdown filter: not in freefall
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.drawdown_max:
            return None

        # ATR for stops and targets
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        target = bar.close + self.target_atr_mult * atr

        # Regime filter: skip DOWN trend
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "UP")
        if trend == "DOWN":
            return None

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "consec_down": self.consec_down_days,
                "pullback_pct": round(pullback, 4),
                "ibs": round(ibs, 3),
                "atr": round(atr, 4),
                "trend": trend,
                "seed": "mean_reversion",
            },
        )
