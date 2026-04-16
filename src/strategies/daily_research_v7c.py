"""Vol-Adaptive IBS Mean Reversion — buy oversold closes with ATR confirmation.

IBS (Internal Bar Strength) = (close - low) / (high - low).
Low IBS means close near day's low (oversold). Buy and target mean reversion.
ATR filter prevents buying into expanding volatility crashes.
Evolved from vol_breakout seed — uses ATR as vol-awareness layer.
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
        self.min_bars = int(config.get("min_bars", 25))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.15))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_avg_period = int(config.get("atr_avg_period", 20))
        self.max_atr_expansion = float(config.get("max_atr_expansion", 1.8))
        self.ema_period = int(config.get("ema_period", 20))
        self.ema_slow_period = int(config.get("ema_slow_period", 50))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 0.5))
        self.max_hold_days = int(config.get("max_hold_days", 3))

    # --- Indicator helpers ---

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = v * mult + ema * (1 - mult)
        return ema

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

        # Skip near-earnings
        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings"):
            return None

        # Skip DOWN trend
        regime_trend = regime_labels.get("regime_trend", "FLAT")
        if regime_trend == "DOWN":
            return None

        # Skip SHOCK volatility
        regime_vol = regime_labels.get("regime_vol", "NORMAL")
        if regime_vol == "SHOCK":
            return None

        # IBS calculation
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range

        # Entry: IBS below threshold (close near day's low = oversold)
        if ibs > self.ibs_threshold:
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # ATR current
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Vol filter: ATR should NOT be expanding rapidly (avoid crashes)
        atr_values = self._atr_series(bars, self.atr_period, self.atr_avg_period)
        if len(atr_values) >= self.atr_avg_period:
            atr_avg = sum(atr_values) / len(atr_values)
            if atr_avg > 1e-9 and atr > self.max_atr_expansion * atr_avg:
                return None  # Vol expanding too fast — likely a crash

        # Trend filter: EMA(20) > EMA(50) or close > EMA(50)
        ema_fast = self._ema(closes, self.ema_period)
        ema_slow = self._ema(closes, self.ema_slow_period)
        if ema_fast is None or ema_slow is None:
            return None

        # Require EMA alignment (uptrend) — strict filter
        if ema_fast <= ema_slow:
            return None

        # Stop and target
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
                "ibs": round(ibs, 3),
                "atr": round(atr, 4),
                "ema_fast": round(ema_fast, 2),
                "ema_slow": round(ema_slow, 2),
                "seed": "vol_breakout_evolved",
            },
        )
