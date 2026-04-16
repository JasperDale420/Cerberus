"""Iteration 1: Keltner Channel + IBS Mean Reversion with Trend Filter.

Buy when price dips below lower Keltner Channel, IBS is low (close near day low),
and price is in an uptrend (above SMA). Skip DOWN+HIGH regimes.
Long-only, daily bars, max_hold_days=5.
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
        self.min_bars = int(config.get("min_bars", 55))
        # Keltner Channel params
        self.kc_ema_period = int(config.get("kc_ema_period", 20))
        self.kc_atr_period = int(config.get("kc_atr_period", 14))
        self.kc_atr_mult = float(config.get("kc_atr_mult", 2.0))
        # IBS threshold (Internal Bar Strength — close relative to day range)
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        # Trend filter
        self.trend_sma_period = int(config.get("trend_sma_period", 50))
        # Consecutive down days required (0 = disabled)
        self.min_down_days = int(config.get("min_down_days", 2))
        # Stop/target
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.15))
        self.max_hold_days = int(config.get("max_hold_days", 5))

    # --- Indicator helpers ---

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        result = values[0]
        for v in values[1:]:
            result = v * mult + result * (1 - mult)
        return result

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
        highs = [b.high for b in bars]
        [b.low for b in bars]

        # --- Regime filter: skip DOWN+HIGH and DOWN+SHOCK ---
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "")
        vol = labels.get("regime_vol", "")
        if trend == "DOWN" and vol in ("HIGH", "SHOCK"):
            return None

        # --- Trend filter: price must be above SMA(50) ---
        sma_trend = self._sma(closes, self.trend_sma_period)
        if sma_trend is None:
            return None
        if bar.close < sma_trend:
            return None

        # --- Keltner Channel ---
        ema_mid = self._ema(closes, self.kc_ema_period)
        atr = self._atr(bars, self.kc_atr_period)
        if ema_mid is None or atr is None or atr < 1e-9:
            return None
        lower_kc = ema_mid - self.kc_atr_mult * atr

        # Price must be below lower Keltner Channel
        if bar.close > lower_kc:
            return None

        # --- IBS filter: close must be near the low of the day ---
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None

        # --- Consecutive down days filter ---
        if self.min_down_days > 0:
            if len(closes) < self.min_down_days + 1:
                return None
            for i in range(1, self.min_down_days + 1):
                if closes[-i] >= closes[-i - 1]:
                    return None

        # --- Drawdown filter: skip if too far from recent high ---
        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.max_drawdown_pct:
            return None

        # --- ATR for stop/target ---
        stop_atr = self._atr(bars, self.atr_period)
        if stop_atr is None or stop_atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * stop_atr
        target = bar.close + self.target_atr_mult * stop_atr

        # Sanity: target must be above entry
        if target <= bar.close:
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
                "ibs": round(ibs, 3),
                "kc_lower": round(lower_kc, 2),
                "ema_mid": round(ema_mid, 2),
                "atr": round(atr, 4),
                "trend": trend,
                "vol": vol,
                "drawdown_from_peak": round((peak - bar.close) / peak, 4),
            },
        )
