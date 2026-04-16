"""Iteration 5: Dual-Mode Regime-Adaptive strategy.

UP trend: buy pullbacks to EMA(20) when EMA(20) > EMA(50).
FLAT trend: buy low-IBS dips after 2 down days (mean reversion).
DOWN trend: skip entirely.
Skip HIGH/SHOCK vol. Skip Monday.
Long-only, daily bars, max_hold_days=5.
"""

from __future__ import annotations

from datetime import datetime
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
        # EMA periods for trend detection
        self.ema_fast = int(config.get("ema_fast", 20))
        self.ema_slow = int(config.get("ema_slow", 50))
        # Pullback threshold (% below fast EMA to buy)
        self.pullback_pct = float(config.get("pullback_pct", 0.02))
        # IBS threshold for mean reversion mode
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        # Consecutive down days for mean reversion mode
        self.min_down_days = int(config.get("min_down_days", 2))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.15))
        # Stop/target in ATR multiples
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))

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

        # --- Regime filter ---
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "")
        vol = labels.get("regime_vol", "")

        # Skip DOWN trend and HIGH/SHOCK vol
        if trend == "DOWN":
            return None
        if vol in ("HIGH", "SHOCK"):
            return None

        # --- Day-of-week filter: skip Monday ---
        bar_time = bar.time
        if isinstance(bar_time, datetime):
            if bar_time.weekday() == 0:
                return None

        # --- Drawdown filter ---
        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs)
        dd = (peak - bar.close) / peak if peak > 0 else 0
        if dd > self.max_drawdown_pct:
            return None

        # --- ATR for stop/target ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # --- Compute EMAs ---
        ema_f = self._ema(closes, self.ema_fast)
        ema_s = self._ema(closes, self.ema_slow)

        mode = None

        # --- Mode 1: Trend Pullback (UP trend with EMA alignment) ---
        if ema_f is not None and ema_s is not None and ema_f > ema_s:
            # Price has pulled back to near or below fast EMA
            pct_from_ema = (bar.close - ema_f) / ema_f
            if -self.pullback_pct <= pct_from_ema <= 0.005:
                mode = "TREND_PULLBACK"

        # --- Mode 2: Mean Reversion (FLAT trend, IBS + down days) ---
        if mode is None and trend == "FLAT":
            bar_range = bar.high - bar.low
            if bar_range > 1e-9:
                ibs = (bar.close - bar.low) / bar_range
                if ibs < self.ibs_threshold:
                    # Check consecutive down days
                    if len(closes) >= self.min_down_days + 1:
                        all_down = True
                        for i in range(1, self.min_down_days + 1):
                            if closes[-i] >= closes[-i - 1]:
                                all_down = False
                                break
                        if all_down:
                            mode = "MEAN_REVERSION"

        if mode is None:
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
                "mode": mode,
                "atr": round(atr, 4),
                "trend": trend,
                "vol": vol,
                "dd": round(dd, 4),
            },
        )
