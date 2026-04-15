"""daily_research_v7a — Trend Pullback with Volatility Squeeze.

Archetype: Trend Pullback (NOT mean reversion)
Buy dips in established uptrends. EMA alignment confirms trend,
ATR-normalized pullback depth finds entries.
Uses volatility contraction (narrowing ATR) as setup condition.
Long-only, daily bars.
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
        self.ema_fast = int(config.get("ema_fast", 10))
        self.ema_slow = int(config.get("ema_slow", 40))
        self.atr_period = int(config.get("atr_period", 14))
        self.pullback_atr = float(config.get("pullback_atr", 1.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 10))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.12))
        self.vol_squeeze_ratio = float(config.get("vol_squeeze_ratio", 1.2))

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
    def _atr_at(bars: list[Bar], end_idx: int, period: int) -> Optional[float]:
        """ATR ending at a specific index."""
        if end_idx < period:
            return None
        trs = []
        for i in range(end_idx - period + 1, end_idx + 1):
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

        # --- EMA trend alignment ---
        ema_f = self._ema(closes, self.ema_fast)
        ema_s = self._ema(closes, self.ema_slow)
        if ema_f is None or ema_s is None:
            return None

        # Must be in uptrend: fast EMA above slow EMA
        if ema_f <= ema_s:
            return None

        # Price must be above slow EMA (confirms we're in the trend)
        if bar.close <= ema_s:
            return None

        # --- ATR ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # --- Pullback detection ---
        # Price pulled back from recent high by at least pullback_atr * ATR
        recent_high = max(highs[-10:])
        pullback_depth = recent_high - bar.close
        if pullback_depth < self.pullback_atr * atr:
            return None  # Not enough pullback

        # But not too deep — shouldn't be more than 3x ATR (trend break)
        if pullback_depth > 3.0 * atr:
            return None

        # --- Volatility squeeze (optional boost) ---
        # Current ATR vs ATR from 20 bars ago — contracting vol suggests impending move
        atr_old = self._atr_at(bars, len(bars) - 20, self.atr_period) if len(bars) > 35 else None
        if atr_old is not None and atr_old > 1e-9:
            vol_ratio = atr / atr_old
            # Skip if volatility is expanding too much (unstable)
            if vol_ratio > self.vol_squeeze_ratio:
                return None

        # --- Drawdown filter ---
        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.max_drawdown_pct:
            return None

        # --- Trend strength: EMA spread must be meaningful ---
        ema_spread = (ema_f - ema_s) / ema_s
        if ema_spread < 0.005:
            return None  # Trend too weak

        # --- Stop and target ---
        stop = bar.close - self.stop_atr_mult * atr
        target = bar.close + self.target_atr_mult * atr

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
                "ema_spread": round(ema_spread, 4),
                "pullback_atr": round(pullback_depth / atr, 2),
                "atr": round(atr, 4),
                "archetype": "trend_pullback",
            },
        )
