"""Trend-pullback strategy: buy dips in confirmed uptrends.

Entry: EMA(20) > EMA(50) (uptrend) + price pulls back near EMA(20) + volume OK.
Naturally avoids DOWN regime windows since EMA alignment won't be bullish.
Asymmetric R:R: stop 1.5 ATR, target 3.0 ATR (2:1 payoff).
Skip SHOCK vol + earnings/FOMC.
Long-only, daily bars, max_hold_days=7.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v9a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.ema_fast = int(config.get("ema_fast", 20))
        self.ema_slow = int(config.get("ema_slow", 50))
        self.pullback_pct = float(config.get("pullback_pct", 0.02))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_avg_mult = float(config.get("vol_avg_mult", 0.8))

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

        # Skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Skip earnings and FOMC
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # EMA crossover: fast > slow = uptrend
        ema_f = self._ema(closes, self.ema_fast)
        ema_s = self._ema(closes, self.ema_slow)
        if ema_f is None or ema_s is None:
            return None
        if ema_f <= ema_s:
            return None  # No uptrend — don't trade

        # Price must have pulled back near or below the fast EMA
        distance = (bar.close - ema_f) / ema_f
        if distance > self.pullback_pct:
            return None  # Too extended above EMA — not a pullback

        # Volume check: at least vol_avg_mult of 20-day average
        if len(volumes) >= self.vol_avg_period:
            avg_vol = sum(volumes[-self.vol_avg_period :]) / self.vol_avg_period
            if avg_vol > 0 and bar.volume < avg_vol * self.vol_avg_mult:
                return None

        # ATR for stop and target
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
                "ema_fast": round(ema_f, 2),
                "ema_slow": round(ema_s, 2),
                "pullback_pct": round(distance, 4),
                "atr": round(atr, 4),
                "seed": "mean_reversion",
            },
        )
