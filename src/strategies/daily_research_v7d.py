"""Trend Pullback — buy dips in confirmed uptrends.

Entry: EMA(10) > EMA(30) (uptrend) AND price pulls back to within
pullback_pct of EMA(10). This naturally filters bear markets since
the EMA crossover won't be bullish.

Second confirmation: prior day was a down close (buying the dip).

Stop: below recent N-day low (swing low).
Target: risk-reward ratio applied to stop distance.

Trades in all regimes but EMA crossover naturally limits DOWN entries.
Skip SHOCK vol, earnings, FOMC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv7dStrategy(BaseStrategy):
    name = "daily_research_v7d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 35))
        self.ema_fast = int(config.get("ema_fast", 10))
        self.ema_slow = int(config.get("ema_slow", 30))
        self.pullback_pct = float(config.get("pullback_pct", 0.02))
        self.atr_period = int(config.get("atr_period", 14))
        self.risk_reward = float(config.get("risk_reward", 2.0))
        self.swing_low_period = int(config.get("swing_low_period", 5))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.03))

    # --- Indicator helpers ---

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = sum(values[:period]) / period
        for v in values[period:]:
            ema = (v - ema) * mult + ema
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
        lows = [b.low for b in bars]

        if len(closes) < self.min_bars:
            return None

        # ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Min price filter
        if bar.close < 5.0:
            return None

        # ATR/price filter
        if atr / bar.close < 0.005:
            return None

        # EMA crossover: fast > slow = uptrend
        ema_f = self._ema(closes, self.ema_fast)
        ema_s = self._ema(closes, self.ema_slow)
        if ema_f is None or ema_s is None:
            return None
        if ema_f <= ema_s:
            return None

        # Pullback: price near or just below the fast EMA
        distance = (bar.close - ema_f) / ema_f
        if distance > self.pullback_pct:
            return None  # Too far above — chasing
        if distance < -self.pullback_pct * 2:
            return None  # Too far below — broken trend

        # Confirmation: prior day was a down close
        if len(closes) >= 2 and closes[-1] >= closes[-2]:
            return None

        # Swing low stop: lowest low of recent N bars
        if len(lows) >= self.swing_low_period:
            swing_low = min(lows[-self.swing_low_period :])
        else:
            swing_low = min(lows)

        # Cap stop loss
        max_stop_price = bar.close * (1.0 - self.max_stop_pct)
        stop = max(swing_low - atr * 0.1, max_stop_price)

        # Ensure stop is below entry
        if stop >= bar.close:
            return None

        # Risk-reward target
        risk = bar.close - stop
        target = bar.close + risk * self.risk_reward

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
                "pullback_dist": round(distance, 4),
                "swing_low": round(swing_low, 2),
                "risk": round(risk, 2),
                "atr": round(atr, 4),
                "seed": "trend_pullback",
            },
        )
