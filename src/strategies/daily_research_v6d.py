"""Daily Research v6d — Dual EMA Momentum with Pullback Entry.

Trend-following: buy pullbacks in confirmed uptrends.
Long-only. Few parameters. Uses symbol_state.bars directly.

Rules:
  ENTRY: EMA(fast) > EMA(slow) AND price within pullback_pct of EMA(fast)
         AND close > prior close (bounce confirmation)
  EXIT: ATR-based stop/target, capped at max_stop_pct
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv6dStrategy(BaseStrategy):
    name = "daily_research_v6d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.ema_fast = int(config.get("ema_fast", 10))
        self.ema_slow = int(config.get("ema_slow", 30))
        self.pullback_pct = float(config.get("pullback_pct", 0.02))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr = float(config.get("stop_atr", 1.5))
        self.target_atr = float(config.get("target_atr", 2.0))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.02))
        self.allow_overnight = True
        self.max_hold_days = int(config.get("max_hold_days", 5))

    def _ema(self, closes: list[float], period: int) -> float | None:
        if len(closes) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = closes[0]
        for c in closes[1:]:
            ema = c * mult + ema * (1 - mult)
        return ema

    def _atr(self, bars: list, period: int = 14) -> float | None:
        if len(bars) < period + 1:
            return None
        tr_vals = []
        for i in range(len(bars) - period, len(bars)):
            hi, lo, pc = bars[i].high, bars[i].low, bars[i - 1].close
            tr_vals.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
        return sum(tr_vals) / len(tr_vals)

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

        if len(closes) < self.min_bars:
            return None

        price = closes[-1]

        # Dual EMA trend filter
        ema_f = self._ema(closes, self.ema_fast)
        ema_s = self._ema(closes, self.ema_slow)
        if ema_f is None or ema_s is None:
            return None
        if ema_f <= ema_s:
            return None

        # Cross-above: price crossed above fast EMA (yesterday below, today above)
        if len(closes) < 2:
            return None
        prev_ema_f = self._ema(closes[:-1], self.ema_fast)
        if prev_ema_f is None:
            return None
        if closes[-2] >= prev_ema_f:
            return None  # Was already above — not a fresh cross
        if price < ema_f:
            return None  # Still below — hasn't crossed yet

        # ATR for stop/target sizing
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 0.01:
            return None

        # Stop/target capped at max_stop_pct of price
        max_dist = price * self.max_stop_pct
        stop_dist = min(atr * self.stop_atr, max_dist)
        target_dist = min(atr * self.target_atr, max_dist)
        stop_price = price - stop_dist
        target_price = price + target_dist

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={"mode": "EMA_PULLBACK", "ema_spread": round((ema_f - ema_s) / ema_s * 100, 2)},
        )
