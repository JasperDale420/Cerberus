"""Daily Research Strategy v6b — Dual-Timeframe RSI Mean Reversion.

iter1: Replace SMA(20) with RSI(14)>50 dual-timeframe filter.
- RSI(14) > 50 confirms medium-term momentum is bullish (Connors method)
- RSI(2) < 25 normal, RSI(2) < 10 in high-vol
- Symmetric 2x ATR stop and target (capped at 4% of price)
- 12% drawdown filter (40-bar lookback)
- 5-day max hold, long-only, daily bars

Hypothesis: RSI(14)>50 is more responsive than SMA(20) — doesn't filter
out valid pullback entries where price dips below MA but momentum is still up.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv6bStrategy(BaseStrategy):
    name = "daily_research_v6b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 20))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_entry = float(config.get("rsi_entry", 25))
        self.rsi_entry_highvol = float(config.get("rsi_entry_highvol", 10))
        self.vol_ratio_threshold = float(config.get("vol_ratio_threshold", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.rsi_long_period = int(config.get("rsi_long_period", 14))
        self.rsi_long_threshold = float(config.get("rsi_long_threshold", 50))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.12))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.04))
        self.allow_overnight = True

    def _rsi(self, closes: list[float], period: int) -> float | None:
        if len(closes) < period + 1:
            return None
        gains = 0.0
        losses = 0.0
        for i in range(len(closes) - period, len(closes)):
            change = closes[i] - closes[i - 1]
            if change > 0:
                gains += change
            else:
                losses -= change
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

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
        highs = [b.high for b in bars]

        if len(closes) < self.min_bars:
            return None

        # Dual-timeframe filter: RSI(14) > 50 confirms bullish momentum
        rsi_long = self._rsi(closes, self.rsi_long_period)
        if rsi_long is not None and rsi_long < self.rsi_long_threshold:
            return None

        # Drawdown filter
        lookback = min(self.drawdown_lookback, len(highs))
        recent_high = max(highs[-lookback:])
        drawdown = 0.0
        if recent_high > 0:
            drawdown = (recent_high - bar.close) / recent_high
            if drawdown > self.max_drawdown_pct:
                return None

        # RSI(2)
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None:
            return None

        # ATR calculations
        atr_short = self._atr(bars, 5)
        atr_long = self._atr(bars, 14)
        if atr_short is None or atr_long is None or atr_long < 0.01:
            return None

        # Volatility-adaptive RSI threshold (ATR ratio only)
        vol_ratio = atr_short / atr_long
        if vol_ratio > self.vol_ratio_threshold:
            effective_rsi_entry = self.rsi_entry_highvol
        else:
            effective_rsi_entry = self.rsi_entry

        if rsi >= effective_rsi_entry:
            return None

        # Stop/target with cap at max_stop_pct of price
        max_dist = bar.close * self.max_stop_pct
        stop_dist = min(atr_long * self.stop_atr_mult, max_dist)
        target_dist = min(atr_long * self.target_atr_mult, max_dist)
        stop_price = bar.close - stop_dist
        target_price = bar.close + target_dist

        self.last_signal_time[symbol] = bar.time
        return Signal(
            symbol=symbol,
            side=OrderSide.BUY,
            size_hint=0.0,
            entry_price=bar.close,
            stop_price=stop_price,
            target_price=target_price,
            strategy=self.name,
            generated_at=bar.time,
            meta={
                "rsi2": round(rsi, 1),
                "vol_ratio": round(vol_ratio, 2),
                "drawdown": round(drawdown, 3),
            },
        )
