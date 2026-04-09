"""Daily Research v6c — RSI(2) Mean Reversion with Momentum Guard.

Iteration 5: Vol-adaptive stops + tighter cap.
Iter 4: min_pf=0.51, 457 trades. DOWN+HIGH avg_pf=0.89 (close!).
One bad DOWN+HIGH window (PF=0.51) drags min_pf.

Changes from iter4:
1. Vol-adaptive: when ATR > 2% of price (high vol), tighten stop
   multiplier to 1.0x ATR to limit per-trade loss.
2. Max stop cap 2.5% (down from 3%).
3. Target stays 1.5x ATR — favorable risk/reward in high vol.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv6cStrategy(BaseStrategy):
    name = "daily_research_v6c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 20))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_entry = float(config.get("rsi_entry", 25))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.12))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.025))
        self.momentum_lookback = int(config.get("momentum_lookback", 5))
        self.high_vol_threshold = float(config.get("high_vol_threshold", 0.02))
        self.high_vol_stop_mult = float(config.get("high_vol_stop_mult", 1.0))
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

        # Drawdown filter
        lookback = min(self.drawdown_lookback, len(highs))
        recent_high = max(highs[-lookback:])
        drawdown = 0.0
        if recent_high > 0:
            drawdown = (recent_high - bar.close) / recent_high
            if drawdown > self.max_drawdown_pct:
                return None

        # Momentum guard: don't buy if price is at a new 5-bar low
        if len(closes) >= self.momentum_lookback + 1:
            recent_min = min(closes[-(self.momentum_lookback + 1) : -1])
            if bar.close < recent_min:
                return None

        # RSI(2) oversold entry
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi >= self.rsi_entry:
            return None

        # ATR for stop/target
        atr = self._atr(bars, 14)
        if atr is None or atr < 0.01:
            return None

        # Vol-adaptive stop: tighter in high-ATR environments
        atr_pct = atr / bar.close
        if atr_pct > self.high_vol_threshold:
            stop_mult = self.high_vol_stop_mult
        else:
            stop_mult = self.stop_atr_mult

        max_dist = bar.close * self.max_stop_pct
        stop_dist = min(atr * stop_mult, max_dist)
        target_dist = min(atr * self.target_atr_mult, max_dist)
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
                "drawdown": round(drawdown, 3),
                "atr_pct": round(atr_pct, 4),
            },
        )
