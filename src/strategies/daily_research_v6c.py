"""Daily Research v6c — RSI(2) + IBS Mean Reversion.

Iteration 3: IBS confirmation + asymmetric risk/reward.
Iter 1: min_pf=0.60, SMA(50) too restrictive, DOWN+HIGH=0.00.
Iter 2: min_pf=0.30, regime gate ineffective (per-stock, not per-window),
        UP+NORMAL windows still inconsistent (PF=0.30-0.48).

Changes: Add IBS < 0.3 filter for quality entries. Asymmetric
stop/target (wider stop 2.5 ATR, tighter target 1.5 ATR) to improve
win rate. Remove regime gate (ineffective). Keep SMA(20) trend filter.
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
        self.min_bars = int(config.get("min_bars", 25))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_entry = float(config.get("rsi_entry", 30))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        self.sma_period = int(config.get("sma_period", 20))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.15))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.05))
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

    def _sma(self, values: list[float], period: int) -> float | None:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

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

        # SMA(20) trend filter — only buy in uptrend
        sma = self._sma(closes, self.sma_period)
        if sma is not None and bar.close < sma:
            return None

        # Drawdown filter — skip stocks in freefall
        lookback = min(self.drawdown_lookback, len(highs))
        recent_high = max(highs[-lookback:])
        drawdown = 0.0
        if recent_high > 0:
            drawdown = (recent_high - bar.close) / recent_high
            if drawdown > self.max_drawdown_pct:
                return None

        # RSI(2) oversold entry
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi >= self.rsi_entry:
            return None

        # IBS confirmation — close must be in bottom of daily range
        day_range = bar.high - bar.low
        if day_range > 0:
            ibs = (bar.close - bar.low) / day_range
            if ibs > self.ibs_threshold:
                return None

        # ATR for stop/target
        atr = self._atr(bars, 14)
        if atr is None or atr < 0.01:
            return None

        # Asymmetric: wider stop (survive), tighter target (take profit)
        max_dist = bar.close * self.max_stop_pct
        stop_dist = min(atr * self.stop_atr_mult, max_dist)
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
                "ibs": round(ibs if day_range > 0 else 0.0, 2),
                "drawdown": round(drawdown, 3),
            },
        )
