"""Daily Research v6c — Composite Mean Reversion.

Session 4, Iteration 1: Composite signal for more trades.
Signal A: IBS < 0.3 + RSI(2) < 50 (proven edge)
Signal B: 2 consecutive down closes (independent edge)
Either signal triggers entry. More trades = more consistent PF per window.
Momentum guard + drawdown filter + ATR stop/target + 2% cap.
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
        self.min_bars = int(config.get("min_bars", 15))
        self.rsi_entry = float(config.get("rsi_entry", 50))
        self.ibs_entry = float(config.get("ibs_entry", 0.3))
        self.momentum_lookback = int(config.get("momentum_lookback", 5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.10))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.02))
        self.consec_down_days = int(config.get("consec_down_days", 2))
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

        # Drawdown filter — skip stocks in freefall
        lookback = min(self.drawdown_lookback, len(highs))
        recent_high = max(highs[-lookback:])
        drawdown = 0.0
        if recent_high > 0:
            drawdown = (recent_high - bar.close) / recent_high
            if drawdown > self.max_drawdown_pct:
                return None

        # Momentum guard — close must be above N days ago
        if len(closes) > self.momentum_lookback:
            if bar.close <= closes[-self.momentum_lookback - 1]:
                return None

        # Signal A: IBS + RSI(2)
        signal_a = False
        bar_range = bar.high - bar.low
        ibs = None
        if bar_range > 0:
            ibs = (bar.close - bar.low) / bar_range
            if ibs < self.ibs_entry:
                rsi = self._rsi(closes, 2)
                if rsi is not None and rsi < self.rsi_entry:
                    signal_a = True

        # Signal B: N consecutive down closes
        signal_b = False
        n = self.consec_down_days
        if len(closes) >= n + 1:
            all_down = True
            for i in range(1, n + 1):
                if closes[-i] >= closes[-i - 1]:
                    all_down = False
                    break
            signal_b = all_down

        if not signal_a and not signal_b:
            return None

        # ATR for stop/target
        atr = self._atr(bars, 14)
        if atr is None or atr < 0.01:
            return None

        # Symmetric stop/target capped at 2%
        max_dist = bar.close * self.max_stop_pct
        stop_dist = min(atr * self.stop_atr_mult, max_dist)
        target_dist = min(atr * self.target_atr_mult, max_dist)
        stop_price = bar.close - stop_dist
        target_price = bar.close + target_dist

        rsi_val = self._rsi(closes, 2)
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
                "rsi2": round(rsi_val, 1) if rsi_val else 0,
                "ibs": round(ibs, 2) if ibs else 0,
                "signal": "ibs_rsi" if signal_a else "consec_down",
                "drawdown": round(drawdown, 3),
            },
        )
