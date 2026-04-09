"""Daily Research Strategy v6b — RSI(2) Mean Reversion with Down-Days Filter.

High win-rate mean reversion:
- RSI(2) < 20 + IBS < 0.5 (oversold + closed near lows)
- 2+ consecutive lower closes (exhaustion selling confirmed)
- Tight target (1x ATR) for quick exits and high win rate
- Moderate stop (2x ATR)
- Drawdown filter prevents crash entries
- Long-only, daily bars.
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
        self.rsi_entry = float(config.get("rsi_entry", 20))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.5))
        self.down_days = int(config.get("down_days", 2))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.0))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.15))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
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

    def _consecutive_down(self, closes: list[float], n: int) -> bool:
        """Check if last n closes are consecutively lower."""
        if len(closes) < n + 1:
            return False
        for i in range(1, n + 1):
            if closes[-i] >= closes[-i - 1]:
                return False
        return True

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

        # Drawdown filter — skip crash scenarios
        lookback = min(self.drawdown_lookback, len(highs))
        recent_high = max(highs[-lookback:])
        if recent_high > 0:
            drawdown = (recent_high - bar.close) / recent_high
            if drawdown > self.max_drawdown_pct:
                return None

        # Consecutive down days — confirm exhaustion selling
        if not self._consecutive_down(closes, self.down_days):
            return None

        # RSI(2) — oversold
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi >= self.rsi_entry:
            return None

        # IBS — must close in lower half of range
        rng = bar.high - bar.low
        ibs = 0.5
        if rng > 0:
            ibs = (bar.close - bar.low) / rng
            if ibs >= self.ibs_threshold:
                return None

        # ATR for stops/targets
        atr = self._atr(bars, 14)
        if atr is None or atr < 0.01:
            return None

        stop_price = bar.close - atr * self.stop_atr_mult
        target_price = bar.close + atr * self.target_atr_mult

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
                "ibs": round(ibs, 2),
                "drawdown": round(drawdown, 3),
            },
        )
