"""Daily Research Strategy v6b — Consecutive Down Days Mean Reversion.

iter3: Replace RSI(2) with consecutive down bars entry signal.
- Buy after 3+ consecutive down closes (close < prev close)
- SMA(20) trend filter (only trade in uptrends)
- Symmetric 2x ATR stop and target (capped at 4%)
- 12% drawdown filter
- 5-day max hold, long-only, daily bars

Hypothesis: 3+ down days is simpler (fewer params), more robust,
and statistically produces higher bounce probability than RSI(2) < 25.
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
        self.consec_down = int(config.get("consec_down", 3))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.sma_period = int(config.get("sma_period", 20))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.12))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.04))
        self.allow_overnight = True

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

        # Price-based trend filter: close > SMA(20)
        if len(closes) >= self.sma_period:
            sma = sum(closes[-self.sma_period :]) / self.sma_period
            if bar.close < sma:
                return None

        # Drawdown filter
        lookback = min(self.drawdown_lookback, len(highs))
        recent_high = max(highs[-lookback:])
        drawdown = 0.0
        if recent_high > 0:
            drawdown = (recent_high - bar.close) / recent_high
            if drawdown > self.max_drawdown_pct:
                return None

        # Consecutive down bars check
        if len(closes) < self.consec_down + 1:
            return None
        for i in range(1, self.consec_down + 1):
            if closes[-i] >= closes[-i - 1]:
                return None

        # ATR for stop/target
        atr = self._atr(bars, 14)
        if atr is None or atr < 0.01:
            return None

        # Stop/target with cap at max_stop_pct of price
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
                "consec_down": self.consec_down,
                "drawdown": round(drawdown, 3),
            },
        )
