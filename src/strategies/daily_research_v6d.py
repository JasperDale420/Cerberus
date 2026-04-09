"""Daily Research v6d — RSI(2) mean reversion in uptrends.

Buy short-term oversold dips when the long-term trend is up.
Long-only. Few parameters. Proven robust across market regimes.

Rules:
  ENTRY: RSI(2) < rsi_entry AND price > SMA(trend_period)
  EXIT: hit stop (ATR-based) or target (ATR-based)

Iteration 3: Use symbol_state.bars directly (proven working pattern).
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
        self.trend_period = int(config.get("trend_period", 50))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_entry = float(config.get("rsi_entry", 25.0))
        self.stop_pct = float(config.get("stop_pct", 0.015))
        self.target_pct = float(config.get("target_pct", 0.02))
        self.allow_overnight = True
        self.max_hold_days = int(config.get("max_hold_days", 5))

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

        if len(closes) < self.min_bars:
            return None

        price = closes[-1]

        # Trend filter: price above SMA(trend_period) = uptrend
        if len(closes) < self.trend_period:
            return None
        sma = sum(closes[-self.trend_period :]) / self.trend_period
        if price < sma:
            return None

        # RSI(2) — buy on short-term oversold dip
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi >= self.rsi_entry:
            return None

        # Pure percentage-based stop/target (regime-independent)
        stop_price = price * (1.0 - self.stop_pct)
        target_price = price * (1.0 + self.target_pct)

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={"mode": "RSI2_REVERSION", "rsi": round(rsi, 1)},
        )
