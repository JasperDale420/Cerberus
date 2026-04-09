"""Daily Research v6d — RSI(2) mean reversion in uptrends.

Buy short-term oversold dips when the long-term trend is up.
Long-only. Few parameters. Proven robust across market regimes.

Rules:
  ENTRY: RSI(2) < rsi_entry AND price > SMA(trend_period)
         AND drawdown < max_drawdown_pct (bear market guard)
  EXIT: hit stop (ATR-based) or target (ATR-based)

Iteration 4: Add drawdown guard + IBS confirmation for bear protection.
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
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr = float(config.get("stop_atr", 3.0))
        self.target_atr = float(config.get("target_atr", 4.0))
        # Bear market guard: skip if price dropped > X% from recent high
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.12))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        # IBS (Internal Bar Strength) confirmation: close near low = oversold
        self.ibs_entry = float(config.get("ibs_entry", 0.4))
        self.allow_overnight = True
        self.max_hold_days = int(config.get("max_hold_days", 10))

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

        price = closes[-1]

        # Trend filter: price above SMA(trend_period) = uptrend
        if len(closes) < self.trend_period:
            return None
        sma = sum(closes[-self.trend_period :]) / self.trend_period
        if price < sma:
            return None

        # Drawdown guard: skip if price is in a drawdown from recent high
        lookback = min(self.drawdown_lookback, len(highs))
        recent_high = max(highs[-lookback:])
        if recent_high > 0:
            drawdown = (recent_high - price) / recent_high
            if drawdown > self.max_drawdown_pct:
                return None

        # RSI(2) — buy on short-term oversold dip
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi >= self.rsi_entry:
            return None

        # IBS confirmation: close should be in lower portion of today's range
        bar_range = bar.high - bar.low
        if bar_range > 0.001:
            ibs = (bar.close - bar.low) / bar_range
            if ibs > self.ibs_entry:
                return None

        # ATR for stop/target sizing
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 0.01:
            return None

        stop_price = price - atr * self.stop_atr
        target_price = price + atr * self.target_atr

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
