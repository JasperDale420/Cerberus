"""Daily Research Strategy v6b — Symmetric-Exit RSI(2) Mean Reversion.

iter2: Disable max_hold_days (set to 0) to let all trades resolve at
stop or target. Time exits with max_hold_days=5 were creating asymmetric
P&L (avg_loss >> avg_win) even with symmetric stops, causing some windows
to have PF=0.58 despite 49% WR (which should give PF~0.96 with true symmetry).
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
        self.max_hold_days = int(config.get("max_hold_days", 0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.sma_period = int(config.get("sma_period", 20))
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
