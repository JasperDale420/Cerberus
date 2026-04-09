"""Daily Research v6c — Triple-Signal Mean Reversion.

Iteration 5: Use OR logic — fire on ANY of:
  1) RSI(2) < 25 AND IBS < 0.5 (strong oversold + close near low)
  2) 2+ consecutive lower closes AND IBS < 0.4 (selling exhaustion)
SMA(20) trend filter on both. Momentum guard on RSI path only.
Symmetric 1.5x ATR, 3% cap. OR logic maximizes trades while each
path has its own quality gate.
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
        self.rsi_entry = float(config.get("rsi_entry", 25))
        self.ibs_rsi = float(config.get("ibs_rsi", 0.5))
        self.consec_down = int(config.get("consec_down", 2))
        self.ibs_consec = float(config.get("ibs_consec", 0.4))
        self.sma_period = int(config.get("sma_period", 20))
        self.momentum_lookback = int(config.get("momentum_lookback", 5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.10))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.03))
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

        # SMA trend filter: only buy if close > SMA(20)
        if len(closes) >= self.sma_period:
            sma = sum(closes[-self.sma_period :]) / self.sma_period
            if bar.close < sma:
                return None

        # IBS calculation
        bar_range = bar.high - bar.low
        ibs = (bar.close - bar.low) / bar_range if bar_range > 0 else 0.5

        # RSI(2)
        rsi = self._rsi(closes, 2)

        # Signal path 1: RSI oversold + IBS confirmation + momentum guard
        signal_rsi = False
        if rsi is not None and rsi < self.rsi_entry and ibs < self.ibs_rsi:
            # Momentum guard only for RSI path
            if len(closes) > self.momentum_lookback:
                if bar.close > closes[-self.momentum_lookback - 1]:
                    signal_rsi = True
            else:
                signal_rsi = True

        # Signal path 2: Consecutive lower closes + tight IBS
        signal_consec = False
        if len(closes) >= self.consec_down + 1 and ibs < self.ibs_consec:
            down_count = 0
            for i in range(len(closes) - 1, max(len(closes) - self.consec_down - 1, 0), -1):
                if closes[i] < closes[i - 1]:
                    down_count += 1
                else:
                    break
            if down_count >= self.consec_down:
                signal_consec = True

        if not signal_rsi and not signal_consec:
            return None

        # ATR for stop/target
        atr = self._atr(bars, 14)
        if atr is None or atr < 0.01:
            return None

        # Symmetric stop/target capped at max_stop_pct
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
                "rsi2": round(rsi, 1) if rsi is not None else None,
                "ibs": round(ibs, 2),
                "signal": "rsi" if signal_rsi else "consec",
                "drawdown": round(drawdown, 3),
            },
        )
