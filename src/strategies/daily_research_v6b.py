"""Daily Research Strategy v6b — RSI(2) Mean Reversion.

Buy when short-term RSI is extremely oversold. Exit after fixed hold period
or when RSI recovers. Long-only for equity upward bias. Uses IBS
(Internal Bar Strength) as confirmation for consistency across regimes.
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
        self.min_bars = int(config.get("min_bars", 25))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_entry = float(config.get("rsi_entry", 15))
        self.ibs_entry = float(config.get("ibs_entry", 0.3))
        self.consecutive_down = int(config.get("consecutive_down", 2))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 3.0))
        self.sma_period = int(config.get("sma_period", 200))
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

    def _sma(self, values: list[float], period: int) -> float | None:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    def _atr(self, bars: list[Bar], period: int = 14) -> float | None:
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

        # Need enough bars for SMA filter
        if len(closes) < max(self.sma_period, self.min_bars):
            # If not enough for SMA, use shorter trend filter
            if len(closes) < self.min_bars:
                return None

        # RSI(2)
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None:
            return None

        # IBS = (close - low) / (high - low)
        ibs = (bar.close - bar.low) / (bar.high - bar.low) if (bar.high - bar.low) > 0.001 else 0.5

        # Count consecutive down days
        down_days = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                down_days += 1
            else:
                break

        # Trend filter: price above SMA (use available length if < sma_period)
        effective_sma_period = min(self.sma_period, len(closes))
        if effective_sma_period >= 50:
            sma = self._sma(closes, effective_sma_period)
            if sma is not None and closes[-1] < sma * 0.92:
                return None  # Too far below SMA — likely in crash, skip

        # === ENTRY CONDITIONS ===
        # RSI(2) oversold + IBS low + consecutive down days
        if rsi < self.rsi_entry and ibs < self.ibs_entry and down_days >= self.consecutive_down:
            atr = self._atr(bars)
            if atr is None or atr < 0.01:
                return None

            stop_price = bar.close - atr * self.stop_atr_mult
            # Target: mean reversion — modest target
            target_price = bar.close + atr * 2.0

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
                    "down_days": down_days,
                },
            )

        return None
