"""Daily Research Strategy v6b — Bollinger Band Mean Reversion.

iter5: Bollinger Band entry with RSI(2) confirmation.
- Entry: price < lower BB(20,2) AND RSI(2) < 30
- Target: BB midline (SMA20) — natural mean reversion target
- Stop: entry - 1.5 * BB_std — adaptive to volatility
- 12% drawdown filter (40-bar lookback)
- No SMA trend filter needed (BB provides implicit trend context)
- 5-day max hold, long-only, daily bars

Key advantage: BB adapts stop/target to current volatility. In low vol,
tight bands → quick resolution. In high vol, wide bands → proportional R:R.
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
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std_mult = float(config.get("bb_std_mult", 2.0))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_entry = float(config.get("rsi_entry", 30))
        self.stop_bb_mult = float(config.get("stop_bb_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
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

        if len(closes) < self.bb_period:
            return None

        # Bollinger Bands
        bb_data = closes[-self.bb_period :]
        sma = sum(bb_data) / self.bb_period
        variance = sum((x - sma) ** 2 for x in bb_data) / self.bb_period
        std = variance**0.5
        if std < 0.01:
            return None
        lower_band = sma - self.bb_std_mult * std

        # Entry: price below lower band
        if bar.close >= lower_band:
            return None

        # RSI(2) confirmation
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi >= self.rsi_entry:
            return None

        # Drawdown filter
        lookback = min(self.drawdown_lookback, len(highs))
        recent_high = max(highs[-lookback:])
        drawdown = 0.0
        if recent_high > 0:
            drawdown = (recent_high - bar.close) / recent_high
            if drawdown > self.max_drawdown_pct:
                return None

        # Target = BB midline (SMA), Stop = entry - 1.5 * std
        target_price = sma
        stop_dist = self.stop_bb_mult * std
        # Cap stop at max_stop_pct of price
        max_dist = bar.close * self.max_stop_pct
        stop_dist = min(stop_dist, max_dist)
        stop_price = bar.close - stop_dist

        # Ensure target is above entry
        if target_price <= bar.close:
            return None

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
                "bb_lower": round(lower_band, 2),
                "bb_mid": round(sma, 2),
                "drawdown": round(drawdown, 3),
            },
        )
