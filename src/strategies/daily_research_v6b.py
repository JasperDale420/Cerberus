"""Daily Research Strategy v6b — Regime-Adaptive RSI(2) with Symmetric Risk.

Combines regime-adaptive entry selectivity with symmetric stop/target:
- Normal regime: RSI(2) < 25 + IBS < 0.5 (generous, more trades)
- DOWN/HIGH regime: RSI(2) < 5 + IBS < 0.3 (ultra-selective, only best setups)
- Symmetric 2x ATR stop and target (needs only 47% WR for PF>0.9)
- Drawdown filter (12%) prevents crash entries
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
        self.rsi_entry_normal = float(config.get("rsi_entry_normal", 25))
        self.rsi_entry_cautious = float(config.get("rsi_entry_cautious", 5))
        self.ibs_normal = float(config.get("ibs_normal", 0.5))
        self.ibs_cautious = float(config.get("ibs_cautious", 0.3))
        self.max_hold_days = int(config.get("max_hold_days", 10))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.08))
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

    def _ibs(self, bar: Bar) -> float:
        rng = bar.high - bar.low
        if rng <= 0:
            return 0.5
        return (bar.close - bar.low) / rng

    def _is_cautious_regime(self, symbol_state: SymbolState) -> bool:
        meta = symbol_state.meta
        regime_labels = meta.get("regime_labels", {})
        trend = str(regime_labels.get("trend", "")).upper()
        vol = str(regime_labels.get("vol", "")).upper()
        return trend == "DOWN" or vol in ("HIGH", "SHOCK")

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
        if recent_high > 0:
            drawdown = (recent_high - bar.close) / recent_high
            if drawdown > self.max_drawdown_pct:
                return None

        # RSI(2)
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None:
            return None

        # IBS
        ibs = self._ibs(bar)

        # Regime-adaptive entry
        cautious = self._is_cautious_regime(symbol_state)
        if cautious:
            if rsi >= self.rsi_entry_cautious or ibs >= self.ibs_cautious:
                return None
        else:
            if rsi >= self.rsi_entry_normal or ibs >= self.ibs_normal:
                return None

        # ATR
        atr = self._atr(bars, 14)
        if atr is None or atr < 0.01:
            return None

        # Cap stop at max_stop_pct of price — limits damage in high-vol
        stop_dist = min(atr * self.stop_atr_mult, bar.close * self.max_stop_pct)
        target_dist = min(atr * self.target_atr_mult, bar.close * self.max_stop_pct)
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
                "ibs": round(ibs, 2),
                "cautious": cautious,
            },
        )
