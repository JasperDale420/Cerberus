"""Regime-Adaptive Multi-Factor Strategy v2.

Switches behavior based on regime_trend labels:
- UP: Buy dips with EMA alignment + mild oversold (IBS + momentum guard)
- FLAT: Mean reversion via Z-score + IBS
- DOWN: Selective deep oversold bounce

Long-only, daily bars. Loose filters to ensure sufficient trade count.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v7a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 25))
        # IBS threshold
        self.ibs_entry = float(config.get("ibs_entry", 0.4))
        # RSI
        self.rsi_period = int(config.get("rsi_period", 5))
        self.rsi_entry = float(config.get("rsi_entry", 45.0))
        # Trend EMA
        self.trend_period = int(config.get("trend_period", 50))
        # ATR / risk
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.025))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.10))
        # Volume filter
        self.vol_mult = float(config.get("vol_mult", 0.5))
        # Momentum guard
        self.momentum_lookback = int(config.get("momentum_lookback", 10))
        # Hold
        self.max_hold_days = int(config.get("max_hold_days", 5))

    # --- Indicators ---

    @staticmethod
    def _rsi(closes: list[float], period: int) -> float | None:
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

    @staticmethod
    def _sma(values: list[float], period: int) -> float | None:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _atr(bars: list, period: int = 14) -> float | None:
        if len(bars) < period + 1:
            return None
        tr_vals = []
        for i in range(len(bars) - period, len(bars)):
            hi, lo, pc = bars[i].high, bars[i].low, bars[i - 1].close
            tr_vals.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
        return sum(tr_vals) / len(tr_vals)

    def _get_regime(self, symbol_state: SymbolState) -> str:
        """Get trend regime from labels."""
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "")
        if trend in ("UP", "DOWN", "FLAT"):
            return trend
        return "UNKNOWN"

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

        # --- Common filters ---

        # Drawdown filter
        lookback = min(self.drawdown_lookback, len(highs))
        recent_high = max(highs[-lookback:])
        drawdown = 0.0
        if recent_high > 0:
            drawdown = (recent_high - bar.close) / recent_high
            if drawdown > self.max_drawdown_pct:
                return None

        # Volume filter
        volumes = [b.volume for b in bars if b.volume and b.volume > 0]
        if len(volumes) >= 20:
            avg_vol = sum(volumes[-20:]) / 20
            if avg_vol > 0 and bar.volume < avg_vol * self.vol_mult:
                return None

        # IBS: close near day's low
        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return None
        ibs = (bar.close - bar.low) / bar_range

        # RSI
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None:
            return None

        # ATR for stop/target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 0.01:
            return None

        # Trend context
        trend_sma = self._sma(closes, self.trend_period)
        regime = self._get_regime(symbol_state)

        # --- Regime-specific entry logic ---

        if regime == "UP" or (regime == "UNKNOWN" and trend_sma is not None and bar.close > trend_sma):
            # UP: buy dips — looser IBS, moderate RSI, momentum guard
            if ibs >= self.ibs_entry:
                return None
            if rsi >= self.rsi_entry:
                return None
            # Momentum guard: price should be above N-day-ago (still in uptrend)
            if len(closes) > self.momentum_lookback:
                if bar.close <= closes[-self.momentum_lookback - 1]:
                    return None
            target_mult = self.target_atr_mult

        elif regime == "FLAT" or (regime == "UNKNOWN" and trend_sma is not None):
            # FLAT: mean reversion — tighter IBS, moderate RSI
            if ibs >= self.ibs_entry * 0.85:  # slightly tighter
                return None
            if rsi >= self.rsi_entry:
                return None
            target_mult = self.target_atr_mult * 0.8  # smaller target in flat

        elif regime == "DOWN":
            # DOWN: very selective — tight IBS, low RSI
            if ibs >= self.ibs_entry * 0.6:
                return None
            if rsi >= self.rsi_entry * 0.6:
                return None
            target_mult = self.target_atr_mult * 0.6  # modest target

        else:
            # Unknown regime without trend_sma: use FLAT-like logic
            if ibs >= self.ibs_entry:
                return None
            if rsi >= self.rsi_entry:
                return None
            target_mult = self.target_atr_mult * 0.8

        # --- Stop/target ---
        max_dist = bar.close * self.max_stop_pct
        stop_dist = min(atr * self.stop_atr_mult, max_dist)
        target_dist = min(atr * target_mult, max_dist * 1.5)
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
                "regime": regime,
                "rsi": round(rsi, 1),
                "ibs": round(ibs, 3),
                "drawdown": round(drawdown, 3),
                "atr": round(atr, 4),
            },
        )
