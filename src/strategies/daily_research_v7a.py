"""Regime-Adaptive IBS+RSI Mean Reversion with Adaptive Targets.

Core entry: IBS < threshold + RSI(2) < threshold + volume filter + drawdown guard.
Regime-adaptive: adjusts target sizing and filters based on trend regime.
- UP: momentum guard, larger targets (trend continuation after bounce)
- FLAT: standard mean reversion targets (SMA reversion)
- DOWN: tighter targets, stricter IBS (quick bounce only)

Based on v6c which generates 175 trades. Adds regime-adaptive target sizing.
Long-only, daily bars.
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
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_entry = float(config.get("rsi_entry", 30.0))
        self.ibs_entry = float(config.get("ibs_entry", 0.4))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.02))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.10))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.vol_mult = float(config.get("vol_mult", 0.6))
        self.momentum_lookback = int(config.get("momentum_lookback", 5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.trend_period = int(config.get("trend_period", 50))

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

    def _sma(self, values: list[float], period: int) -> float | None:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    def _get_regime(self, symbol_state: SymbolState, closes: list[float]) -> str:
        """Get trend regime from labels, fallback to SMA-based detection."""
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "")
        if trend in ("UP", "DOWN", "FLAT"):
            return trend
        # Fallback: use SMA trend
        sma = self._sma(closes, self.trend_period)
        if sma is None:
            return "FLAT"
        if closes[-1] > sma * 1.02:
            return "UP"
        elif closes[-1] < sma * 0.98:
            return "DOWN"
        return "FLAT"

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

        # --- Drawdown filter ---
        lookback = min(self.drawdown_lookback, len(highs))
        recent_high = max(highs[-lookback:])
        drawdown = 0.0
        if recent_high > 0:
            drawdown = (recent_high - bar.close) / recent_high
            if drawdown > self.max_drawdown_pct:
                return None

        # --- Volume filter ---
        volumes = [b.volume for b in bars if b.volume and b.volume > 0]
        if len(volumes) >= 20:
            avg_vol = sum(volumes[-20:]) / 20
            if avg_vol > 0 and bar.volume < avg_vol * self.vol_mult:
                return None

        # --- IBS: close near day's low ---
        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return None
        ibs = (bar.close - bar.low) / bar_range

        # --- RSI ---
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None:
            return None

        # --- ATR ---
        atr = self._atr(bars, 14)
        if atr is None or atr < 0.01:
            return None

        # --- Regime detection ---
        regime = self._get_regime(symbol_state, closes)

        # --- Regime-adaptive entry and target logic ---
        if regime == "UP":
            # Looser IBS, apply momentum guard, larger target
            if ibs >= self.ibs_entry:
                return None
            if rsi >= self.rsi_entry * 1.5:  # RSI up to 45 in uptrend
                return None
            # Momentum guard: price above N-day-ago (still in uptrend structure)
            if len(closes) > self.momentum_lookback:
                if bar.close <= closes[-self.momentum_lookback - 1]:
                    return None
            target_mult = self.target_atr_mult * 1.3  # bigger target in uptrend

        elif regime == "DOWN":
            # Tighter IBS, lower RSI, smaller target
            if ibs >= self.ibs_entry * 0.7:
                return None
            if rsi >= self.rsi_entry * 0.8:
                return None
            target_mult = self.target_atr_mult * 0.7  # modest target

        else:  # FLAT
            # Standard mean reversion
            if ibs >= self.ibs_entry:
                return None
            if rsi >= self.rsi_entry:
                return None
            target_mult = self.target_atr_mult

        # --- Stop/target with cap ---
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
            },
        )
