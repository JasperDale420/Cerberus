"""daily_research_v7a — IBS + Z-Score Mean Reversion in Uptrends.

Archetype: Mean reversion gated by trend filter.
Entry: IBS < threshold AND BB z-score < -1.5 (NOT RSI).
Trend: price must be above SMA(50) — avoids DOWN regime losses.
Target: BB midline. Stop: 2x ATR.
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
        self.min_bars = int(config.get("min_bars", 55))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std_mult = float(config.get("bb_std_mult", 2.0))
        self.zscore_entry = float(config.get("zscore_entry", 1.5))
        self.ibs_entry = float(config.get("ibs_entry", 0.35))
        self.trend_period = int(config.get("trend_period", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.15))
        self.vol_mult = float(config.get("vol_mult", 0.5))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        subset = values[-period:]
        mean = sum(subset) / period
        variance = sum((v - mean) ** 2 for v in subset) / period
        return variance**0.5

    @staticmethod
    def _atr(bars: list[Bar], period: int) -> Optional[float]:
        if len(bars) < period + 1:
            return None
        trs = []
        for i in range(-period, 0):
            b = bars[i]
            prev_close = bars[i - 1].close
            tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
            trs.append(tr)
        return sum(trs) / period

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
        volumes = [b.volume for b in bars]

        # --- TREND GATE: price must be above SMA(50) ---
        trend_sma = self._sma(closes, self.trend_period)
        if trend_sma is None:
            return None
        if bar.close < trend_sma:
            return None  # Skip when below trend — avoids DOWN regime

        # --- BB z-score entry (replaces RSI2 — anti-convergence) ---
        bb_sma = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_sma is None or bb_std is None or bb_std < 1e-9:
            return None
        z_score = (bar.close - bb_sma) / bb_std
        if z_score > -self.zscore_entry:
            return None  # Not oversold enough

        # --- IBS (Internal Bar Strength): must close near day low ---
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_entry:
            return None

        # --- Volume filter ---
        if len(volumes) >= 20:
            avg_vol = sum(volumes[-20:]) / 20
            if avg_vol > 0 and bar.volume < avg_vol * self.vol_mult:
                return None

        # --- Drawdown filter ---
        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.max_drawdown_pct:
            return None

        # --- ATR for stop ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        target = bb_sma  # BB midline

        # Only enter if target is above entry
        if target <= bar.close:
            return None

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "z_score": round(z_score, 2),
                "ibs": round(ibs, 3),
                "atr": round(atr, 4),
                "trend_pct": round((bar.close - trend_sma) / trend_sma, 4),
                "archetype": "zscore_mr_uptrend",
            },
        )
