"""Regime-Adaptive Volatility Strategy.

Dual-mode: volatility breakout in trending markets, ATR-contraction
mean-reversion in flat/range-bound markets.  Adapts entry thresholds
and risk sizing to regime_trend and regime_vol labels.

Long-only, daily bars, max_hold_days configurable.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v7c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.atr_period = int(config.get("atr_period", 14))

        # Breakout mode params
        self.atr_expansion = float(config.get("atr_expansion", 1.3))
        self.breakout_lookback = int(config.get("breakout_lookback", 10))
        self.vol_surge_mult = float(config.get("vol_surge_mult", 1.3))

        # Mean-reversion mode params
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std_mult = float(config.get("bb_std_mult", 2.0))
        self.mr_atr_contraction = float(config.get("mr_atr_contraction", 0.9))

        # Risk params
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 7))

    # --- Indicator helpers ---

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

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        data = values[-period:]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return variance**0.5

    def _atr_series(self, bars: list[Bar], period: int, count: int) -> list[float]:
        result = []
        for i in range(count):
            end_idx = len(bars) - count + i + 1
            if end_idx < period + 1:
                continue
            sub = bars[:end_idx]
            val = self._atr(sub, period)
            if val is not None:
                result.append(val)
        return result

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

        regime_labels = symbol_state.meta.get("regime_labels", {})

        # Skip near-earnings and near-FOMC
        if regime_labels.get("near_earnings") or regime_labels.get("near_fomc"):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # Current ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # ATR average for expansion/contraction detection
        atr_values = self._atr_series(bars, self.atr_period, 20)
        if len(atr_values) < 10:
            return None
        atr_avg = sum(atr_values) / len(atr_values)
        if atr_avg < 1e-9:
            return None
        atr_ratio = atr / atr_avg

        # Regime
        trend = regime_labels.get("regime_trend", "FLAT")
        vol_regime = regime_labels.get("regime_vol", "NORMAL")

        # Skip SHOCK volatility — too chaotic
        if vol_regime == "SHOCK":
            return None

        # Volume average
        avg_vol = self._sma(volumes, 20)
        if avg_vol is None or avg_vol < 1e-9:
            return None
        vol_ratio = bar.volume / avg_vol

        # Try breakout mode (UP or FLAT with expanding vol)
        signal = self._try_breakout(symbol, bar, bars, closes, atr, atr_ratio, vol_ratio, trend, market_state)
        if signal is not None:
            return signal

        # Try mean-reversion mode (FLAT or DOWN with contracting vol)
        signal = self._try_mean_reversion(symbol, bar, bars, closes, atr, atr_ratio, vol_ratio, trend, market_state)
        return signal

    def _try_breakout(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        atr: float,
        atr_ratio: float,
        vol_ratio: float,
        trend: str,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """Volatility breakout: expanding ATR + new highs + volume surge."""
        # Require ATR expansion
        if atr_ratio < self.atr_expansion:
            return None

        # Breakout: close above N-day high
        if len(bars) < self.breakout_lookback + 1:
            return None
        lookback_highs = [b.high for b in bars[-(self.breakout_lookback + 1) : -1]]
        high_nd = max(lookback_highs)
        if bar.close <= high_nd:
            return None

        # Volume confirmation
        if vol_ratio < self.vol_surge_mult:
            return None

        # In DOWN trend, require stronger breakout
        if trend == "DOWN" and atr_ratio < self.atr_expansion * 1.3:
            return None

        # Stop/target
        stop = bar.close - self.stop_atr_mult * atr
        target = bar.close + self.target_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "mode": "BREAKOUT",
                "atr_ratio": round(atr_ratio, 2),
                "vol_ratio": round(vol_ratio, 2),
                "trend": trend,
            },
        )

    def _try_mean_reversion(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        atr: float,
        atr_ratio: float,
        vol_ratio: float,
        trend: str,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """Mean reversion: price below lower BB in contracting/normal ATR."""
        # Mean reversion works best in FLAT/non-trending — skip strong UP
        if trend == "UP" and atr_ratio > 1.2:
            return None

        # Require ATR not expanding wildly (contracting or normal)
        if atr_ratio > 1.2:
            return None

        # Bollinger Bands
        bb_mean = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_mean is None or bb_std is None or bb_std < 0.01:
            return None

        lower_band = bb_mean - self.bb_std_mult * bb_std
        price = bar.close

        # Price below lower band
        if price >= lower_band:
            return None

        # Confirmation: price above previous close (bounce starting)
        if len(closes) >= 2 and price <= closes[-2]:
            return None

        # In DOWN trend, require deeper oversold (more below band)
        if trend == "DOWN":
            deeper_band = bb_mean - (self.bb_std_mult + 0.5) * bb_std
            if price >= deeper_band:
                return None

        # Stop/target — tighter for mean reversion
        stop = price - self.stop_atr_mult * atr
        target = bb_mean  # Target the mean

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "mode": "MEAN_REV",
                "atr_ratio": round(atr_ratio, 2),
                "z_score": round((price - bb_mean) / bb_std, 2) if bb_std > 0 else 0,
                "trend": trend,
            },
        )
