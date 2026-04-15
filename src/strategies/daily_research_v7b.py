"""Regime-Adaptive Dual-Mode: Trend Pullback + IBS Mean Reversion.

UP regime: Buy pullbacks to EMA(20) in confirmed uptrends.
DOWN/FLAT regime: Buy IBS oversold bounces (close near day low).
Event filters: skip earnings and FOMC.
ATR-based stops/targets, max_hold_days=7.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v7b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))

        # Trend pullback params (UP regime)
        self.sma_slow_period = int(config.get("sma_slow_period", 50))
        self.ema_fast_period = int(config.get("ema_fast_period", 20))
        self.pullback_max_pct = float(config.get("pullback_max_pct", 0.03))

        # IBS mean reversion params (DOWN/FLAT regime)
        self.ibs_threshold = float(config.get("ibs_threshold", 0.25))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std_mult = float(config.get("bb_std_mult", 1.5))

        # Shared params
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.7))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = sum(values[:period]) / period
        for v in values[period:]:
            ema = (v - ema) * mult + ema
        return ema

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        data = values[-period:]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
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

    @staticmethod
    def _ibs(bar: Bar) -> Optional[float]:
        """Internal Bar Strength: (Close - Low) / (High - Low). Range [0, 1]."""
        range_ = bar.high - bar.low
        if range_ < 1e-9:
            return None
        return (bar.close - bar.low) / range_

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

        # Skip SHOCK volatility regime
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Skip earnings and FOMC
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # Volume filter
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # ATR for stop and target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Get regime
        regime_trend = labels.get("regime_trend", "FLAT")

        # Route to appropriate mode
        if regime_trend == "UP":
            return self._trend_pullback_signal(symbol, bar, bars, closes, atr, market_state)
        else:
            # DOWN or FLAT: use IBS mean reversion
            return self._ibs_reversion_signal(symbol, bar, bars, closes, atr, market_state, regime_trend)

    def _trend_pullback_signal(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        atr: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """Buy pullbacks to EMA(20) in confirmed uptrends."""
        sma50 = self._sma(closes, self.sma_slow_period)
        if sma50 is None or bar.close <= sma50:
            return None

        ema20 = self._ema(closes, self.ema_fast_period)
        if ema20 is None or ema20 <= sma50:
            return None

        # Pullback: close within range of EMA20 (not too far above or below)
        if ema20 < 1e-9:
            return None
        dist_pct = (bar.close - ema20) / ema20
        if dist_pct < -self.pullback_max_pct or dist_pct > 0.01:
            return None

        # IBS confirmation: prefer close in lower half of bar (pullback day)
        ibs = self._ibs(bar)
        if ibs is not None and ibs > 0.6:
            return None

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
                "mode": "TREND_PULLBACK",
                "ema20": round(ema20, 2),
                "sma50": round(sma50, 2),
                "pullback_pct": round(dist_pct, 4),
                "atr": round(atr, 4),
            },
        )

    def _ibs_reversion_signal(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        atr: float,
        market_state: MarketState,
        regime_trend: str,
    ) -> Optional[Signal]:
        """Buy IBS oversold bounces in DOWN/FLAT regimes."""
        ibs = self._ibs(bar)
        if ibs is None or ibs > self.ibs_threshold:
            return None

        # Bollinger Band confirmation: price near or below lower band
        sma = self._sma(closes, self.bb_period)
        std = self._std(closes, self.bb_period)
        if sma is None or std is None or std < 1e-9:
            return None

        lower_band = sma - self.bb_std_mult * std
        if bar.close > lower_band:
            return None

        # Tighter stop in DOWN regime, wider in FLAT
        if regime_trend == "DOWN":
            stop_mult = self.stop_atr_mult * 0.8
            target_mult = self.target_atr_mult * 0.8
        else:
            stop_mult = self.stop_atr_mult
            target_mult = self.target_atr_mult

        stop = bar.close - stop_mult * atr
        target = bar.close + target_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "mode": "IBS_REVERSION",
                "regime": regime_trend,
                "ibs": round(ibs, 3),
                "lower_bb": round(lower_band, 2),
                "atr": round(atr, 4),
            },
        )
