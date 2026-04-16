"""Multi-Factor Mean Reversion with Regime Awareness.

Unified approach: buy multi-day weakness (consecutive down closes + low IBS)
near Bollinger Band lower zone. ATR-based risk management.
Adapts stop/target tightness by regime. Skips SHOCK vol.
Long-only, daily bars.
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
        # Entry filters
        self.consec_down_days = int(config.get("consec_down_days", 2))
        self.ibs_entry_threshold = float(config.get("ibs_entry_threshold", 0.35))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.bb_proximity = float(config.get("bb_proximity", 0.5))  # how close to lower BB (0=at band, 1=at mean)
        # Risk management
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        # Volatility scaling
        self.high_vol_stop_scale = float(config.get("high_vol_stop_scale", 0.8))
        self.high_vol_target_scale = float(config.get("high_vol_target_scale", 0.7))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

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
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        data = values[-period:]
        mean = sum(data) / period
        variance = sum((x - mean) ** 2 for x in data) / period
        return variance**0.5

    @staticmethod
    def _ibs(bar: Bar) -> float:
        """Internal Bar Strength: (close - low) / (high - low)."""
        rng = bar.high - bar.low
        if rng < 1e-9:
            return 0.5
        return (bar.close - bar.low) / rng

    @staticmethod
    def _consec_down(closes: list[float], min_days: int) -> bool:
        """Check if last N closes were each lower than the previous."""
        if len(closes) < min_days + 1:
            return False
        for i in range(-min_days, 0):
            if closes[i] >= closes[i - 1]:
                return False
        return True

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

        # Skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # --- Entry Condition 1: Consecutive down closes ---
        if not self._consec_down(closes, self.consec_down_days):
            return None

        # --- Entry Condition 2: Low IBS (closed near the low) ---
        ibs = self._ibs(bar)
        if ibs > self.ibs_entry_threshold:
            return None

        # --- Entry Condition 3: Price near lower Bollinger Band ---
        bb_mean = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_mean is None or bb_std is None or bb_std < 1e-9:
            return None

        lower_band = bb_mean - self.bb_std * bb_std
        # bb_proximity: 0.0 = at lower band, 1.0 = at mean
        # Price must be below: lower_band + proximity * (mean - lower_band)
        threshold = lower_band + self.bb_proximity * (bb_mean - lower_band)
        if bar.close > threshold:
            return None

        # --- Risk Management ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Scale stops/targets in high vol regime
        labels = symbol_state.meta.get("regime_labels", {})
        vol_regime = labels.get("regime_vol", "NORMAL")
        is_high_vol = vol_regime in ("HIGH", "SHOCK")

        stop_mult = self.stop_atr_mult * (self.high_vol_stop_scale if is_high_vol else 1.0)
        target_mult = self.target_atr_mult * (self.high_vol_target_scale if is_high_vol else 1.0)

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
                "mode": "multi_factor_mr",
                "ibs": round(ibs, 2),
                "consec_down": self.consec_down_days,
                "bb_zone": round((bar.close - lower_band) / (bb_mean - lower_band), 2) if bb_mean > lower_band else 0,
                "vol_regime": vol_regime,
            },
        )
