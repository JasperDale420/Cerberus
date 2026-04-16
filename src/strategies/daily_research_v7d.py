"""Regime-Adaptive IBS + Consecutive Down Days Strategy.

Long-only mean reversion using IBS (Internal Bar Strength) as core signal,
with consecutive down days for confirmation. Regime-adaptive thresholds:
  UP   — moderate IBS threshold, 1+ down days, above SMA(50)
  FLAT — tighter IBS, 2+ down days
  DOWN — extreme oversold only, 3+ down days, skip DOWN+HIGH

No shorts — empirically inconsistent across time periods.
Stops/targets scaled by ATR with vol adjustment.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedRegimeSwitchStrategy(BaseStrategy):
    name = "daily_research_v7d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.sma_period = int(config.get("sma_period", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.base_stop_atr_mult = float(config.get("base_stop_atr_mult", 1.5))
        self.base_target_atr_mult = float(config.get("base_target_atr_mult", 2.5))
        self.vol_stop_scale_factor = float(config.get("vol_stop_scale_factor", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        # IBS thresholds per regime
        self.ibs_up = float(config.get("ibs_up", 0.3))
        self.ibs_flat = float(config.get("ibs_flat", 0.25))
        self.ibs_down = float(config.get("ibs_down", 0.15))
        # Consecutive down days required per regime
        self.down_days_up = int(config.get("down_days_up", 1))
        self.down_days_flat = int(config.get("down_days_flat", 2))
        self.down_days_down = int(config.get("down_days_down", 3))
        # Volume filter
        self.vol_lookback = int(config.get("vol_lookback", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.5))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.12))

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
    def _ibs(bar: Bar) -> Optional[float]:
        """Internal Bar Strength: (close - low) / (high - low)."""
        rng = bar.high - bar.low
        if rng < 1e-9:
            return None
        return (bar.close - bar.low) / rng

    @staticmethod
    def _consecutive_down_days(bars: list[Bar]) -> int:
        """Count consecutive days where close < previous close, from most recent."""
        count = 0
        for i in range(len(bars) - 1, 0, -1):
            if bars[i].close < bars[i - 1].close:
                count += 1
            else:
                break
        return count

    def _vol_adjusted_stop_mult(self, market_state: MarketState) -> float:
        realized_vol = market_state.realized_vol
        if realized_vol <= 0:
            return self.base_stop_atr_mult
        return self.base_stop_atr_mult * (1.0 + realized_vol * self.vol_stop_scale_factor)

    def _check_drawdown(self, closes: list[float]) -> bool:
        """Return False if recent drawdown exceeds threshold."""
        lookback = min(self.drawdown_lookback, len(closes))
        if lookback < 5:
            return True
        recent = closes[-lookback:]
        peak = max(recent)
        if peak <= 0:
            return True
        dd = (peak - closes[-1]) / peak
        return dd <= self.max_drawdown_pct

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

        # Core indicators
        sma50 = self._sma(closes, self.sma_period)
        atr = self._atr(bars, self.atr_period)
        ibs = self._ibs(bar)

        if any(v is None for v in (sma50, atr, ibs)) or atr < 1e-9:
            return None

        # Volume filter: skip abnormally low volume days
        if len(bars) >= self.vol_lookback:
            avg_vol = sum(b.volume for b in bars[-self.vol_lookback :]) / self.vol_lookback
            if avg_vol > 0 and bar.volume < avg_vol * self.vol_min_ratio:
                return None

        # Drawdown filter
        if not self._check_drawdown(closes):
            return None

        # Consecutive down days
        down_days = self._consecutive_down_days(bars)

        # Read regime
        regime_labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = regime_labels.get("regime_trend", "FLAT").upper()
        regime_vol = regime_labels.get("regime_vol", "NORMAL").upper()

        # Skip DOWN+HIGH — too volatile and unreliable
        if regime_trend == "DOWN" and regime_vol in ("HIGH", "SHOCK"):
            return None

        # Skip earnings/FOMC
        if regime_labels.get("near_earnings", False):
            return None
        if regime_labels.get("near_fomc", False):
            return None

        # Determine entry thresholds based on regime
        if regime_trend == "UP":
            ibs_threshold = self.ibs_up
            min_down_days = self.down_days_up
            # Must be above SMA(50) — buying dips in uptrend
            if bar.close < sma50:
                return None
        elif regime_trend == "DOWN":
            ibs_threshold = self.ibs_down
            min_down_days = self.down_days_down
        else:  # FLAT
            ibs_threshold = self.ibs_flat
            min_down_days = self.down_days_flat

        # Entry conditions
        if ibs > ibs_threshold:
            return None
        if down_days < min_down_days:
            return None

        # Position sizing via stop/target
        stop_mult = self._vol_adjusted_stop_mult(market_state)
        stop = bar.close - stop_mult * atr
        target = bar.close + self.base_target_atr_mult * atr

        if target <= bar.close or stop >= bar.close:
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
                "regime": regime_trend,
                "regime_vol": regime_vol,
                "ibs": round(ibs, 3),
                "down_days": down_days,
                "sma50": round(sma50, 2),
                "stop_mult": round(stop_mult, 3),
                "seed": "regime_switch",
            },
        )
