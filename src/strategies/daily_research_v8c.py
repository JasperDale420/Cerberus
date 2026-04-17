"""Dual-mode ATR strategy — mean reversion + momentum, regime-gated.

Two entry modes using ATR (volatility DNA):
  Mode A (Mean Reversion): When price dips > atr_drop_mult*ATR below SMA,
    buy the dip with SMA as target. Works best in FLAT/UP regimes.
  Mode B (Momentum): When price breaks above N-day high with ATR expansion,
    buy breakout with trailing ATR target. Works in UP regimes.

Regime gate: skip HIGH/SHOCK vol, skip near earnings/FOMC.
Both modes use ATR for stops and targets — pure volatility-based sizing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v8c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.sma_period = int(config.get("sma_period", 20))
        self.atr_period = int(config.get("atr_period", 14))
        # Mean reversion params
        self.atr_drop_mult = float(config.get("atr_drop_mult", 1.2))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        # Momentum breakout params
        self.breakout_lookback = int(config.get("breakout_lookback", 20))
        self.atr_expansion_mult = float(config.get("atr_expansion_mult", 1.3))
        self.breakout_target_atr = float(config.get("breakout_target_atr", 2.0))
        # Shared
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))

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

        # Skip HIGH and SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol in (VolRegime.HIGH, VolRegime.SHOCK):
            return None

        # Event filters
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        sma = self._sma(closes, self.sma_period)
        if sma is None or sma < 1e-9:
            return None

        # Try Mode A first (mean reversion), then Mode B (breakout)
        signal = self._try_mean_reversion(symbol, bar, bars, closes, volumes, atr, sma, market_state)
        if signal is not None:
            return signal

        return self._try_breakout(symbol, bar, bars, closes, volumes, atr, sma, market_state)

    def _try_mean_reversion(
        self,
        symbol: str,
        bar: Bar,
        bars: list,
        closes: list,
        volumes: list,
        atr: float,
        sma: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # IBS filter: close near day's low
        bar_range = bar.high - bar.low
        if bar_range > 1e-9:
            ibs = (bar.close - bar.low) / bar_range
            if ibs > self.ibs_threshold:
                return None

        # Price dipped below SMA by ATR-scaled amount
        dip = sma - bar.close
        if dip < self.atr_drop_mult * atr:
            return None

        # Target: SMA (mean reversion)
        target = sma
        if target <= bar.close:
            return None

        stop = bar.close - self.stop_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={"mode": "mean_rev", "dip_atr": round(dip / atr, 2), "seed": "vol_dual"},
        )

    def _try_breakout(
        self,
        symbol: str,
        bar: Bar,
        bars: list,
        closes: list,
        volumes: list,
        atr: float,
        sma: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # Only breakout in UP trend
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.trend and snapshot.trend.name != "UP":
            return None

        # Breakout: close above N-day high
        if len(bars) < self.breakout_lookback + 1:
            return None
        lookback_highs = [b.high for b in bars[-(self.breakout_lookback + 1) : -1]]
        high_nd = max(lookback_highs)
        if bar.close <= high_nd:
            return None

        # ATR expansion: current ATR > mult * average ATR
        atr_values = self._atr_series(bars, self.atr_period, self.vol_avg_period)
        if len(atr_values) < self.vol_avg_period:
            return None
        atr_avg = sum(atr_values) / len(atr_values)
        if atr_avg < 1e-9 or atr < self.atr_expansion_mult * atr_avg:
            return None

        # Volume confirmation
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < avg_vol:
            return None

        stop = bar.low
        target = bar.close + self.breakout_target_atr * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={"mode": "breakout", "atr_exp": round(atr / atr_avg, 2), "seed": "vol_dual"},
        )
