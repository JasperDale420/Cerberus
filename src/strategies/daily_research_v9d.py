"""Regime-Adaptive ROC Mean Reversion — Buy oversold bounces measured by
Rate of Change, with regime-scaled thresholds and volume confirmation.

Evolved from seed_regime_switch. Uses ROC (5-day price change %) to detect
oversold conditions instead of Keltner bands. Regime-adaptive entry:

  UP   — moderate oversold (ROC < -4%) + volume surge
  FLAT — deeper oversold (ROC < -5%) + volume surge
  DOWN — SKIP entirely (catch falling knives)

Long-only. Event/calendar filters. ATR stops, SMA(20) target.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedRegimeSwitchStrategy(BaseStrategy):
    name = "daily_research_v9d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 30))
        self.roc_period = int(config.get("roc_period", 5))
        self.sma_period = int(config.get("sma_period", 20))
        self.atr_period = int(config.get("atr_period", 14))
        # Regime-adaptive ROC thresholds (must be negative — how much price fell)
        self.roc_up = float(config.get("roc_up", -0.04))
        self.roc_flat = float(config.get("roc_flat", -0.05))
        # IBS confirmation
        self.ibs_threshold = float(config.get("ibs_threshold", 0.30))
        # Volume surge: require above-avg volume
        self.vol_surge_mult = float(config.get("vol_surge_mult", 1.0))
        # Stop/target
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.0))
        self.max_hold_days = int(config.get("max_hold_days", 3))
        # Filters
        self.max_atr_pct = float(config.get("max_atr_pct", 0.04))
        self.max_risk_pct = float(config.get("max_risk_pct", 0.025))
        self.min_price = float(config.get("min_price", 15.0))
        self.min_consec_down = int(config.get("min_consec_down", 2))

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
    def _count_consec_down(closes: list[float], max_look: int = 10) -> int:
        count = 0
        for i in range(1, min(len(closes), max_look)):
            if closes[-i] < closes[-i - 1]:
                count += 1
            else:
                break
        return count

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

        # Event filters
        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings"):
            return None
        if regime_labels.get("near_fomc"):
            return None
        if regime_labels.get("opex_week"):
            return None

        # Vol filter — skip SHOCK and HIGH
        regime_vol = regime_labels.get("regime_vol", "NORMAL")
        if regime_vol in ("SHOCK", "HIGH"):
            return None

        # Regime trend — skip DOWN entirely
        regime_trend = regime_labels.get("regime_trend", "FLAT").upper()
        if regime_trend == "DOWN":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        if len(closes) < max(self.sma_period, self.roc_period) + 1:
            return None

        # Min price filter
        if bar.close < self.min_price:
            return None

        # ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Skip very volatile stocks
        if bar.close > 0 and atr / bar.close > self.max_atr_pct:
            return None

        # Rate of Change
        roc_ref = closes[-(self.roc_period + 1)]
        if roc_ref < 1e-9:
            return None
        roc = (bar.close - roc_ref) / roc_ref

        # Regime-adaptive ROC threshold
        if regime_trend == "UP":
            if roc > self.roc_up:
                return None
        else:  # FLAT
            if roc > self.roc_flat:
                return None

        # Consecutive down days
        consec_down = self._count_consec_down(closes)
        if consec_down < self.min_consec_down:
            return None

        # IBS: close near daily low (capitulation)
        daily_range = bar.high - bar.low
        if daily_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / daily_range
        if ibs > self.ibs_threshold:
            return None

        # Volume surge confirmation
        if len(volumes) >= 20:
            avg_vol = sum(volumes[-20:]) / 20
            if avg_vol > 0 and bar.volume < avg_vol * self.vol_surge_mult:
                return None

        # Skip extremely wide bars
        if daily_range > 2.0 * atr:
            return None

        # Stop and target
        stop = bar.close - self.stop_atr_mult * atr
        sma = self._sma(closes, self.sma_period)
        if sma is None:
            return None
        target = sma  # Revert to mean

        if target <= bar.close:
            return None

        # Cap risk per trade
        risk = (bar.close - stop) / bar.close if bar.close > 0 else 1.0
        if risk > self.max_risk_pct:
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
                "roc": round(roc, 4),
                "ibs": round(ibs, 3),
                "consec_down": consec_down,
                "atr": round(atr, 4),
                "sma20": round(sma, 2),
                "seed": "regime_switch_roc_v1",
            },
        )
