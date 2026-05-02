"""Regime-Adaptive Dip Buyer — Unified entry with regime-scaled thresholds.

Evolved from seed_regime_switch. Instead of 3 completely different strategies
per regime, uses ONE core signal (consecutive down + low IBS + below Keltner band)
with regime-adaptive entry strictness:

  UP   — easiest entry (2 down days, IBS<0.35, below KC lower)
  FLAT — moderate entry (2 down days, IBS<0.25, below KC lower)
  DOWN — strict entry (3 down days, IBS<0.20, below KC lower, tighter target)

Long-only. Event filters (earnings, FOMC). Vol filter (skip SHOCK).
Stops: tight ATR-based below bar low. Target: Keltner midline.
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
        self.kc_period = int(config.get("kc_period", 20))
        self.kc_mult = float(config.get("kc_mult", 1.5))
        self.atr_period = int(config.get("atr_period", 14))
        # Regime-adaptive IBS thresholds
        self.ibs_up = float(config.get("ibs_up", 0.30))
        self.ibs_flat = float(config.get("ibs_flat", 0.25))
        self.ibs_down = float(config.get("ibs_down", 0.20))
        # Consecutive down day requirements
        self.consec_down_up = int(config.get("consec_down_up", 2))
        self.consec_down_flat = int(config.get("consec_down_flat", 2))
        self.consec_down_down = int(config.get("consec_down_down", 3))
        # Stop/target
        self.stop_atr_mult = float(config.get("stop_atr_mult", 0.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 4))
        # Filters
        self.max_atr_pct = float(config.get("max_atr_pct", 0.04))
        self.max_risk_pct = float(config.get("max_risk_pct", 0.025))
        self.min_price = float(config.get("min_price", 10.0))
        self.min_rr_ratio = float(config.get("min_rr_ratio", 2.0))

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

        # Vol filter — skip SHOCK and HIGH
        regime_vol = regime_labels.get("vol_regime_symbol", "NORMAL")
        if regime_vol in ("SHOCK", "HIGH"):
            return None
        # Calendar filters
        if regime_labels.get("opex_week"):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        if len(closes) < self.kc_period + 1:
            return None

        # Min price
        if bar.close < self.min_price:
            return None

        # ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Skip very volatile stocks
        if bar.close > 0 and atr / bar.close > self.max_atr_pct:
            return None

        # Keltner Channel
        kc_mid = self._sma(closes, self.kc_period)
        if kc_mid is None:
            return None
        kc_lower = kc_mid - self.kc_mult * atr

        # Price must be below lower Keltner band
        if bar.close >= kc_lower:
            return None

        # IBS: close near daily low (capitulation)
        daily_range = bar.high - bar.low
        if daily_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / daily_range

        # Consecutive down days
        consec_down = self._count_consec_down(closes)

        # Regime-adaptive thresholds
        regime_trend = regime_labels.get("trend_regime_symbol", "FLAT").upper()

        if regime_trend == "UP":
            if consec_down < self.consec_down_up:
                return None
            if ibs > self.ibs_up:
                return None
        elif regime_trend == "DOWN":
            if consec_down < self.consec_down_down:
                return None
            if ibs > self.ibs_down:
                return None
        else:  # FLAT
            if consec_down < self.consec_down_flat:
                return None
            if ibs > self.ibs_flat:
                return None

        # Skip extremely wide bars (news-driven)
        if daily_range > 2.0 * atr:
            return None

        # Stop below bar low, target Keltner midline
        stop = bar.low - self.stop_atr_mult * atr
        target = kc_mid

        if target <= bar.close:
            return None

        # Cap risk per trade
        risk_dollar = bar.close - stop
        if risk_dollar <= 0:
            return None
        risk_pct = risk_dollar / bar.close if bar.close > 0 else 1.0
        if risk_pct > self.max_risk_pct:
            return None

        # Min reward:risk ratio
        reward = target - bar.close
        if reward / risk_dollar < self.min_rr_ratio:
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
                "atr": round(atr, 4),
                "kc_mid": round(kc_mid, 2),
                "kc_lower": round(kc_lower, 2),
                "ibs": round(ibs, 3),
                "consec_down": consec_down,
                "seed": "regime_switch_v2",
            },
        )
