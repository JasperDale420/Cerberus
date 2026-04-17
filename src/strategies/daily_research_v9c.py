"""Keltner Dip Buy — Buy oversold dips at lower Keltner Channel band.

Evolved from seed_vol_breakout (volatility family). Uses ATR-based Keltner
Channels to identify when price is stretched below its normal range, combined
with IBS (Internal Bar Strength) to confirm capitulation selling.

Entry:
1. Price < lower Keltner band (oversold relative to ATR-defined range)
2. IBS < threshold (close near daily low — selling exhaustion)
3. Regime filter: skip SHOCK vol, skip earnings/FOMC
4. Calendar: skip opex week, quad witch week

Exit: ATR-based stop and target (reversion to Keltner midline).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v9c"
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
        self.ibs_threshold = float(config.get("ibs_threshold", 0.30))
        self.min_consec_down = int(config.get("min_consec_down", 2))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 0.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 4))
        self.max_atr_pct = float(config.get("max_atr_pct", 0.05))
        self.max_risk_pct = float(config.get("max_risk_pct", 0.025))
        self.min_price = float(config.get("min_price", 10.0))

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

        # Skip cheap stocks
        if bar.close < self.min_price:
            return None

        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings"):
            return None
        if regime_labels.get("near_fomc"):
            return None
        if regime_labels.get("opex_week"):
            return None
        if regime_labels.get("quad_witch_week"):
            return None

        regime_vol = regime_labels.get("regime_vol", "NORMAL")
        if regime_vol in ("SHOCK", "HIGH"):
            return None

        regime_trend = regime_labels.get("regime_trend", "FLAT")
        if regime_trend == "DOWN":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        if len(closes) < self.kc_period + 1:
            return None

        # Consecutive down days filter
        if len(closes) >= self.min_consec_down + 1:
            consec_down = 0
            for i in range(1, min(len(closes), 10)):
                if closes[-i] < closes[-i - 1]:
                    consec_down += 1
                else:
                    break
            if consec_down < self.min_consec_down:
                return None

        # ATR for Keltner Channel width
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Skip very volatile stocks (ATR > max_atr_pct of price)
        if bar.close > 0 and atr / bar.close > self.max_atr_pct:
            return None

        # Keltner Channel
        kc_mid = self._sma(closes, self.kc_period)
        if kc_mid is None:
            return None
        kc_lower = kc_mid - self.kc_mult * atr

        # Price below lower Keltner band
        if bar.close >= kc_lower:
            return None

        # IBS: close near daily low (capitulation)
        daily_range = bar.high - bar.low
        if daily_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / daily_range
        if ibs > self.ibs_threshold:
            return None

        # Filter out extremely wide bars (news-driven, unreliable)
        if daily_range > 2.0 * atr:
            return None

        # Stop just below bar low, target Keltner midline
        stop = bar.low - self.stop_atr_mult * atr
        target = kc_mid

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
                "atr": round(atr, 4),
                "kc_mid": round(kc_mid, 2),
                "kc_lower": round(kc_lower, 2),
                "ibs": round(ibs, 3),
                "daily_range_atr": round(daily_range / atr, 2),
                "regime_vol": regime_vol,
                "seed": "keltner_dip_v1",
            },
        )
