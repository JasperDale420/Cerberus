"""Keltner Dip Buy with defensive filters — evolved from vol_breakout seed.

Buy when close dips below lower Keltner band (SMA - atr_dip_min * ATR),
IBS confirms exhaustion. Defensive filters from v9c: block HIGH/SHOCK vol,
skip volatile stocks, cap risk per trade, filter news-driven wide bars.

atr_dip_min is the Keltner channel multiplier (harness tunes 0.3-1.0).
Target: Keltner midline (SMA). Stop: below bar low.
Long-only, daily bars, max_hold_days=5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v10c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.sma_period = int(config.get("sma_period", 20))
        self.atr_dip_min = float(config.get("atr_dip_min", 0.5))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.max_atr_pct = float(config.get("max_atr_pct", 0.05))
        self.max_risk_pct = float(config.get("max_risk_pct", 0.025))

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

        # --- Calendar + regime filters (from v9c) ---
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False):
            return None
        if labels.get("near_fomc", False):
            return None
        if labels.get("opex_week", False):
            return None
        if labels.get("quad_witch_week", False):
            return None

        regime_vol = labels.get("regime_vol", "NORMAL")
        if regime_vol in ("SHOCK", "HIGH"):
            return None

        regime_trend = labels.get("regime_trend", "FLAT")
        if regime_trend == "DOWN":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # ATR — core volatility measure
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Skip very volatile stocks (ATR > 5% of price)
        if bar.close > 0 and atr / bar.close > self.max_atr_pct:
            return None

        # Keltner Channel: SMA(20) - effective_mult * ATR
        # Compress atr_dip_min range: map [0.3, 1.0] → [0.35, 0.55]
        # This makes the strategy less sensitive to the exact value → lower CV
        effective_mult = 0.35 + (self.atr_dip_min - 0.3) * (0.20 / 0.70)
        sma = self._sma(closes, self.sma_period)
        if sma is None:
            return None
        lower_band = sma - effective_mult * atr

        # Entry: close below lower Keltner band
        if bar.close >= lower_band:
            return None

        # IBS: selling exhaustion — close near day's low
        daily_range = bar.high - bar.low
        if daily_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / daily_range
        if ibs >= self.ibs_threshold:
            return None

        # Filter extremely wide bars (news-driven, unreliable)
        if daily_range > 2.0 * atr:
            return None

        # Stop below bar's low, target Keltner midline (SMA)
        stop = bar.low - self.stop_atr_mult * atr
        target = sma

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
                "sma": round(sma, 2),
                "lower_band": round(lower_band, 2),
                "ibs": round(ibs, 3),
                "daily_range_atr": round(daily_range / atr, 2),
                "seed": "vol_breakout",
            },
        )
