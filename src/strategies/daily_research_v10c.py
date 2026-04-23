"""Keltner Channel Dip Buy — evolved from vol_breakout seed.

Buy when price closes below the lower Keltner Channel band (EMA - ATR*mult),
indicating an ATR-normalized oversold dip. Target the middle band (EMA).
Uses ATR for both channel width and stop sizing — maintains volatility DNA.

Filters: Skip earnings/FOMC, skip SHOCK vol regime.
Long-only, daily bars, max_hold_days=5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
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
        self.ema_period = int(config.get("ema_period", 20))
        self.atr_period = int(config.get("atr_period", 14))
        self.keltner_mult = float(config.get("keltner_mult", 2.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.trend_sma_period = int(config.get("trend_sma_period", 50))

    # --- Indicator helpers ---

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = v * mult + ema * (1 - mult)
        return ema

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

        # --- Regime filters ---
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False):
            return None
        if labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Keltner Channel middle line (EMA)
        ema = self._ema(closes, self.ema_period)
        if ema is None:
            return None

        # ATR for channel bands
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Keltner bands
        lower_band = ema - self.keltner_mult * atr

        # Entry: close below lower Keltner band (dip below volatility envelope)
        if bar.close >= lower_band:
            return None

        # Trend context: price should be above SMA(50) — dip in an uptrend
        sma_trend = self._sma(closes, self.trend_sma_period)
        if sma_trend is None:
            return None
        if bar.close <= sma_trend * 0.95:
            return None  # Too far below trend — not a dip, it's a breakdown

        # Stop: 1.5 ATR below entry
        stop = bar.close - self.stop_atr_mult * atr
        # Target: middle Keltner band (EMA) — natural mean reversion target
        target = ema

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "ema": round(ema, 2),
                "atr": round(atr, 4),
                "lower_band": round(lower_band, 2),
                "depth": round((ema - bar.close) / atr, 2),
                "seed": "vol_breakout",
            },
        )
