"""Research v7a: Volatility Breakout — ATR expansion + N-day high + volume.

Entry: price closes above highest high of lookback period,
       current ATR > expansion_mult * avg ATR (volatility expanding),
       volume above average, price above SMA (trend confirmation).
Filters: regime (skip SHOCK, DOWN), earnings/FOMC, drawdown, ATR/price.
Exit: ATR-based stop/target, max hold days.
Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v7a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 40))
        self.breakout_lookback = int(config.get("breakout_lookback", 10))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_expansion_mult = float(config.get("atr_expansion_mult", 1.2))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.trend_period = int(config.get("trend_period", 20))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 30))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.12))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.max_atr_pct = float(config.get("max_atr_pct", 0.04))

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
    def _highest_high(bars: list[Bar], lookback: int) -> Optional[float]:
        if len(bars) < lookback + 1:
            return None
        return max(b.high for b in bars[-(lookback + 1) : -1])

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

        # --- Regime filter: skip SHOCK vol and DOWN trend ---
        labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = labels.get("regime_trend", "FLAT")
        regime_vol = labels.get("regime_vol", "NORMAL")
        if regime_vol == "SHOCK":
            return None
        if regime_trend == "DOWN":
            return None

        # --- Event filter: skip near earnings and FOMC ---
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        # --- Trend confirmation: price above SMA ---
        sma = self._sma(closes, self.trend_period)
        if sma is None or bar.close < sma:
            return None

        # --- ATR ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # ATR/price filter — skip overly volatile stocks
        if bar.close > 0 and atr / bar.close > self.max_atr_pct:
            return None

        # --- Volatility expansion: current ATR must be above avg ATR ---
        avg_atr_bars_needed = self.atr_period + self.vol_avg_period + 1
        if len(bars) < avg_atr_bars_needed:
            return None
        recent_atrs = []
        for offset in range(self.vol_avg_period):
            idx = len(bars) - 1 - offset
            if idx < self.atr_period + 1:
                break
            sub_bars = bars[: idx + 1]
            a = self._atr(sub_bars, self.atr_period)
            if a is not None:
                recent_atrs.append(a)
        if len(recent_atrs) < self.vol_avg_period:
            return None
        avg_atr = sum(recent_atrs) / len(recent_atrs)
        if avg_atr < 1e-9 or atr < avg_atr * self.atr_expansion_mult:
            return None

        # --- Price breakout: close above highest high of lookback period ---
        prev_high = self._highest_high(bars, self.breakout_lookback)
        if prev_high is None or bar.close <= prev_high:
            return None

        # --- Volume confirmation: today's volume above 20-day average ---
        volumes = [b.volume for b in bars]
        vol_avg = self._sma(volumes, 20)
        if vol_avg is None or vol_avg < 1 or bar.volume < vol_avg:
            return None

        # --- Drawdown filter ---
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.max_drawdown_pct:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        max_stop_loss = bar.close * 0.02
        if bar.close - stop > max_stop_loss:
            stop = bar.close - max_stop_loss
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
                "atr": round(atr, 4),
                "atr_expansion": round(atr / avg_atr, 3),
                "breakout_high": round(prev_high, 2),
                "vol_ratio": round(bar.volume / vol_avg, 2) if vol_avg else 0,
                "regime": f"{regime_trend}+{regime_vol}",
                "seed": "vol_breakout",
            },
        )
