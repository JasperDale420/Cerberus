"""Regime-Adaptive Keltner Channel Strategy (long-only).

Different entry logic per regime, unified Keltner Channel framework:
  UP   — buy shallow dips to lower Keltner with volume contraction (healthy pullback)
  DOWN — buy only at extreme lower Keltner (capitulation) with volume spike
  FLAT — buy at lower Keltner with mean-reversion confirmation (consecutive downs)

All entries are long-only. Stops scaled by ATR, targets regime-dependent.
Skips SHOCK vol, earnings, FOMC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedRegimeSwitchStrategy(BaseStrategy):
    name = "daily_research_v8d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        # Keltner Channel params
        self.kc_ema_period = int(config.get("kc_ema_period", 20))
        self.atr_period = int(config.get("atr_period", 14))
        self.kc_mult = float(config.get("kc_mult", 2.0))
        # Entry depth per regime (how far below lower KC to trigger)
        self.up_depth = float(config.get("up_depth", 0.0))  # at or below lower KC
        self.down_depth = float(config.get("down_depth", 0.5))  # deeper for DOWN regime
        self.flat_depth = float(config.get("flat_depth", 0.0))  # at lower KC
        # Volume filters
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.up_vol_max = float(config.get("up_vol_max", 1.0))  # UP: low volume = healthy dip
        self.down_vol_min = float(config.get("down_vol_min", 1.2))  # DOWN: high volume = capitulation
        # Consecutive down filter for FLAT regime
        self.flat_consec_min = int(config.get("flat_consec_min", 2))
        # Stop/target
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.up_target_atr_mult = float(config.get("up_target_atr_mult", 2.5))
        self.down_target_atr_mult = float(config.get("down_target_atr_mult", 1.5))
        self.flat_target_atr_mult = float(config.get("flat_target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))

    # --- Indicator helpers ---

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
    def _consecutive_downs(closes: list[float]) -> int:
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _ibs(bar: Bar) -> float:
        rng = bar.high - bar.low
        if rng < 1e-9:
            return 0.5
        return (bar.close - bar.low) / rng

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

        # Event filters
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # Core indicators
        ema = self._ema(closes, self.kc_ema_period)
        atr = self._atr(bars, self.atr_period)
        if ema is None or atr is None or atr < 1e-9:
            return None

        # Keltner Channel
        lower_kc = ema - self.kc_mult * atr

        # Volume ratio
        if len(volumes) >= self.vol_avg_period:
            avg_vol = sum(volumes[-self.vol_avg_period :]) / self.vol_avg_period
        else:
            avg_vol = 0
        vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 1.0

        # Regime from labels
        regime_trend = labels.get("regime_trend", "FLAT").upper()

        # Distance below lower KC in ATR units
        kc_distance = (lower_kc - bar.close) / atr if atr > 0 else 0

        if regime_trend == "UP":
            # Buy shallow dips — price at or below lower KC, low volume (healthy pullback)
            if kc_distance < self.up_depth:
                return None
            if vol_ratio > self.up_vol_max:
                return None  # Too much selling pressure

            stop = bar.close - self.stop_atr_mult * atr
            target = bar.close + self.up_target_atr_mult * atr

            self.last_signal_time[symbol] = bar.time
            return self._create_signal(
                symbol,
                OrderSide.BUY,
                bar,
                market_state,
                stop_price=stop,
                target_price=target,
                meta={
                    "regime": "UP",
                    "kc_dist": round(kc_distance, 2),
                    "vol_ratio": round(vol_ratio, 2),
                    "atr": round(atr, 4),
                    "seed": "keltner_regime_switch",
                },
            )

        elif regime_trend == "DOWN":
            # Buy extreme dips only — deeper below KC, with volume spike (capitulation)
            if kc_distance < self.down_depth:
                return None
            if vol_ratio < self.down_vol_min:
                return None  # Need capitulation volume

            stop = bar.close - self.stop_atr_mult * atr
            target = bar.close + self.down_target_atr_mult * atr

            self.last_signal_time[symbol] = bar.time
            return self._create_signal(
                symbol,
                OrderSide.BUY,
                bar,
                market_state,
                stop_price=stop,
                target_price=target,
                meta={
                    "regime": "DOWN",
                    "kc_dist": round(kc_distance, 2),
                    "vol_ratio": round(vol_ratio, 2),
                    "atr": round(atr, 4),
                    "seed": "keltner_regime_switch",
                },
            )

        else:
            # FLAT: mean reversion — at lower KC + consecutive downs
            if kc_distance < self.flat_depth:
                return None
            consec = self._consecutive_downs(closes)
            if consec < self.flat_consec_min:
                return None

            stop = bar.close - self.stop_atr_mult * atr
            target = ema  # Target the EMA (midline)

            if target <= bar.close:
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
                    "regime": "FLAT",
                    "kc_dist": round(kc_distance, 2),
                    "consec_down": consec,
                    "atr": round(atr, 4),
                    "seed": "keltner_regime_switch",
                },
            )

        return None
