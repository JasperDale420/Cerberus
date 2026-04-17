"""Regime-Adaptive Keltner + IBS Strategy (long-only).

Different entry logic per regime, unified Keltner Channel framework + IBS filter:
  UP   — buy dips below lower KC + IBS < threshold + 1+ consecutive down days
  DOWN — buy extreme dips below KC + high volume (capitulation) + low IBS
  FLAT — buy at lower KC + 2+ consecutive downs + very low IBS

All entries are long-only. Stops scaled by ATR, targets regime-dependent.
Skips SHOCK vol, earnings, FOMC. Skips HIGH vol in UP regime.
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
        # IBS thresholds per regime
        self.up_ibs_max = float(config.get("up_ibs_max", 0.35))
        self.down_ibs_max = float(config.get("down_ibs_max", 0.30))
        self.flat_ibs_max = float(config.get("flat_ibs_max", 0.30))
        # Entry depth per regime (ATR units below lower KC)
        self.up_depth = float(config.get("up_depth", 0.0))
        self.down_depth = float(config.get("down_depth", 0.3))
        self.flat_depth = float(config.get("flat_depth", 0.0))
        # Consecutive down requirements
        self.up_consec_min = int(config.get("up_consec_min", 1))
        self.flat_consec_min = int(config.get("flat_consec_min", 2))
        # Volume filter for DOWN regime
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.down_vol_min = float(config.get("down_vol_min", 1.2))
        # Stop/target
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.up_target_atr_mult = float(config.get("up_target_atr_mult", 2.0))
        self.down_target_atr_mult = float(config.get("down_target_atr_mult", 1.5))
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

        # IBS
        ibs = self._ibs(bar)

        # Regime from labels
        regime_trend = labels.get("regime_trend", "FLAT").upper()
        regime_vol = labels.get("regime_vol", "NORMAL").upper()

        # Distance below lower KC in ATR units
        kc_distance = (lower_kc - bar.close) / atr if atr > 0 else 0

        # Consecutive down days
        consec = self._consecutive_downs(closes)

        if regime_trend == "UP":
            # Skip HIGH vol in UP — inconsistent
            if regime_vol == "HIGH":
                return None
            # Buy dips: below lower KC + low IBS + at least 1 consec down
            if kc_distance < self.up_depth:
                return None
            if ibs > self.up_ibs_max:
                return None
            if consec < self.up_consec_min:
                return None

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
                    "ibs": round(ibs, 3),
                    "consec": consec,
                    "seed": "keltner_regime_switch",
                },
            )

        elif regime_trend == "DOWN":
            # Buy extreme dips: deeper below KC + capitulation volume + low IBS
            if kc_distance < self.down_depth:
                return None
            if ibs > self.down_ibs_max:
                return None

            # Volume ratio check
            if len(volumes) >= self.vol_avg_period:
                avg_vol = sum(volumes[-self.vol_avg_period :]) / self.vol_avg_period
            else:
                avg_vol = 0
            vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 1.0
            if vol_ratio < self.down_vol_min:
                return None

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
                    "ibs": round(ibs, 3),
                    "vol_ratio": round(vol_ratio, 2),
                    "seed": "keltner_regime_switch",
                },
            )

        else:
            # FLAT: mean reversion — at lower KC + consecutive downs + low IBS
            if kc_distance < self.flat_depth:
                return None
            if ibs > self.flat_ibs_max:
                return None
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
                    "ibs": round(ibs, 3),
                    "consec": consec,
                    "seed": "keltner_regime_switch",
                },
            )

        return None
