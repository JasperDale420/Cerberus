"""IBS Mean Reversion with Regime-Adaptive Sizing.

Core signal: Internal Bar Strength (IBS) = (close - low) / (high - low).
IBS near 0 = closed near low = oversold bounce expected next day.
Combined with consecutive down days for confirmation.

Regime adaptation: adjust stop/target multipliers per trend regime.
  UP   — wider target (let winners run), normal stop
  DOWN — tighter target (take profit quick), tighter stop
  FLAT — balanced

Long-only. Skip SHOCK vol, earnings, FOMC.
Short hold (max 3 days) for consistency across all windows.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv7dStrategy(BaseStrategy):
    name = "daily_research_v7d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 30))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.2))
        self.consec_down = int(config.get("consec_down", 1))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 3))
        self.sma_period = int(config.get("sma_period", 50))
        # Regime-adaptive multipliers
        self.up_target_scale = float(config.get("up_target_scale", 1.3))
        self.down_target_scale = float(config.get("down_target_scale", 0.8))
        self.down_stop_scale = float(config.get("down_stop_scale", 0.8))

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
        """Internal Bar Strength: (close - low) / (high - low). 0=closed at low, 1=closed at high."""
        rng = bar.high - bar.low
        if rng < 1e-9:
            return None
        return (bar.close - bar.low) / rng

    def _count_consecutive_down(self, closes: list[float]) -> int:
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
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

        # Skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Skip earnings and FOMC
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        if len(closes) < self.min_bars:
            return None

        # Core indicator: ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Min price filter to avoid penny stocks
        if bar.close < 5.0:
            return None

        # ATR/price filter: skip if ATR too small relative to price (dead stocks)
        if atr / bar.close < 0.005:
            return None

        # IBS signal
        ibs = self._ibs(bar)
        if ibs is None or ibs >= self.ibs_threshold:
            return None

        # Consecutive down day confirmation
        consec = self._count_consecutive_down(closes)
        if consec < self.consec_down:
            return None

        # Trend context for regime-adaptive sizing
        regime_trend = labels.get("regime_trend", "FLAT").upper()

        # Regime-adaptive stop/target
        stop_mult = self.stop_atr_mult
        target_mult = self.target_atr_mult

        if regime_trend == "UP":
            target_mult *= self.up_target_scale
        elif regime_trend == "DOWN":
            target_mult *= self.down_target_scale
            stop_mult *= self.down_stop_scale

        stop = bar.close - stop_mult * atr
        target = bar.close + target_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "ibs": round(ibs, 3),
                "consec_down": consec,
                "regime": regime_trend,
                "atr": round(atr, 4),
                "stop_mult": round(stop_mult, 3),
                "target_mult": round(target_mult, 3),
                "seed": "ibs_mean_reversion",
            },
        )
