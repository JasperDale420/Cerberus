"""Confirmed Reversal — buy the bounce day after an oversold setup.

Two-bar pattern:
  Day 1 (setup): 2+ consecutive down closes + low IBS + near lower BB
  Day 2 (entry): close > previous close (reversal confirmed)

This avoids buying into continued declines. We only enter when the
bounce has actually started. Structurally different from buying the
oversold day itself.

Skip SHOCK vol, earnings, FOMC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v8b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True
        # Track which symbols had a valid setup yesterday
        self._setup_active: dict[str, bool] = {}
        self._setup_bb_sma: dict[str, float] = {}
        self._setup_atr: dict[str, float] = {}

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.consec_down_min = int(config.get("consec_down_min", 2))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.25))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.bb_proximity = 1.0

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        subset = values[-period:]
        mean = sum(subset) / period
        variance = sum((v - mean) ** 2 for v in subset) / period
        return variance**0.5

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

    def _check_setup(self, symbol: str, bar: Bar, closes: list[float], bars: list[Bar]) -> bool:
        """Check if today qualifies as a setup day (oversold conditions)."""
        # Consecutive down closes
        consec = self._consecutive_downs(closes)
        if consec < self.consec_down_min:
            return False

        # Low IBS
        ibs = self._ibs(bar)
        if ibs > self.ibs_threshold:
            return False

        # Near lower BB
        bb_sma = self._sma(closes, self.bb_period)
        bb_std_val = self._std(closes, self.bb_period)
        if bb_sma is None or bb_std_val is None or bb_std_val < 1e-9:
            return False

        lower_bb = bb_sma - self.bb_std * bb_std_val
        bb_distance = (bar.close - lower_bb) / bb_std_val
        if bb_distance > self.bb_proximity:
            return False

        # ATR for later use
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return False

        # Store for entry day
        self._setup_bb_sma[symbol] = bb_sma
        self._setup_atr[symbol] = atr
        return True

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
            self._setup_active[symbol] = False
            return None

        # Event filters
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            self._setup_active[symbol] = False
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Check if we had a setup yesterday and today confirms reversal
        signal = None
        if self._setup_active.get(symbol, False) and len(closes) >= 2:
            # Reversal confirmation: today's close > yesterday's close
            if bar.close > closes[-2]:
                atr = self._setup_atr.get(symbol, 0)
                bb_sma = self._setup_bb_sma.get(symbol, 0)
                if atr > 1e-9:
                    stop = bar.close - self.stop_atr_mult * atr
                    target_bb = bb_sma
                    target_atr = bar.close + self.target_atr_mult * atr
                    target = min(target_bb, target_atr)

                    if target > bar.close:
                        self.last_signal_time[symbol] = bar.time
                        signal = self._create_signal(
                            symbol,
                            OrderSide.BUY,
                            bar,
                            market_state,
                            stop_price=stop,
                            target_price=target,
                            meta={
                                "atr": round(atr, 4),
                                "seed": "confirmed_reversal",
                            },
                        )

        # Check if today is a new setup day (regardless of whether we traded)
        self._setup_active[symbol] = self._check_setup(symbol, bar, closes, bars)

        return signal
