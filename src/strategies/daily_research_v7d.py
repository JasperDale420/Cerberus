"""Regime-switch strategy: different entry logic per market regime.

UP regime: Buy pullbacks to SMA(20) in established uptrends (EMA20 > EMA50).
FLAT regime: Mean reversion via Bollinger Band fade + IBS confirmation.
DOWN regime: Skip entirely — no consistent edge found.

True regime-adaptive: the ENTRY LOGIC changes, not just thresholds.
Long-only for consistency. Event and regime filters.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedRegimeSwitchStrategy(BaseStrategy):
    name = "daily_research_v7d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        # Shared
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        # UP regime: trend pullback
        self.sma_fast = int(config.get("sma_fast", 20))
        self.sma_slow = int(config.get("sma_slow", 50))
        self.pullback_pct = float(config.get("pullback_pct", 0.02))
        self.target_atr_up = float(config.get("target_atr_up", 2.5))
        # FLAT regime: BB mean reversion
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        self.target_atr_flat = float(config.get("target_atr_flat", 2.0))

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
    def _ibs(bar: Bar) -> float:
        rng = bar.high - bar.low
        if rng < 1e-9:
            return 0.5
        return (bar.close - bar.low) / rng

    @staticmethod
    def _consecutive_downs(closes: list[float]) -> int:
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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Event filter
        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings", False):
            return None
        if regime_labels.get("near_fomc", False):
            return None

        # Regime filter: skip SHOCK vol entirely
        regime_vol = regime_labels.get("regime_vol", "NORMAL").upper()
        if regime_vol == "SHOCK":
            return None

        regime_trend = regime_labels.get("regime_trend", "FLAT").upper()

        # DOWN regime: skip entirely — no consistent edge
        if regime_trend == "DOWN":
            return None

        # Common indicators
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Route to regime-specific logic
        if regime_trend == "UP":
            return self._up_regime_signal(symbol, bar, bars, closes, atr, regime_vol, market_state)
        else:
            return self._flat_regime_signal(symbol, bar, bars, closes, atr, regime_vol, market_state)

    def _up_regime_signal(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        atr: float,
        regime_vol: str,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """UP regime: buy pullbacks to SMA(20) in established uptrends."""
        sma_fast = self._sma(closes, self.sma_fast)
        sma_slow = self._sma(closes, self.sma_slow)
        if sma_fast is None or sma_slow is None:
            return None

        # Require established uptrend: fast > slow
        if sma_fast <= sma_slow:
            return None

        # Price must have pulled back near or below fast SMA
        distance_pct = (bar.close - sma_fast) / sma_fast
        if distance_pct > self.pullback_pct:
            return None  # Too far above — chasing

        # Price must be above slow SMA (still in trend)
        if bar.close < sma_slow:
            return None

        # Require at least 1 down day (actual pullback, not just hovering)
        if len(closes) >= 2 and closes[-1] >= closes[-2]:
            return None

        # In HIGH vol, require deeper pullback (more margin of safety)
        if regime_vol == "HIGH" and distance_pct > 0.0:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        target = bar.close + self.target_atr_up * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "mode": "UP_PULLBACK",
                "regime_vol": regime_vol,
                "distance_pct": round(distance_pct, 4),
                "sma_spread": round((sma_fast - sma_slow) / sma_slow, 4),
            },
        )

    def _flat_regime_signal(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        atr: float,
        regime_vol: str,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """FLAT regime: Bollinger Band mean reversion with IBS confirmation."""
        bb_sma = self._sma(closes, self.bb_period)
        bb_std_val = self._std(closes, self.bb_period)
        if bb_sma is None or bb_std_val is None or bb_std_val < 1e-9:
            return None

        lower_bb = bb_sma - self.bb_std * bb_std_val

        # Price must be near or below lower BB
        if bar.close > lower_bb:
            return None

        # IBS confirmation: closed near the low
        ibs = self._ibs(bar)
        if ibs > self.ibs_threshold:
            return None

        # Require at least 2 consecutive down days
        consec = self._consecutive_downs(closes)
        if consec < 2:
            return None

        # Target: BB midline (mean reversion)
        target = bb_sma
        if target <= bar.close:
            return None

        # Cap target at ATR-based level
        target = min(target, bar.close + self.target_atr_flat * atr)

        stop = bar.close - self.stop_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "mode": "FLAT_REVERSION",
                "regime_vol": regime_vol,
                "ibs": round(ibs, 3),
                "consec_down": consec,
                "bb_distance": round((bar.close - lower_bb) / bb_std_val, 2),
            },
        )
