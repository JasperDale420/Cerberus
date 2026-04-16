"""Pure Selective Mean Reversion — Extreme Oversold Bounce.

Buy only when multiple extreme conditions align:
1. Price at/below BB lower band
2. 2-3 consecutive down closes
3. Very low IBS (closed near the low)
4. ATR/price not too extreme
No regime filtering — these conditions are self-selecting for oversold environments.
Target: BB mean. Stop: 1.5 ATR. Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v7b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        # Entry filters
        self.consec_down_days = int(config.get("consec_down_days", 2))
        self.ibs_entry_threshold = float(config.get("ibs_entry_threshold", 0.3))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std_mult = float(config.get("bb_std_mult", 2.0))
        # Risk management
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.max_atr_pct = float(config.get("max_atr_pct", 0.04))
        # Volume
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.5))

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
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        data = values[-period:]
        mean = sum(data) / period
        variance = sum((x - mean) ** 2 for x in data) / period
        return variance**0.5

    @staticmethod
    def _ibs(bar: Bar) -> float:
        rng = bar.high - bar.low
        if rng < 1e-9:
            return 0.5
        return (bar.close - bar.low) / rng

    @staticmethod
    def _consec_down(closes: list[float], min_days: int) -> bool:
        if len(closes) < min_days + 1:
            return False
        for i in range(-min_days, 0):
            if closes[i] >= closes[i - 1]:
                return False
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

        # Only skip SHOCK — let strategy logic self-select in all other regimes
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # --- ATR filter ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None
        atr_pct = atr / bar.close
        if atr_pct > self.max_atr_pct:
            return None

        # --- Condition 1: Consecutive down closes ---
        if not self._consec_down(closes, self.consec_down_days):
            return None

        # --- Condition 2: Low IBS (closed near the low) ---
        ibs = self._ibs(bar)
        if ibs > self.ibs_entry_threshold:
            return None

        # --- Condition 3: Price at/below BB lower band ---
        bb_mean = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_mean is None or bb_std is None or bb_std < 1e-9:
            return None

        lower_band = bb_mean - self.bb_std_mult * bb_std
        if bar.close > lower_band:
            return None

        # --- Volume confirmation ---
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is not None and avg_vol > 0 and bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # --- Target: BB mean, Stop: ATR-based ---
        stop = bar.close - self.stop_atr_mult * atr
        target = bb_mean

        # Minimum reward:risk of 1.0
        upside = target - bar.close
        downside = bar.close - stop
        if downside > 0 and upside / downside < 1.0:
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
                "mode": "extreme_mr",
                "ibs": round(ibs, 2),
                "consec_down": self.consec_down_days,
                "z_score": round((bar.close - bb_mean) / bb_std, 2),
                "rr_ratio": round(upside / downside, 2) if downside > 0 else 0,
            },
        )
