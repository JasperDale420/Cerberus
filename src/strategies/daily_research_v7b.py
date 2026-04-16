"""Consecutive-Down Bounce with Tight Vol + Trend Filter.

Buy after N consecutive down closes with low IBS.
Tight ATR/price vol filter (2.5%) + rising SMA trend guard.
BB mean as target for natural reversion. Long-only, daily bars.
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
        self.consec_down_days = int(config.get("consec_down_days", 3))
        self.ibs_entry_threshold = float(config.get("ibs_entry_threshold", 0.3))
        # ATR-based volatility filter
        self.atr_period = int(config.get("atr_period", 14))
        self.max_atr_pct = float(config.get("max_atr_pct", 0.025))
        # BB for target calculation
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        # Risk management
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        # SMA trend filter
        self.sma_period = int(config.get("sma_period", 50))

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

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        data = values[-period:]
        mean = sum(data) / period
        variance = sum((x - mean) ** 2 for x in data) / period
        return variance**0.5

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

        # SHOCK regime — always skip
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # --- Tight ATR-based vol filter ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None
        atr_pct = atr / bar.close
        if atr_pct > self.max_atr_pct:
            return None

        # --- SMA trend filter: price not too far below SMA ---
        sma = self._sma(closes, self.sma_period)
        if sma is not None and bar.close < sma * 0.93:
            return None

        # --- Also check SMA is not in steep decline ---
        if sma is not None and len(closes) >= self.sma_period + 5:
            sma_5ago = self._sma(closes[:-5], self.sma_period)
            if sma_5ago is not None and sma < sma_5ago * 0.98:
                return None  # SMA declining > 2% over 5 bars — bearish

        # --- Entry Condition 1: Consecutive down closes ---
        if not self._consec_down(closes, self.consec_down_days):
            return None

        # --- Entry Condition 2: Low IBS ---
        ibs = self._ibs(bar)
        if ibs > self.ibs_entry_threshold:
            return None

        # --- Target: BB mean (natural reversion target) ---
        bb_mean = self._sma(closes, self.bb_period)
        if bb_mean is None:
            return None

        # Only take trade if there's enough upside to BB mean
        upside = bb_mean - bar.close
        if upside < atr * 0.5:
            return None  # Not enough upside to mean

        stop = bar.close - self.stop_atr_mult * atr
        target = bb_mean  # Revert to mean

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "mode": "consec_down_to_mean",
                "ibs": round(ibs, 2),
                "consec_down": self.consec_down_days,
                "atr_pct": round(atr_pct, 4),
                "upside_atr": round(upside / atr, 2),
            },
        )
