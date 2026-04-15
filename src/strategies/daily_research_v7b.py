"""Mean-Reversion-to-SMA Strategy.

Buys when price dips below its short-term moving average with intraday
capitulation (IBS confirmation). Targets the SMA itself as exit, creating
a natural mean-reversion mechanic that works across all regimes.

Entry logic:
  - Price is below SMA(20) by at least min_dip_pct
  - Close in lower portion of day's range (IBS < threshold)
  - At least 1 consecutive down close (selling pressure)
  - Volume above minimum threshold
  - Skip SHOCK volatility

Exit logic:
  - Target: SMA(20) — the mean itself
  - Stop: entry - stop_atr_mult * ATR(14)
  - Max hold: 5 days

Long-only, daily bars. Few parameters for robustness.
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
        self.min_bars = int(config.get("min_bars", 25))
        # SMA period for mean target
        self.sma_period = int(config.get("sma_period", 20))
        # Minimum dip below SMA to trigger (pct)
        self.min_dip_pct = float(config.get("min_dip_pct", 0.01))
        # Max dip — avoid catching falling knives
        self.max_dip_pct = float(config.get("max_dip_pct", 0.08))
        # IBS threshold — close in lower portion of range
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        # ATR for stop
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        # Volume filter
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.7))
        # Max hold
        self.max_hold_days = int(config.get("max_hold_days", 5))
        # Min consecutive down days
        self.min_down_days = int(config.get("min_down_days", 1))
        # Max consecutive down days — avoid free-falls
        self.max_down_days = int(config.get("max_down_days", 5))

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
        rng = bar.high - bar.low
        if rng < 1e-9:
            return None
        return (bar.close - bar.low) / rng

    @staticmethod
    def _count_down_days(bars: list[Bar], max_lookback: int = 6) -> int:
        count = 0
        for i in range(len(bars) - 1, 0, -1):
            if bars[i].close < bars[i - 1].close:
                count += 1
            else:
                break
            if count >= max_lookback:
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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # SMA — our mean-reversion target
        sma = self._sma(closes, self.sma_period)
        if sma is None or sma < 1e-9:
            return None

        # Price must be BELOW the SMA (dip)
        dip_pct = (sma - bar.close) / sma
        if dip_pct < self.min_dip_pct or dip_pct > self.max_dip_pct:
            return None

        # IBS confirmation — close near low of day
        ibs = self._ibs(bar)
        if ibs is None or ibs > self.ibs_threshold:
            return None

        # Consecutive down days — selling pressure confirmation
        down_days = self._count_down_days(bars)
        if down_days < self.min_down_days or down_days > self.max_down_days:
            return None

        # Volume — must have participation
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # ATR for stop
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Target = SMA (the mean we're reverting to)
        target = sma
        # Stop = below entry by ATR multiple
        stop = bar.close - self.stop_atr_mult * atr

        # Sanity: target must be above entry
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
                "dip_pct": round(dip_pct, 4),
                "ibs": round(ibs, 3),
                "down_days": down_days,
                "sma": round(sma, 2),
                "atr": round(atr, 4),
                "archetype": "mean_reversion_to_sma",
            },
        )
