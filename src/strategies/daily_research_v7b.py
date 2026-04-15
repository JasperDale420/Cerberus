"""Multi-Factor Confluence Scoring Strategy.

Uses a weighted scoring system across multiple oversold/pullback indicators.
Enters when total score exceeds a regime-adaptive threshold.
No single indicator can block entry — diversity prevents fragility.

Factors:
1. IBS (Internal Bar Strength) — close near day low
2. Z-score (distance from 20-SMA in std devs) — statistical oversold
3. 3-day return — recent pullback magnitude
4. Volume ratio — participation confirmation
5. Prior day pattern — down close confirms selling pressure

Regime adaptation: lower threshold in FLAT (more mean-reversion friendly),
higher in DOWN (more selective). UP regime gets slightly higher threshold.
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
        # Scoring thresholds
        self.score_threshold = float(config.get("score_threshold", 55.0))
        # Z-score params
        self.zscore_period = int(config.get("zscore_period", 20))
        # IBS weight
        self.ibs_weight = float(config.get("ibs_weight", 30.0))
        # Shared
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))

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
        data = values[-period:]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
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

        if len(closes) < self.min_bars:
            return None

        # ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # --- Compute factor scores ---
        score = 0.0

        # Factor 1: IBS (Internal Bar Strength) — low = oversold
        rng = bar.high - bar.low
        if rng > 1e-9:
            ibs = (bar.close - bar.low) / rng
            if ibs < 0.15:
                score += self.ibs_weight  # Strong oversold
            elif ibs < 0.30:
                score += self.ibs_weight * 0.6
            elif ibs < 0.45:
                score += self.ibs_weight * 0.2
        else:
            return None  # No range = no trade

        # Factor 2: Z-score (distance from mean in std devs)
        sma = self._sma(closes, self.zscore_period)
        std = self._std(closes, self.zscore_period)
        if sma is not None and std is not None and std > 1e-9:
            zscore = (bar.close - sma) / std
            if zscore < -2.0:
                score += 25.0
            elif zscore < -1.0:
                score += 15.0
            elif zscore < -0.5:
                score += 8.0

        # Factor 3: 3-day return (recent pullback)
        if len(closes) >= 4:
            ret_3d = (bar.close - closes[-4]) / closes[-4]
            if ret_3d < -0.05:
                score += 20.0
            elif ret_3d < -0.02:
                score += 12.0
            elif ret_3d < -0.01:
                score += 5.0

        # Factor 4: Volume above average (participation)
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is not None and avg_vol > 0:
            vol_ratio = bar.volume / avg_vol
            if vol_ratio > 1.5:
                score += 15.0
            elif vol_ratio > 1.0:
                score += 8.0
            elif vol_ratio > 0.7:
                score += 3.0
            else:
                return None  # Minimum volume gate

        # Factor 5: Prior day was a down day (selling pressure confirmation)
        if len(closes) >= 2 and closes[-2] > bar.close:
            score += 10.0

        # --- Regime-adaptive threshold ---
        regime_trend = labels.get("regime_trend", "FLAT")
        if regime_trend == "FLAT":
            threshold = self.score_threshold * 0.9  # More permissive
        elif regime_trend == "DOWN":
            threshold = self.score_threshold * 1.0
        else:  # UP
            threshold = self.score_threshold * 1.1  # More selective

        if score < threshold:
            return None

        # --- Entry ---
        stop = bar.close - self.stop_atr_mult * atr
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
                "score": round(score, 1),
                "threshold": round(threshold, 1),
                "regime": regime_trend,
                "atr": round(atr, 4),
            },
        )
