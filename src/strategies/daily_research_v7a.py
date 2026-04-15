"""Multi-Factor Scoring Mean Reversion.

Scores multiple independent oversold signals (IBS, consecutive down days,
z-score, volume contraction) and enters when the composite score exceeds
a threshold. No single indicator is a hard gate — diversity = consistency.

Long-only, daily bars, max_hold_days=5.
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
        self.min_bars = int(config.get("min_bars", 50))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        self.score_threshold = float(config.get("score_threshold", 35.0))
        self.down_days_min = int(config.get("down_days_min", 2))
        self.vol_avg_mult = float(config.get("vol_avg_mult", 0.3))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.drawdown_max = float(config.get("drawdown_max", 0.25))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))

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
    def _consecutive_down_days(closes: list[float]) -> int:
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
        highs = [b.high for b in bars]
        volumes = [b.volume for b in bars]

        # --- Compute indicators ---
        sma = self._sma(closes, self.bb_period)
        std = self._std(closes, self.bb_period)
        if sma is None or std is None or std < 1e-9:
            return None

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        z_score = (bar.close - sma) / std

        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range

        down_days = self._consecutive_down_days(closes)

        avg_vol = self._sma(volumes, 20)
        vol_ratio = bar.volume / avg_vol if avg_vol and avg_vol > 0 else 1.0

        # Drawdown filter
        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs) if lookback_highs else bar.close
        drawdown = (peak - bar.close) / peak if peak > 0 else 0.0
        if drawdown > self.drawdown_max:
            return None

        # --- Regime filter ---
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("regime_vol") == "SHOCK":
            return None
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        # --- Multi-factor scoring ---
        score = 0.0

        # Factor 1: IBS — close near low of day (0-30 points)
        if ibs < self.ibs_threshold:
            score += 30.0 * (1.0 - ibs / self.ibs_threshold)

        # Factor 2: Consecutive down days (0-25 points)
        if down_days >= self.down_days_min:
            score += min(25.0, 10.0 * down_days)

        # Factor 3: Z-score — how oversold vs BB (0-25 points)
        if z_score < -1.0:
            score += min(25.0, 10.0 * abs(z_score))

        # Factor 4: Volume contraction (0-10 points)
        if vol_ratio < self.vol_avg_mult:
            score += 10.0

        # Factor 5: Price above long-term SMA (uptrend intact) (0-10 points)
        sma50 = self._sma(closes, 50)
        if sma50 is not None and bar.close > sma50:
            score += 10.0

        if score < self.score_threshold:
            return None

        # Target: BB midline + small offset
        target = sma + std * self.target_atr_mult
        if target <= bar.close:
            target = sma
        if target <= bar.close:
            return None

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
                "score": round(score, 1),
                "ibs": round(ibs, 3),
                "z_score": round(z_score, 2),
                "down_days": down_days,
                "vol_ratio": round(vol_ratio, 2),
                "drawdown": round(drawdown, 4),
                "seed": "mean_reversion",
            },
        )
