"""Daily Research V7A — Multi-Factor Scoring Mean Reversion.

Confluence scoring: IBS + consecutive down days + BB z-score + volume spike.
Long-only, daily bars, regime-adaptive.
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

        # IBS threshold — close must be in bottom portion of day's range
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))

        # Consecutive down days minimum
        self.down_days_min = int(config.get("down_days_min", 2))

        # Bollinger Band params (for z-score, not hard gate)
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))

        # Volume filter — current vol must be above this fraction of avg
        self.vol_avg_mult = float(config.get("vol_avg_mult", 0.3))

        # Confluence score threshold to enter
        self.score_threshold = float(config.get("score_threshold", 35.0))

        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.drawdown_max = float(config.get("drawdown_max", 0.25))

        # ATR for stop/target
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

    def _count_down_days(self, bars: list[Bar], max_lookback: int = 10) -> int:
        """Count consecutive down days (close < prior close) ending at current bar."""
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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]

        # --- IBS (Internal Bar Strength) ---
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range

        # --- Consecutive down days ---
        down_days = self._count_down_days(bars)

        # --- Bollinger Band z-score ---
        sma = self._sma(closes, self.bb_period)
        std = self._std(closes, self.bb_period)
        if sma is None or std is None or std < 1e-9:
            return None
        z_score = (bar.close - sma) / std

        # --- Volume ---
        volumes = [b.volume for b in bars]
        avg_vol = self._sma(volumes, 20)
        if avg_vol is None or avg_vol < 1:
            return None
        vol_ratio = bar.volume / avg_vol

        # --- ATR ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # --- Drawdown filter ---
        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.drawdown_max:
            return None

        # --- Trend filter: SMA(50) uptrend ---
        sma50 = self._sma(closes, 50)
        if sma50 is None:
            return None

        # --- CONFLUENCE SCORING ---
        score = 0.0

        # Factor 1: IBS — close near low of day (0–25 pts)
        if ibs < self.ibs_threshold:
            score += 25.0 * (1.0 - ibs / self.ibs_threshold)

        # Factor 2: Consecutive down days (0–25 pts)
        if down_days >= self.down_days_min:
            score += min(25.0, 10.0 * down_days)

        # Factor 3: BB z-score — how far below mean (0–25 pts)
        if z_score < -1.0:
            score += min(25.0, 10.0 * abs(z_score))

        # Factor 4: Volume participation (0–15 pts)
        if vol_ratio >= self.vol_avg_mult:
            score += min(15.0, 5.0 * vol_ratio)

        # Factor 5: Price above SMA(50) — uptrend bonus (0–10 pts)
        if bar.close > sma50:
            score += 10.0

        # --- Regime adjustments ---
        labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = labels.get("regime_trend", "FLAT")
        regime_vol = labels.get("regime_vol", "NORMAL")

        # Boost score in favorable regimes
        if regime_trend == "UP":
            score *= 1.1
        elif regime_trend == "DOWN":
            score *= 0.8  # penalize but don't block

        # Reduce in shock vol
        if regime_vol == "SHOCK":
            score *= 0.5

        if score < self.score_threshold:
            return None

        # --- Entry: target = BB midline, stop = ATR-based ---
        stop = bar.close - self.stop_atr_mult * atr
        target = sma  # BB midline

        # Target must be above entry
        if target <= bar.close:
            # Use ATR-based target instead
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
                "ibs": round(ibs, 3),
                "down_days": down_days,
                "z_score": round(z_score, 2),
                "vol_ratio": round(vol_ratio, 2),
                "atr": round(atr, 4),
                "seed": "mean_reversion",
            },
        )
