"""Multi-Factor Mean Reversion with Scoring.

Combines multiple weak mean-reversion signals into a composite score:
- IBS (Internal Bar Strength) — low close relative to range
- Consecutive down days — multi-day selloff
- Bollinger z-score — distance below mean
- Volume surge — participation confirmation
- Drawdown filter — avoid catching falling knives

Long-only, daily bars, max_hold_days configurable.
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
        self.down_days_min = int(config.get("down_days_min", 2))
        self.score_threshold = float(config.get("score_threshold", 35.0))
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
    def _count_down_days(closes: list[float], max_look: int = 5) -> int:
        """Count consecutive down closes from the most recent bar backward."""
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
            if count >= max_look:
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

        # Bollinger Band calculation
        sma = self._sma(closes, self.bb_period)
        std = self._std(closes, self.bb_period)
        if sma is None or std is None or std < 1e-9:
            return None

        lower_bb = sma - self.bb_std * std
        z_score = (bar.close - sma) / std

        # ATR for stop/target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # --- Multi-factor scoring ---
        score = 0.0

        # Factor 1: IBS (Internal Bar Strength) — low = oversold
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs < self.ibs_threshold:
            score += 20.0
        if ibs < 0.15:
            score += 10.0  # Extra credit for very low IBS

        # Factor 2: Consecutive down days
        down_days = self._count_down_days(closes)
        if down_days >= self.down_days_min:
            score += 15.0
        if down_days >= 3:
            score += 10.0  # Extra for extended selloff

        # Factor 3: Bollinger z-score (negative = below mean)
        if z_score < -1.0:
            score += 15.0
        if z_score < -2.0:
            score += 10.0  # Deep below lower band

        # Factor 4: Price below lower Bollinger Band
        if bar.close < lower_bb:
            score += 10.0

        # Factor 5: Volume surge (participation)
        volumes = [b.volume for b in bars[-20:]]
        if len(volumes) >= 10:
            avg_vol = sum(volumes) / len(volumes)
            if avg_vol > 0 and bar.volume > avg_vol * (1.0 + self.vol_avg_mult):
                score += 10.0

        # --- Filters (hard gates) ---

        # Drawdown filter: skip if price dropped > drawdown_max from lookback high
        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs) if lookback_highs else bar.close
        drawdown = (peak - bar.close) / peak if peak > 0 else 0
        if drawdown > self.drawdown_max:
            return None

        # Regime filter: avoid SHOCK volatility
        labels = symbol_state.meta.get("regime_labels", {})
        vol_regime = labels.get("regime_vol", "NORMAL")
        if vol_regime == "SHOCK":
            return None

        # Skip near earnings (high-risk event)
        if labels.get("near_earnings", False):
            return None

        # --- Score gate ---
        if score < self.score_threshold:
            return None

        # Target: BB midline (SMA) with ATR multiplier as alternative
        target = max(sma, bar.close + self.target_atr_mult * atr)
        stop = bar.close - self.stop_atr_mult * atr

        # Only enter if target above entry
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
                "score": round(score, 1),
                "ibs": round(ibs, 3),
                "z_score": round(z_score, 2),
                "down_days": down_days,
                "drawdown": round(drawdown, 4),
                "atr": round(atr, 4),
                "vol_regime": vol_regime,
                "seed": "mean_reversion",
            },
        )
