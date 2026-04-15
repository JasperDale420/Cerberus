"""daily_research_v7a — Regime-adaptive multi-factor mean reversion.

Trades in ALL market regimes with adapted entry/exit logic:
- UP: buy pullbacks (moderate oversold, wide target)
- FLAT: classic mean reversion (z-score + IBS scoring)
- DOWN: buy extreme oversold bounces (tight target, tight stop)

Multi-factor score: z-score + IBS + down days + volume.
No single-indicator gating — composite scoring only.
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
    def _count_down_days(closes: list[float], max_look: int = 6) -> int:
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
            if count >= max_look:
                break
        return count

    @staticmethod
    def _pct_rank(values: list[float], current: float) -> float:
        """Percentile rank of current value within historical values."""
        if len(values) < 2:
            return 0.5
        below = sum(1 for v in values if v < current)
        return below / len(values)

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

        # --- Core indicators ---
        sma = self._sma(closes, self.bb_period)
        std = self._std(closes, self.bb_period)
        if sma is None or std is None or std < 1e-9:
            return None

        z_score = (bar.close - sma) / std
        lower_bb = sma - self.bb_std * std

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # IBS
        bar_range = bar.high - bar.low
        ibs = (bar.close - bar.low) / bar_range if bar_range > 1e-9 else 0.5

        # Down days
        down_days = self._count_down_days(closes)

        # Close percentile rank over lookback (low = oversold)
        close_rank = self._pct_rank(closes[-40:], bar.close)

        # Volume ratio
        volumes = [b.volume for b in bars[-20:]]
        avg_vol = sum(volumes) / len(volumes) if len(volumes) >= 5 else 1
        vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 1.0

        # --- Regime detection ---
        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "FLAT")
        vol_regime = labels.get("regime_vol", "NORMAL")

        # Hard filters: skip near earnings/FOMC only
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        # --- Multi-factor score (0-100 scale) ---
        score = 0.0

        # Z-score oversold (0-25 pts)
        if z_score < -0.5:
            score += min(25.0, 10.0 * abs(z_score))

        # Low IBS (0-20 pts)
        if ibs < self.ibs_threshold:
            score += 15.0 * (1.0 - ibs / max(self.ibs_threshold, 0.01))
        if ibs < 0.15:
            score += 5.0

        # Down days (0-15 pts)
        if down_days >= 2:
            score += min(15.0, 5.0 * down_days)

        # Close rank oversold (0-15 pts)
        if close_rank < 0.25:
            score += 15.0 * (1.0 - close_rank / 0.25)

        # Volume participation (0-10 pts)
        if vol_ratio > 1.0:
            score += min(10.0, 5.0 * (vol_ratio - 1.0))

        # Below lower BB bonus (0-10 pts)
        if bar.close < lower_bb:
            score += 10.0

        # --- Drawdown guard (avoid catching falling knives) ---
        peak = max(highs[-40:]) if len(highs) >= 40 else max(highs)
        dd_pct = (peak - bar.close) / peak if peak > 0 else 0
        if dd_pct > 0.30:
            return None

        # --- Regime-adaptive thresholds and risk ---
        if trend == "UP":
            entry_threshold = 30.0
            stop_mult = self.stop_atr_mult
            target = max(sma, bar.close + self.target_atr_mult * atr)
        elif trend == "FLAT":
            entry_threshold = 33.0
            stop_mult = self.stop_atr_mult
            target = sma
        else:
            # DOWN: still require strong signal but allow trades
            entry_threshold = 40.0
            stop_mult = self.stop_atr_mult * 0.8
            target = bar.close + 1.2 * atr

        # HIGH/SHOCK vol: tighten stop, raise threshold slightly
        if vol_regime in ("HIGH", "SHOCK"):
            stop_mult *= 0.85
            entry_threshold += 5.0

        if score < entry_threshold:
            return None

        stop = bar.close - stop_mult * atr

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
                "z_score": round(z_score, 2),
                "ibs": round(ibs, 3),
                "down_days": down_days,
                "close_rank": round(close_rank, 3),
                "trend": trend,
                "vol_regime": vol_regime,
            },
        )
