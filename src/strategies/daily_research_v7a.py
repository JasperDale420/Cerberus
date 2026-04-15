"""daily_research_v7a — Regime-adaptive multi-factor mean reversion.

Multi-factor scoring: z-score + IBS + down days + close rank + volume.
Internal trend detection via SMA(20) vs SMA(50) — no reliance on regime labels.
Regime-adapted entry/exit:
- UP: easier entry, wide target
- FLAT: moderate entry, BB midline target
- DOWN: require strong oversold, tight quick target
Long-only, daily bars.
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
        self.min_bars = int(config.get("min_bars", 55))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        self.trend_fast = int(config.get("trend_fast", 20))
        self.trend_slow = int(config.get("trend_slow", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.15))

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
    def _count_down_days(closes: list[float]) -> int:
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
            if count >= 6:
                break
        return count

    @staticmethod
    def _pct_rank(values: list[float], current: float) -> float:
        if len(values) < 2:
            return 0.5
        below = sum(1 for v in values if v < current)
        return below / len(values)

    def _detect_trend(self, closes: list[float]) -> str:
        sma_f = self._sma(closes, self.trend_fast)
        sma_s = self._sma(closes, self.trend_slow)
        if sma_f is None or sma_s is None:
            return "FLAT"
        spread = (sma_f - sma_s) / sma_s if sma_s > 0 else 0.0
        if spread > 0.01:
            return "UP"
        elif spread < -0.01:
            return "DOWN"
        return "FLAT"

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

        sma = self._sma(closes, self.bb_period)
        std = self._std(closes, self.bb_period)
        if sma is None or std is None or std < 1e-9:
            return None

        z_score = (bar.close - sma) / std
        lower_bb = sma - self.bb_std * std

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        bar_range = bar.high - bar.low
        ibs = (bar.close - bar.low) / bar_range if bar_range > 1e-9 else 0.5

        down_days = self._count_down_days(closes)
        close_rank = self._pct_rank(closes[-40:], bar.close)

        volumes = [b.volume for b in bars[-20:]]
        avg_vol = sum(volumes) / len(volumes) if len(volumes) >= 5 else 1
        vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 1.0

        trend = self._detect_trend(closes)

        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs)
        dd_pct = (peak - bar.close) / peak if peak > 0 else 0
        if dd_pct > self.max_drawdown_pct:
            return None

        score = 0.0
        if z_score < -0.5:
            score += min(25.0, 10.0 * abs(z_score))
        if ibs < self.ibs_threshold:
            score += 15.0 * (1.0 - ibs / max(self.ibs_threshold, 0.01))
        if ibs < 0.15:
            score += 5.0
        if down_days >= 2:
            score += min(15.0, 5.0 * down_days)
        if close_rank < 0.25:
            score += 15.0 * (1.0 - close_rank / 0.25)
        if vol_ratio > 1.0:
            score += min(10.0, 5.0 * (vol_ratio - 1.0))
        if bar.close < lower_bb:
            score += 10.0

        if trend == "UP":
            entry_threshold = 30.0
            stop_mult = self.stop_atr_mult
            target = max(sma, bar.close + self.target_atr_mult * atr)
        elif trend == "FLAT":
            entry_threshold = 35.0
            stop_mult = self.stop_atr_mult
            target = sma if sma > bar.close else bar.close + 1.5 * atr
        else:
            entry_threshold = 45.0
            stop_mult = self.stop_atr_mult * 0.75
            target = bar.close + 1.0 * atr

        if score < entry_threshold:
            return None

        stop = bar.close - stop_mult * atr

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
            },
        )
