"""daily_research_v7a — Multi-factor scoring aligned with WFO param space.

Composite score from: z-score + IBS + RSI(2) + consecutive down days + volume.
Regime-adaptive threshold. Param names match optuna_harness search space.
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
        # Locked by harness
        self.rsi_period = int(config.get("rsi_period", 2))
        self.trend_period = int(config.get("trend_period", 50))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.10))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.02))
        # Optimized by harness
        self.rsi_entry = float(config.get("rsi_entry", 25.0))
        self.ibs_entry = float(config.get("ibs_entry", 0.35))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.vol_mult = float(config.get("vol_mult", 0.5))
        # Internal
        self.min_bars = int(config.get("min_bars", 55))
        self.bb_period = int(config.get("bb_period", 20))
        self.atr_period = int(config.get("atr_period", 14))

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
    def _rsi(closes: list[float], period: int) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        gains = []
        losses = []
        for i in range(-period, 0):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss < 1e-9:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

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
            if count >= 5:
                break
        return count

    def _detect_trend(self, closes: list[float]) -> str:
        sma_f = self._sma(closes, 20)
        sma_s = self._sma(closes, self.trend_period)
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

        # BB z-score
        sma = self._sma(closes, self.bb_period)
        std = self._std(closes, self.bb_period)
        if sma is None or std is None or std < 1e-9:
            return None
        zscore = (bar.close - sma) / std

        # ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # RSI(2)
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None:
            return None

        # IBS
        bar_range = bar.high - bar.low
        ibs = (bar.close - bar.low) / bar_range if bar_range > 1e-9 else 0.5

        # Volume
        volumes = [b.volume for b in bars[-20:]]
        avg_vol = sum(volumes) / len(volumes) if len(volumes) >= 5 else 1
        vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 1.0

        # Drawdown filter (locked param)
        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs)
        drawdown = (peak - bar.close) / peak if peak > 0 else 0
        if drawdown > self.max_drawdown_pct:
            return None

        # Regime filters
        labels = symbol_state.meta.get("regime_labels", {})
        vol_regime = labels.get("regime_vol", "NORMAL")
        if vol_regime == "SHOCK":
            return None
        if labels.get("near_earnings", False):
            return None

        # Internal trend detection
        trend = self._detect_trend(closes)
        down_days = self._count_down_days(closes)

        # --- Multi-factor composite score ---
        score = 0.0

        # Factor 1: RSI(2) oversold (0-25 pts) — uses rsi_entry from harness
        if rsi < self.rsi_entry:
            score += 15.0 + 10.0 * (1.0 - rsi / max(self.rsi_entry, 1.0))

        # Factor 2: Z-score oversold (0-25 pts)
        if zscore < -1.0:
            score += min(25.0, 10.0 * abs(zscore))
        elif zscore < -0.5:
            score += 5.0

        # Factor 3: IBS near low (0-20 pts) — uses ibs_entry from harness
        if ibs < self.ibs_entry:
            score += 20.0 * (1.0 - ibs / max(self.ibs_entry, 0.01))

        # Factor 4: Consecutive down days (0-15 pts)
        if down_days >= 2:
            score += min(15.0, 5.0 * down_days)

        # Factor 5: Volume participation (0-10 pts) — uses vol_mult from harness
        if vol_ratio > self.vol_mult:
            score += min(10.0, 5.0 * (vol_ratio - self.vol_mult))

        # Factor 6: Close rank in 40-day range (0-10 pts)
        if len(closes) >= 40:
            below = sum(1 for c in closes[-40:] if c < bar.close)
            rank = below / 40.0
            if rank < 0.25:
                score += 10.0 * (1.0 - rank / 0.25)

        # Regime-adaptive threshold
        if trend == "UP":
            threshold = 30.0
        elif trend == "FLAT":
            threshold = 35.0
        else:  # DOWN
            threshold = 45.0

        if score < threshold:
            return None

        # Target and stop — use harness-tuned params
        if trend == "DOWN":
            target = bar.close + 1.0 * atr
            stop_mult = self.stop_atr_mult * 0.75
        elif trend == "FLAT":
            target = sma if sma > bar.close else bar.close + self.target_atr_mult * atr
            stop_mult = self.stop_atr_mult
        else:  # UP
            target = max(sma, bar.close + self.target_atr_mult * atr)
            stop_mult = self.stop_atr_mult

        if target <= bar.close:
            return None

        stop = bar.close - stop_mult * atr
        # Cap stop at max_stop_pct
        max_stop = bar.close * (1.0 - self.max_stop_pct)
        if stop < max_stop:
            stop = max_stop

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
                "rsi2": round(rsi, 1),
                "zscore": round(zscore, 2),
                "ibs": round(ibs, 3),
                "down_days": down_days,
                "trend": trend,
            },
        )
