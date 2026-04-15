"""Connors RSI(2) Mean Reversion v2 — tighter for consistency.

Entry: 2+ consecutive down closes AND RSI(2) < threshold.
In HIGH vol: require 3+ down days and use tighter stops.
Regime-neutral (works in UP, DOWN, FLAT).
Skips SHOCK vol, earnings, FOMC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv7bStrategy(BaseStrategy):
    name = "daily_research_v7b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.consec_down_days = int(config.get("consec_down_days", 2))
        self.consec_down_days_high_vol = int(config.get("consec_down_days_high_vol", 3))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_max = float(config.get("rsi_max", 10.0))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 3))
        self.sma_period = int(config.get("sma_period", 50))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.7))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

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

    def _count_consecutive_down(self, closes: list[float]) -> int:
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

        # Skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Skip earnings and FOMC
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        if len(closes) < self.min_bars:
            return None

        # Volume filter: need decent participation
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is not None and avg_vol > 0 and bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # Determine if we're in HIGH vol
        is_high_vol = snapshot and snapshot.vol == VolRegime.HIGH

        # Core signal: consecutive down days
        consec = self._count_consecutive_down(closes)
        required_down = self.consec_down_days_high_vol if is_high_vol else self.consec_down_days
        if consec < required_down:
            return None

        # RSI(2) oversold confirmation
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi > self.rsi_max:
            return None

        # ATR for stop and target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Tighter stop in HIGH vol
        stop_mult = self.stop_atr_mult * 0.8 if is_high_vol else self.stop_atr_mult
        stop = bar.close - stop_mult * atr
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
                "consec_down": consec,
                "rsi2": round(rsi, 2),
                "high_vol": is_high_vol,
                "atr": round(atr, 4),
                "seed": "connors_rsi2_mr_v2",
            },
        )
