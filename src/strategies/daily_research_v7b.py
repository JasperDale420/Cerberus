"""Connors RSI consecutive-down mean reversion — buy oversold bounces in uptrends.

Entry: 2+ consecutive down closes AND RSI < threshold AND price > SMA (uptrend).
Uses ATR-based vol filter and IBS confirmation.
Skips earnings, FOMC, and high-volatility stocks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
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
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_max = float(config.get("rsi_max", 10.0))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 3))
        self.sma_period = int(config.get("sma_period", 50))
        self.ibs_entry_threshold = float(config.get("ibs_entry_threshold", 0.3))
        self.mr_ibs_threshold = float(config.get("mr_ibs_threshold", 0.2))
        self.max_realized_vol = float(config.get("max_realized_vol", 0.25))
        self.max_atr_pct = float(config.get("max_atr_pct", 0.03))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.06))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 30))
        self.pullback_atr_min = float(config.get("pullback_atr_min", 0.3))
        self.sma_fast_period = int(config.get("sma_fast_period", 20))

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
        """Count consecutive down closes ending at the most recent bar."""
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

        # Skip earnings and FOMC via labels (if available)
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        if len(closes) < self.min_bars:
            return None

        # Dual trend filter: price above SMA(20) AND SMA(50)
        sma = self._sma(closes, self.sma_period)
        if sma is None:
            return None
        if bar.close < sma:
            return None
        sma_fast = self._sma(closes, self.sma_fast_period)
        if sma_fast is None:
            return None
        if bar.close < sma_fast:
            return None
        # Fast SMA must be above slow SMA (trend alignment)
        if sma_fast < sma:
            return None

        # Max drawdown filter: skip if stock has dropped too much from 30-day high
        lookback = min(self.drawdown_lookback, len(closes))
        recent_high = max(closes[-lookback:])
        drawdown = (recent_high - bar.close) / recent_high
        if drawdown > self.max_drawdown_pct:
            return None

        # ATR for volatility filter and stops
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Pullback must leave price well above SMA (not rolling over)
        dist_from_sma = (bar.close - sma) / atr
        if dist_from_sma < self.pullback_atr_min:
            return None

        # Vol filter: skip excessively volatile stocks (ATR/price too high)
        atr_pct = atr / bar.close
        if atr_pct > self.max_atr_pct:
            return None

        # Core signal: consecutive down days
        consec = self._count_consecutive_down(closes)
        if consec < self.consec_down_days:
            return None

        # RSI oversold confirmation
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi > self.rsi_max:
            return None

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
                "consec_down": consec,
                "rsi": round(rsi, 2),
                "atr_pct": round(atr_pct, 4),
                "atr": round(atr, 4),
                "seed": "connors_rsi2_mr",
            },
        )
