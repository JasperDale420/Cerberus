"""daily_research_v7a — Multi-Factor Mean Reversion with Trend Filter.

IBS (Internal Bar Strength) + RSI(2) + Bollinger Band with trend context.
Long-only, daily bars. Regime-aware via SMA trend filter.
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
        self.min_bars = int(config.get("min_bars", 25))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_entry = float(config.get("rsi_entry", 30.0))
        self.ibs_entry = float(config.get("ibs_entry", 0.4))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.trend_period = int(config.get("trend_period", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.10))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.02))
        self.vol_mult = float(config.get("vol_mult", 0.6))
        self.momentum_lookback = int(config.get("momentum_lookback", 5))

    # --- Indicator helpers ---

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

        # --- Trend filter: SMA(trend_period) ---
        # Allow entry when price is above SMA (uptrend) or within 3% below (mild pullback)
        trend_sma = self._sma(closes, self.trend_period)
        if trend_sma is None:
            return None
        trend_pct = (bar.close - trend_sma) / trend_sma
        # Skip deep downtrends (more than 5% below trend SMA)
        if trend_pct < -0.05:
            return None

        # --- RSI(2) filter ---
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi >= self.rsi_entry:
            return None

        # --- IBS (Internal Bar Strength): must be low ---
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_entry:
            return None

        # --- Volume filter: require minimum participation ---
        if len(volumes) >= 20:
            avg_vol = sum(volumes[-20:]) / 20
            if avg_vol > 0 and bar.volume < avg_vol * self.vol_mult:
                return None

        # --- Drawdown filter: skip if price dropped too far from lookback high ---
        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.max_drawdown_pct:
            return None

        # --- Bollinger Band for target ---
        bb_sma = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_sma is None or bb_std is None or bb_std < 1e-9:
            return None
        lower_bb = bb_sma - self.bb_std * bb_std

        # --- ATR for stop ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Stop: ATR-based, capped at max_stop_pct
        raw_stop = self.stop_atr_mult * atr
        max_stop = bar.close * self.max_stop_pct
        stop_dist = min(raw_stop, max_stop) if max_stop > 0 else raw_stop
        stop = bar.close - stop_dist

        # Target: ATR-based
        target = bar.close + self.target_atr_mult * atr

        # Only enter if target is above entry (positive expectancy)
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
                "rsi2": round(rsi, 2),
                "ibs": round(ibs, 3),
                "lower_bb": round(lower_bb, 2),
                "atr": round(atr, 4),
                "trend_pct": round(trend_pct, 4),
                "seed": "mean_reversion",
            },
        )
