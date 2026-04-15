"""Daily Research v7a — Multi-Factor Mean Reversion.

Combines IBS (Internal Bar Strength) + Bollinger Band z-score + volume confirmation.
Long-only, daily bars. Avoids earnings/FOMC. Drawdown filter for falling knives.
ATR-based stops, BB midline target.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v7a"
    allow_overnight: bool = True

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)
        self.allow_overnight = True
        self._daily_closes: dict[str, deque[float]] = {}
        self._daily_highs: dict[str, deque[float]] = {}
        self._daily_lows: dict[str, deque[float]] = {}
        self._daily_volumes: dict[str, deque[float]] = {}
        self._last_bar_date: dict[str, date | None] = {}
        self._intraday_high: dict[str, float] = {}
        self._intraday_low: dict[str, float] = {}
        self._intraday_volume: dict[str, float] = {}
        self._intraday_close: dict[str, float] = {}

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        # Bollinger Bands
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        # IBS threshold (buy when IBS < threshold = closed near low)
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        # BB z-score entry threshold (buy when z < -threshold)
        self.zscore_entry = float(config.get("zscore_entry", 1.5))
        # Volume filter
        self.vol_avg_mult = float(config.get("vol_avg_mult", 0.5))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.drawdown_max = float(config.get("drawdown_max", 0.15))
        # ATR for stops
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        # Hold
        self.max_hold_days = int(config.get("max_hold_days", 5))

    def _ensure_symbol(self, symbol: str) -> None:
        if symbol not in self._daily_closes:
            lookback = max(self.bb_period + 10, self.drawdown_lookback + 5, 60)
            self._daily_closes[symbol] = deque(maxlen=lookback)
            self._daily_highs[symbol] = deque(maxlen=lookback)
            self._daily_lows[symbol] = deque(maxlen=lookback)
            self._daily_volumes[symbol] = deque(maxlen=lookback)
            self._last_bar_date[symbol] = None
            self._intraday_high[symbol] = 0.0
            self._intraday_low[symbol] = float("inf")
            self._intraday_volume[symbol] = 0.0
            self._intraday_close[symbol] = 0.0

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> float | None:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _std(values: list[float], period: int) -> float | None:
        if len(values) < period:
            return None
        subset = values[-period:]
        mean = sum(subset) / period
        variance = sum((v - mean) ** 2 for v in subset) / period
        return variance**0.5

    @staticmethod
    def _atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float | None:
        if len(highs) < period + 1:
            return None
        trs = []
        for i in range(-period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        return sum(trs) / period

    # --- Bar processing (daily accumulation pattern) ---

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        self._ensure_symbol(symbol)
        current_date = bar.time.date() if isinstance(bar.time, datetime) else bar.time

        if self._last_bar_date[symbol] is not None and current_date != self._last_bar_date[symbol]:
            # New day — save previous day
            self._daily_closes[symbol].append(self._intraday_close[symbol])
            self._daily_highs[symbol].append(self._intraday_high[symbol])
            self._daily_lows[symbol].append(self._intraday_low[symbol])
            self._daily_volumes[symbol].append(self._intraday_volume[symbol])

            # Reset for new day
            self._intraday_high[symbol] = bar.high
            self._intraday_low[symbol] = bar.low
            self._intraday_volume[symbol] = bar.volume
            self._intraday_close[symbol] = bar.close

            # Evaluate signal
            signal = self._evaluate(symbol, bar, symbol_state, market_state)
            self._last_bar_date[symbol] = current_date
            return signal
        else:
            # Same day — accumulate
            if self._last_bar_date[symbol] is None:
                self._intraday_high[symbol] = bar.high
                self._intraday_low[symbol] = bar.low
                self._intraday_volume[symbol] = bar.volume
            else:
                self._intraday_high[symbol] = max(self._intraday_high[symbol], bar.high)
                self._intraday_low[symbol] = min(self._intraday_low[symbol], bar.low)
                self._intraday_volume[symbol] += bar.volume
            self._intraday_close[symbol] = bar.close
            self._last_bar_date[symbol] = current_date
            return None

    def _evaluate(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        closes = list(self._daily_closes[symbol])
        highs = list(self._daily_highs[symbol])
        lows = list(self._daily_lows[symbol])
        volumes = list(self._daily_volumes[symbol])

        if len(closes) < self.min_bars:
            return None
        if not self._check_cooldown(symbol, bar.time):
            return None

        # Skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot is not None and snapshot.vol == VolRegime.SHOCK:
            return None

        # Skip earnings/FOMC events
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        price = closes[-1]

        # --- IBS (Internal Bar Strength) on last completed day ---
        day_high = highs[-1]
        day_low = lows[-1]
        day_range = day_high - day_low
        if day_range < 1e-9:
            return None
        ibs = (closes[-1] - day_low) / day_range

        # Must close near day low
        if ibs >= self.ibs_threshold:
            return None

        # --- Bollinger Band z-score ---
        bb_mean = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_mean is None or bb_std is None or bb_std < 0.01:
            return None

        z_score = (price - bb_mean) / bb_std
        # Must be below -zscore_entry (oversold)
        if z_score > -self.zscore_entry:
            return None

        # --- Volume confirmation ---
        avg_vol = sum(volumes[-20:]) / min(20, len(volumes[-20:])) if len(volumes) >= 5 else 0
        current_vol = volumes[-1] if volumes else 0
        if avg_vol > 0 and current_vol < avg_vol * self.vol_avg_mult:
            return None

        # --- Drawdown filter (avoid catching falling knives) ---
        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs)
        if peak > 0 and (peak - price) / peak > self.drawdown_max:
            return None

        # --- ATR for stop ---
        atr = self._atr(highs, lows, closes, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop_price = price - self.stop_atr_mult * atr
        target_price = bb_mean  # BB midline

        # Only enter if target above entry
        if target_price <= price:
            return None

        self.last_signal_time[symbol] = bar.time
        return Signal(
            symbol=symbol,
            side=OrderSide.BUY,
            size_hint=0.0,
            entry_price=price,
            stop_price=stop_price,
            target_price=target_price,
            strategy=self.name,
            generated_at=bar.time,
            meta={
                "ibs": round(ibs, 3),
                "z_score": round(z_score, 2),
                "atr": round(atr, 4),
                "dd_from_peak": round((peak - price) / peak, 4),
            },
        )
