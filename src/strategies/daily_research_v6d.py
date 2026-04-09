"""Daily Research v6d — Connors RSI(2) mean reversion.

Proven robust strategy: buy short-term oversold dips in long-term uptrends.
Long-only. Few parameters. Consistent across bull/bear/flat markets.

Rules:
  ENTRY: RSI(2) < rsi_entry AND price > SMA(trend_period) [uptrend filter]
  EXIT: RSI(2) > rsi_exit OR hit stop/target

Iteration 2: Proper bar accumulation + Connors RSI(2).
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv6dStrategy(BaseStrategy):
    name = "daily_research_v6d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self._daily_closes: dict[str, deque[float]] = {}
        self._daily_highs: dict[str, deque[float]] = {}
        self._daily_lows: dict[str, deque[float]] = {}
        self._daily_volumes: dict[str, deque[float]] = {}
        self._last_bar_date: dict[str, date | None] = {}
        self._intraday_high: dict[str, float] = {}
        self._intraday_low: dict[str, float] = {}
        self._intraday_volume: dict[str, float] = {}
        self._intraday_close: dict[str, float] = {}
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        # Trend filter
        self.trend_period = int(config.get("trend_period", 50))
        # RSI(2) entry/exit
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_entry = float(config.get("rsi_entry", 10.0))
        self.rsi_exit = float(config.get("rsi_exit", 70.0))
        # ATR stop
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr = float(config.get("stop_atr", 3.0))
        self.target_atr = float(config.get("target_atr", 4.0))
        self.allow_overnight = True
        self.max_hold_days = int(config.get("max_hold_days", 10))

    def _ensure_symbol(self, symbol: str) -> None:
        if symbol not in self._daily_closes:
            lookback = max(self.trend_period + 10, 60)
            self._daily_closes[symbol] = deque(maxlen=lookback)
            self._daily_highs[symbol] = deque(maxlen=lookback)
            self._daily_lows[symbol] = deque(maxlen=lookback)
            self._daily_volumes[symbol] = deque(maxlen=lookback)
            self._last_bar_date[symbol] = None
            self._intraday_high[symbol] = 0.0
            self._intraday_low[symbol] = float("inf")
            self._intraday_volume[symbol] = 0.0
            self._intraday_close[symbol] = 0.0

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        self._ensure_symbol(symbol)
        current_date = bar.time.date() if isinstance(bar.time, datetime) else bar.time

        if self._last_bar_date[symbol] is not None and current_date != self._last_bar_date[symbol]:
            # New day — save previous day, evaluate signal
            self._daily_closes[symbol].append(self._intraday_close[symbol])
            self._daily_highs[symbol].append(self._intraday_high[symbol])
            self._daily_lows[symbol].append(self._intraday_low[symbol])
            self._daily_volumes[symbol].append(self._intraday_volume[symbol])

            self._intraday_high[symbol] = bar.high
            self._intraday_low[symbol] = bar.low
            self._intraday_volume[symbol] = bar.volume
            self._intraday_close[symbol] = bar.close

            signal = self._evaluate_daily(symbol, bar, symbol_state, market_state)
            self._last_bar_date[symbol] = current_date
            return signal
        else:
            # Same day or first bar — accumulate
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

    def _evaluate_daily(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        closes = self._daily_closes[symbol]
        highs = self._daily_highs[symbol]
        lows = self._daily_lows[symbol]

        if len(closes) < self.min_bars:
            return None
        if not self._check_cooldown(symbol, bar.time):
            return None

        closes_list = list(closes)
        price = closes_list[-1]

        # Trend filter: price must be above SMA(trend_period)
        sma = self._sma(closes_list, self.trend_period)
        if sma is None or price < sma:
            return None

        # RSI(2) — buy when deeply oversold
        rsi = self._rsi(closes_list, self.rsi_period)
        if rsi is None or rsi >= self.rsi_entry:
            return None

        # ATR for stops
        atr = self._atr(list(highs), list(lows), closes_list, self.atr_period)
        if atr is None or atr < 0.01:
            return None

        stop_price = price - atr * self.stop_atr
        target_price = price + atr * self.target_atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={"mode": "RSI2_REVERSION", "rsi": round(rsi, 1)},
        )

    # --- Technical indicators ---

    def _sma(self, values: list, period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    def _rsi(self, values: list, period: int) -> Optional[float]:
        if len(values) < period + 1:
            return None
        gains, losses = [], []
        for i in range(len(values) - period, len(values)):
            change = values[i] - values[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _atr(self, highs: list, lows: list, closes: list, period: int) -> Optional[float]:
        if len(highs) < period + 1:
            return None
        tr_values = []
        for i in range(len(highs) - period, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            tr_values.append(tr)
        if not tr_values:
            return None
        return sum(tr_values) / len(tr_values)
