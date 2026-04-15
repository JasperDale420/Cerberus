"""IBS + RSI(2) Mean Reversion with Loss Capping.

Core signal: IBS < threshold AND RSI(2) < oversold AND 2+ consecutive down days.
This triple confirmation filters out weak setups.

Loss control: max_stop_pct caps per-trade loss regardless of ATR.
Short hold (max 3 days) for consistency across all windows.

Regime adaptation:
  DOWN — tighter target and stop (take profits fast, cut losses fast)
  UP   — slightly wider target (let winners run)
  FLAT — balanced defaults

Long-only. Skip SHOCK vol, earnings, FOMC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv7dStrategy(BaseStrategy):
    name = "daily_research_v7d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 30))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.2))
        self.rsi_period = 2  # Fixed — RSI(2) is the classic short-term reversal indicator
        self.rsi_oversold = float(config.get("rsi_oversold", 15.0))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_hold_days = 3  # Fixed — short hold for consistency
        self.max_stop_pct = float(config.get("max_stop_pct", 0.03))  # 3% max loss per trade
        # Regime-adaptive multipliers
        self.up_target_scale = float(config.get("up_target_scale", 1.3))
        self.down_target_scale = float(config.get("down_target_scale", 0.7))
        self.down_stop_scale = float(config.get("down_stop_scale", 0.7))

    # --- Indicator helpers ---

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
    def _ibs(bar: Bar) -> Optional[float]:
        """Internal Bar Strength: (close - low) / (high - low)."""
        rng = bar.high - bar.low
        if rng < 1e-9:
            return None
        return (bar.close - bar.low) / rng

    @staticmethod
    def _rsi(closes: list[float], period: int) -> Optional[float]:
        """Simple RSI calculation."""
        if len(closes) < period + 1:
            return None
        gains = []
        losses = []
        for i in range(-period, 0):
            change = closes[i] - closes[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss < 1e-9:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

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

        if len(closes) < self.min_bars:
            return None

        # Core indicator: ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Min price filter
        if bar.close < 5.0:
            return None

        # ATR/price filter: skip dead stocks
        if atr / bar.close < 0.005:
            return None

        # --- Triple confirmation ---

        # 1. IBS signal (closed near day's low)
        ibs = self._ibs(bar)
        if ibs is None or ibs >= self.ibs_threshold:
            return None

        # 2. RSI(2) oversold
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi >= self.rsi_oversold:
            return None

        # 3. Consecutive down days (hardcoded 2 for stability)
        consec = self._count_consecutive_down(closes)
        if consec < 2:
            return None

        # Trend context for regime-adaptive sizing
        regime_trend = labels.get("regime_trend", "FLAT").upper()

        # Regime-adaptive stop/target
        stop_mult = self.stop_atr_mult
        target_mult = self.target_atr_mult

        if regime_trend == "UP":
            target_mult *= self.up_target_scale
        elif regime_trend == "DOWN":
            target_mult *= self.down_target_scale
            stop_mult *= self.down_stop_scale

        stop_atr = bar.close - stop_mult * atr
        target = bar.close + target_mult * atr

        # Cap the stop loss at max_stop_pct of entry price
        max_stop = bar.close * (1.0 - self.max_stop_pct)
        stop = max(stop_atr, max_stop)

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "ibs": round(ibs, 3),
                "rsi2": round(rsi, 1),
                "consec_down": consec,
                "regime": regime_trend,
                "atr": round(atr, 4),
                "stop_mult": round(stop_mult, 3),
                "target_mult": round(target_mult, 3),
                "seed": "ibs_rsi2_mean_reversion",
            },
        )
