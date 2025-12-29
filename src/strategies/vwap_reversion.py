from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy
from src.strategies.config_models import VWAPReversionConfig


class VWAPReversionStrategy(BaseStrategy):
    """VWAP Reversion (Mean Reversion off VWAP Bands)."""

    name: str = "vwap_reversion"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        cfg = VWAPReversionConfig(**config)
        self.band_sigma = cfg.effective_band_sigma
        self.risk_reward = cfg.risk_reward
        self.time_window_start = cfg.time_window_start
        self.time_window_end = cfg.time_window_end
        self.max_hold_minutes = cfg.max_hold_minutes
        self.confirmation = cfg.confirmation.lower()
        self.rsi_len = cfg.rsi_len
        self.rsi_oversold = cfg.rsi_oversold
        self.rsi_overbought = cfg.rsi_overbought

    def _in_time_window(self, dt: datetime) -> bool:
        """Check if datetime is within configured trading window."""
        return time_utils.in_time_window_str(
            dt, self.time_window_start, self.time_window_end
        )

    def _rsi(self, closes: np.ndarray, length: int) -> Optional[float]:
        if length <= 0:
            return None
        if closes.size < length + 1:
            return None
        diffs = np.diff(closes.astype(float))
        gains = np.maximum(diffs, 0.0)
        losses = np.maximum(-diffs, 0.0)
        avg_gain = float(np.mean(gains[-length:])) if gains.size >= length else 0.0
        avg_loss = float(np.mean(losses[-length:])) if losses.size >= length else 0.0
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _confirm_reversal(
        self, closes: np.ndarray, side: OrderSide
    ) -> tuple[bool, dict]:
        if self.confirmation == "none":
            return True, {"confirmation": "none"}

        if self.confirmation != "rsi":
            return False, {
                "confirmation": self.confirmation,
                "error": "unsupported_confirmation",
            }

        if closes.size < self.rsi_len + 2:
            return False, {"confirmation": "rsi", "error": "insufficient_history"}

        prev_rsi = self._rsi(closes[:-1], self.rsi_len)
        curr_rsi = self._rsi(closes, self.rsi_len)
        if prev_rsi is None or curr_rsi is None:
            return False, {"confirmation": "rsi", "error": "rsi_unavailable"}

        if side == OrderSide.BUY:
            ok = prev_rsi < self.rsi_oversold and curr_rsi > self.rsi_oversold
        else:
            ok = prev_rsi > self.rsi_overbought and curr_rsi < self.rsi_overbought

        return (
            bool(ok),
            {
                "confirmation": "rsi",
                "rsi_len": self.rsi_len,
                "rsi_prev": float(prev_rsi),
                "rsi_curr": float(curr_rsi),
                "rsi_oversold": float(self.rsi_oversold),
                "rsi_overbought": float(self.rsi_overbought),
            },
        )

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # Only trade in CHOP regime
        if market_state.regime != Regime.CHOP:
            return None
        # 3) Require time window (to avoid gap-up extremes, etc.)
        if not self._in_time_window(bar.time):
            return None

        if not self._require_min_bars(symbol_state, 20):
            return None

        # Prefer intraday/session VWAP injected by the engine (PRD 7.2).
        vwap = getattr(bar, "vwap", None)
        if vwap is None:
            vwap = symbol_state.indicators.get("session_vwap")
        try:
            vwap = float(vwap) if vwap is not None else None
        except Exception:
            vwap = None

        # Fallback: compute VWAP over currently available bars.
        # VWAP = Sum(Typical_Price * Volume) / Sum(Volume)

        bars = list(symbol_state.bars)
        typical_prices = np.array([(b.high + b.low + b.close) / 3.0 for b in bars])
        volumes = np.array([b.volume for b in bars])

        # Avoid division by zero
        total_volume = np.sum(volumes)
        if total_volume == 0:
            return None

        if vwap is None:
            vwap = np.sum(typical_prices * volumes) / total_volume

        # Calculate Std Dev for Bands (Standard Deviation of Close prices)
        # Alternatively, could use Std Dev of (Price - VWAP)
        # Common simplified implementation: VWAP +/- 2 * StdDev(Close)
        closes = np.array([b.close for b in bars])
        std = np.std(closes)

        upper = vwap + self.band_sigma * std
        lower = vwap - self.band_sigma * std

        current_price = bar.close

        # Deterministic time
        now = market_state.time
        if not self._in_time_window(now):
            return None

        if current_price < lower:
            ok, confirm_meta = self._confirm_reversal(closes, OrderSide.BUY)
            if not ok:
                return None
            # Entry Long (Reversion to mean)
            # Stop: A bit below the recent low or a fixed ATR multiple
            # For this slice, using Std Dev based stop
            stop_loss = current_price - (std * 0.5)
            risk = current_price - stop_loss
            take_profit = current_price + (risk * self.risk_reward)

            signal = Signal(
                symbol=symbol,
                side=OrderSide.BUY,
                size_hint=0,
                entry_price=current_price,
                stop_price=stop_loss,
                target_price=take_profit,
                strategy=self.name,
                regime=market_state.regime,
                generated_at=now,
                meta={
                    "reason": "price_below_lower_vwap_band",
                    "vwap": float(vwap),
                    "lower_band": float(lower),
                    "upper_band": float(upper),
                    **confirm_meta,
                },
            )

        elif current_price > upper:
            ok, confirm_meta = self._confirm_reversal(closes, OrderSide.SELL)
            if not ok:
                return None
            # Entry Short (Reversion to mean)
            stop_loss = current_price + (std * 0.5)
            risk = stop_loss - current_price
            take_profit = current_price - (risk * self.risk_reward)

            signal = Signal(
                symbol=symbol,
                side=OrderSide.SELL,
                size_hint=0,
                entry_price=current_price,
                stop_price=stop_loss,
                target_price=take_profit,
                strategy=self.name,
                regime=market_state.regime,
                generated_at=now,
                meta={
                    "reason": "price_above_upper_vwap_band",
                    "vwap": float(vwap),
                    "lower_band": float(lower),
                    "upper_band": float(upper),
                    **confirm_meta,
                },
            )

        return signal
