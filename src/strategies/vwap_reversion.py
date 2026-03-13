from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.data.calculator import FeatureCalculator
from src.data.requirements import DataRequirements
from src.strategies.base import BaseStrategy
from src.strategies.config_models import VWAPReversionConfig


class VWAPReversionStrategy(BaseStrategy):
    """VWAP Reversion (Mean Reversion off VWAP Bands)."""

    name: str = "vwap_reversion"
    data_requirements = DataRequirements(streams=["bars", "quotes"], on_scan=["flow"])

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)

    def _set_params(self, config: Dict[str, Any]) -> None:
        """Override to parse strategy-specific parameters."""
        super()._set_params(config)
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
        # Mean reversion: only trade when higher TF is flat
        self.tf_alignment_mode = "mean_reversion"

    def _in_time_window(self, dt: datetime) -> bool:
        """Check if datetime is within configured trading window."""
        return time_utils.in_time_window_str(dt, self.time_window_start, self.time_window_end)

    def _confirm_reversal(self, closes: np.ndarray, side: OrderSide) -> tuple[bool, dict]:
        """Check if RSI confirms the reversal signal."""
        if self.confirmation == "none":
            return True, {"confirmation": "none"}

        if self.confirmation != "rsi":
            return False, {
                "confirmation": self.confirmation,
                "error": "unsupported_confirmation",
            }

        if closes.size < self.rsi_len + 2:
            return False, {"confirmation": "rsi", "error": "insufficient_history"}

        # Use centralized calculator
        closes_list = closes.tolist()
        prev_rsi = FeatureCalculator.calculate_rsi(closes_list[:-1], self.rsi_len)
        curr_rsi = FeatureCalculator.calculate_rsi(closes_list, self.rsi_len)
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
        # Legacy standalone behavior: this strategy only runs in CHOP.
        regime_name = str(getattr(market_state.regime, "value", market_state.regime))
        if regime_name != "chop":
            return None

        # P4 fix: Check cooldown before processing
        if not self._check_cooldown(symbol, bar.time):
            return None
        # Regime gating removed - handled by strategy routing at engine level
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

        # P1 fix: Filter to current session bars for std calculation
        current_date = bar.time.date()
        session_bars = [b for b in bars if b.time.date() == current_date]
        if len(session_bars) < 5:
            # Not enough session bars for meaningful std
            session_bars = bars[-20:]  # Fallback to recent 20 bars

        typical_prices = np.array([(b.high + b.low + b.close) / 3.0 for b in bars])
        volumes = np.array([b.volume for b in bars])

        # Avoid division by zero
        total_volume = np.sum(volumes)
        if total_volume == 0:
            return None

        if vwap is None:
            vwap = np.sum(typical_prices * volumes) / total_volume

        # P1 fix: Calculate Std Dev from session bars only
        session_closes = np.array([b.close for b in session_bars])
        std = np.std(session_closes)
        # Use session closes for confirmation as well
        closes = session_closes

        upper = vwap + self.band_sigma * std
        lower = vwap - self.band_sigma * std

        current_price = bar.close
        signal: Optional[Signal] = None

        # Deterministic time
        now = market_state.time
        if not self._in_time_window(now):
            return None

        # === QLIB-INSPIRED CONFIRMATIONS ===

        # 1. RVOL check - low volume extremes often don't revert well
        rvol = symbol_state.meta.get("rvol", 0.0) or 0.0
        bar_volume = bar.volume if hasattr(bar, "volume") else 0
        avg_volume = symbol_state.meta.get("avg_volume_20", 0) or 0
        if avg_volume > 0 and bar_volume > 0:
            current_rvol = bar_volume / avg_volume
        else:
            current_rvol = rvol

        # Skip very low volume conditions (< 0.5x avg) - thin liquidity
        if current_rvol > 0 and current_rvol < 0.5:
            return None

        # 2. Trend alignment - mean reversion works best in flat/choppy markets
        regime_trend = None
        if market_state.regime_snapshot:
            regime_trend = market_state.regime_snapshot.trend.value if market_state.regime_snapshot.trend else None

        # Skip strong trends - mean reversion is risky
        # (Soft filter - still allow but log for analysis)
        in_strong_trend = regime_trend in ("up", "down")

        # Debug: Log periodically when price is within bands (no signal condition)
        if lower <= current_price <= upper:
            if len(symbol_state.bars) % 200 == 0:
                self.logger.debug(
                    "VWAPReversion: Price within bands",
                    symbol=symbol,
                    price=round(current_price, 2),
                    vwap=round(float(vwap), 2),
                    lower=round(float(lower), 2),
                    upper=round(float(upper), 2),
                    band_sigma=self.band_sigma,
                )

        if current_price < lower:
            ok, confirm_meta = self._confirm_reversal(closes, OrderSide.BUY)
            if not ok:
                return None
            # Higher TF alignment (mean reversion: requires flat higher TF)
            if not self._check_higher_tf_alignment(symbol_state, OrderSide.BUY):
                return None
            # Entry Long (Reversion to mean)
            # Stop: A bit below the recent low or a fixed ATR multiple
            # P0.2 fix: Cap stop distance at 2% to prevent excessive risk
            max_stop_pct = 0.02
            base_distance = min(std * 0.5, current_price * max_stop_pct)
            stop_distance = self._apply_regime_volatility_multiplier(base_distance, market_state)
            stop_loss = current_price - stop_distance
            risk = current_price - stop_loss
            take_profit = current_price + (risk * self.risk_reward)

            # P2.6 fix: Use _create_signal for consistent correlation_id
            signal = self._create_signal(
                symbol=symbol,
                side=OrderSide.BUY,
                bar=bar,
                market_state=market_state,
                stop_price=stop_loss,
                target_price=take_profit,
                meta={
                    "reason": "price_below_lower_vwap_band",
                    "vwap": float(vwap),
                    "lower_band": float(lower),
                    "upper_band": float(upper),
                    "rvol": current_rvol,
                    "regime_trend": regime_trend,
                    "in_trend": in_strong_trend,
                    **confirm_meta,
                },
            )

        elif current_price > upper:
            ok, confirm_meta = self._confirm_reversal(closes, OrderSide.SELL)
            if not ok:
                return None
            # Higher TF alignment (mean reversion: requires flat higher TF)
            if not self._check_higher_tf_alignment(symbol_state, OrderSide.SELL):
                return None
            # Entry Short (Reversion to mean)
            # P0.2 fix: Cap stop distance at 2% to prevent excessive risk
            max_stop_pct = 0.02
            base_distance = min(std * 0.5, current_price * max_stop_pct)
            stop_distance = self._apply_regime_volatility_multiplier(base_distance, market_state)
            stop_loss = current_price + stop_distance
            risk = stop_loss - current_price
            take_profit = current_price - (risk * self.risk_reward)

            # P2.6 fix: Use _create_signal for consistent correlation_id
            signal = self._create_signal(
                symbol=symbol,
                side=OrderSide.SELL,
                bar=bar,
                market_state=market_state,
                stop_price=stop_loss,
                target_price=take_profit,
                meta={
                    "reason": "price_above_upper_vwap_band",
                    "vwap": float(vwap),
                    "lower_band": float(lower),
                    "upper_band": float(upper),
                    "rvol": current_rvol,
                    "regime_trend": regime_trend,
                    "in_trend": in_strong_trend,
                    **confirm_meta,
                },
            )

        return signal
