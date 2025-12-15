from typing import Any, Dict, Optional

import numpy as np

from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class VWAPReversionStrategy(BaseStrategy):
    name: str = "vwap_reversion"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.band_sigma = config.get("band_sigma", 2.0)
        self.risk_reward = config.get("risk_reward", 2.0)

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

        # Need enough bars
        if not symbol_state.bars or len(symbol_state.bars) < 20:
            return None

        # Calculate Cumulative VWAP over the visible window (or use bar.vwap if available/accumulated)
        # For this implementation, we calculate VWAP over the loaded deque of bars
        # VWAP = Sum(Typical_Price * Volume) / Sum(Volume)

        bars = list(symbol_state.bars)
        typical_prices = np.array([(b.high + b.low + b.close) / 3.0 for b in bars])
        volumes = np.array([b.volume for b in bars])

        # Avoid division by zero
        total_volume = np.sum(volumes)
        if total_volume == 0:
            return None

        vwap = np.sum(typical_prices * volumes) / total_volume

        # Calculate Std Dev for Bands (Standard Deviation of Close prices)
        # Alternatively, could use Std Dev of (Price - VWAP)
        # Common simplified implementation: VWAP +/- 2 * StdDev(Close)
        closes = np.array([b.close for b in bars])
        std = np.std(closes)

        upper = vwap + self.band_sigma * std
        lower = vwap - self.band_sigma * std

        current_price = bar.close

        signal = None
        signal = None
        # FIX: deterministic time
        now = market_state.time

        if current_price < lower:
            # Entry Long (Reversion to mean)
            # Stop: A bit below the recent low or a fixed ATR multiple
            # For this slice, using Std Dev based stop
            stop_loss = current_price - (std * 0.5)
            risk = current_price - stop_loss
            take_profit = current_price + (risk * self.risk_reward)

            signal = Signal(
                symbol=symbol,
                side=OrderSide.BUY,
                size_hint=1.0,
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
                },
            )

        elif current_price > upper:
            # Entry Short (Reversion to mean)
            stop_loss = current_price + (std * 0.5)
            risk = stop_loss - current_price
            take_profit = current_price - (risk * self.risk_reward)

            signal = Signal(
                symbol=symbol,
                side=OrderSide.SELL,
                size_hint=1.0,
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
                },
            )

        return signal
