from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class FailedBreakoutStrategy(BaseStrategy):
    """
    Failed Breakout Strategy (Fade).
    Enters when price breaks a key level (Prior High/Low) but fails to hold, reversing back.
    Ideally for CHOP regimes.
    """

    name = "failed_breakout"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.lookback_days = config.get("lookback_days", 1)  # Usually 1 for PDH/PDL
        self.risk_reward = config.get("risk_reward", 2.0)

        # We need to know the Key Levels.
        # Ideally these come from SymbolFeatures (Scanner) or we re-calc.
        # Strategies don't see SymbolFeatures directly in on_bar usually,
        # unless passed or we compute locally.
        # We'll compute locally from bars if enough history, or rely on 'meta' passed in SymbolState.
        # Let's rely on calculating PDH/PDL from historical bars in SymbolState.

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # 1. Identify Key Levels (Prior Day High/Low)
        # We need daily bars. If 'bars' are minute bars, we need to aggregate.
        # For efficiency, we ideally cache this.
        # Let's check if we have them in symbol_state.indicators (populated by some other process?)
        # Or just compute.

        # Optimization: Only compute once per day/session using _get_daily_levels helper
        pdh = symbol_state.indicators.get("prior_day_high")
        pdl = symbol_state.indicators.get("prior_day_low")

        if pdh is None or pdl is None:
            # Try to compute from bars
            levels = self._compute_prior_levels(symbol_state.bars)
            if not levels:
                return None
            pdh, pdl = levels
            # Cache them
            symbol_state.indicators["prior_day_high"] = pdh
            symbol_state.indicators["prior_day_low"] = pdl

        # 2. State Tracking: Did we breach?
        has_breached_high = symbol_state.indicators.get("breached_pdh", False)
        has_breached_low = symbol_state.indicators.get("breached_pdl", False)

        if bar.high > pdh:
            has_breached_high = True
            symbol_state.indicators["breached_pdh"] = True

        if bar.low < pdl:
            has_breached_low = True
            symbol_state.indicators["breached_pdl"] = True

        # 3. Check for Failure (Fade Setup)
        signal = self._check_bearish_fade(
            symbol, bar, symbol_state, market_state, pdh, has_breached_high
        )
        if signal:
            return signal

        signal = self._check_bullish_fade(
            symbol, bar, symbol_state, market_state, pdl, has_breached_low
        )
        return signal

    def _check_bearish_fade(
        self, symbol, bar, symbol_state, market_state, pdh, has_breached_high
    ) -> Optional[Signal]:
        if not (
            has_breached_high and market_state.regime in [Regime.CHOP, Regime.BEAR]
        ):
            return None

        # Trigger: Price closes back inside range ( < PDH )
        if bar.close >= pdh:
            return None

        is_crossing_down = False
        if len(symbol_state.bars) >= 2:
            prev_bar = symbol_state.bars[-2]
            # Check if we were above or rejected intra-bar
            if (prev_bar.close >= pdh or bar.open >= pdh) or (bar.high > pdh):
                is_crossing_down = True

        # If history is small, assume crossing if high was above
        elif bar.high > pdh:
            is_crossing_down = True

        if not is_crossing_down:
            return None

        # ENTRY SHORT
        stop_price = bar.high
        if stop_price <= bar.close:
            stop_price = bar.close * 1.005

        risk = stop_price - bar.close
        target_price = bar.close - (risk * self.risk_reward)

        return Signal(
            symbol=symbol,
            side=OrderSide.SELL,
            size_hint=0,
            entry_price=bar.close,
            stop_price=stop_price,
            target_price=target_price,
            strategy=self.name,
            regime=market_state.regime,
            generated_at=bar.time,
            meta={"pdh": pdh, "type": "fade_high"},
            correlation_id=f"{self.name}-short-{symbol}-{bar.time.timestamp()}",
        )

    def _check_bullish_fade(
        self, symbol, bar, symbol_state, market_state, pdl, has_breached_low
    ) -> Optional[Signal]:
        if not (has_breached_low and market_state.regime in [Regime.CHOP, Regime.BULL]):
            return None

        # Trigger: Price closes back inside range ( > PDL )
        if bar.close <= pdl:
            return None

        is_crossing_up = False
        if len(symbol_state.bars) >= 2:
            prev_bar = symbol_state.bars[-2]
            if (prev_bar.close <= pdl or bar.open <= pdl) or (bar.low < pdl):
                is_crossing_up = True

        elif bar.low < pdl:
            is_crossing_up = True

        if not is_crossing_up:
            return None

        # ENTRY LONG
        stop_price = bar.low
        if stop_price >= bar.close:
            stop_price = bar.close * 0.995

        risk = bar.close - stop_price
        target_price = bar.close + (risk * self.risk_reward)

        return Signal(
            symbol=symbol,
            side=OrderSide.BUY,
            size_hint=0,
            entry_price=bar.close,
            stop_price=stop_price,
            target_price=target_price,
            strategy=self.name,
            regime=market_state.regime,
            generated_at=bar.time,
            meta={"pdl": pdl, "type": "fade_low"},
            correlation_id=f"{self.name}-long-{symbol}-{bar.time.timestamp()}",
        )

    def _compute_prior_levels(self, bars) -> Optional[tuple]:
        # Need to find boundaries of previous day.
        # Assuming bars are datetime sorted.
        if not bars:
            return None

        last_bar = bars[-1]
        current_date = last_bar.time.date()

        # Filter for previous day (not current date)
        # This is a naive heuristic if we only have one contiguous list.
        # We traverse backwards.

        prior_date = None
        for b in reversed(bars):
            if b.time.date() < current_date:
                prior_date = b.time.date()
                break

        if not prior_date:
            return None

        # Collect highs/lows for prior_date
        highs = []
        lows = []
        for b in bars:
            if b.time.date() == prior_date:
                highs.append(b.high)
                lows.append(b.low)

        if not highs:
            return None

        return (max(highs), min(lows))
