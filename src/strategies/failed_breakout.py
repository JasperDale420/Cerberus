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
        # We need to track strictly "Did we go ABOVE PDH today?"
        # symbol_state.indicators is good for this.

        has_breached_high = symbol_state.indicators.get("breached_pdh", False)
        has_breached_low = symbol_state.indicators.get("breached_pdl", False)

        # Check current bar for new breaches
        if bar.high > pdh:
            has_breached_high = True
            symbol_state.indicators["breached_pdh"] = True

        if bar.low < pdl:
            has_breached_low = True
            symbol_state.indicators["breached_pdl"] = True

        # 3. Check for Failure (Fade Setup)

        # Bearish Fade: We breached High, but now we are Closing BELOW High (or some confirmation)
        # To avoid noise, we might want "Breached by X amount" or "Time spent above".
        # Simple Logic: If breached previously, and NOW Close < PDH, and Regime is CHOP/BEAR.

        # We need to ensure we don't just spam signals. We need a "Setup -> Trigger" flow.
        # Setup: Breached High.
        # Trigger: Close crosses below High.

        # We can verify the "Cross Below" by looking at previous bar close vs this bar close?
        # Or just "Is currently below" + "Was detecting breach".
        # Risk: If it chops around PDH, we get many signals.
        # Fix: Check if we have an open position? (Handled by engine usually)
        # Fix: Check if specific 'trigger' happened THIS BAR (e.g. Open > PDH, Close < PDH).

        signal = None

        # BEARISH FADE (Failed Breakout High)
        if has_breached_high and market_state.regime in [Regime.CHOP, Regime.BEAR]:
            # Trigger: Price closes back inside range ( < PDH )
            if bar.close < pdh:
                # Was previous bar above? Or high of this bar above?
                # Stronger signal: This bar made a new high (or tested high) and closed weak.

                # Check if we just crossed down
                # If we were already below, maybe we shouldn't enter late.
                # Let's require strictly: Close < PDH < High (shooting star / reversal candle logic)
                # OR prev_close > PDH and curr_close < PDH.

                is_crossing_down = False
                if len(symbol_state.bars) >= 2:
                    prev_bar = symbol_state.bars[-2]
                    if prev_bar.close >= pdh or bar.open >= pdh:
                        is_crossing_down = True
                    elif bar.high > pdh:  # Intra-bar rejection
                        is_crossing_down = True

                if is_crossing_down:
                    # ENTRY SHORT
                    # Stop: Day's High (which should be > PDH)
                    # ENTRY SHORT
                    # Stop: Day's High (which should be > PDH)
                    # days_high intent removed as unused
                    # Check global high tracking roughly? Or just this bar high if it's the pivot?
                    # Safer: Use this bar's high or a recent swing high.
                    stop_price = bar.high
                    if stop_price <= bar.close:
                        stop_price = bar.close * 1.005

                    risk = stop_price - bar.close
                    target_price = bar.close - (risk * self.risk_reward)

                    signal = Signal(
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

        # BULLISH FADE (Failed Breakout Low)
        if has_breached_low and market_state.regime in [Regime.CHOP, Regime.BULL]:
            # Trigger: Price closes back inside range ( > PDL )
            if bar.close > pdl:
                is_crossing_up = False
                if len(symbol_state.bars) >= 2:
                    prev_bar = symbol_state.bars[-2]
                    if prev_bar.close <= pdl or bar.open <= pdl:
                        is_crossing_up = True
                    elif bar.low < pdl:
                        is_crossing_up = True

                if is_crossing_up:
                    # ENTRY LONG
                    stop_price = bar.low
                    if stop_price >= bar.close:
                        stop_price = bar.close * 0.995

                    risk = bar.close - stop_price
                    target_price = bar.close + (risk * self.risk_reward)

                    signal = Signal(
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

        return signal

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
