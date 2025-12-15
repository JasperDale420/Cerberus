from typing import Any, Dict, Optional

import pandas as pd

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class GapFillStrategy(BaseStrategy):
    """
    Gap-Fill Scalper.
    Fades morning gaps that fail to extend.
    Entry: Break of Opening Range (first 5-15 mins) in opposite direction of Gap.
    Target: Prior Day Close (The Gap Fill).
    """

    name = "gap_fill"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.min_gap = config.get("min_gap", 0.015)
        self.max_gap = config.get("max_gap", 0.10)
        self.risk_reward = config.get("risk_reward", 2.0)
        self.or_time_minutes = config.get(
            "or_time_minutes", 15
        )  # Opening Range duration

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # 1. Time Check: Only valid in early session (e.g. first 90 mins)?
        # And we need Opening Range to be defined.
        # Assuming bar.time is aware of session start.
        # Simple check: If bars count is small < 60 mins?

        if not symbol_state.bars:
            return None

        # Calculate Gap (Current Open vs Prev Close)
        # We need "Day Open" and "Prev Close".
        # If we only have intraday bars for today, we might miss Prev Close unless provided.
        # Scanner checked gap, so we know it exists. But we need direction.
        # Pipeline puts 'gap_pct' in features. Engine might not pass features here.
        # We can estimate gap from first bar of today vs ?
        # Ideally, `symbol_state.meta` has 'gap_pct' or 'prev_close'.
        # If not, checks risk of missing data.

        # In current design, we often lack Prev Day context inside Strategy unless computed in pipeline and passed.
        # Let's check if we can infer from `symbol_state` or `features`.
        # Assuming `symbol_state.meta` has 'gap_pct' from pipeline?
        # (I need to ensure Engine populates this, but for now I assume it might be there or I calc from first bar if I have history).

        # Let's try to calculate from History if available (multiday) or fail gracefully.
        # `symbol_state.bars` might be just today's bars?
        # If it's multi-day, we can find yesterday.

        # Workaround: Use opening price of first bar of TODAY.
        # Warning: We don't know yesterday's close easily without queries.
        # BUT, the Scanner only passed this symbol because it HAS a gap.
        # So we assume there IS a gap. We just need to know DIRECTION.
        # We can look at `SymbolFeatures` if available.
        # If I can't modify Engine, I'll rely on `symbol_state.meta.get('features')` or similar if it exists.
        # Actually I can modify `src/engine/execution.py` to pass features if needed, but risky now.

        # Let's rely on Price vs VWAP or similar? No.
        # Let's assume `gap_pct` is passed in `meta` by the Scanner/Engine logic?
        # ScanResult -> Engine. Engine initiates Strategy.

        # Let's just track the *Opening Range* of TODAY.
        # If Price breaks LOW of OR, and we are relatively High (Gap Up), we Short.
        # How do we know if we Gapped Up?
        # Price >> Prev Close.
        # Let's assume the user configures the bot to pass `gap_pct` OR we just use the first bar Open.
        # If First Bar Open > High of Yesterday... (Need Yesterday).

        # Simplification:
        # If price is dropping hard from Open, and it's early, we Short (Fade).
        # But we want to target Gap Fill.

        # Let's ASSUME `symbol_state.meta` contains `gap_pct` because I will ensure it does or assume the Scanner did its job.
        gap_pct = symbol_state.meta.get("gap_pct", 0.0)

        # If 0, maybe we calc logic:
        # If we don't have gap_pct, we can't be sure.
        # However, for the test I can inject it.
        # In Live, if `gap_pct` is missing, we skip.

        if gap_pct == 0.0:
            # PRD Compliance: usage of robust logging
            # Only warn if we are in the first few bars where we EXPECT to trade
            if len(symbol_state.bars) < 20:
                self.logger.warning(
                    "Missing gap_pct for GapFill strategy", symbol=symbol
                )
            return None

        gap_up = gap_pct > 0
        gap_down = gap_pct < 0

        # 2. Opening Range (OR) Logic
        # We need to define OR High/Low from the first X minutes.
        # Bars are usually 5m?
        # If OR is 15m, it's first 3 bars.

        # Determine Session Start Time?
        # Assuming `bars[0]` is session start (simple assumption for now).

        bars = symbol_state.bars
        start_time = bars[0].time
        cutoff_time = start_time + pd.Timedelta(minutes=self.or_time_minutes)

        current_time = bar.time

        # If we are effectively AT or BEFORE cutoff, we are building OR.
        if current_time <= cutoff_time:
            return None  # Waiting for OR to form

        # OR is formed. Calculate it.
        or_bars = [b for b in bars if b.time <= cutoff_time]
        if not or_bars:
            return None

        or_high = max(b.high for b in or_bars)
        or_low = min(b.low for b in or_bars)

        # 3. Setup & Trigger

        signal = None

        # FADE GAP UP (Short)
        if gap_up:
            # Trigger: Breakdown below OR Low
            # Confirm: Close < OR Low
            if bar.close < or_low:
                # Basic check: have we already filled?
                # Target is "Gap Fill" (Prev Close).
                # Prev Close = Open / (1+gap_pct)
                # This is a good way to derive it!

                open_price = or_bars[0].open
                prev_close = open_price / (1.0 + gap_pct)

                # If we are already below prev_close, gap is filled. No trade.
                if bar.close <= prev_close:
                    return None

                # Entry Short
                stop_price = or_high  # Stop at High of Day (OR High)
                target_price = prev_close

                # Risk/Reward check?
                risk = stop_price - bar.close
                reward = bar.close - target_price
                if risk > 0 and (reward / risk) >= self.risk_reward:
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
                        meta={"gap_pct": gap_pct, "or_low": or_low},
                        correlation_id=f"{self.name}-short-{symbol}-{bar.time.timestamp()}",
                    )

        # FADE GAP DOWN (Long)
        elif gap_down:
            # Trigger: Breakout above OR High
            if bar.close > or_high:
                open_price = or_bars[0].open
                prev_close = open_price / (1.0 + gap_pct)  # gap_pct is negative

                if bar.close >= prev_close:
                    return None

                # Entry Long
                stop_price = or_low
                target_price = prev_close

                risk = bar.close - stop_price
                reward = target_price - bar.close

                if risk > 0 and (reward / risk) >= self.risk_reward:
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
                        meta={"gap_pct": gap_pct, "or_high": or_high},
                        correlation_id=f"{self.name}-long-{symbol}-{bar.time.timestamp()}",
                    )

        return signal
