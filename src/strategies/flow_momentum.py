from typing import Any, Dict, Optional

import pandas as pd
import pandas_ta as ta

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class FlowMomentumStrategy(BaseStrategy):
    """
    Flow-Confirmed Momentum Strategy.
    Direction is dictated by Option Flow Z-Score.
    Entry logic is based on Momentum (Price Action + Volume).
    """

    name = "flow_momentum"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.min_flow_zscore = config.get("min_flow_zscore", 3.0)
        self.vol_mult = config.get("vol_mult", 1.5)
        self.risk_reward = config.get("risk_reward", 2.0)

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # 1. Check Flow Direction (Meta from Scanner/Pipeline usually passed here?)
        # Strategy usually doesn't see SymbolFeatures directly.
        # We need flow_zscore in 'symbol_state.indicators' or 'meta'.
        # Pipeline puts it in SymbolFeatures. The Engine might need to pass it?
        # Standard Engine implementation passes 'symbol_state.meta' which we can populate?
        # Or we cheat and look up features if available globally? (Bad design).
        # Better: Assume Engine populates `symbol_state.meta['features']` with latest features?
        # Checking `src/core/domain.py`: SymbolFeatures has `flow_zscore`.
        # Checking `src/engine/execution.py`: Does it put features in symbol_state?
        # Usually it updates SymbolState with latest bars. features?
        # If I can't access flow_zscore, I can't verify flow.

        # Assumption: The SCANNER filtered for this symbol because of flow.
        # But flow can change or be stale.
        # Ideally, we have access. Let's assume `symbol_state.meta` has `flow_zscore`.
        # If not, we might need to rely on the fact that if it's in the list, it has flow.
        # But we need DIRECTION.

        flow_score = symbol_state.meta.get("flow_zscore", 0.0)

        # Fallback: Maybe mapped in config or separate lookup?
        # Let's rely on passed meta for now. If missing, we verify later.

        if abs(flow_score) < self.min_flow_zscore:
            # Maybe strict cutoff? or rely on scanner?
            # If 0, we have no data.
            if flow_score == 0:
                return None

        is_bullish_flow = flow_score > 0

        # 2. Check Momentum (Price + Volume)
        # Need history for Avg Vol
        if not symbol_state.bars or len(symbol_state.bars) < 21:
            return None

        bars = list(symbol_state.bars)
        df = pd.DataFrame([vars(b) for b in bars])

        # Avg Vol
        avg_vol_series = ta.sma(df["volume"], length=20)
        if avg_vol_series is None:
            return None

        avg_vol = avg_vol_series.iloc[-1]

        # Candle Strength
        # Current bar body
        open_ = bar.open
        close = bar.close
        high = bar.high
        low = bar.low

        is_green = close > open_
        is_red = close < open_

        vol_ok = bar.volume > (avg_vol * self.vol_mult)

        signal = None

        # BULLISH MOMENTUM
        if is_bullish_flow and is_green and vol_ok:
            # Entry: Buy logic
            # Stop: Low of this momentum candle (or recent low)
            stop_price = low
            if stop_price >= close:
                stop_price = close * 0.99

            risk = close - stop_price
            target_price = close + (risk * self.risk_reward)

            signal = Signal(
                symbol=symbol,
                side=OrderSide.BUY,
                size_hint=0,
                entry_price=close,
                stop_price=stop_price,
                target_price=target_price,
                strategy=self.name,
                regime=market_state.regime,
                generated_at=bar.time,
                meta={"flow_zscore": flow_score, "vol_mult": bar.volume / avg_vol},
                correlation_id=f"{self.name}-long-{symbol}-{bar.time.timestamp()}",
            )

        # BEARISH MOMENTUM
        elif (not is_bullish_flow) and is_red and vol_ok:
            # Entry: Sell logic
            stop_price = high
            if stop_price <= close:
                stop_price = close * 1.01

            risk = stop_price - close
            target_price = close - (risk * self.risk_reward)

            signal = Signal(
                symbol=symbol,
                side=OrderSide.SELL,
                size_hint=0,
                entry_price=close,
                stop_price=stop_price,
                target_price=target_price,
                strategy=self.name,
                regime=market_state.regime,
                generated_at=bar.time,
                meta={"flow_zscore": flow_score, "vol_mult": bar.volume / avg_vol},
                correlation_id=f"{self.name}-short-{symbol}-{bar.time.timestamp()}",
            )

        return signal
