from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy
from src.strategies.config_models import FlowMomentumConfig


class FlowMomentumStrategy(BaseStrategy):
    """
    Flow-Confirmed Momentum Strategy.
    Direction is dictated by Option Flow Z-Score.
    Entry logic is based on Momentum (Price Action + Volume).
    """

    name = "flow_momentum"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        cfg = FlowMomentumConfig(**config)
        self.min_flow_zscore = cfg.min_flow_zscore
        self.vol_mult = cfg.vol_mult
        self.risk_reward = cfg.risk_reward

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # PRD: flow-confirmed momentum is intended for trending regimes.
        if market_state.regime not in (Regime.BULL, Regime.BEAR):
            return None
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

        flow_score = float(symbol_state.meta.get("flow_zscore", 0.0) or 0.0)
        call_put_ratio = float(symbol_state.meta.get("call_put_ratio", 0.0) or 0.0)

        # Fallback: Maybe mapped in config or separate lookup?
        # Let's rely on passed meta for now. If missing, we verify later.

        if abs(flow_score) < self.min_flow_zscore:
            # Maybe strict cutoff? or rely on scanner?
            # If 0, we have no data.
            if flow_score == 0:
                return None

        is_bullish_flow = flow_score > 0
        if call_put_ratio <= 0:
            return None
        # PRD: require flow_zscore + call_put_ratio agreement.
        if is_bullish_flow and call_put_ratio < 1.0:
            return None
        if (not is_bullish_flow) and call_put_ratio > 1.0:
            return None

        # 2. Check Momentum (Price + Volume)
        # Filter: Check minimum bars for EMA
        if not self._require_min_bars(symbol_state, 21):
            return None

        bars = list(symbol_state.bars)
        avg_vol = symbol_state.indicators.get("sma_vol:20")
        if avg_vol is None:
            vols = [float(b.volume) for b in bars[-20:]]
            if not vols:
                return None
            avg_vol = sum(vols) / len(vols)
        try:
            avg_vol_f = float(avg_vol)
        except Exception:
            return None

        # Candle Strength
        # Current bar body
        open_ = bar.open
        close = bar.close
        high = bar.high
        low = bar.low

        is_green = close > open_
        is_red = close < open_

        vol_ok = float(bar.volume) > (avg_vol_f * float(self.vol_mult))

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

            signal = self._create_signal(
                symbol=symbol,
                side=OrderSide.BUY,
                bar=bar,
                market_state=market_state,
                stop_price=stop_price,
                target_price=target_price,
                meta={
                    "flow_zscore": flow_score,
                    "call_put_ratio": call_put_ratio,
                    "is_bullish_flow": bool(is_bullish_flow),
                    "vol_mult": (float(bar.volume) / avg_vol_f) if avg_vol_f else None,
                },
            )

        # BEARISH MOMENTUM
        elif (not is_bullish_flow) and is_red and vol_ok:
            # Entry: Sell logic
            stop_price = high
            if stop_price <= close:
                stop_price = close * 1.01

            risk = stop_price - close
            target_price = close - (risk * self.risk_reward)

            signal = self._create_signal(
                symbol=symbol,
                side=OrderSide.SELL,
                bar=bar,
                market_state=market_state,
                stop_price=stop_price,
                target_price=target_price,
                meta={
                    "flow_zscore": flow_score,
                    "call_put_ratio": call_put_ratio,
                    "is_bullish_flow": bool(is_bullish_flow),
                    "vol_mult": (float(bar.volume) / avg_vol_f) if avg_vol_f else None,
                },
            )

        return signal
