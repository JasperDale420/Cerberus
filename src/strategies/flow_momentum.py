from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy
from src.strategies.config_models import FlowMomentumConfig

# L3 fix: Named constants for emergency stop buffers when bar extremes are invalid
# 1% buffer from close when stop would be at or beyond entry
STOP_BUFFER_LONG = 0.99  # Long: stop 1% below close
STOP_BUFFER_SHORT = 1.01  # Short: stop 1% above close


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

    def _validate_flow_direction(
        self, symbol_state: SymbolState
    ) -> tuple[bool, float, float]:
        """
        Validate flow zscore and call/put ratio agreement.

        Returns:
            Tuple of (is_valid, flow_score, call_put_ratio).
            is_valid is False if flow is insufficient or misaligned.
        """
        flow_score = float(symbol_state.meta.get("flow_zscore", 0.0) or 0.0)
        call_put_ratio = float(symbol_state.meta.get("call_put_ratio", 0.0) or 0.0)

        # Reject weak flow
        if abs(flow_score) < self.min_flow_zscore:
            return False, flow_score, call_put_ratio

        # Require valid ratio
        if call_put_ratio <= 0:
            return False, flow_score, call_put_ratio

        # Require flow/ratio agreement
        is_bullish_flow = flow_score > 0
        if is_bullish_flow and call_put_ratio < 1.0:
            return False, flow_score, call_put_ratio
        if not is_bullish_flow and call_put_ratio > 1.0:
            return False, flow_score, call_put_ratio

        return True, flow_score, call_put_ratio

    def _get_average_volume(self, symbol_state: SymbolState) -> Optional[float]:
        """Get 20-bar average volume from indicators or calculate directly."""
        avg_vol = symbol_state.indicators.get("sma_vol:20")
        if avg_vol is None:
            bars = list(symbol_state.bars)
            vols = [float(b.volume) for b in bars[-20:]]
            if not vols:
                return None
            avg_vol = sum(vols) / len(vols)
        try:
            return float(avg_vol)
        except Exception:
            return None

    def _build_signal(
        self,
        symbol: str,
        bar: Bar,
        market_state: MarketState,
        side: OrderSide,
        flow_score: float,
        call_put_ratio: float,
        avg_vol_f: float,
    ) -> Signal:
        """Build a signal for the given side with proper stop/target."""
        close = bar.close

        if side == OrderSide.BUY:
            stop_price = bar.low
            if stop_price >= close:
                stop_price = close * STOP_BUFFER_LONG
            risk = close - stop_price
            target_price = close + (risk * self.risk_reward)
        else:
            stop_price = bar.high
            if stop_price <= close:
                stop_price = close * STOP_BUFFER_SHORT
            risk = stop_price - close
            target_price = close - (risk * self.risk_reward)

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={
                "flow_zscore": flow_score,
                "call_put_ratio": call_put_ratio,
                "is_bullish_flow": flow_score > 0,
                "vol_mult": (float(bar.volume) / avg_vol_f) if avg_vol_f else None,
            },
        )

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # P1 fix: Add cooldown check to prevent rapid-fire signals
        if not self._check_cooldown(symbol, bar.time):
            return None

        # Regime gating removed - handled by strategy routing at engine level

        # Validate flow direction and agreement
        is_valid, flow_score, call_put_ratio = self._validate_flow_direction(
            symbol_state
        )
        if not is_valid:
            return None

        # Check minimum bars
        if not self._require_min_bars(symbol_state, 21):
            return None

        # Get average volume
        avg_vol_f = self._get_average_volume(symbol_state)
        if avg_vol_f is None:
            return None

        # Check volume confirmation
        vol_ok = float(bar.volume) > (avg_vol_f * float(self.vol_mult))
        if not vol_ok:
            return None

        # Determine direction from candle and flow
        is_bullish_flow = flow_score > 0
        is_green = bar.close > bar.open
        is_red = bar.close < bar.open

        # Generate signal if conditions align
        if is_bullish_flow and is_green:
            return self._build_signal(
                symbol,
                bar,
                market_state,
                OrderSide.BUY,
                flow_score,
                call_put_ratio,
                avg_vol_f,
            )
        if not is_bullish_flow and is_red:
            return self._build_signal(
                symbol,
                bar,
                market_state,
                OrderSide.SELL,
                flow_score,
                call_put_ratio,
                avg_vol_f,
            )

        return None
