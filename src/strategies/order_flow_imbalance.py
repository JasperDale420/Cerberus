from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy
from src.strategies.config_models import OrderFlowImbalanceConfig


class OrderFlowImbalanceStrategy(BaseStrategy):
    """
    Order Flow Imbalance (OFI) Strategy.

    Entry logic:
    1.  TFI (Trade Flow Imbalance) is highly skewed (abs(tfi) > tfi_threshold).
    2.  Directional Options Flow bias aligns with TFI direction.

    Exit logic:
    -   ATR-based dynamic stops and targets.
    """

    name = "order_flow_imbalance"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)

    def _set_params(self, config: Dict[str, Any]) -> None:
        """Override to parse strategy-specific parameters."""
        super()._set_params(config)
        cfg = OrderFlowImbalanceConfig(**config)
        self.tfi_threshold = cfg.tfi_threshold
        self.min_flow_bias = cfg.min_flow_bias
        self.stop_atr_mult = cfg.stop_atr_mult
        self.target_atr_mult = cfg.target_atr_mult

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # Check cooldown
        if not self._check_cooldown(symbol, bar.time):
            return None

        # Check existing position for this strategy
        if symbol_state.position and symbol_state.position.strategy == self.name:
            return None

        # Require minimum bars
        if not self._require_min_bars(symbol_state, 14):
            return None

        # Extract features
        tfi = float(symbol_state.meta.get("tfi", 0.0) or 0.0)
        flow_bias = float(symbol_state.meta.get("flow_bias", 0.0) or 0.0)

        # Entry Conditions
        is_bullish = tfi >= self.tfi_threshold and flow_bias >= self.min_flow_bias
        is_bearish = tfi <= -self.tfi_threshold and flow_bias <= -self.min_flow_bias

        if not (is_bullish or is_bearish):
            return None

        side = OrderSide.BUY if is_bullish else OrderSide.SELL

        # Get ATR from scanner/pipeline meta or calculate basic fallback
        atr = float(symbol_state.meta.get("atr", 0.0) or 0.0)
        if atr <= 0:
            # Fallback to 1% of price as a unit of risk
            atr = bar.close * 0.01

        if side == OrderSide.BUY:
            stop_price = bar.close - (atr * self.stop_atr_mult)
            target_price = bar.close + (atr * self.target_atr_mult)
        else:
            stop_price = bar.close + (atr * self.stop_atr_mult)
            target_price = bar.close - (atr * self.target_atr_mult)

        self.logger.info(
            "Order Flow Imbalance entry triggered",
            symbol=symbol,
            tfi=round(tfi, 3),
            flow_bias=round(flow_bias, 3),
            side=side.value,
        )

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={
                "tfi": tfi,
                "flow_bias": flow_bias,
                "atr": atr,
            },
        )
