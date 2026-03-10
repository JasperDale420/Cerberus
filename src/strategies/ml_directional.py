from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.strategies.base import BaseStrategy


class MLDirectionalStrategy(BaseStrategy):
    """
    ML Enhanced Directional Strategy

    Uses standard trend alignment but requires a Machine Learning prediction
    score above a certain threshold to generate a valid signal.
    """
    name = "ml_directional"

    def __init__(self, config: dict[str, Any], logger: Any):
        super().__init__(config, logger)
        self.ml_threshold = float(self.config.get("ml_threshold", 0.75))
        self.require_trend_alignment = bool(self.config.get("require_trend_alignment", True))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        if not self._require_min_bars(symbol_state, 20, log=False):
            return None

        if not self._check_cooldown(symbol, bar.time):
            return None

        features = symbol_state.meta.get("features")
        if not features:
            return None

        ml_prediction = getattr(features, "ml_prediction", 0.0)

        # We need a strong directional prediction
        if abs(ml_prediction) < self.ml_threshold:
            return None

        side = OrderSide.BUY if ml_prediction > 0 else OrderSide.SELL

        # Check higher timeframe trend alignment if configured
        if self.require_trend_alignment and not self._check_higher_tf_alignment(symbol_state, side):
            return None

        # Basic stop/target logic based on ATR
        atr = features.atr
        if atr <= 0:
            return None

        # Dynamically scale stops based on volatility regime
        base_stop_dist = atr * float(self.config.get("stop_atr_multiplier", 1.5))
        base_target_dist = base_stop_dist * float(self.config.get("risk_reward_ratio", 2.0))

        stop_dist = self._apply_regime_volatility_multiplier(base_stop_dist, market_state)
        # Apply dynamic RR based on regime (e.g., lower RR in chop, higher in trend)
        dynamic_rr = self._get_dynamic_risk_reward(base_target_dist / base_stop_dist, market_state)
        target_dist = stop_dist * dynamic_rr

        if side == OrderSide.BUY:
            stop_price = bar.close - stop_dist
            target_price = bar.close + target_dist
        else:
            stop_price = bar.close + stop_dist
            target_price = bar.close - target_dist

        signal = self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            entry_price=bar.close,
            meta={"ml_prediction": ml_prediction, "dynamic_rr": dynamic_rr}
        )

        self.logger.info("ML Directional signal generated", symbol=symbol, side=side.value, prediction=ml_prediction)
        return signal
