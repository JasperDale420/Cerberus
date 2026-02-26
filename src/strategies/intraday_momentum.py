from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy
from src.strategies.config_models import IntradayMomentumConfig


class IntradayMomentumStrategy(BaseStrategy):
    """
    Intraday Time-Series Momentum (ITSM) Strategy.

    Entry logic:
    1.  Activates after the Opening Session.
    2.  Checks for strong ADX and EMA trend strength.
    3.  Enters in the direction of the trend if price breaks opening range.

    Exit logic:
    -   ATR-based dynamic stops and targets.
    -   Designed to hold through midday/power_hour.
    """

    name = "intraday_momentum"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)

    def _set_params(self, config: Dict[str, Any]) -> None:
        """Override to parse strategy-specific parameters."""
        super()._set_params(config)
        cfg = IntradayMomentumConfig(**config)
        self.min_adx = cfg.min_adx
        self.min_trend_strength = cfg.min_trend_strength
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
        if not self._require_min_bars(symbol_state, 20):  # Need ADX/EMA history
            return None

        # Session Regimes checks (Must not be premarket/opening)
        if market_state.regime_snapshot:
            session = market_state.regime_snapshot.session.value if market_state.regime_snapshot.session else None
            # Do not trade this strategy during the opening 30 minutes
            if session in ("premarket", "opening"):
                return None

        # Extract technical features
        adx = float(symbol_state.meta.get("adx", 0.0) or 0.0)
        trend_strength = float(symbol_state.meta.get("ema_trend_strength", 0.0) or 0.0)

        # Verify sufficient trend
        if adx < self.min_adx or abs(trend_strength) < self.min_trend_strength:
            return None

        # Compare to opening range breakout
        orb_high = symbol_state.meta.get("orb_high")
        orb_low = symbol_state.meta.get("orb_low")

        if not orb_high or not orb_low or orb_high == float("-inf"):
            return None

        is_breakout_high = bar.close > orb_high
        is_breakout_low = bar.close < orb_low

        if not (is_breakout_high or is_breakout_low):
            return None

        # Ensure breakout aligns with trend strength direction
        if is_breakout_high and trend_strength > 0:
            side = OrderSide.BUY
        elif is_breakout_low and trend_strength < 0:
            side = OrderSide.SELL
        else:
            return None

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
            "Intraday Time-Series Momentum entry triggered",
            symbol=symbol,
            adx=round(adx, 2),
            trend_strength=round(trend_strength, 2),
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
                "adx": adx,
                "trend_strength": trend_strength,
                "atr": atr,
                "orb_high": orb_high,
                "orb_low": orb_low,
            },
        )
