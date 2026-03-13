from datetime import datetime, time, timedelta
from typing import Any, Dict, Optional

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.data.requirements import DataRequirements
from src.strategies.base import BaseStrategy
from src.strategies.config_models import FusionStrategyConfig


class FusionStrategyV1(BaseStrategy):
    """
    Fusion V1 (Multi-Factor) Strategy.

    Entry logic:
    1.  Price Breakout: Break above/below opening range.
    2.  Relative Strength: Stock must outperform SPY (RS > threshold).
    3.  Directional Options Flow: Flow bias must align with breakout direction
        and DOF score must exceed threshold (institutional conviction).

    Exit logic:
    -   ATR-based dynamic stops and targets.
    """

    name = "fusion_v1"
    data_requirements = DataRequirements(streams=["bars", "quotes"], on_scan=["flow", "gex"])

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        cfg = FusionStrategyConfig(**config)
        self.orb_minutes = cfg.orb_minutes
        self.min_dof_score = cfg.min_dof_score
        self.min_flow_bias = cfg.min_flow_bias
        self.min_relative_strength = cfg.min_relative_strength
        self.require_flow = cfg.require_flow
        self.atr_period = cfg.atr_period
        self.stop_atr_mult = cfg.stop_atr_mult
        self.target_atr_mult = cfg.target_atr_mult

        # Set timing windows
        self.orb_start = time_utils.parse_time_string("09:30")

        # Calculate orb_end time
        base_dt = datetime(2000, 1, 1, self.orb_start.hour, self.orb_start.minute)
        self.orb_end = (base_dt + timedelta(minutes=self.orb_minutes)).time()

        self.entry_start = time_utils.parse_time_string(cfg.entry_window_start)
        self.entry_end = time_utils.parse_time_string(cfg.entry_window_end)

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        if not bar:
            return None

        t = time_utils.get_eastern_time_of_day(bar.time)

        # 1. Update Opening Range
        if self.orb_start <= t < self.orb_end:
            self._update_opening_range(symbol_state, bar)
            return None

        if t >= self.orb_end:
            symbol_state.indicators["orb_complete"] = True

        # 2. Global Execution Filters (Entry Window & Position)
        if not (self.entry_start <= t <= self.entry_end):
            return None

        if symbol_state.position and symbol_state.position.strategy == self.name:
            return None

        # --- Factor 3: Multi-Factor Fusion Signal ---
        res = self._check_fusion_entry(symbol, bar, symbol_state, market_state)

        # Targeted tracing for AAPL to verify metadata penetration
        if symbol == "AAPL" and t == time(10, 0):
            rs = float(symbol_state.meta.get("relative_strength", 0.0) or 0.0)
            oh = float(symbol_state.meta.get("orb_high", 0.0) or 0.0)
            self.logger.info(
                "AAPL Trace: 10AM",
                close=bar.close,
                rs=round(rs, 6),
                orb_h=round(oh, 2),
                orb_complete=symbol_state.indicators.get("orb_complete", False),
            )
        return res

    def _update_opening_range(self, symbol_state: SymbolState, bar: Bar):
        """Track high/low for the specified opening range duration."""
        bar_date = bar.time.date()
        stored_date = symbol_state.indicators.get("orb_date")

        # Reset range if new day
        if stored_date != bar_date:
            symbol_state.indicators["orb_high"] = float("-inf")
            symbol_state.indicators["orb_low"] = float("inf")
            symbol_state.indicators["orb_date"] = bar_date
            symbol_state.indicators["orb_complete"] = False

        current_high = symbol_state.indicators.get("orb_high", float("-inf"))
        current_low = symbol_state.indicators.get("orb_low", float("inf"))

        symbol_state.indicators["orb_high"] = max(current_high, bar.high)
        symbol_state.indicators["orb_low"] = min(current_low, bar.low)

    def _check_fusion_entry(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """Aligns price, relative strength, and options flow factors."""
        if not symbol_state.indicators.get("orb_complete") and not symbol_state.meta.get("scanner_bypass"):
            return None

        orb_high = symbol_state.indicators.get("orb_high")
        orb_low = symbol_state.indicators.get("orb_low")

        # Fallback to metadata from scanner (PRD Phase 2)
        if orb_high is None or orb_low is None or orb_high == float("-inf"):
            orb_high = symbol_state.meta.get("orb_high")
            orb_low = symbol_state.meta.get("orb_low")

        if not orb_high or not orb_low:
            return None

        # --- Factor 1: Price Breakout ---
        is_breakout_high = bar.close > orb_high
        is_breakout_low = bar.close < orb_low

        if not (is_breakout_high or is_breakout_low):
            return None

        # Diagnostic log (Price Breakout detected) - Promoted to INFO for backtest visibility
        self.logger.info(
            "Price Breakout detected",
            symbol=symbol,
            close=bar.close,
            orb_high=orb_high,
            orb_low=orb_low,
            side="buy" if is_breakout_high else "sell",
        )

        # --- Factor 2: Relative Strength (Alpha Filter) ---
        # Compare performance vs SPY index
        rs = float(symbol_state.meta.get("relative_strength", 0.0) or 0.0)
        if rs < self.min_relative_strength:
            return None

        # --- Factor 3: Directional Options Flow (Fusion Signal) ---
        dof_score = float(symbol_state.meta.get("dof_score", 0.0) or 0.0)
        flow_bias = float(symbol_state.meta.get("flow_bias", 0.0) or 0.0)

        flow_aligned = False
        if is_breakout_high:
            if flow_bias >= self.min_flow_bias and dof_score >= self.min_dof_score:
                flow_aligned = True
            side = OrderSide.BUY

        elif is_breakout_low:
            if flow_bias <= -self.min_flow_bias and dof_score >= self.min_dof_score:
                flow_aligned = True
            side = OrderSide.SELL
        else:
            return None

        # If flow is required, it MUST be aligned.
        # If not required, we allow entry on price+RS alone if flow is 0.0 (unobserved)
        if self.require_flow and not flow_aligned:
            return None

        # Even if not required, if flow IS present but counter-aligned, we should probably skip.
        if not self.require_flow and not flow_aligned:
            # If flow exists but is wrong direction, skip
            if is_breakout_high and flow_bias < -0.1:
                return None
            if is_breakout_low and flow_bias > 0.1:
                return None
            # Otherwise (flow is 0.0), we proceed

        # --- Risk Management: ATR-based Dynamic Targets ---
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

        # Ensure stop is not cross-breakout (optional but safer)
        # For long: stop should be above orb_low or at leastorb_low
        # We'll stick to pure ATR for now as it's more 'quant'

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={
                "rs": rs,
                "dof_score": dof_score,
                "flow_bias": flow_bias,
                "atr": atr,
                "orb_high": orb_high,
                "orb_low": orb_low,
                "factor_alignment": True,
            },
        )
