from datetime import datetime, timedelta
from datetime import time as time_type
from typing import Any, Dict, Optional

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy
from src.strategies.config_models import ORBConfig


class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout (ORB) Strategy.
    Uses the first N minutes (default 15) to define the range.
    Breaks above/below that range trigger signals.
    """

    name = "orb"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        cfg = ORBConfig(**config)
        self.orb_start = time_type(9, 30)
        self.orb_minutes = cfg.orb_minutes
        base = datetime(2000, 1, 1, self.orb_start.hour, self.orb_start.minute)
        self.orb_end = (base + timedelta(minutes=self.orb_minutes)).time()
        self.entry_window_end = time_type(10, 30)  # Don't enter after this

        # Pull from config or defaults
        self.risk_reward = cfg.risk_reward
        self.stop_loss_pct = cfg.stop_loss_pct
        self.min_gap_pct = cfg.min_gap_pct
        self.min_flow_zscore = cfg.min_flow_zscore
        self.min_premarket_volume = cfg.min_premarket_volume

    def _to_et_time(self, dt: datetime) -> time_type:
        """Convert datetime to US/Eastern time-of-day."""
        return time_utils.get_eastern_time_of_day(dt)

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        if not bar:
            return None

        t = self._to_et_time(bar.time)

        # 1. Update Opening Range
        if self.orb_start <= t < self.orb_end:
            self._update_opening_range(symbol_state, bar)
            return None

        # 2. Mark Completion
        if t >= self.orb_end:
            symbol_state.indicators["orb_complete"] = True

        # 3. Check Breakout
        return self._check_breakout(symbol, bar, symbol_state, market_state)

    def _update_opening_range(self, symbol_state: SymbolState, bar: Bar):
        current_high = symbol_state.indicators.get("orb_high", float("-inf"))
        current_low = symbol_state.indicators.get("orb_low", float("inf"))

        symbol_state.indicators["orb_high"] = max(current_high, bar.high)
        symbol_state.indicators["orb_low"] = min(current_low, bar.low)
        symbol_state.indicators["orb_complete"] = False

    def _check_breakout(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        t = self._to_et_time(bar.time)

        if not symbol_state.indicators.get("orb_complete"):
            return None

        if t > self.entry_window_end:
            return None

        if symbol_state.position and symbol_state.position.strategy == self.name:
            return None

        # PRD 7.2 ORB filters (best-effort, uses scanner/pipeline meta).
        gap_pct = float(symbol_state.meta.get("gap_pct", 0.0) or 0.0)
        flow_zscore = float(symbol_state.meta.get("flow_zscore", 0.0) or 0.0)
        premarket_volume = float(symbol_state.meta.get("premarket_volume", 0.0) or 0.0)
        if self.min_gap_pct > 0 and abs(gap_pct) < self.min_gap_pct:
            return None
        if self.min_flow_zscore > 0 and abs(flow_zscore) < self.min_flow_zscore:
            return None
        if (
            self.min_premarket_volume > 0
            and premarket_volume < self.min_premarket_volume
        ):
            return None

        orb_high = symbol_state.indicators.get("orb_high")
        orb_low = symbol_state.indicators.get("orb_low")

        if not orb_high or not orb_low:
            return None

        # Long Breakout (PRD: BULL)
        if bar.close > orb_high and market_state.regime == Regime.BULL:
            return self._create_orb_signal(
                symbol,
                bar,
                OrderSide.BUY,
                orb_low,
                market_state,
                orb_high,
                orb_low,
                meta={
                    "gap_pct": gap_pct,
                    "flow_zscore": flow_zscore,
                    "premarket_volume": premarket_volume,
                },
            )

        # Short Breakout (PRD: BEAR)
        if bar.close < orb_low and market_state.regime == Regime.BEAR:
            return self._create_orb_signal(
                symbol,
                bar,
                OrderSide.SELL,
                orb_high,
                market_state,
                orb_high,
                orb_low,
                meta={
                    "gap_pct": gap_pct,
                    "flow_zscore": flow_zscore,
                    "premarket_volume": premarket_volume,
                },
            )

        return None

    def _create_orb_signal(
        self,
        symbol: str,
        bar: Bar,
        side: OrderSide,
        stop_price: float,
        market_state: MarketState,
        orb_high: float,
        orb_low: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Signal]:
        risk = abs(bar.close - stop_price)
        if risk <= 0:
            return None

        if side == OrderSide.BUY:
            target_price = bar.close + (risk * self.risk_reward)
        else:
            target_price = bar.close - (risk * self.risk_reward)

        # Build metadata
        out_meta = {
            "or_high": float(orb_high),
            "or_low": float(orb_low),
            "breakout_type": "high" if side == OrderSide.BUY else "low",
        }
        if isinstance(meta, dict):
            out_meta.update(meta)

        # Use base class helper
        return super()._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta=out_meta,
        )
