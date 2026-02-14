import math
from datetime import datetime, timedelta
from datetime import time as time_type
from typing import Any, Dict, Optional

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
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

    def _set_params(self, config: Dict[str, Any]) -> None:
        """Override to parse strategy-specific parameters."""
        super()._set_params(config)
        cfg = ORBConfig(**config)
        self.orb_start = time_type(9, 30)
        self.orb_minutes = cfg.orb_minutes
        base = datetime(2000, 1, 1, self.orb_start.hour, self.orb_start.minute)
        self.orb_end = (base + timedelta(minutes=self.orb_minutes)).time()
        self.entry_window_end = time_type(10, 30)  # Don't enter after this

        # Pull from config or defaults
        self.risk_reward = cfg.risk_reward
        self.stop_loss_pct = cfg.stop_loss_pct
        self.min_or_range_pct = cfg.min_or_range_pct
        self.min_gap_pct = cfg.min_gap_pct
        self.min_flow_zscore = cfg.min_flow_zscore
        self.min_premarket_volume = cfg.min_premarket_volume
        self.min_or_range_pct = cfg.min_or_range_pct
        # P2 fix: ATR-based buffer for stop placement
        self.stop_buffer_atr_mult = cfg.stop_buffer_atr_mult

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
        # P1.1 fix: Reset ORB range at session start to prevent stale data
        bar_date = bar.time.date()
        stored_date = symbol_state.indicators.get("orb_date")
        if stored_date != bar_date:
            symbol_state.indicators["orb_high"] = float("-inf")
            symbol_state.indicators["orb_low"] = float("inf")
            symbol_state.indicators["orb_date"] = bar_date
            symbol_state.indicators["orb_complete"] = False

        current_high = symbol_state.indicators.get("orb_high", float("-inf"))
        current_low = symbol_state.indicators.get("orb_low", float("inf"))

        symbol_state.indicators["orb_high"] = max(current_high, bar.high)
        symbol_state.indicators["orb_low"] = min(current_low, bar.low)
        symbol_state.indicators["orb_complete"] = False

    def _calculate_atr(self, symbol_state: SymbolState, period: int = 14) -> float:
        """Calculate ATR from available bars for stop buffer."""
        bars = list(symbol_state.bars)
        if len(bars) < 2:
            return 0.0

        trs = []
        for i in range(1, min(len(bars), period + 1)):
            high = bars[-i].high
            low = bars[-i].low
            prev_close = bars[-i - 1].close if i < len(bars) else bars[-i].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)

        return sum(trs) / len(trs) if trs else 0.0

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
        if self.min_premarket_volume > 0 and premarket_volume < self.min_premarket_volume:
            return None

        orb_high = symbol_state.indicators.get("orb_high")
        orb_low = symbol_state.indicators.get("orb_low")

        if orb_high is None or orb_low is None:
            return None
        if orb_low <= 0:
            return None
        if self.min_or_range_pct > 0:
            or_range_pct = (orb_high - orb_low) / orb_low
            if or_range_pct < self.min_or_range_pct:
                return None

        # === QLIB-INSPIRED CONFIRMATIONS ===

        # 1. VWAP Confirmation - price should be on correct side of VWAP
        vwap = getattr(bar, "vwap", None) or symbol_state.meta.get("vwap")

        # 2. Relative Volume (RVOL) - breakout bar should have elevated volume
        rvol = symbol_state.meta.get("rvol", 0.0) or 0.0
        bar_volume = bar.volume if hasattr(bar, "volume") else 0
        avg_volume = symbol_state.meta.get("avg_volume_20", 0) or 0
        if avg_volume > 0 and bar_volume > 0:
            breakout_rvol = bar_volume / avg_volume
        else:
            breakout_rvol = rvol  # fallback to pre-computed RVOL

        # Long Breakout
        if bar.close > orb_high_f:
            # Confirmation 1: VWAP - price should be above VWAP for longs
            if vwap and bar.close < vwap:
                return None  # Weak long - price below VWAP

            # Confirmation 2: RVOL - require elevated volume (> 1.0)
            if breakout_rvol > 0 and breakout_rvol < 1.0:
                return None  # Low volume breakout - likely false

            # Confirmation 3: Gap alignment - for longs, prefer positive gaps
            # (soft filter - still allow if gap < 0 but > -1%)
            if gap_pct < -0.01:  # More than 1% down gap
                return None  # Counter-trend long after large down gap

            # P2 fix: Apply ATR buffer to stop
            stop = orb_low_f
            if self.stop_buffer_atr_mult > 0:
                atr = self._calculate_atr(symbol_state)
                stop = orb_low_f - (atr * self.stop_buffer_atr_mult)
            stop = self._cap_stop_loss(OrderSide.BUY, bar.close, stop, symbol, bar.time)
            return self._create_orb_signal(
                symbol,
                bar,
                OrderSide.BUY,
                stop,
                market_state,
                orb_high_f,
                orb_low_f,
                meta={
                    "gap_pct": gap_pct,
                    "flow_zscore": flow_zscore,
                    "premarket_volume": premarket_volume,
                    "vwap": vwap,
                    "breakout_rvol": breakout_rvol,
                },
            )

        # Short Breakout
        if bar.close < orb_low_f:
            # Confirmation 1: VWAP - price should be below VWAP for shorts
            if vwap and bar.close > vwap:
                return None  # Weak short - price above VWAP

            # Confirmation 2: RVOL - require elevated volume (> 1.0)
            if breakout_rvol > 0 and breakout_rvol < 1.0:
                return None  # Low volume breakout - likely false

            # Confirmation 3: Gap alignment - for shorts, prefer negative gaps
            # (soft filter - still allow if gap > 0 but < 1%)
            if gap_pct > 0.01:  # More than 1% up gap
                return None  # Counter-trend short after large up gap

            # P2 fix: Apply ATR buffer to stop
            stop = orb_high_f
            if self.stop_buffer_atr_mult > 0:
                atr = self._calculate_atr(symbol_state)
                stop = orb_high_f + (atr * self.stop_buffer_atr_mult)
            stop = self._cap_stop_loss(OrderSide.SELL, bar.close, stop, symbol, bar.time)
            return self._create_orb_signal(
                symbol,
                bar,
                OrderSide.SELL,
                stop,
                market_state,
                orb_high_f,
                orb_low_f,
                meta={
                    "gap_pct": gap_pct,
                    "flow_zscore": flow_zscore,
                    "premarket_volume": premarket_volume,
                    "vwap": vwap,
                    "breakout_rvol": breakout_rvol,
                },
            )

        return None

    def _orb_range_ready(self, orb_high: Any, orb_low: Any) -> bool:
        if orb_high is None or orb_low is None:
            return False
        try:
            return math.isfinite(float(orb_high)) and math.isfinite(float(orb_low))
        except (TypeError, ValueError):
            return False

    def _log_missing_orb_range(
        self,
        symbol_state: SymbolState,
        symbol: str,
        bar_time: datetime,
        orb_high: Any,
        orb_low: Any,
    ) -> None:
        bar_date = bar_time.date()
        if symbol_state.indicators.get("orb_missing_range_logged") == bar_date:
            return
        symbol_state.indicators["orb_missing_range_logged"] = bar_date
        self.logger.warning(
            f"{self.name}: missing opening range data",
            symbol=symbol,
            orb_high=orb_high,
            orb_low=orb_low,
            orb_complete=symbol_state.indicators.get("orb_complete"),
        )

    def _cap_stop_loss(
        self,
        side: OrderSide,
        entry_price: float,
        stop_price: float,
        symbol: str,
        bar_time: datetime,
    ) -> float:
        if self.stop_loss_pct <= 0:
            return stop_price
        entry = float(entry_price)
        if entry <= 0:
            return stop_price
        max_risk = entry * float(self.stop_loss_pct)
        if max_risk <= 0:
            return stop_price

        if side == OrderSide.BUY:
            capped_stop = max(float(stop_price), entry - max_risk)
        else:
            capped_stop = min(float(stop_price), entry + max_risk)

        if capped_stop != float(stop_price):
            self.logger.info(
                f"{self.name}: stop_loss_pct cap applied",
                symbol=symbol,
                entry_price=entry,
                stop_price=float(stop_price),
                capped_stop=capped_stop,
                stop_loss_pct=self.stop_loss_pct,
                bar_time=bar_time.isoformat(),
            )
        return capped_stop

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
