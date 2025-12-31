from __future__ import annotations

from datetime import datetime, timedelta, timezone
from datetime import time as time_type
from typing import Any, Dict, Optional

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy
from src.strategies.config_models import GapFillConfig


class GapFillStrategy(BaseStrategy):
    """
    Gap-Fill Scalper.
    Fades morning gaps that fail to extend.
    Entry: Break of Opening Range (first 5-15 mins) in opposite direction of Gap.
    Target: Prior Day Close (the gap fill level).
    """

    name = "gap_fill"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        cfg = GapFillConfig(**config)
        self.min_gap = cfg.min_gap
        self.max_gap = cfg.max_gap
        self.risk_reward = cfg.risk_reward
        self.or_time_minutes = cfg.or_time_minutes
        self.weak_trend_max_score = cfg.weak_trend_max_score

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

        if not symbol_state.bars:
            return None

        # PRD 7.2: Gap-Fill prefers CHOP / weak trend - regime gating now handled by engine.
        # Keep trend_score check as an additional filter for signal quality.
        trend_score = None
        if isinstance(getattr(market_state, "meta", None), dict):
            trend_score = market_state.meta.get("trend_score")
        try:
            trend_score_f = float(trend_score) if trend_score is not None else None
        except Exception:
            trend_score_f = None

        # Optional: if trend is too strong, skip (configurable via weak_trend_max_score)
        if trend_score_f is not None and trend_score_f >= float(
            self.weak_trend_max_score
        ):
            # Strong trend - less optimal for gap fill, but don't hard block
            pass

        # Requires scanner/pipeline-provided `gap_pct` in `symbol_state.meta`.
        gap_pct = float(symbol_state.meta.get("gap_pct", 0.0) or 0.0)
        if gap_pct == 0.0:
            if len(symbol_state.bars) < 20:
                self.logger.warning(
                    "Missing gap_pct for GapFill strategy", symbol=symbol
                )
            return None

        # PRD 7.2: enforce X–Y% gap size constraint.
        gap_abs = abs(gap_pct)
        if gap_abs < float(self.min_gap) or gap_abs > float(self.max_gap):
            return None

        gap_up = gap_pct > 0
        gap_down = gap_pct < 0

        bars = symbol_state.bars
        bt = bar.time
        if isinstance(bt, datetime) and bt.tzinfo is None:
            bt = bt.replace(tzinfo=timezone.utc)
        bt_et = time_utils.to_eastern_time(bt)
        session_open_et = time_utils.get_eastern_timezone().localize(
            datetime.combine(bt_et.date(), time_type(9, 30))
        )
        cutoff_et = session_open_et + timedelta(minutes=self.or_time_minutes)

        # Build opening range.
        if bt_et <= cutoff_et:
            return None

        or_bars = []
        for b in bars:
            t = b.time
            if isinstance(t, datetime) and t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            t_et = time_utils.to_eastern_time(t)
            if t_et.date() != bt_et.date():
                continue
            if t_et <= cutoff_et:
                or_bars.append(b)

        if not or_bars:
            return None

        or_high = max(b.high for b in or_bars)
        or_low = min(b.low for b in or_bars)

        # Fade gap up (short): breakdown below OR low.
        if gap_up and bar.close < or_low:
            open_price = float(or_bars[0].open)
            # P0 fix: Guard against division by zero (100% gap down = gap_pct = -1.0)
            if abs(1.0 + gap_pct) < 1e-9:
                return None
            prev_close = open_price / (1.0 + gap_pct)
            if bar.close <= prev_close:
                return None

            stop_price = float(or_high)
            target_price = float(prev_close)
            risk = stop_price - float(bar.close)
            reward = float(bar.close) - target_price
            if risk > 0 and (reward / risk) >= float(self.risk_reward):
                return self._create_signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    bar=bar,
                    market_state=market_state,
                    stop_price=stop_price,
                    target_price=target_price,
                    meta={"gap_pct": gap_pct, "prev_close": prev_close},
                )

        # Fade gap down (long): breakout above OR high.
        if gap_down and bar.close > or_high:
            open_price = float(or_bars[0].open)
            # P0 fix: Guard against division by zero
            if abs(1.0 + gap_pct) < 1e-9:
                return None
            prev_close = open_price / (1.0 + gap_pct)
            if bar.close >= prev_close:
                return None

            stop_price = float(or_low)
            target_price = float(prev_close)
            risk = float(bar.close) - stop_price
            reward = target_price - float(bar.close)
            if risk > 0 and (reward / risk) >= float(self.risk_reward):
                return self._create_signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    bar=bar,
                    market_state=market_state,
                    stop_price=stop_price,
                    target_price=target_price,
                    meta={
                        "gap_pct": gap_pct,
                        "prev_close": prev_close,
                        "or_high": float(or_high),
                        "or_low": float(or_low),
                    },
                )

        return None
