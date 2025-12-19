from __future__ import annotations

from datetime import datetime, timedelta, timezone
from datetime import time as time_type
from typing import Any, Dict, Optional

import pytz  # type: ignore

from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


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
        self.min_gap = float(config.get("min_gap", 0.015) or 0.015)
        self.max_gap = float(config.get("max_gap", 0.10) or 0.10)
        self.risk_reward = float(config.get("risk_reward", 2.0) or 2.0)
        self.or_time_minutes = int(config.get("or_time_minutes", 15) or 15)
        self.weak_trend_max_score = float(
            config.get("weak_trend_max_score", 1.0) or 1.0
        )

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        if not symbol_state.bars:
            return None

        # PRD 7.2: Gap-Fill is for CHOP / weak trend.
        trend_score = None
        if isinstance(getattr(market_state, "meta", None), dict):
            trend_score = market_state.meta.get("trend_score")
        try:
            trend_score_f = float(trend_score) if trend_score is not None else None
        except Exception:
            trend_score_f = None

        is_weak_trend = trend_score_f is not None and trend_score_f < float(
            self.weak_trend_max_score
        )
        if not (market_state.regime == Regime.CHOP or is_weak_trend):
            return None

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
        et = pytz.timezone("US/Eastern")
        bt_et = bt.astimezone(et)
        session_open_et = et.localize(datetime.combine(bt_et.date(), time_type(9, 30)))
        cutoff_et = session_open_et + timedelta(minutes=self.or_time_minutes)

        # Build opening range.
        if bt_et <= cutoff_et:
            return None

        or_bars = []
        for b in bars:
            t = b.time
            if isinstance(t, datetime) and t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            t_et = t.astimezone(et)
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
            prev_close = open_price / (1.0 + gap_pct)
            if bar.close <= prev_close:
                return None

            stop_price = float(or_high)
            target_price = float(prev_close)
            risk = stop_price - float(bar.close)
            reward = float(bar.close) - target_price
            if risk > 0 and (reward / risk) >= float(self.risk_reward):
                return Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    size_hint=0,
                    entry_price=float(bar.close),
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy=self.name,
                    regime=market_state.regime,
                    generated_at=bar.time,
                    meta={
                        "gap_pct": gap_pct,
                        "or_low": float(or_low),
                        "or_high": float(or_high),
                    },
                )

        # Fade gap down (long): breakout above OR high.
        if gap_down and bar.close > or_high:
            open_price = float(or_bars[0].open)
            prev_close = open_price / (1.0 + gap_pct)
            if bar.close >= prev_close:
                return None

            stop_price = float(or_low)
            target_price = float(prev_close)
            risk = float(bar.close) - stop_price
            reward = target_price - float(bar.close)
            if risk > 0 and (reward / risk) >= float(self.risk_reward):
                return Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    size_hint=0,
                    entry_price=float(bar.close),
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy=self.name,
                    regime=market_state.regime,
                    generated_at=bar.time,
                    meta={
                        "gap_pct": gap_pct,
                        "or_high": float(or_high),
                        "or_low": float(or_low),
                    },
                )

        return None
