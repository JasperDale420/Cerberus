from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

from src.core.domain import Bar, MarketState, OrderSide, Regime, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.failed_breakout import FailedBreakoutStrategy


def _make_bar(
    symbol: str,
    *,
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Bar:
    return Bar(
        symbol=symbol,
        time=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1000,
    )


def _make_state(symbol: str, bars: list[Bar], pdh: float, pdl: float) -> SymbolState:
    return SymbolState(
        symbol=symbol,
        bars=deque(bars),
        indicators={"prior_day_high": pdh, "prior_day_low": pdl},
        position=None,
        open_orders={},
        allowed_strategies=["failed_breakout"],
        meta={},
    )


def _simulate_short_pnl(signal, next_bar: Bar) -> float:
    if signal is None:
        return 0.0
    entry = float(signal.entry_price)
    stop = float(signal.stop_price)
    target = float(signal.target_price)

    if next_bar.high >= stop:
        return entry - stop
    if next_bar.low <= target:
        return entry - target
    return next_bar.close - entry


def test_failed_breakout_min_reclaim_improves_pnl() -> None:
    logger = StructuredLogger("Test", level="INFO")
    symbol = "AAPL"
    pdh = 100.0
    pdl = 90.0

    bar_prev = _make_bar(
        symbol,
        ts=datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc),
        open_=100.5,
        high=101.2,
        low=99.8,
        close=100.6,
    )
    bar_current = _make_bar(
        symbol,
        ts=datetime(2026, 2, 12, 10, 1, tzinfo=timezone.utc),
        open_=100.4,
        high=100.8,
        low=99.5,
        close=99.9,
    )
    bar_next = _make_bar(
        symbol,
        ts=datetime(2026, 2, 12, 10, 2, tzinfo=timezone.utc),
        open_=100.0,
        high=101.0,
        low=99.7,
        close=100.5,
    )

    market_state = MarketState(time=bar_current.time, regime=Regime.CHOP)
    symbol_state = _make_state(symbol, [bar_prev, bar_current], pdh, pdl)

    base_strategy = FailedBreakoutStrategy(
        {"risk_reward": 2.0, "cooldown_bars": 0, "min_reclaim_pct": 0.0},
        logger,
    )
    filter_strategy = FailedBreakoutStrategy(
        {"risk_reward": 2.0, "cooldown_bars": 0, "min_reclaim_pct": 0.002},
        logger,
    )

    base_signal = base_strategy.on_bar(symbol, bar_current, symbol_state, market_state)
    filter_signal = filter_strategy.on_bar(symbol, bar_current, symbol_state, market_state)

    assert base_signal is not None
    assert base_signal.side == OrderSide.SELL

    base_pnl = _simulate_short_pnl(base_signal, bar_next)
    filtered_pnl = _simulate_short_pnl(filter_signal, bar_next)

    assert filtered_pnl >= base_pnl
