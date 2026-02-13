from collections import deque
from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock

import pytest
import pytz

from src.backtest.stats import BacktestAnalyzer
from src.core.domain import Bar, MarketState, OrderSide, Regime, RiskMode, Signal, SymbolState
from src.strategies.gap_fill import GapFillStrategy


def _build_gap_up_state(
    open_dt: datetime,
    or_high: float,
    or_low: float,
    open_price: float,
    gap_pct: float,
) -> SymbolState:
    bars = deque(
        [
            Bar("AAPL", open_dt, open_price, or_high, or_low, open_price, 1000),
            Bar(
                "AAPL",
                open_dt + timedelta(minutes=5),
                open_price,
                or_high,
                or_low,
                open_price,
                1000,
            ),
        ],
        maxlen=100,
    )
    return SymbolState("AAPL", bars, {}, None, {}, [], {"gap_pct": gap_pct})


def _fills_from_signals(signals: List[Signal], profits_per_share: List[float]) -> List[dict]:
    fills: List[dict] = []
    now = datetime(2023, 10, 23, 10, 0, tzinfo=pytz.UTC)
    for idx, sig in enumerate(signals):
        entry_price = sig.entry_price
        profit_per_share = profits_per_share[idx]
        if sig.side == OrderSide.SELL:
            exit_price = entry_price - profit_per_share
            entry_side = "sell"
            exit_side = "buy"
        else:
            exit_price = entry_price + profit_per_share
            entry_side = "buy"
            exit_side = "sell"
        fills.append(
            {
                "symbol": sig.symbol,
                "side": entry_side,
                "qty": 10,
                "fill_price": entry_price,
                "filled_at": now + timedelta(minutes=idx),
                "strategy": sig.strategy,
            }
        )
        fills.append(
            {
                "symbol": sig.symbol,
                "side": exit_side,
                "qty": 10,
                "fill_price": exit_price,
                "filled_at": now + timedelta(minutes=idx + 1),
                "strategy": sig.strategy,
            }
        )
    return fills


@pytest.mark.unit
def test_gap_fill_blocks_small_opening_range() -> None:
    logger = MagicMock()
    config = {
        "min_gap": 0.02,
        "max_gap": 0.05,
        "risk_reward": 0.5,
        "or_time_minutes": 15,
        "min_or_range_pct": 0.005,
    }
    strat = GapFillStrategy(config, logger)

    et = pytz.timezone("US/Eastern")
    open_dt = et.localize(datetime(2023, 10, 23, 9, 30, 0))
    state = _build_gap_up_state(open_dt, or_high=101.2, or_low=101.0, open_price=101.1, gap_pct=0.03)

    bar = Bar("AAPL", open_dt + timedelta(minutes=20), 101.1, 101.1, 100.8, 100.9, 500)
    market = MarketState(time=bar.time, regime=Regime.CHOP, risk_mode=RiskMode.NORMAL)

    sig = strat.on_bar("AAPL", bar, state, market)

    assert sig is None


@pytest.mark.unit
def test_gap_fill_or_range_filter_improves_backtest_pnl() -> None:
    logger = MagicMock()
    baseline = GapFillStrategy(
        {
            "min_gap": 0.02,
            "max_gap": 0.05,
            "risk_reward": 0.5,
            "or_time_minutes": 15,
            "min_or_range_pct": 0.0,
        },
        logger,
    )
    filtered = GapFillStrategy(
        {
            "min_gap": 0.02,
            "max_gap": 0.05,
            "risk_reward": 0.5,
            "or_time_minutes": 15,
            "min_or_range_pct": 0.005,
        },
        logger,
    )

    et = pytz.timezone("US/Eastern")
    open_dt = et.localize(datetime(2023, 10, 23, 9, 30, 0))

    small_range_state = _build_gap_up_state(open_dt, or_high=101.2, or_low=101.0, open_price=101.1, gap_pct=0.03)
    large_range_state = _build_gap_up_state(open_dt, or_high=103.0, or_low=101.0, open_price=102.0, gap_pct=0.03)

    small_range_bar = Bar("AAPL", open_dt + timedelta(minutes=20), 101.1, 101.1, 100.8, 100.9, 500)
    large_range_bar = Bar("AAPL", open_dt + timedelta(minutes=20), 102.0, 102.0, 100.5, 100.8, 500)
    market = MarketState(time=small_range_bar.time, regime=Regime.CHOP, risk_mode=RiskMode.NORMAL)

    baseline_signals = [
        s
        for s in [
            baseline.on_bar("AAPL", small_range_bar, small_range_state, market),
            baseline.on_bar("AAPL", large_range_bar, large_range_state, market),
        ]
        if s is not None
    ]
    filtered_signals = [
        s
        for s in [
            filtered.on_bar("AAPL", small_range_bar, small_range_state, market),
            filtered.on_bar("AAPL", large_range_bar, large_range_state, market),
        ]
        if s is not None
    ]

    baseline_fills = _fills_from_signals(baseline_signals, [-2.0, 2.0])
    filtered_fills = _fills_from_signals(filtered_signals, [2.0])

    analyzer = BacktestAnalyzer(initial_cash=10000.0)
    baseline_stats = analyzer.calculate_statistics(baseline_fills, {})
    filtered_stats = analyzer.calculate_statistics(filtered_fills, {})

    assert baseline_stats["total_closed_pnl"] < filtered_stats["total_closed_pnl"]
