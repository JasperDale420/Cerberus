"""Unit tests for multi-timeframe bar aggregation in backtest runner."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.backtest.runner import _maybe_aggregate_bar
from src.core.domain import Bar, SymbolState


def _make_bar(minute: int, close: float = 100.0) -> Bar:
    """Create a 1m bar at the given minute of hour 14 on 2024-01-02."""
    return Bar(
        symbol="SPY",
        time=datetime(2024, 1, 2, 14, minute, tzinfo=timezone.utc),
        open=close - 0.1,
        high=close + 0.2,
        low=close - 0.3,
        close=close,
        volume=1000.0,
        vwap=close,
    )


@pytest.mark.unit
def test_maybe_aggregate_bar_emits_5m_bar_on_boundary_crossing() -> None:
    """After 5 consecutive 1m bars (minute 0-4), crossing to minute 5
    should emit one aggregated 5m bar on the target deque."""
    state = SymbolState(symbol="SPY")
    target = state.bars_5m

    # Feed minutes 0 through 4 (all belong to the 14:00 5m bucket)
    for m in range(5):
        _maybe_aggregate_bar(state, _make_bar(m, close=100.0 + m), 5, target)

    # No bar emitted yet — the 14:00 bucket is still accumulating
    assert len(target) == 0

    # Feed minute 5 — this is a new bucket (14:05), so the 14:00 bar is flushed
    _maybe_aggregate_bar(state, _make_bar(5, close=105.0), 5, target)

    assert len(target) == 1
    agg = target[0]
    assert agg.symbol == "SPY"
    assert agg.time.minute == 0  # 14:00 bucket
    assert agg.open == pytest.approx(99.9)  # first bar's open
    assert agg.close == pytest.approx(104.0)  # last bar's close (m=4)
    assert agg.high == pytest.approx(104.2)  # max high across bars
    assert agg.low == pytest.approx(99.7)  # min low across bars
    assert agg.volume == pytest.approx(5000.0)  # sum of 5 bars


@pytest.mark.unit
def test_maybe_aggregate_bar_emits_15m_bars() -> None:
    """15m aggregation should emit a bar after crossing 15-minute boundaries."""
    state = SymbolState(symbol="SPY")
    target = state.bars_15m

    # Feed 15 bars (minute 0-14) for the 14:00 bucket
    for m in range(15):
        _maybe_aggregate_bar(state, _make_bar(m, close=100.0 + m), 15, target)

    assert len(target) == 0  # still accumulating

    # Minute 15 crosses to a new bucket
    _maybe_aggregate_bar(state, _make_bar(15, close=115.0), 15, target)

    assert len(target) == 1
    agg = target[0]
    assert agg.time.minute == 0  # 14:00 bucket
    assert agg.volume == pytest.approx(15000.0)


@pytest.mark.unit
def test_maybe_aggregate_bar_multiple_buckets() -> None:
    """Verify that multiple 5m bars are emitted across several buckets."""
    state = SymbolState(symbol="AAPL")
    target = state.bars_5m

    # Feed 16 bars (minute 0-15 → should flush buckets at m=5, m=10, m=15)
    for m in range(16):
        _maybe_aggregate_bar(state, _make_bar(m, close=100.0), 5, target)

    # Buckets 14:00 (flushed at m=5), 14:05 (flushed at m=10), 14:10 (flushed at m=15)
    assert len(target) == 3
    assert target[0].time.minute == 0
    assert target[1].time.minute == 5
    assert target[2].time.minute == 10
