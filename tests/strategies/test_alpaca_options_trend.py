from collections import deque
from datetime import UTC, datetime, timedelta

import pytest

from src.core.domain import Bar, MarketState, OrderSide, Regime, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.alpaca_options_trend import AlpacaOptionsTrend


@pytest.fixture
def logger():
    return StructuredLogger("TestLogger")


@pytest.fixture
def market_state():
    return MarketState(
        time=datetime(2023, 1, 1, 10, 0, tzinfo=UTC),
        index_price=400.0,
        regime=Regime.BULL,
        meta={},
    )


def _make_bar(symbol: str, close: float, offset_minutes: int = 0) -> Bar:
    return Bar(
        symbol=symbol,
        time=datetime(2023, 1, 1, 10, 0, tzinfo=UTC) + timedelta(minutes=offset_minutes),
        open=close - 1.0,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1000,
    )


def _create_symbol_state_with_bars(bars: list[Bar]) -> SymbolState:
    d = deque(bars, maxlen=500)
    return SymbolState(
        symbol="SPY",
        bars=d,
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["alpaca_options_trend"],
        meta={},
    )


def test_options_trend_no_signal_insufficient_bars(logger, market_state):
    """No signal when there are fewer than 5 bars in history."""
    strategy = AlpacaOptionsTrend({}, logger)
    bar = _make_bar("SPY", 400.0)
    state = _create_symbol_state_with_bars([bar])

    signal = strategy.on_bar("SPY", bar, state, market_state)
    assert signal is None


def test_options_trend_no_signal_flat_prices(logger, market_state):
    """No signal when price momentum is below threshold."""
    strategy = AlpacaOptionsTrend({}, logger)
    # 6 bars with flat prices (0% return)
    bars = [_make_bar("SPY", 400.0, i) for i in range(6)]
    state = _create_symbol_state_with_bars(bars)
    current_bar = bars[-1]

    signal = strategy.on_bar("SPY", current_bar, state, market_state)
    assert signal is None


def test_options_trend_creates_signal_strong_momentum(logger, market_state):
    """Signal fires when price momentum exceeds threshold."""
    strategy = AlpacaOptionsTrend({"trend_threshold": 1.5}, logger)
    # Create bars where the last bar shows >1.5% return from 5 bars ago
    prices = [400.0, 401.0, 402.0, 403.0, 404.0, 408.0]  # 2% return from bar[-5]
    bars = [_make_bar("SPY", p, i) for i, p in enumerate(prices)]
    state = _create_symbol_state_with_bars(bars)
    current_bar = bars[-1]

    signal = strategy.on_bar("SPY", current_bar, state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.target_price == current_bar.close * 1.05
    assert signal.stop_price == current_bar.close * 0.95
    assert signal.meta["tape_signal"] == "bull_sweeps"
