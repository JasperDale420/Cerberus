from collections import deque
from datetime import datetime, timezone

import pytest

from src.analysis.regime import Regime
from src.core.domain import Bar, OrderSide
from src.core.logger import StructuredLogger
from src.strategies.base import MarketState, SymbolState
from src.strategies.vwap_reversion import VWAPReversionStrategy

# Helper to create a Bar with defaults
MARKET_TIME = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)
PREMARKET_TIME = datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc)  # 07:00 ET


def create_bar(close, volume=100, t: datetime = MARKET_TIME):
    return Bar(
        symbol="TEST",
        time=t,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


@pytest.mark.unit
def test_vwap_reversion_signal_long():
    logger = StructuredLogger("test")
    strategy = VWAPReversionStrategy(
        {"band_sigma": 2.0, "confirmation": "none"}, logger
    )

    # Setup state
    symbol = "AAPL"
    bars = deque([create_bar(100) for _ in range(20)])  # Stable price
    symbol_state = SymbolState(
        symbol=symbol,
        bars=bars,
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    market_state = MarketState(time=MARKET_TIME, regime=Regime.CHOP)

    # Current price drops significantly below mean (100)
    # Mean=100, Std=0 (approx), so any drop might trigger if std was non-zero.
    # Let's make the history have some variance so bands are valid.

    bars = deque(
        [create_bar(100 + (i % 2) * 2) for i in range(20)]
    )  # 100, 102, 100, 102...
    # Mean approx 101, Std approx 1.
    # Lower band approx 101 - 2*1 = 99.

    symbol_state.bars = bars

    # New bar at 98 (below lower band)
    new_bar = create_bar(98)

    signal = strategy.on_bar(symbol, new_bar, symbol_state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert (
        signal.reason == "price_below_lower_band" if hasattr(signal, "reason") else True
    )  # Check meta if reason not on object


@pytest.mark.unit
def test_vwap_reversion_no_signal_wrong_regime():
    logger = StructuredLogger("test")
    strategy = VWAPReversionStrategy({"confirmation": "none"}, logger)

    symbol = "AAPL"
    bars = deque([create_bar(100) for _ in range(20)])
    symbol_state = SymbolState(
        symbol=symbol,
        bars=bars,
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    market_state = MarketState(time=MARKET_TIME, regime=Regime.BULL)  # Wrong regime

    new_bar = create_bar(90)  # Deep drop

    signal = strategy.on_bar(symbol, new_bar, symbol_state, market_state)

    assert signal is None


@pytest.mark.unit
def test_vwap_reversion_signal_short():
    logger = StructuredLogger("test")
    strategy = VWAPReversionStrategy(
        {"band_sigma": 2.0, "confirmation": "none"}, logger
    )

    # Setup state with variance
    bars = deque(
        [create_bar(100 + (i % 2) * 2) for i in range(20)]
    )  # 100, 102... Mean~101, Std~1
    symbol_state = SymbolState(
        symbol="AAPL",
        bars=bars,
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    market_state = MarketState(time=MARKET_TIME, regime=Regime.CHOP)

    # Upper band approx 101 + 2*1 = 103.

    # New bar at 104 (above upper band)
    new_bar = create_bar(104)

    signal = strategy.on_bar("AAPL", new_bar, symbol_state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.SELL


@pytest.mark.unit
def test_vwap_reversion_respects_time_window_et():
    logger = StructuredLogger("test")
    strategy = VWAPReversionStrategy(
        {
            "band_sigma": 2.0,
            "confirmation": "none",
            "time_window_start": "09:45",
            "time_window_end": "15:45",
        },
        logger,
    )

    symbol = "AAPL"
    bars = deque([create_bar(100 + (i % 2) * 2, t=PREMARKET_TIME) for i in range(20)])
    symbol_state = SymbolState(
        symbol=symbol,
        bars=bars,
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    market_state = MarketState(time=PREMARKET_TIME, regime=Regime.CHOP)

    # Would otherwise trigger a long, but should be blocked by ET time window.
    new_bar = create_bar(98, t=PREMARKET_TIME)
    signal = strategy.on_bar(symbol, new_bar, symbol_state, market_state)
    assert signal is None
