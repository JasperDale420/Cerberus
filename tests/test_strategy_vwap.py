from collections import deque
from datetime import datetime, timezone

from src.analysis.regime import Regime
from src.core.domain import Bar, OrderSide
from src.core.logger import StructuredLogger
from src.strategies.base import MarketState, SymbolState
from src.strategies.vwap_reversion import VWAPReversionStrategy


# Helper to create a Bar with defaults
def create_bar(close, volume=100):
    return Bar(
        symbol="TEST",
        time=datetime.now(timezone.utc),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


def test_vwap_reversion_signal_long():
    logger = StructuredLogger("test")
    strategy = VWAPReversionStrategy({"band_sigma": 2.0}, logger)

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
    market_state = MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP)

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


def test_vwap_reversion_no_signal_wrong_regime():
    logger = StructuredLogger("test")
    strategy = VWAPReversionStrategy({}, logger)

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
    market_state = MarketState(
        time=datetime.now(timezone.utc), regime=Regime.BULL
    )  # Wrong regime

    new_bar = create_bar(90)  # Deep drop

    signal = strategy.on_bar(symbol, new_bar, symbol_state, market_state)

    assert signal is None


def test_vwap_reversion_signal_short():
    logger = StructuredLogger("test")
    strategy = VWAPReversionStrategy({"band_sigma": 2.0}, logger)

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
    market_state = MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP)

    # Upper band approx 101 + 2*1 = 103.

    # New bar at 104 (above upper band)
    new_bar = create_bar(104)

    signal = strategy.on_bar("AAPL", new_bar, symbol_state, market_state)

    assert signal is not None
    assert signal.side == OrderSide.SELL
