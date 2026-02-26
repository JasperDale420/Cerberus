from datetime import datetime, timezone

import pytest

from src.core.domain import Bar, MarketState, OrderSide, Regime, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.intraday_momentum import IntradayMomentumStrategy


@pytest.fixture
def strategy():
    config = {"min_adx": 25.0, "min_trend_strength": 1.5, "stop_atr_mult": 1.5, "target_atr_mult": 3.0}
    logger = StructuredLogger("TestLogger")
    return IntradayMomentumStrategy(config, logger)


@pytest.fixture
def bar():
    return Bar(
        symbol="AAPL",
        time=datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc),
        open=150.0,
        high=151.0,
        low=149.0,
        close=150.5,
        volume=1000,
    )


@pytest.fixture
def market_state():
    return MarketState(
        time=datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc),
        index_price=400.0,
        regime=Regime.BULL,
        meta={},
    )


def test_itsm_long_signal(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["intraday_momentum"],
        meta={"adx": 30.0, "ema_trend_strength": 2.0, "atr": 2.0, "orb_high": 150.0, "orb_low": 140.0},
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.stop_price == 150.5 - (1.5 * 2.0)
    assert signal.target_price == 150.5 + (3.0 * 2.0)
    assert signal.meta["adx"] == 30.0


def test_itsm_short_signal(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["intraday_momentum"],
        meta={"adx": 30.0, "ema_trend_strength": -2.0, "atr": 2.0, "orb_low": 151.0, "orb_high": 160.0},
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is not None
    assert signal.side == OrderSide.SELL
    assert signal.stop_price == 150.5 + (1.5 * 2.0)
    assert signal.target_price == 150.5 - (3.0 * 2.0)
    assert signal.meta["adx"] == 30.0


def test_itsm_no_signal_weak_adx(strategy, bar, market_state):
    from collections import deque

    symbol_state = SymbolState(
        symbol="AAPL",
        bars=deque([bar] * 20),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["intraday_momentum"],
        meta={"adx": 20.0, "ema_trend_strength": 2.0, "atr": 2.0, "orb_high": 150.0, "orb_low": 140.0},
    )

    signal = strategy.on_bar("AAPL", bar, symbol_state, market_state)
    assert signal is None
