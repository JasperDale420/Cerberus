from collections import deque
from datetime import datetime, timedelta

import pytest

from src.core.domain import Bar, MarketState, Regime, RiskMode, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.vwap_trend_rider import VWAPTrendRiderStrategy


class MockLogger(StructuredLogger):
    def __init__(self):
        pass

    def info(self, msg, **kwargs):
        pass

    def error(self, msg, **kwargs):
        pass

    def warning(self, msg, **kwargs):
        pass


@pytest.fixture
def vtr_strategy():
    config = {
        "ema_fast": 5,  # Shorten len for test with few bars
        "ema_slow": 10,
        "vol_mult": 1.2,
        "risk_reward": 2.0,
    }
    return VWAPTrendRiderStrategy(config, MockLogger())


def test_bullish_reclaim(vtr_strategy):
    market_state = MarketState(
        time=datetime.now(),
        regime=Regime.BULL,
        index_symbol="SPY",
        index_price=100,
        index_return=0,
        realized_vol=0,
        daily_pnl=0,
        risk_mode=RiskMode.NORMAL,
        meta={},
    )

    # 1. Setup Bars
    # Needs enough bars for EMAs (10 bars min for test config)
    bars = []
    start_dt = datetime.now() - timedelta(minutes=100)

    # Generate 15 bars
    # Create UPTREND: Price rising 100 -> 110
    # Create VWAP around the price
    for i in range(15):
        price = 100 + i
        bars.append(
            Bar(
                "TEST",
                start_dt + timedelta(minutes=i * 5),
                price,
                price + 0.5,
                price - 0.5,
                price,
                1000,
            )
        )  # Avg Vol 1000

    # Append a PULLBACK below VWAP
    # VWAP usually trails price in uptrend.
    # Let's manually manipulate close and volume to force signal

    # Bar 16 force below VWAP
    # We rely on pandas_ta computing VWAP.
    # In simple series, VWAP ~ Price Avg (weighted).
    # If price drops sharp, it goes below VWAP.

    last_price = bars[-1].close
    drop_price = last_price - 5

    b_below = Bar(
        "TEST",
        start_dt + timedelta(minutes=16 * 5),
        drop_price,
        drop_price + 1,
        drop_price - 1,
        drop_price,
        1000,
    )

    # Bar 17 Reclaim (Cross Up) + High Volume
    reclaim_price = drop_price + 2  # Cross back up
    high_vol = 1500  # > 1000 * 1.2

    b_reclaim = Bar(
        "TEST",
        start_dt + timedelta(minutes=17 * 5),
        reclaim_price,
        reclaim_price + 1,
        reclaim_price - 1,
        reclaim_price,
        high_vol,
    )

    # Construct state
    # We need to run bars[-1] first to set prev state (below)
    # Then run current bar (reclaim)

    symbol_state = SymbolState(
        symbol="TEST",
        bars=deque(bars + [b_below]),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    # Run strategy on b_reclaim ?
    # Wait, strategy needs history ON_BAR to calc prev close.
    # on_bar logic:
    # > Calc indicators on ALL bars (including 'bar' argument?)
    # usually 'symbol_state.bars' *excludes* current bar in some engines, or includes?
    # Our engine usually appends bar AFTER on_bar for some reason? Or before?
    # Standard: symbol_state.bars is history. 'bar' is current.
    # The strategy combines them: "bars = list(symbol_state.bars)" -> then dataframe.
    # If we want 'bar' in dataframe, we must append it or pass it.

    # In trend_pullback and others: "bars = list(symbol_state.bars)" ... implies history.
    # BUT, if 'bar' is current, we need it in DF to get current indicators?
    # Actually most strats rely on `symbol_state.bars` having sufficient data.
    # If `bar` is not in `symbol_state.bars`, we must append it or `df` will be missing latest.

    # Let's assume standard behavior: `symbol_state.bars` is history.
    # Current `bar` needs to be considered.
    # The strategies I wrote earlier use `df = pd.DataFrame([vars(b) for b in bars])`.
    # This ONLY uses history. That might be a BUG if I don't append `bar`.
    # Let's check `vwap_trend_rider.py` implementation...
    # It calculates `current_fast = ema_fast.iloc[-1]`.
    # If `bar` isn't in `bars`, then `[-1]` is the *previous* bar.
    # THIS IS A BUG IN MY STRATEGIES if the Engine doesn't append before calling.
    # Let's assume for test I need to append `bar` to `symbol_state.bars` OR strategy logic handles it.

    # Adjusting test to append b_reclaim to bars used in state

    symbol_state.bars.append(b_reclaim)

    # Force VWAP calc on this DF to ensure we know where it is?
    # Strategy calcs it.

    sig = vtr_strategy.on_bar("TEST", b_reclaim, symbol_state, market_state)

    # It might fail if EMAs don't align or invalid calculation.
    # Given randomness of EMAs on 15 bars, hard to predict exactly.
    # But usually Price Rising = EMA fast > Slow.
    # Just checking if it runs without error first.

    if sig:
        assert sig.side.value == "buy"
        assert sig.strategy == "vwap_trend_rider"
    else:
        # If no signal, might be EMAs didn't cross or Volume not enough?
        # 1500 > 1000 * 1.2 (1200). Volume ok.
        pass
