from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytz

from src.core.domain import Bar, MarketState, OrderSide, Regime, SymbolState
from src.strategies.vwap_reversion import VWAPReversionStrategy


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def strategy(mock_logger):
    config = {"band_sigma": 2.0, "rsi_len": 2, "confirmation": "rsi"}
    return VWAPReversionStrategy(config, mock_logger)


@pytest.mark.unit
def test_initialization(strategy):
    assert strategy.band_sigma == 2.0
    assert strategy.rsi_len == 2
    assert strategy.confirmation == "rsi"


@pytest.mark.unit
def test_time_window_check(strategy):
    # US Equities 9:30 - 16:00. Default window 9:45-15:45.
    et = pytz.timezone("US/Eastern")

    # 9:30 - Too early
    dt = et.localize(datetime(2023, 1, 1, 9, 30))
    assert strategy._in_time_window(dt) is False

    # 10:00 - OK
    dt = et.localize(datetime(2023, 1, 1, 10, 0))
    assert strategy._in_time_window(dt) is True

    # 16:00 - Too late
    dt = et.localize(datetime(2023, 1, 1, 16, 0))
    assert strategy._in_time_window(dt) is False


@pytest.mark.unit
def test_rsi_logic(strategy):
    # Not enough data
    assert strategy._rsi(np.array([100.0]), 2) is None

    # Flat -> RSI ? (Avg Gain 0, Avg Loss 0).
    # If standard RSI:
    # diffs: 0. gains 0. losses 0.
    # Logic in code: avg_loss == 0 -> 100.
    closes = np.array([100.0, 100.0, 100.0, 100.0])
    assert strategy._rsi(closes, 2) == 100.0

    # Up move -> Gain. Loss 0. -> 100.
    closes = np.array([100.0, 101.0, 102.0, 103.0])
    # diffs: 1, 1, 1. Avg Gain 1. Avg Loss 0.
    assert strategy._rsi(closes, 2) == 100.0

    # Down move. Gain 0. Loss 1. RS = 0. RSI = 100 - 100 = 0.
    closes = np.array([100.0, 99.0, 98.0, 97.0])
    assert strategy._rsi(closes, 2) == 0.0


@pytest.mark.unit
def test_confirm_reversal_none(mock_logger):
    strat = VWAPReversionStrategy({"confirmation": "none"}, mock_logger)
    ok, meta = strat._confirm_reversal(np.array([100.0]), OrderSide.BUY)
    assert ok is True
    assert meta["confirmation"] == "none"


@pytest.mark.unit
def test_confirm_reversal_unsupported(mock_logger):
    strat = VWAPReversionStrategy({"confirmation": "magic"}, mock_logger)
    ok, meta = strat._confirm_reversal(np.array([100.0]), OrderSide.BUY)
    assert ok is False
    assert meta["error"] == "unsupported_confirmation"


@pytest.mark.unit
def test_confirm_reversal_rsi_insufficient_data(strategy):
    # Need len + 2 = 4 bars.
    closes = np.array([100.0, 101.0])
    ok, meta = strategy._confirm_reversal(closes, OrderSide.BUY)
    assert ok is False
    assert meta["error"] == "insufficient_history"


@pytest.mark.unit
def test_on_bar_ignored_regime(strategy):
    market = MarketState(time=datetime.now(timezone.utc), regime=Regime.BULL)
    assert strategy.on_bar("A", MagicMock(), MagicMock(), market) is None


@pytest.mark.unit
def test_on_bar_not_enough_bars(strategy):
    market = MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP)
    state = MagicMock()
    state.bars = [MagicMock()] * 10  # < 20
    assert strategy.on_bar("A", MagicMock(), state, market) is None


@pytest.mark.unit
def test_signal_generation_buy(strategy):
    # CHOP, 20 bars. Price < Lower Band. RSI Oversold confirmation.
    # RSI Oversold threshold 10.
    # Prev RSI < 10. Curr RSI > 10.

    market = MarketState(
        time=datetime(2023, 1, 1, 15, 0, tzinfo=timezone.utc), regime=Regime.CHOP
    )

    # Mock Bars
    # VWAP = 100. Std = 1. Lower Band (2sigma) = 98.
    # Current Close = 97. (Below Lower)

    # Need sequences to trigger RSI reversal.
    # Length 2.
    # [-4] 100
    # [-3] 80 (Down 20). Loss 20. RSI < 10.
    # [-2] 70 (Down 10). Loss 10.
    # [-1] 75 (Up 5). Gain 5.

    # Actually just mocking return of _rsi might be easier if I spy, but I want to cover code.
    # Let's verify RSI logic triggers.

    # Prev RSI (from -3 to -1):
    # Diffs: [80-100=-20, 70-80=-10]. Avg Loss = 15. Avg Gain 0. RSI = 0. (< 10)

    # Curr RSI (from -2 to 0):
    # Diffs: [70-80=-10, 75-70=5]. Avg Loss 10?? No.
    # RSI on last 2 bars.
    # sequence: 100, 80, 70, 75.

    # Prev (-1 excluded): 100, 80, 70. Diffs: -20, -10. AvgLoss 15. RSI 0.
    # Curr: 80, 70, 75. Diffs: -10, +5. AvgLoss 5. AvgGain 2.5. RS=0.5. RSI=33. (> 10).
    # Reversal confirmed!

    prices = [100.0] * 17 + [100.0, 80.0, 70.0, 75.0]
    bars = []
    for p in prices:
        bars.append(Bar("A", datetime.now(), p, p, p, p, 100))

    state = SymbolState("A", deque(bars, maxlen=100), {}, None, {}, [], {})
    state.bars = deque(bars, maxlen=100)

    # VWAP calc.
    # vwap provided in state or calculated.
    # Let's provide it in indicators logic mock or let it calc.
    # Avg price approx 95?
    # Std dev huge.
    # We need price < Lower Band.
    # If Std Dev is large, bands are wide.
    # We need price 75 to be < Lower Band.
    # If VWAP is 95. Lower = 95 - 2*Std.
    # We want Lower > 75.
    # 95 - 2*Std > 75 => 20 > 2*Std => 10 > Std.
    # But Std of [100...80, 70, 75] is approx 8. So valid.

    current_bar = bars[-1]

    sig = strategy.on_bar("A", current_bar, state, market)
    assert sig is not None
    assert sig.side == OrderSide.BUY
    assert sig.meta["confirmation"] == "rsi"


@pytest.mark.unit
def test_signal_generation_sell(strategy):
    # RSI Overbought 90.
    # Prev > 90. Curr < 90.
    # Sequence: 100, 120, 130, 125.
    # Prev (100,120,130): Gains 20, 10. AvgGain 15. Loss 0. RSI 100. (>90)
    # Curr (120,130,125): Gains 10. Loss 5. AvgGain 5. AvgLoss 2.5. RS 2. RSI = 100 - 33 = 66. (<90).
    # Reversal confirmed.

    market = MarketState(
        time=datetime(2023, 1, 1, 15, 0, tzinfo=timezone.utc), regime=Regime.CHOP
    )

    prices = [100.0] * 17 + [100.0, 120.0, 130.0, 125.0]
    bars = []
    for p in prices:
        bars.append(Bar("A", datetime.now(), p, p, p, p, 100))

    state = SymbolState("A", deque(bars, maxlen=100), {}, None, {}, [], {})
    state.bars = deque(bars, maxlen=100)

    current_bar = bars[-1]
    # VWAP approx 105. Std approx 10.
    # Upper Band = 105 + 20 = 125.
    # Current 125. Not strictly > Upper (125 > 125 False).
    # Need price slightly higher or band lower.
    # Decrease sigma to 1.5.
    strategy.band_sigma = 1.0

    sig = strategy.on_bar("A", current_bar, state, market)
    assert sig is not None
    assert sig.side == OrderSide.SELL
