from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import pytz  # type: ignore

from src.backtest.stats import BacktestAnalyzer
from src.core.domain import Bar, MarketState, OrderSide, Regime, RiskMode, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.orb import ORBStrategy


class MockLogger(StructuredLogger):
    def __init__(self):
        """Mock implementation."""
        pass

    def info(self, msg, **kwargs):
        """Mock implementation."""
        pass

    def error(self, msg, **kwargs):
        """Mock implementation."""
        pass

    def warning(self, msg, **kwargs):
        """Mock implementation."""
        pass


@pytest.fixture
def orb_strategy():
    config = {"orb_minutes": 15, "risk_reward": 2.0, "stop_loss_pct": 0.01}
    return ORBStrategy(config, MockLogger())


def create_bar(t_str: str, o, h, low_px, c, vwap=None):
    # Interpret input as a US/Eastern session time and convert to UTC timestamps
    # to match production Alpaca bar timestamps.
    et = pytz.timezone("US/Eastern")
    dt_et = et.localize(datetime.strptime(f"2023-10-27 {t_str}", "%Y-%m-%d %H:%M:%S"))
    dt_utc = dt_et.astimezone(timezone.utc)
    return Bar(symbol="TEST", time=dt_utc, open=o, high=h, low=low_px, close=c, volume=1000, vwap=vwap)


@pytest.mark.unit
def test_orb_logic(orb_strategy):
    market_state = MarketState(
        time=datetime.now(timezone.utc),
        regime=Regime.BULL,
        index_symbol="SPY",
        index_price=100,
        index_return=0,
        realized_vol=0,
        daily_pnl=0,
        risk_mode=RiskMode.NORMAL,
        meta={},
    )
    symbol_state = SymbolState(
        symbol="TEST",
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={"gap_pct": 0.02, "flow_zscore": 3.0, "premarket_volume": 12345.0},
    )

    # 1. Opening Range (09:30 - 09:45)
    # 09:30
    b1 = create_bar("09:30:00", 100, 105, 99, 102)
    sig = orb_strategy.on_bar("TEST", b1, symbol_state, market_state)
    assert sig is None
    assert symbol_state.indicators["orb_high"] == 105
    assert symbol_state.indicators["orb_low"] == 99

    # 09:40 update
    b2 = create_bar("09:40:00", 102, 106, 101, 104)
    sig = orb_strategy.on_bar("TEST", b2, symbol_state, market_state)
    assert sig is None
    assert symbol_state.indicators["orb_high"] == 106  # New High

    # 2. Breakout (09:46)
    b3 = create_bar("09:46:00", 105, 108, 105, 107)  # Close 107 > 106

    # Check assertions again here if needed, but for now logic holds.

    # Force complete flag (logic in on_bar handles this if time >= 09:45)
    # The strategies logic for 'completion' relies on time check.
    # We call with 09:46, it should set complete=True then check breakout.

    sig = orb_strategy.on_bar("TEST", b3, symbol_state, market_state)

    assert sig is not None
    assert sig.side.value == "buy"
    assert sig.strategy == "orb"
    assert sig.stop_price == 99  # Low of range (from b1)
    assert sig.meta.get("gap_pct") == pytest.approx(0.02)
    assert sig.meta.get("flow_zscore") == pytest.approx(3.0)
    assert sig.meta.get("premarket_volume") == pytest.approx(12345.0)


@pytest.mark.unit
def test_orb_bearish_breakout(orb_strategy):
    market_state = MarketState(
        time=datetime.now(timezone.utc),
        regime=Regime.BEAR,
        index_symbol="SPY",
        index_price=100,
        index_return=0,
        realized_vol=0,
        daily_pnl=0,
        risk_mode=RiskMode.NORMAL,
        meta={},
    )
    symbol_state = SymbolState(
        symbol="TEST",
        bars=deque(),
        indicators={"orb_high": 105, "orb_low": 100, "orb_complete": True},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={"gap_pct": -0.02, "flow_zscore": -3.0, "premarket_volume": 20000.0},
    )

    # Breakout Down
    b = create_bar("09:46:00", 100, 100, 98, 99)  # Close 99 < 100

    sig = orb_strategy.on_bar("TEST", b, symbol_state, market_state)
    assert sig is not None
    assert sig.side.value == "sell"
    assert sig.meta.get("gap_pct") == pytest.approx(-0.02)
    assert sig.meta.get("flow_zscore") == pytest.approx(-3.0)
    assert sig.meta.get("premarket_volume") == pytest.approx(20000.0)


def _make_symbol_state(orb_high: float, orb_low: float) -> SymbolState:
    return SymbolState(
        symbol="TEST",
        bars=deque(),
        indicators={"orb_high": orb_high, "orb_low": orb_low, "orb_complete": True},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={
            "gap_pct": 0.02,
            "rvol": 2.0,
            "avg_volume_20": 1000.0,
        },
    )


def _fills_from_signal(signal, exit_price: float, exit_time: datetime) -> list[dict[str, Any]]:
    if signal is None:
        return []
    qty = 1000.0
    entry = {
        "symbol": signal.symbol,
        "side": signal.side.value,
        "qty": qty,
        "fill_price": signal.entry_price,
        "filled_at": signal.generated_at,
        "strategy": signal.strategy,
    }
    exit_side = "sell" if signal.side == OrderSide.BUY else "buy"
    exit_fill = {
        "symbol": signal.symbol,
        "side": exit_side,
        "qty": qty,
        "fill_price": exit_price,
        "filled_at": exit_time,
        "strategy": signal.strategy,
    }
    return [entry, exit_fill]


@pytest.mark.unit
def test_orb_min_or_range_pct_blocks_tight_range():
    strategy = ORBStrategy(
        {
            "orb_minutes": 15,
            "risk_reward": 2.0,
            "stop_loss_pct": 0.01,
            "min_or_range_pct": 0.006,
        },
        MockLogger(),
    )
    market_state = MarketState(
        time=datetime.now(timezone.utc),
        regime=Regime.BULL,
        index_symbol="SPY",
        index_price=100,
        index_return=0,
        realized_vol=0,
        daily_pnl=0,
        risk_mode=RiskMode.NORMAL,
        meta={},
    )

    symbol_state = _make_symbol_state(orb_high=100.2, orb_low=100.0)
    bar = create_bar("09:46:00", 100.1, 100.3, 100.0, 100.25, vwap=100.1)

    sig = strategy.on_bar("TEST", bar, symbol_state, market_state)
    assert sig is None


@pytest.mark.unit
def test_orb_min_or_range_pct_improves_micro_backtest_pnl():
    base_strategy = ORBStrategy(
        {
            "orb_minutes": 15,
            "risk_reward": 2.0,
            "stop_loss_pct": 0.01,
            "min_or_range_pct": 0.0,
        },
        MockLogger(),
    )
    filtered_strategy = ORBStrategy(
        {
            "orb_minutes": 15,
            "risk_reward": 2.0,
            "stop_loss_pct": 0.01,
            "min_or_range_pct": 0.006,
        },
        MockLogger(),
    )
    market_state = MarketState(
        time=datetime.now(timezone.utc),
        regime=Regime.BULL,
        index_symbol="SPY",
        index_price=100,
        index_return=0,
        realized_vol=0,
        daily_pnl=0,
        risk_mode=RiskMode.NORMAL,
        meta={},
    )

    # Low-range breakout (expected loss, filtered out by min_or_range_pct)
    low_range_state = _make_symbol_state(orb_high=100.2, orb_low=100.0)
    low_range_bar = create_bar("09:46:00", 100.1, 100.3, 100.0, 100.25, vwap=100.1)
    low_signal_base = base_strategy.on_bar("TEST", low_range_bar, low_range_state, market_state)
    assert low_signal_base is not None
    low_exit_time = low_range_bar.time + timedelta(minutes=5)
    low_fills_base = _fills_from_signal(low_signal_base, low_signal_base.stop_price, low_exit_time)
    low_signal_filtered = filtered_strategy.on_bar("TEST", low_range_bar, low_range_state, market_state)
    assert low_signal_filtered is None
    low_fills_filtered: list[dict[str, Any]] = []

    # Healthy-range breakout (expected win, retained by filter)
    high_range_state = _make_symbol_state(orb_high=101.0, orb_low=99.0)
    high_range_bar = create_bar("09:47:00", 101.0, 101.7, 100.8, 101.5, vwap=101.2)
    high_signal_base = base_strategy.on_bar("TEST", high_range_bar, high_range_state, market_state)
    assert high_signal_base is not None
    high_exit_time = high_range_bar.time + timedelta(minutes=5)
    high_fills_base = _fills_from_signal(high_signal_base, high_signal_base.target_price, high_exit_time)
    high_signal_filtered = filtered_strategy.on_bar("TEST", high_range_bar, high_range_state, market_state)
    assert high_signal_filtered is not None
    high_fills_filtered = _fills_from_signal(
        high_signal_filtered,
        high_signal_filtered.target_price,
        high_exit_time,
    )

    analyzer = BacktestAnalyzer()
    base_stats = analyzer.calculate_statistics(low_fills_base + high_fills_base)
    filtered_stats = analyzer.calculate_statistics(low_fills_filtered + high_fills_filtered)

    assert filtered_stats["total_pnl"] > base_stats["total_pnl"]
    assert filtered_stats["max_drawdown_pct"] < base_stats["max_drawdown_pct"]
