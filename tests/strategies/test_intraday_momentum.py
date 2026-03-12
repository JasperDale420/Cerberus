"""Tests for IntradayMomentumStrategy (Gao et al. 2018 design).

These tests validate the power-hour entry logic with morning/first-half
return tracking and regime gating.
"""

from collections import deque
from datetime import datetime, timezone

import pytest

from src.core.domain import (
    Bar,
    LiquidityRegime,
    MarketRegimeSnapshot,
    MarketState,
    OrderSide,
    Regime,
    RiskRegime,
    SessionRegime,
    SymbolState,
    TrendRegime,
    VolRegime,
)
from src.core.logger import StructuredLogger
from src.strategies.intraday_momentum import IntradayMomentumStrategy


@pytest.fixture
def strategy():
    config = {
        "min_move_threshold": 0.003,
        "confluence_threshold": 40.0,  # low threshold to make signals easier to trigger
        "stop_atr_mult": 1.5,
    }
    logger = StructuredLogger("TestLogger")
    return IntradayMomentumStrategy(config, logger)


def _make_regime_snapshot(
    trend=TrendRegime.UP,
    vol=VolRegime.NORMAL,
    session=SessionRegime.POWER_HOUR,
):
    return MarketRegimeSnapshot(
        time=datetime(2023, 1, 3, 19, 30, tzinfo=timezone.utc),
        index_symbol="SPY",
        vol_symbol="VXX",
        trend=trend,
        vol=vol,
        liquidity=LiquidityRegime.GOOD,
        risk=RiskRegime.RISK_ON,
        session=session,
    )


def _make_bar(symbol, utc_hour, utc_minute, open_p, close_p, volume=1000):
    """Create a bar at a specific UTC time."""
    return Bar(
        symbol=symbol,
        time=datetime(2023, 1, 3, utc_hour, utc_minute, tzinfo=timezone.utc),
        open=open_p,
        high=max(open_p, close_p) + 0.5,
        low=min(open_p, close_p) - 0.5,
        close=close_p,
        volume=volume,
    )


def _build_session_with_returns(strategy, symbol, morning_ret_positive=True):
    """Feed morning and first-half bars to establish session returns.

    Morning (9:30-10:00 ET = 14:30-15:00 UTC) and first half (9:30-12:30 ET = 14:30-17:30 UTC).
    """
    # Market state without regime (session tracking doesn't need it)
    ms = MarketState(
        time=datetime(2023, 1, 3, 14, 30, tzinfo=timezone.utc),
        regime=Regime.BULL,
    )

    bars_list = []
    if morning_ret_positive:
        # Morning: open 100, close 101 → +1% return
        morning_bar = _make_bar(symbol, 14, 35, 100.0, 101.0)
        # Mid-morning continuation
        midday_bar = _make_bar(symbol, 16, 0, 101.0, 101.5)
        # End of first half: close at 102 → ~+2% from open
        first_half_end = _make_bar(symbol, 17, 25, 101.5, 102.0)
    else:
        # Morning: open 100, close 99 → -1% return
        morning_bar = _make_bar(symbol, 14, 35, 100.0, 99.0)
        # Mid-morning continuation
        midday_bar = _make_bar(symbol, 16, 0, 99.0, 98.5)
        # End of first half: close at 98 → ~-2% from open
        first_half_end = _make_bar(symbol, 17, 25, 98.5, 98.0)

    ss = SymbolState(
        symbol=symbol,
        bars=deque([morning_bar] * 30),
        indicators={"atr_14": 1.5},
        meta={"avg_volume_20": 500},
    )

    for b in [morning_bar, midday_bar, first_half_end]:
        strategy.on_bar(symbol, b, ss, ms)
        bars_list.append(b)

    return ss, bars_list


def test_itsm_long_signal(strategy):
    """Bullish morning+first-half returns should produce a BUY signal in power hour."""
    symbol = "AAPL"
    ss, _ = _build_session_with_returns(strategy, symbol, morning_ret_positive=True)

    # Power hour bar (14:30 ET = 19:30 UTC)
    power_bar = _make_bar(symbol, 19, 30, 102.0, 102.5)
    ss.bars_1m.append(power_bar)

    ms = MarketState(
        time=power_bar.time,
        regime=Regime.BULL,
        regime_snapshot=_make_regime_snapshot(
            trend=TrendRegime.UP,
            session=SessionRegime.POWER_HOUR,
        ),
    )

    signal = strategy.on_bar(symbol, power_bar, ss, ms)
    assert signal is not None
    assert signal.side == OrderSide.BUY
    assert signal.stop_price < power_bar.close
    assert signal.target_price > power_bar.close


def test_itsm_short_signal(strategy):
    """Bearish morning+first-half returns should produce a SELL signal in power hour."""
    symbol = "AAPL"
    ss, _ = _build_session_with_returns(strategy, symbol, morning_ret_positive=False)

    # Power hour bar
    power_bar = _make_bar(symbol, 19, 30, 98.0, 97.5)
    ss.bars_1m.append(power_bar)

    ms = MarketState(
        time=power_bar.time,
        regime=Regime.BEAR,
        regime_snapshot=_make_regime_snapshot(
            trend=TrendRegime.DOWN,
            session=SessionRegime.POWER_HOUR,
        ),
    )

    signal = strategy.on_bar(symbol, power_bar, ss, ms)
    assert signal is not None
    assert signal.side == OrderSide.SELL
    assert signal.stop_price > power_bar.close
    assert signal.target_price < power_bar.close


def test_itsm_no_signal_outside_power_hour(strategy):
    """No signal should be generated outside the 14:30-15:50 ET window."""
    symbol = "AAPL"
    ss, _ = _build_session_with_returns(strategy, symbol, morning_ret_positive=True)

    # 10:00 ET = 15:00 UTC — too early for power hour entry
    early_bar = _make_bar(symbol, 15, 0, 102.0, 102.5)

    ms = MarketState(
        time=early_bar.time,
        regime=Regime.BULL,
        regime_snapshot=_make_regime_snapshot(session=SessionRegime.OPENING),
    )

    signal = strategy.on_bar(symbol, early_bar, ss, ms)
    assert signal is None


def test_itsm_no_signal_shock_vol(strategy):
    """No signal when volatility regime is SHOCK."""
    symbol = "AAPL"
    ss, _ = _build_session_with_returns(strategy, symbol, morning_ret_positive=True)

    power_bar = _make_bar(symbol, 19, 30, 102.0, 102.5)

    ms = MarketState(
        time=power_bar.time,
        regime=Regime.BULL,
        regime_snapshot=_make_regime_snapshot(vol=VolRegime.SHOCK),
    )

    signal = strategy.on_bar(symbol, power_bar, ss, ms)
    assert signal is None
