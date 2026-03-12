"""Unit tests for IntradayMomentumStrategy (Gao et al. 2018).

Tests cover session tracking, signal generation, time gating, regime filtering,
confluence scoring, and day-reset logic.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock

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
from src.strategies.intraday_momentum import IntradayMomentumStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# January 15 2024 is EST (UTC-5).
# 9:30 ET = 14:30 UTC, 10:00 ET = 15:00 UTC, 12:30 ET = 17:30 UTC
# 14:30 ET = 19:30 UTC, 15:00 ET = 20:00 UTC, 15:50 ET = 20:50 UTC

_BASE_DATE = datetime(2024, 1, 15, tzinfo=timezone.utc)


def _utc(hour: int, minute: int = 0) -> datetime:
    """Shortcut for a UTC datetime on the base date."""
    return _BASE_DATE.replace(hour=hour, minute=minute)


def _make_logger() -> MagicMock:
    return MagicMock()


def _make_bar(
    symbol: str = "TEST",
    time: datetime | None = None,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: float = 50000.0,
) -> Bar:
    t = time or _utc(19, 30)  # default 14:30 ET
    return Bar(symbol=symbol, time=t, open=open_, high=high, low=low, close=close, volume=volume)


def _make_symbol_state(
    symbol: str = "TEST",
    indicators: dict | None = None,
    meta: dict | None = None,
) -> SymbolState:
    return SymbolState(
        symbol=symbol,
        bars_1m=deque(maxlen=1440),
        bars_5m=deque(maxlen=288),
        bars_15m=deque(maxlen=96),
        bars_1d=deque(maxlen=252),
        indicators=indicators or {"atr_14": 1.5, "adx": 30.0},
        meta=meta or {"avg_volume_20": 100000},
    )


def _make_market_state(
    trend: TrendRegime = TrendRegime.UP,
    vol: VolRegime = VolRegime.NORMAL,
    session: SessionRegime = SessionRegime.POWER_HOUR,
    bar_time: datetime | None = None,
) -> MarketState:
    t = bar_time or _utc(19, 30)
    snap = MarketRegimeSnapshot(
        time=t,
        index_symbol="SPY",
        vol_symbol="VXX",
        trend=trend,
        vol=vol,
        liquidity=LiquidityRegime.GOOD,
        risk=RiskRegime.RISK_ON,
        session=session,
    )
    return MarketState(time=t, regime=Regime.BULL, regime_snapshot=snap)


def _make_strategy(**overrides) -> IntradayMomentumStrategy:
    config = {
        "min_move_threshold": 0.003,
        "confluence_threshold": 60.0,
        "time_window_start": "14:30",
        "time_window_end": "15:50",
        "max_hold_minutes": 80,
        "stop_atr_mult": 1.5,
        "cooldown_bars": 0,
    }
    config.update(overrides)
    return IntradayMomentumStrategy(config, _make_logger())


def _feed_morning_bars(strategy: IntradayMomentumStrategy, symbol: str = "TEST") -> None:
    """Feed bars from 9:30-10:00 ET with a +0.5% morning move."""
    ss = _make_symbol_state(symbol)
    ms = _make_market_state(session=SessionRegime.OPENING)
    # 9:30 ET = 14:30 UTC — opening bar
    bar_open = _make_bar(symbol, time=_utc(14, 30), open_=100.0, close=100.1)
    strategy.on_bar(symbol, bar_open, ss, ms)
    # 9:45 ET = 14:45 UTC
    bar_mid = _make_bar(symbol, time=_utc(14, 45), open_=100.1, close=100.3)
    strategy.on_bar(symbol, bar_mid, ss, ms)
    # 9:59 ET = 14:59 UTC — last morning bar
    bar_end = _make_bar(symbol, time=_utc(14, 59), open_=100.3, close=100.5)
    strategy.on_bar(symbol, bar_end, ss, ms)


def _feed_first_half_bars(strategy: IntradayMomentumStrategy, symbol: str = "TEST") -> None:
    """Feed bars from 10:00-12:29 ET continuing the uptrend to +1%."""
    ss = _make_symbol_state(symbol)
    ms = _make_market_state(session=SessionRegime.MIDDAY)
    # 10:30 ET = 15:30 UTC
    bar1 = _make_bar(symbol, time=_utc(15, 30), open_=100.5, close=100.7)
    strategy.on_bar(symbol, bar1, ss, ms)
    # 12:00 ET = 17:00 UTC
    bar2 = _make_bar(symbol, time=_utc(17, 0), open_=100.7, close=101.0)
    strategy.on_bar(symbol, bar2, ss, ms)


def _feed_bearish_morning(strategy: IntradayMomentumStrategy, symbol: str = "TEST") -> None:
    """Feed bearish morning bars: -0.5% move."""
    ss = _make_symbol_state(symbol)
    ms = _make_market_state(session=SessionRegime.OPENING, trend=TrendRegime.DOWN)
    bar_open = _make_bar(symbol, time=_utc(14, 30), open_=100.0, close=99.8)
    strategy.on_bar(symbol, bar_open, ss, ms)
    bar_end = _make_bar(symbol, time=_utc(14, 59), open_=99.8, close=99.5)
    strategy.on_bar(symbol, bar_end, ss, ms)


def _feed_bearish_first_half(strategy: IntradayMomentumStrategy, symbol: str = "TEST") -> None:
    """Feed bearish midday bars continuing to -1%."""
    ss = _make_symbol_state(symbol)
    ms = _make_market_state(session=SessionRegime.MIDDAY, trend=TrendRegime.DOWN)
    bar1 = _make_bar(symbol, time=_utc(15, 30), open_=99.5, close=99.3)
    strategy.on_bar(symbol, bar1, ss, ms)
    bar2 = _make_bar(symbol, time=_utc(17, 0), open_=99.3, close=99.0)
    strategy.on_bar(symbol, bar2, ss, ms)


# ===========================================================================
# Tests
# ===========================================================================


class TestIntradayMomentum:
    # -----------------------------------------------------------------------
    # 1. Morning return tracking
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_morning_return_tracking(self):
        """Feed bars in the 9:30-10:00 window and verify morning return is captured."""
        strategy = _make_strategy()
        _feed_morning_bars(strategy)

        session = strategy._symbol_sessions["TEST"]
        assert session.morning_open == 100.0
        assert session.morning_close == 100.5
        r = IntradayMomentumStrategy._compute_return(session.morning_open, session.morning_close)
        assert r is not None
        assert abs(r - 0.005) < 1e-6

    # -----------------------------------------------------------------------
    # 2. First-half return tracking
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_first_half_return_tracking(self):
        """Feed bars from 9:30-12:30 and verify the full first-half return."""
        strategy = _make_strategy()
        _feed_morning_bars(strategy)
        _feed_first_half_bars(strategy)

        session = strategy._symbol_sessions["TEST"]
        assert session.first_half_open == 100.0
        assert session.first_half_close == 101.0
        r = IntradayMomentumStrategy._compute_return(session.first_half_open, session.first_half_close)
        assert r is not None
        assert abs(r - 0.01) < 1e-6

    # -----------------------------------------------------------------------
    # 3. Signal generation — bullish agreement
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_signal_bullish_agreement(self):
        """Both morning and midday positive -> BUY signal at 14:30 ET."""
        strategy = _make_strategy()
        _feed_morning_bars(strategy)
        _feed_first_half_bars(strategy)

        ss = _make_symbol_state()
        ms = _make_market_state(trend=TrendRegime.UP)
        # 14:30 ET = 19:30 UTC — power hour entry bar
        entry_bar = _make_bar("TEST", time=_utc(19, 30), open_=101.0, close=101.2, high=101.5, low=100.8)
        signal = strategy.on_bar("TEST", entry_bar, ss, ms)

        assert signal is not None
        assert signal.side == OrderSide.BUY
        assert signal.strategy == "intraday_momentum"
        assert signal.stop_price < entry_bar.close
        assert signal.target_price > entry_bar.close
        assert signal.meta["r_morning"] > 0
        assert signal.meta["r_first_half"] > 0

    # -----------------------------------------------------------------------
    # 4. Signal generation — bearish agreement
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_signal_bearish_agreement(self):
        """Both morning and midday negative -> SELL signal."""
        strategy = _make_strategy()
        _feed_bearish_morning(strategy)
        _feed_bearish_first_half(strategy)

        ss = _make_symbol_state()
        ms = _make_market_state(trend=TrendRegime.DOWN)
        entry_bar = _make_bar("TEST", time=_utc(19, 30), open_=99.0, close=98.8, high=99.2, low=98.5)
        signal = strategy.on_bar("TEST", entry_bar, ss, ms)

        assert signal is not None
        assert signal.side == OrderSide.SELL
        assert signal.stop_price > entry_bar.close
        assert signal.target_price < entry_bar.close

    # -----------------------------------------------------------------------
    # 5. No signal when morning/midday disagree
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_no_signal_when_directions_disagree(self):
        """Morning positive but midday negative -> no signal."""
        strategy = _make_strategy()
        # Feed positive morning
        _feed_morning_bars(strategy)

        # Now feed bearish midday bars (overwrite first_half_close)
        ss = _make_symbol_state()
        ms = _make_market_state(session=SessionRegime.MIDDAY, trend=TrendRegime.DOWN)
        # These bars come after morning, dragging first_half_close below open
        bar1 = _make_bar("TEST", time=_utc(15, 30), open_=100.5, close=99.5)
        strategy.on_bar("TEST", bar1, ss, ms)
        bar2 = _make_bar("TEST", time=_utc(17, 0), open_=99.5, close=99.0)
        strategy.on_bar("TEST", bar2, ss, ms)

        # Verify: morning positive, first_half negative
        session = strategy._symbol_sessions["TEST"]
        r_morning = IntradayMomentumStrategy._compute_return(session.morning_open, session.morning_close)
        r_first_half = IntradayMomentumStrategy._compute_return(session.first_half_open, session.first_half_close)
        assert r_morning > 0
        assert r_first_half < 0

        # Try entry in power hour
        ms_ph = _make_market_state(trend=TrendRegime.DOWN)
        entry_bar = _make_bar("TEST", time=_utc(19, 30), open_=99.0, close=98.8)
        signal = strategy.on_bar("TEST", entry_bar, ss, ms_ph)
        assert signal is None

    # -----------------------------------------------------------------------
    # 6. No signal before 14:30 ET
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_no_signal_before_time_window(self):
        """Bars at 13:00 ET (18:00 UTC) should not generate signals."""
        strategy = _make_strategy()
        _feed_morning_bars(strategy)
        _feed_first_half_bars(strategy)

        ss = _make_symbol_state()
        # 13:00 ET is MIDDAY session, which also fails the regime gate
        ms = _make_market_state(session=SessionRegime.MIDDAY)
        early_bar = _make_bar("TEST", time=_utc(18, 0), open_=101.0, close=101.2)
        signal = strategy.on_bar("TEST", early_bar, ss, ms)
        assert signal is None

    # -----------------------------------------------------------------------
    # 7. No signal after 15:50 ET
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_no_signal_after_time_window(self):
        """Bars at 15:55 ET (20:55 UTC) should not generate signals."""
        strategy = _make_strategy()
        _feed_morning_bars(strategy)
        _feed_first_half_bars(strategy)

        ss = _make_symbol_state()
        ms = _make_market_state(session=SessionRegime.CLOSE)
        late_bar = _make_bar("TEST", time=_utc(20, 55), open_=101.0, close=101.2)
        signal = strategy.on_bar("TEST", late_bar, ss, ms)
        assert signal is None

    # -----------------------------------------------------------------------
    # 8. Session reset on new trading day
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_session_reset_on_new_day(self):
        """When a bar arrives for a new date, session data should reset."""
        strategy = _make_strategy()
        _feed_morning_bars(strategy)

        session_day1 = strategy._symbol_sessions["TEST"]
        assert session_day1.morning_open is not None

        # Feed a bar on the next day (Jan 16) at 9:30 ET = 14:30 UTC
        next_day = datetime(2024, 1, 16, 14, 30, tzinfo=timezone.utc)
        ss = _make_symbol_state()
        ms = _make_market_state(session=SessionRegime.OPENING, bar_time=next_day)
        new_day_bar = _make_bar("TEST", time=next_day, open_=102.0, close=102.1)
        strategy.on_bar("TEST", new_day_bar, ss, ms)

        session_day2 = strategy._symbol_sessions["TEST"]
        # New session should have a different date
        assert session_day2.date != session_day1.date
        assert session_day2.morning_open == 102.0

    # -----------------------------------------------------------------------
    # 9. Confluence scoring — below threshold yields no signal
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_no_signal_below_confluence_threshold(self):
        """Setting a very high threshold should prevent signal generation."""
        strategy = _make_strategy(confluence_threshold=99.0)
        _feed_morning_bars(strategy)
        _feed_first_half_bars(strategy)

        ss = _make_symbol_state()
        ms = _make_market_state(trend=TrendRegime.UP)
        entry_bar = _make_bar("TEST", time=_utc(19, 30), open_=101.0, close=101.2, high=101.5, low=100.8)
        signal = strategy.on_bar("TEST", entry_bar, ss, ms)
        assert signal is None

    # -----------------------------------------------------------------------
    # 10. Regime gate — SHOCK volatility blocks signal
    # -----------------------------------------------------------------------

    @pytest.mark.unit
    def test_no_signal_in_shock_volatility(self):
        """SHOCK vol regime should block all entries."""
        strategy = _make_strategy()
        _feed_morning_bars(strategy)
        _feed_first_half_bars(strategy)

        ss = _make_symbol_state()
        ms = _make_market_state(vol=VolRegime.SHOCK)
        entry_bar = _make_bar("TEST", time=_utc(19, 30), open_=101.0, close=101.2, high=101.5, low=100.8)
        signal = strategy.on_bar("TEST", entry_bar, ss, ms)
        assert signal is None
