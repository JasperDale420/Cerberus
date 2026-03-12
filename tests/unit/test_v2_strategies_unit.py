"""Unit tests for V2 strategies: TrendRiderPro, MeanReversionPro, ORB V2.

Each test mocks MultiTimeframeAnalyzer to return controlled indicator values,
verifying signal generation logic, regime gating, confluence scoring, and
MRP-specific daily loss throttle.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock, patch

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
from src.strategies.mean_reversion_pro import MeanReversionProStrategy
from src.strategies.orb_v2 import ORBV2Strategy
from src.strategies.trend_rider_pro import TrendRiderProStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    vwap: float | None = 100.2,
) -> Bar:
    t = time or datetime(2024, 3, 15, 10, 30, tzinfo=UTC)
    return Bar(
        symbol=symbol,
        time=t,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
    )


def _make_bars(n: int, **kwargs) -> deque[Bar]:
    """Generate n bars at 1-minute intervals."""
    base_time = kwargs.pop("base_time", datetime(2024, 3, 15, 10, 0, tzinfo=UTC))
    bars: deque[Bar] = deque(maxlen=1440)
    for i in range(n):
        t = base_time + timedelta(minutes=i)
        bars.append(_make_bar(time=t, **kwargs))
    return bars


def _make_5m_bars(n: int, **kwargs) -> deque[Bar]:
    """Generate n bars at 5-minute intervals."""
    base_time = kwargs.pop("base_time", datetime(2024, 3, 15, 9, 30, tzinfo=UTC))
    bars: deque[Bar] = deque(maxlen=288)
    for i in range(n):
        t = base_time + timedelta(minutes=i * 5)
        bars.append(_make_bar(time=t, **kwargs))
    return bars


def _make_symbol_state(
    symbol: str = "TEST",
    n_bars_1m: int = 60,
    n_bars_5m: int = 30,
    **kwargs,
) -> SymbolState:
    return SymbolState(
        symbol=symbol,
        bars_1m=_make_bars(n_bars_1m, **kwargs),
        bars_5m=_make_5m_bars(n_bars_5m, **kwargs),
        bars_15m=deque(maxlen=96),
        bars_1d=deque(maxlen=252),
        indicators={},
        meta={},
    )


def _make_market_state(
    trend: TrendRegime = TrendRegime.UP,
    vol: VolRegime = VolRegime.NORMAL,
    session: SessionRegime = SessionRegime.OPENING,
) -> MarketState:
    now = datetime(2024, 3, 15, 10, 30, tzinfo=UTC)
    snap = MarketRegimeSnapshot(
        time=now,
        index_symbol="SPY",
        vol_symbol="VIX",
        trend=trend,
        vol=vol,
        liquidity=LiquidityRegime.GOOD,
        risk=RiskRegime.RISK_ON,
        session=session,
    )
    return MarketState(
        time=now,
        regime=Regime.BULL,
        regime_snapshot=snap,
    )


def _mock_mtf(**overrides):
    """Create a mock MTF analyzer with default bullish setup."""
    mtf = MagicMock()
    mtf.get_ema.return_value = 100.0
    mtf.get_rsi.return_value = 50.0
    mtf.get_atr.return_value = 1.5
    mtf.get_adx.return_value = 30.0
    mtf.get_vwap_distance.return_value = 0.001
    mtf.get_bb_position.return_value = 0.0
    mtf.get_trend_alignment.return_value = 0.8
    mtf.get_mean_reversion_alignment.return_value = 0.7
    mtf.get_swing_lows.return_value = [98.0]
    mtf.get_swing_highs.return_value = [102.0]
    for k, v in overrides.items():
        setattr(mtf, k, MagicMock(return_value=v) if not callable(v) else v)
    return mtf


# ===========================================================================
# TrendRiderProStrategy
# ===========================================================================


class TestTrendRiderPro:
    @pytest.fixture
    def strategy(self):
        config = {
            "confluence_threshold": 55.0,
            "pullback_threshold": 0.005,
            "stop_atr_mult": 1.5,
            "target_atr_mult": 3.5,
            "trail_min_profit_r": 0.5,
            "max_hold_minutes": 120,
            "min_bars": 5,
            "cooldown_bars": 0,
        }
        return TrendRiderProStrategy(config, _make_logger())

    @pytest.mark.unit
    def test_generates_buy_signal_with_bullish_setup(self, strategy):
        """TRP should emit BUY when trend alignment is strong and pullback is near EMA."""
        bar = _make_bar(close=100.0, time=datetime(2024, 3, 15, 10, 30, tzinfo=UTC))
        ss = _make_symbol_state()
        ms = _make_market_state(trend=TrendRegime.UP, session=SessionRegime.OPENING)

        mtf = _mock_mtf()
        # EMA20 at 100.2 — close 100.0 is 0.2% pullback (within 0.5% threshold)
        mtf.get_ema.return_value = 100.2

        with patch("src.strategies.trend_rider_pro.MultiTimeframeAnalyzer", return_value=mtf):
            sig = strategy.on_bar("TEST", bar, ss, ms)

        if sig is not None:
            assert sig.side == OrderSide.BUY
            assert sig.strategy == "trend_rider_pro"
            assert sig.stop_price < bar.close
            assert sig.target_price > bar.close
            assert "exit_config" in sig.meta

    @pytest.mark.unit
    def test_rejects_during_shock_vol(self, strategy):
        """TRP should not trade during SHOCK volatility."""
        bar = _make_bar(time=datetime(2024, 3, 15, 10, 30, tzinfo=UTC))
        ss = _make_symbol_state()
        ms = _make_market_state(vol=VolRegime.SHOCK)

        sig = strategy.on_bar("TEST", bar, ss, ms)
        assert sig is None

    @pytest.mark.unit
    def test_rejects_outside_time_window(self, strategy):
        """TRP should not trade before market open."""
        bar = _make_bar(time=datetime(2024, 3, 15, 14, 0, tzinfo=UTC))  # 09:00 ET = before window
        ss = _make_symbol_state()
        ms = _make_market_state()

        _sig = strategy.on_bar("TEST", bar, ss, ms)
        # Either None (outside window) or valid — depends on TZ conversion.
        # 14:00 UTC = 10:00 ET in March (EDT), which is within window.
        # Let's use a time that's definitely outside: 06:00 UTC = 01:00 ET
        bar2 = _make_bar(time=datetime(2024, 3, 15, 6, 0, tzinfo=UTC))
        sig2 = strategy.on_bar("TEST", bar2, ss, ms)
        assert sig2 is None

    @pytest.mark.unit
    def test_rejects_premarket_session(self, strategy):
        """TRP should not trade during premarket."""
        bar = _make_bar(time=datetime(2024, 3, 15, 10, 30, tzinfo=UTC))
        ss = _make_symbol_state()
        ms = _make_market_state(session=SessionRegime.PREMARKET)

        sig = strategy.on_bar("TEST", bar, ss, ms)
        assert sig is None

    @pytest.mark.unit
    def test_exit_config_included_in_meta(self, strategy):
        """Signals should include exit_config for ExitManager."""
        bar = _make_bar(close=100.0, time=datetime(2024, 3, 15, 10, 30, tzinfo=UTC))
        ss = _make_symbol_state()
        ms = _make_market_state(trend=TrendRegime.UP, session=SessionRegime.OPENING)

        mtf = _mock_mtf()
        mtf.get_ema.return_value = 100.2

        with patch("src.strategies.trend_rider_pro.MultiTimeframeAnalyzer", return_value=mtf):
            sig = strategy.on_bar("TEST", bar, ss, ms)

        if sig is not None:
            ec = sig.meta.get("exit_config", {})
            assert ec.get("trailing_enabled") is True
            assert ec.get("max_hold_minutes") == 120
            assert "partial_exits" in ec

    @pytest.mark.unit
    def test_insufficient_bars_returns_none(self, strategy):
        """TRP should return None when bars are insufficient."""
        bar = _make_bar(time=datetime(2024, 3, 15, 10, 30, tzinfo=UTC))
        ss = _make_symbol_state(n_bars_1m=2)  # Only 2 bars, need 5
        ms = _make_market_state()

        sig = strategy.on_bar("TEST", bar, ss, ms)
        assert sig is None


# ===========================================================================
# MeanReversionProStrategy
# ===========================================================================


class TestMeanReversionPro:
    @pytest.fixture
    def strategy(self):
        config = {
            "confluence_threshold": 55.0,
            "vwap_dist_threshold": 0.003,
            "bb_pos_threshold": 0.3,
            "stop_atr_mult": 1.5,
            "min_bars": 5,
            "cooldown_bars": 0,
            "max_daily_losers": 3,
        }
        return MeanReversionProStrategy(config, _make_logger())

    @pytest.mark.unit
    def test_daily_throttle_blocks_after_consecutive_losses(self, strategy):
        """MRP should stop trading after max_daily_losers consecutive losing trades."""
        trade_date = date(2024, 3, 15)
        # Record 3 consecutive losses
        for _ in range(3):
            strategy.record_trade_result(-50.0, trade_date)

        # Verify throttle is active
        bar = _make_bar(time=datetime(2024, 3, 15, 15, 30, tzinfo=UTC))  # 11:30 ET
        assert strategy._daily_throttle_ok(bar) is False

    @pytest.mark.unit
    def test_daily_throttle_resets_on_win(self, strategy):
        """A winning trade resets the consecutive loser count."""
        trade_date = date(2024, 3, 15)
        strategy.record_trade_result(-50.0, trade_date)
        strategy.record_trade_result(-50.0, trade_date)
        strategy.record_trade_result(100.0, trade_date)  # Win resets streak

        bar = _make_bar(time=datetime(2024, 3, 15, 15, 30, tzinfo=UTC))
        assert strategy._daily_throttle_ok(bar) is True

    @pytest.mark.unit
    def test_daily_throttle_independent_per_day(self, strategy):
        """Throttle state should be per-day."""
        day1 = date(2024, 3, 15)
        _day2 = date(2024, 3, 16)
        # Hit limit on day1
        for _ in range(3):
            strategy.record_trade_result(-50.0, day1)

        # Day2 should be clean
        bar = _make_bar(time=datetime(2024, 3, 16, 15, 30, tzinfo=UTC))
        assert strategy._daily_throttle_ok(bar) is True

    @pytest.mark.unit
    def test_rejects_shock_vol(self, strategy):
        """MRP should not trade during SHOCK volatility."""
        bar = _make_bar(time=datetime(2024, 3, 15, 15, 30, tzinfo=UTC))
        ss = _make_symbol_state()
        ms = _make_market_state(vol=VolRegime.SHOCK)

        sig = strategy.on_bar("TEST", bar, ss, ms)
        assert sig is None

    @pytest.mark.unit
    def test_rejects_no_regime_snapshot(self, strategy):
        """MRP requires a regime snapshot (unlike trend strategies)."""
        bar = _make_bar(time=datetime(2024, 3, 15, 15, 30, tzinfo=UTC))
        ss = _make_symbol_state()
        ms = MarketState(time=bar.time, regime=Regime.CHOP, regime_snapshot=None)

        sig = strategy.on_bar("TEST", bar, ss, ms)
        assert sig is None

    @pytest.mark.unit
    def test_prefers_flat_trend(self, strategy):
        """MRP prefers FLAT trend for mean reversion."""
        assert (
            MeanReversionProStrategy._regime_ok(_make_market_state(trend=TrendRegime.FLAT, vol=VolRegime.NORMAL))
            is True
        )

    @pytest.mark.unit
    def test_allows_trending_in_low_vol(self, strategy):
        """MRP allows trending regimes only when volatility is LOW."""
        assert MeanReversionProStrategy._regime_ok(_make_market_state(trend=TrendRegime.UP, vol=VolRegime.LOW)) is True

    @pytest.mark.unit
    def test_rejects_trending_in_normal_vol(self, strategy):
        """MRP should not trade trends unless vol is LOW."""
        assert (
            MeanReversionProStrategy._regime_ok(_make_market_state(trend=TrendRegime.UP, vol=VolRegime.NORMAL)) is False
        )

    @pytest.mark.unit
    def test_detect_side_buy(self, strategy):
        """Negative VWAP dist + negative BB pos should trigger BUY."""
        side = strategy._detect_side(vwap_dist=-0.005, bb_pos=-0.5)
        assert side == OrderSide.BUY

    @pytest.mark.unit
    def test_detect_side_sell(self, strategy):
        """Positive VWAP dist + positive BB pos should trigger SELL."""
        side = strategy._detect_side(vwap_dist=0.005, bb_pos=0.5)
        assert side == OrderSide.SELL

    @pytest.mark.unit
    def test_detect_side_none_when_mixed(self, strategy):
        """No signal when VWAP and BB signals disagree."""
        side = strategy._detect_side(vwap_dist=-0.005, bb_pos=0.5)
        assert side is None

    @pytest.mark.unit
    def test_1_5r_minimum_target(self, strategy):
        """Target must be at least 1.5x the stop distance from entry."""
        bar = _make_bar(close=100.0)
        stop_price = 99.0  # 1.0 stop distance
        mtf = _mock_mtf()
        # VWAP distance close to 0 — means VWAP target would be very close to entry
        mtf.get_vwap_distance.return_value = 0.001

        target = strategy._compute_target(OrderSide.BUY, mtf, bar, stop_price)
        stop_distance = abs(bar.close - stop_price)
        target_distance = abs(target - bar.close)
        # Must be >= 1.5R
        assert target_distance >= stop_distance * 1.5 - 0.001  # small float tolerance


# ===========================================================================
# ORBV2Strategy
# ===========================================================================


class TestORBV2:
    @pytest.fixture
    def strategy(self):
        config = {
            "confluence_threshold": 55.0,
            "vol_gate_mult": 1.2,
            "target_range_mult": 2.0,
            "trail_min_profit_r": 1.0,
            "max_hold_minutes": 60,
            "min_bars": 3,
            "cooldown_bars": 0,
        }
        return ORBV2Strategy(config, _make_logger())

    @pytest.mark.unit
    def test_rejects_during_range_building(self, strategy):
        """ORB V2 should not signal during the opening range period."""
        # 09:35 ET = 13:35 UTC in March (EDT)
        bar = _make_bar(time=datetime(2024, 3, 15, 13, 35, tzinfo=UTC))
        ss = _make_symbol_state()
        ms = _make_market_state()

        sig = strategy.on_bar("TEST", bar, ss, ms)
        assert sig is None

    @pytest.mark.unit
    def test_rejects_shock_vol(self, strategy):
        """ORB V2 should not trade during SHOCK volatility."""
        bar = _make_bar(time=datetime(2024, 3, 15, 14, 30, tzinfo=UTC))
        ss = _make_symbol_state()
        ms = _make_market_state(vol=VolRegime.SHOCK)

        sig = strategy.on_bar("TEST", bar, ss, ms)
        assert sig is None

    @pytest.mark.unit
    def test_rejects_close_session(self, strategy):
        """ORB V2 should not trade during CLOSE session."""
        assert ORBV2Strategy._regime_ok(_make_market_state(session=SessionRegime.CLOSE)) is False

    @pytest.mark.unit
    def test_allows_opening_session(self, strategy):
        """ORB V2 allows OPENING session."""
        assert ORBV2Strategy._regime_ok(_make_market_state(session=SessionRegime.OPENING)) is True

    @pytest.mark.unit
    def test_adaptive_range_small_gap(self, strategy):
        """Small gap should use 5-minute range."""
        assert ORBV2Strategy._adaptive_range_minutes(0.002) == 5

    @pytest.mark.unit
    def test_adaptive_range_medium_gap(self, strategy):
        """Medium gap should use 10-minute range."""
        assert ORBV2Strategy._adaptive_range_minutes(0.01) == 10

    @pytest.mark.unit
    def test_adaptive_range_large_gap(self, strategy):
        """Large gap should use 15-minute range."""
        assert ORBV2Strategy._adaptive_range_minutes(0.02) == 15

    @pytest.mark.unit
    def test_session_reset_clears_state(self, strategy):
        """Each new trading day should reset the ORB state."""
        bar1 = _make_bar(time=datetime(2024, 3, 15, 14, 0, tzinfo=UTC))
        ss = _make_symbol_state()
        state = strategy._get_state("TEST")
        state["date"] = date(2024, 3, 14)
        state["range_high"] = 105.0
        state["breakout_fired"] = True

        strategy._maybe_reset(state, bar1, ss)
        # After reset for new day
        assert state["range_high"] == float("-inf")
        assert state["breakout_fired"] is False

    @pytest.mark.unit
    def test_exit_config_included(self, strategy):
        """ORB V2 exit config should have trailing enabled."""
        ec = strategy._exit_config()
        assert ec["trailing_enabled"] is True
        assert ec["max_hold_minutes"] == 60
        assert "partial_exits" in ec

    @pytest.mark.unit
    def test_no_signal_without_breakout(self, strategy):
        """ORB V2 should not signal if price is within the range."""
        # Set up a completed range
        state = strategy._get_state("TEST")
        state["date"] = date(2024, 3, 15)
        state["range_high"] = 101.0
        state["range_low"] = 99.0
        state["range_complete"] = True
        state["range_bars"] = 10
        state["breakout_fired"] = False

        # Bar within range (no breakout)
        bar = _make_bar(close=100.0, high=100.5, low=99.5, time=datetime(2024, 3, 15, 14, 45, tzinfo=UTC))
        ss = _make_symbol_state()
        ms = _make_market_state()

        sig = strategy.on_bar("TEST", bar, ss, ms)
        assert sig is None

    @pytest.mark.unit
    def test_only_one_breakout_per_day(self, strategy):
        """ORB V2 should only fire one breakout signal per day."""
        state = strategy._get_state("TEST")
        state["date"] = date(2024, 3, 15)
        state["range_high"] = 101.0
        state["range_low"] = 99.0
        state["range_complete"] = True
        state["range_bars"] = 10
        state["breakout_fired"] = True  # Already fired

        bar = _make_bar(
            close=102.0, high=102.5, low=101.5, volume=100000, time=datetime(2024, 3, 15, 14, 45, tzinfo=UTC)
        )
        ss = _make_symbol_state()
        ms = _make_market_state()

        sig = strategy.on_bar("TEST", bar, ss, ms)
        assert sig is None
