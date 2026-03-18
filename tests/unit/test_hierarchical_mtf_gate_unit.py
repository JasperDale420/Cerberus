"""Unit tests for hierarchical multi-timeframe gates in BaseStrategy.

Tests _require_higher_tf_trend() and _require_higher_tf_flat() which provide
hard gates based on the 15m timeframe — replacing the blended alignment score
for regime filtering.

TDD: these tests are written BEFORE the implementation.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
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


# ---------------------------------------------------------------------------
# Helpers (same patterns as test_v2_strategies_unit.py)
# ---------------------------------------------------------------------------


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
    return Bar(symbol=symbol, time=t, open=open_, high=high, low=low, close=close, volume=volume, vwap=vwap)


def _make_bars(n: int, interval_minutes: int = 1, **kwargs) -> deque[Bar]:
    base_time = kwargs.pop("base_time", datetime(2024, 3, 15, 10, 0, tzinfo=UTC))
    bars: deque[Bar] = deque(maxlen=1440)
    for i in range(n):
        t = base_time + timedelta(minutes=i * interval_minutes)
        bars.append(_make_bar(time=t, **kwargs))
    return bars


def _make_symbol_state(
    symbol: str = "TEST",
    n_bars_1m: int = 60,
    n_bars_5m: int = 30,
    n_bars_15m: int = 10,
    **kwargs,
) -> SymbolState:
    return SymbolState(
        symbol=symbol,
        bars_1m=_make_bars(n_bars_1m, interval_minutes=1, **kwargs),
        bars_5m=_make_bars(n_bars_5m, interval_minutes=5, **kwargs),
        bars_15m=_make_bars(n_bars_15m, interval_minutes=15, **kwargs),
        bars_1d=deque(maxlen=252),
        indicators={},
        meta={},
    )


def _mock_mtf(**overrides):
    """Create a mock MTF analyzer with controllable per-timeframe indicators."""
    mtf = MagicMock()
    # Default: all timeframes show uptrend (EMA20 > EMA50), ADX=30
    mtf.get_ema.return_value = 100.0
    mtf.get_adx.return_value = 30.0
    mtf.get_atr.return_value = 1.5
    mtf.get_rsi.return_value = 50.0
    mtf.get_vwap_distance.return_value = 0.001
    mtf.get_bb_position.return_value = 0.0
    mtf.get_trend_alignment.return_value = 0.8
    mtf.get_mean_reversion_alignment.return_value = 0.7
    mtf.get_swing_lows.return_value = [98.0]
    mtf.get_swing_highs.return_value = [102.0]
    for k, v in overrides.items():
        setattr(mtf, k, MagicMock(return_value=v) if not callable(v) else v)
    return mtf


def _make_strategy_with_mtf_gate():
    """Create a TrendRiderProStrategy instance for testing MTF gates."""
    from src.strategies.trend_rider_pro import TrendRiderProStrategy

    config = {
        "confluence_threshold": 55.0,
        "pullback_threshold": 0.005,
        "stop_atr_mult": 1.5,
        "target_atr_mult": 3.5,
        "trail_min_profit_r": 0.5,
        "max_hold_minutes": 120,
        "min_bars": 5,
        "cooldown_bars": 0,
        "signal_timeframe": 1,
    }
    return TrendRiderProStrategy(config, _make_logger())


def _make_mrp_with_mtf_gate():
    """Create a MeanReversionProStrategy instance for testing MTF gates."""
    from src.strategies.mean_reversion_pro import MeanReversionProStrategy

    config = {
        "confluence_threshold": 50.0,
        "cooldown_bars": 0,
        "max_hold_minutes": 30,
        "higher_tf_alignment": False,
        "min_bars": 5,
        "signal_timeframe": 1,
    }
    return MeanReversionProStrategy(config, _make_logger())


def _make_logger() -> MagicMock:
    return MagicMock()


# ===========================================================================
# Tests for _require_higher_tf_trend()
# ===========================================================================


class TestRequireHigherTfTrend:
    """15m timeframe must confirm trend direction — hard gate for trend strategies."""

    def test_passes_when_15m_ema20_above_ema50_for_buy(self):
        """BUY allowed when 15m EMA20 > EMA50 (uptrend on higher TF)."""
        from src.strategies.base import BaseStrategy

        strategy = _make_strategy_with_mtf_gate()
        mtf = MagicMock()
        # 15m EMA20=102, EMA50=100 → uptrend
        def get_ema_side_effect(tf, period):
            if tf == "15m" and period == 20:
                return 102.0
            if tf == "15m" and period == 50:
                return 100.0
            return 100.0
        mtf.get_ema.side_effect = get_ema_side_effect
        mtf.get_adx.return_value = 30.0

        result = strategy._require_higher_tf_trend(mtf, OrderSide.BUY)
        assert result is True

    def test_rejects_when_15m_downtrend_for_buy(self):
        """BUY rejected when 15m EMA20 < EMA50 (downtrend on higher TF)."""
        strategy = _make_strategy_with_mtf_gate()
        mtf = MagicMock()
        # 15m EMA20=98, EMA50=100 → downtrend
        def get_ema_side_effect(tf, period):
            if tf == "15m" and period == 20:
                return 98.0
            if tf == "15m" and period == 50:
                return 100.0
            return 100.0
        mtf.get_ema.side_effect = get_ema_side_effect
        mtf.get_adx.return_value = 30.0

        result = strategy._require_higher_tf_trend(mtf, OrderSide.BUY)
        assert result is False

    def test_passes_when_15m_ema20_below_ema50_for_sell(self):
        """SELL allowed when 15m EMA20 < EMA50 (downtrend on higher TF)."""
        strategy = _make_strategy_with_mtf_gate()
        mtf = MagicMock()
        def get_ema_side_effect(tf, period):
            if tf == "15m" and period == 20:
                return 98.0
            if tf == "15m" and period == 50:
                return 100.0
            return 100.0
        mtf.get_ema.side_effect = get_ema_side_effect
        mtf.get_adx.return_value = 30.0

        result = strategy._require_higher_tf_trend(mtf, OrderSide.SELL)
        assert result is True

    def test_rejects_when_15m_uptrend_for_sell(self):
        """SELL rejected when 15m shows uptrend."""
        strategy = _make_strategy_with_mtf_gate()
        mtf = MagicMock()
        def get_ema_side_effect(tf, period):
            if tf == "15m" and period == 20:
                return 102.0
            if tf == "15m" and period == 50:
                return 100.0
            return 100.0
        mtf.get_ema.side_effect = get_ema_side_effect
        mtf.get_adx.return_value = 30.0

        result = strategy._require_higher_tf_trend(mtf, OrderSide.SELL)
        assert result is False

    def test_rejects_when_15m_adx_too_low(self):
        """Even if EMAs align, ADX < 20 means no confirmed trend — reject."""
        strategy = _make_strategy_with_mtf_gate()
        mtf = MagicMock()
        def get_ema_side_effect(tf, period):
            if tf == "15m" and period == 20:
                return 102.0
            if tf == "15m" and period == 50:
                return 100.0
            return 100.0
        mtf.get_ema.side_effect = get_ema_side_effect
        mtf.get_adx.return_value = 15.0  # too low — no confirmed trend

        result = strategy._require_higher_tf_trend(mtf, OrderSide.BUY)
        assert result is False

    def test_allows_when_no_15m_data(self):
        """When 15m data unavailable (early session), allow through."""
        strategy = _make_strategy_with_mtf_gate()
        mtf = MagicMock()
        mtf.get_ema.return_value = None  # no data
        mtf.get_adx.return_value = None

        result = strategy._require_higher_tf_trend(mtf, OrderSide.BUY)
        assert result is True

    def test_allows_when_adx_unavailable(self):
        """When ADX unavailable but EMAs align, allow through (early session grace)."""
        strategy = _make_strategy_with_mtf_gate()
        mtf = MagicMock()
        def get_ema_side_effect(tf, period):
            if tf == "15m" and period == 20:
                return 102.0
            if tf == "15m" and period == 50:
                return 100.0
            return 100.0
        mtf.get_ema.side_effect = get_ema_side_effect
        mtf.get_adx.return_value = None

        result = strategy._require_higher_tf_trend(mtf, OrderSide.BUY)
        assert result is True


# ===========================================================================
# Tests for _require_higher_tf_flat()
# ===========================================================================


class TestRequireHigherTfFlat:
    """15m timeframe must be flat/choppy — hard gate for mean reversion strategies."""

    def test_passes_when_15m_emas_converged_and_adx_low(self):
        """MR allowed when 15m EMAs within 0.15% and ADX < 25."""
        strategy = _make_mrp_with_mtf_gate()
        mtf = MagicMock()
        def get_ema_side_effect(tf, period):
            if tf == "15m" and period == 20:
                return 100.1  # 0.1% from EMA50 — converged
            if tf == "15m" and period == 50:
                return 100.0
            return 100.0
        mtf.get_ema.side_effect = get_ema_side_effect
        mtf.get_adx.return_value = 18.0  # low ADX — choppy

        result = strategy._require_higher_tf_flat(mtf)
        assert result is True

    def test_rejects_when_15m_emas_diverged(self):
        """MR rejected when 15m EMAs spread > 0.3% — trending, not flat."""
        strategy = _make_mrp_with_mtf_gate()
        mtf = MagicMock()
        def get_ema_side_effect(tf, period):
            if tf == "15m" and period == 20:
                return 100.5  # 0.5% from EMA50 — diverged
            if tf == "15m" and period == 50:
                return 100.0
            return 100.0
        mtf.get_ema.side_effect = get_ema_side_effect
        mtf.get_adx.return_value = 18.0

        result = strategy._require_higher_tf_flat(mtf)
        assert result is False

    def test_rejects_when_15m_adx_high(self):
        """MR rejected when ADX > 25 even if EMAs converged — strong trend."""
        strategy = _make_mrp_with_mtf_gate()
        mtf = MagicMock()
        def get_ema_side_effect(tf, period):
            if tf == "15m" and period == 20:
                return 100.1
            if tf == "15m" and period == 50:
                return 100.0
            return 100.0
        mtf.get_ema.side_effect = get_ema_side_effect
        mtf.get_adx.return_value = 30.0  # high ADX — trending

        result = strategy._require_higher_tf_flat(mtf)
        assert result is False

    def test_allows_when_no_15m_data(self):
        """When 15m data unavailable (early session), allow through."""
        strategy = _make_mrp_with_mtf_gate()
        mtf = MagicMock()
        mtf.get_ema.return_value = None
        mtf.get_adx.return_value = None

        result = strategy._require_higher_tf_flat(mtf)
        assert result is True

    def test_allows_when_adx_unavailable_but_emas_flat(self):
        """When ADX unavailable but EMAs converged, allow (early session grace)."""
        strategy = _make_mrp_with_mtf_gate()
        mtf = MagicMock()
        def get_ema_side_effect(tf, period):
            if tf == "15m" and period == 20:
                return 100.1
            if tf == "15m" and period == 50:
                return 100.0
            return 100.0
        mtf.get_ema.side_effect = get_ema_side_effect
        mtf.get_adx.return_value = None  # no ADX yet

        result = strategy._require_higher_tf_flat(mtf)
        assert result is True


# ===========================================================================
# Integration: verify gates are wired into on_bar()
# ===========================================================================


class TestTRPUsesHigherTfTrend:
    """TRP.on_bar() must call _require_higher_tf_trend() and reject on False."""

    def test_trp_rejects_buy_when_15m_downtrend(self):
        """TRP should not generate BUY when 15m shows downtrend."""
        from unittest.mock import patch

        from src.core.domain import TrendRegime
        from src.strategies.trend_rider_pro import TrendRiderProStrategy

        config = {
            "confluence_threshold": 55.0,
            "pullback_threshold": 0.005,
            "stop_atr_mult": 1.5,
            "target_atr_mult": 3.5,
            "trail_min_profit_r": 0.5,
            "max_hold_minutes": 120,
            "min_bars": 5,
            "cooldown_bars": 0,
            "signal_timeframe": 1,
        }
        strategy = TrendRiderProStrategy(config, _make_logger())

        bar = _make_bar(close=100.0, time=datetime(2024, 3, 15, 14, 35, tzinfo=UTC))
        ss = _make_symbol_state()
        ms = MarketState(
            time=bar.time,
            regime=Regime.BULL,
            regime_snapshot=MarketRegimeSnapshot(
                time=bar.time,
                index_symbol="SPY",
                vol_symbol="VIX",
                trend=TrendRegime.UP,
                vol=VolRegime.NORMAL,
                liquidity=LiquidityRegime.GOOD,
                risk=RiskRegime.RISK_ON,
                session=SessionRegime.OPENING,
            ),
        )

        # MTF that would normally generate a BUY signal, BUT 15m is downtrend
        mtf = _mock_mtf()

        def get_ema_for_15m(tf, period):
            if tf == "15m" and period == 20:
                return 98.0  # 15m downtrend
            if tf == "15m" and period == 50:
                return 100.0
            return 100.2  # 5m/1m pullback near EMA
        mtf.get_ema.side_effect = get_ema_for_15m
        mtf.get_adx.return_value = 30.0

        with patch("src.strategies.trend_rider_pro.MultiTimeframeAnalyzer", return_value=mtf):
            sig = strategy.on_bar("TEST", bar, ss, ms)

        assert sig is None, "TRP should reject BUY when 15m shows downtrend"


class TestMRPUsesHigherTfFlat:
    """MRP.on_bar() must call _require_higher_tf_flat() and reject on False."""

    def test_mrp_rejects_when_15m_trending(self):
        """MRP should not generate signal when 15m shows strong trend."""
        from unittest.mock import patch

        from src.core.domain import TrendRegime
        from src.strategies.mean_reversion_pro import MeanReversionProStrategy

        config = {
            "confluence_threshold": 50.0,
            "cooldown_bars": 0,
            "max_hold_minutes": 30,
            "higher_tf_alignment": False,
            "min_bars": 5,
            "signal_timeframe": 1,
        }
        strategy = MeanReversionProStrategy(config, _make_logger())

        bar = _make_bar(close=100.0, time=datetime(2024, 3, 15, 14, 35, tzinfo=UTC))
        ss = _make_symbol_state()
        ms = MarketState(
            time=bar.time,
            regime=Regime.BULL,
            regime_snapshot=MarketRegimeSnapshot(
                time=bar.time,
                index_symbol="SPY",
                vol_symbol="VIX",
                trend=TrendRegime.FLAT,
                vol=VolRegime.LOW,
                liquidity=LiquidityRegime.GOOD,
                risk=RiskRegime.NEUTRAL,
                session=SessionRegime.MIDDAY,
            ),
        )

        # MTF that would pass mean reversion BUT 15m is trending
        mtf = _mock_mtf()
        mtf.get_vwap_distance.return_value = -0.005  # BUY signal (below VWAP)
        mtf.get_bb_position.return_value = -0.8  # near lower band
        mtf.get_rsi.return_value = 25.0  # oversold
        mtf.get_mean_reversion_alignment.return_value = 0.8

        # BUT 15m shows strong trend — EMAs diverged + high ADX
        def get_ema_for_15m(tf, period):
            if tf == "15m" and period == 20:
                return 102.0  # 2% diverged
            if tf == "15m" and period == 50:
                return 100.0
            return 100.0
        mtf.get_ema.side_effect = get_ema_for_15m
        mtf.get_adx.return_value = 35.0

        with patch("src.strategies.mean_reversion_pro.MultiTimeframeAnalyzer", return_value=mtf):
            sig = strategy.on_bar("TEST", bar, ss, ms)

        assert sig is None, "MRP should reject when 15m shows strong trend"
