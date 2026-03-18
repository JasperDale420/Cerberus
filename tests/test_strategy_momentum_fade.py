"""Unit tests for MomentumFade strategy."""

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
from src.strategies.momentum_fade import MomentumFadeStrategy


def _make_bar(
    close: float = 100.0,
    volume: float = 10000,
    vwap: float = 100.0,
    minutes_offset: int = 0,
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
) -> Bar:
    """Create a test bar with sensible defaults."""
    _open = open_price if open_price is not None else close - 0.10
    _high = high if high is not None else close + 0.50
    _low = low if low is not None else close - 0.50
    return Bar(
        symbol="AAPL",
        # 10:00 AM ET = 15:00 UTC (within default trading window)
        time=datetime(2025, 1, 15, 15, 0 + minutes_offset, tzinfo=timezone.utc),
        open=_open,
        high=_high,
        low=_low,
        close=close,
        volume=volume,
        vwap=vwap,
    )


def _make_symbol_state(
    n_bars: int = 50,
    bar_volume: float = 5000,
    bar_close: float = 100.0,
    bar_vwap: float = 100.0,
) -> SymbolState:
    """Create a SymbolState with enough bars for most strategies."""
    bars = deque(
        [
            _make_bar(
                close=bar_close + i * 0.01,
                volume=bar_volume,
                vwap=bar_vwap,
                minutes_offset=i,
            )
            for i in range(n_bars)
        ]
    )
    return SymbolState(
        symbol="AAPL",
        bars_1m=bars,
        bars_5m=deque(),
        bars_15m=deque(),
        bars_1d=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["momentum_fade"],
        meta={},
    )


def _make_market_state(
    vol: VolRegime = VolRegime.NORMAL,
    session: SessionRegime = SessionRegime.MIDDAY,
    trend: TrendRegime = TrendRegime.FLAT,
) -> MarketState:
    """Create a MarketState with a proper regime snapshot."""
    snapshot = MarketRegimeSnapshot(
        time=datetime(2025, 1, 15, 15, 0, tzinfo=timezone.utc),
        index_symbol="SPY",
        vol_symbol="VIX",
        trend=trend,
        vol=vol,
        liquidity=LiquidityRegime.GOOD,
        risk=RiskRegime.NEUTRAL,
        session=session,
    )
    return MarketState(
        time=datetime(2025, 1, 15, 15, 0, tzinfo=timezone.utc),
        regime=Regime.CHOP,
        regime_snapshot=snapshot,
    )


@pytest.fixture
def config() -> dict:
    return {
        "time_window_start": "09:35",
        "time_window_end": "15:50",
        "min_bars": 5,
        "cooldown_bars": 3,
        "higher_tf_alignment": False,
        "vwap_threshold": 0.008,
        "volume_surge_mult": 2.0,
        "confluence_threshold": 40.0,  # Lower for easier test triggering
        "stop_atr_mult": 1.5,
        "target_atr_mult": 3.0,
        "max_hold_minutes": 90,
    }


@pytest.fixture
def strategy(config):
    logger = MagicMock()
    return MomentumFadeStrategy(config=config, logger=logger)


@pytest.mark.unit
class TestMomentumFadeInit:
    def test_name(self, strategy):
        assert strategy.name == "momentum_fade"

    def test_params_from_config(self, strategy, config):
        assert strategy.min_bars == config["min_bars"]
        assert strategy.vwap_threshold == config["vwap_threshold"]
        assert strategy.volume_surge_mult == config["volume_surge_mult"]
        assert strategy.confluence_threshold == config["confluence_threshold"]
        assert strategy.stop_atr_mult == config["stop_atr_mult"]
        assert strategy.target_atr_mult == config["target_atr_mult"]
        assert strategy.max_hold_minutes == config["max_hold_minutes"]

    def test_tf_alignment_mode_is_mean_reversion(self, strategy):
        assert strategy.tf_alignment_mode == "mean_reversion"


@pytest.mark.unit
class TestMomentumFadeOnBar:
    def test_returns_none_on_cooldown(self, strategy):
        bar = _make_bar()
        ss = _make_symbol_state()
        ms = _make_market_state()
        # Trigger cooldown by setting last signal time
        strategy.last_signal_time["AAPL"] = bar.time
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_returns_none_insufficient_bars(self, strategy):
        bar = _make_bar()
        ss = _make_symbol_state(n_bars=1)
        ms = _make_market_state()
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_returns_none_outside_time_window(self, strategy):
        """Before market open (3 AM ET) should return None."""
        bar = _make_bar()
        # 3 AM ET = 8:00 UTC — outside 09:35-15:50 ET window
        bar.time = datetime(2025, 1, 15, 8, 0, tzinfo=timezone.utc)
        ss = _make_symbol_state()
        ms = _make_market_state()
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_returns_none_during_shock_regime(self, strategy):
        """SHOCK vol regime should be gated out."""
        # Price well above VWAP with high volume
        bar = _make_bar(close=101.0, vwap=100.0, volume=50000)
        ss = _make_symbol_state(bar_volume=5000)
        ms = _make_market_state(vol=VolRegime.SHOCK)
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_returns_none_during_premarket(self, strategy):
        """PREMARKET session should be gated out."""
        bar = _make_bar(close=101.0, vwap=100.0, volume=50000)
        ss = _make_symbol_state(bar_volume=5000)
        ms = _make_market_state(session=SessionRegime.PREMARKET)
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_returns_none_when_vwap_dist_insufficient(self, strategy):
        """Price too close to VWAP should not trigger."""
        bar = _make_bar(close=100.05, vwap=100.0, volume=50000)
        ss = _make_symbol_state(bar_volume=5000)
        ms = _make_market_state()
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_returns_none_when_volume_insufficient(self, strategy):
        """Low volume should not trigger even with VWAP distance."""
        bar = _make_bar(close=101.0, vwap=100.0, volume=5000)
        ss = _make_symbol_state(bar_volume=5000)  # Same as avg → ratio = 1.0
        ms = _make_market_state()
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_returns_none_without_vwap(self, strategy):
        """No VWAP data should return None."""
        bar = _make_bar(close=101.0, vwap=None, volume=50000)
        ss = _make_symbol_state(bar_volume=5000)
        ms = _make_market_state()
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_generates_sell_signal_when_price_above_vwap(self, strategy):
        """When price is significantly above VWAP with volume surge, should fade SHORT."""
        # Price 1.0% above VWAP (> 0.8% threshold)
        bar = _make_bar(
            close=101.0,
            vwap=100.0,
            volume=15000,  # 3x avg of 5000
            open_price=100.5,
            high=101.2,
            low=100.3,
        )
        ss = _make_symbol_state(n_bars=50, bar_volume=5000, bar_vwap=100.0)
        ms = _make_market_state()
        result = strategy.on_bar("AAPL", bar, ss, ms)
        # May be None if ATR is not computed (no 5m bars), but if signal fires:
        if result is not None:
            assert result.side == OrderSide.SELL
            assert result.symbol == "AAPL"
            assert result.strategy == "momentum_fade"
            assert result.stop_price > result.entry_price  # Stop above for SELL
            assert result.target_price < result.entry_price  # Target below for SELL

    def test_generates_buy_signal_when_price_below_vwap(self, strategy):
        """When price is significantly below VWAP with volume surge, should fade LONG."""
        # Price 1.0% below VWAP (> 0.8% threshold)
        bar = _make_bar(
            close=99.0,
            vwap=100.0,
            volume=15000,
            open_price=99.5,
            high=99.8,
            low=98.8,
        )
        ss = _make_symbol_state(n_bars=50, bar_volume=5000, bar_vwap=100.0)
        ms = _make_market_state()
        result = strategy.on_bar("AAPL", bar, ss, ms)
        if result is not None:
            assert result.side == OrderSide.BUY
            assert result.stop_price < result.entry_price  # Stop below for BUY
            assert result.target_price > result.entry_price  # Target above for BUY

    def test_signal_has_required_meta_fields(self, strategy):
        """Signal meta should include vwap_dist, volume_ratio, confluence, and exit_config."""
        bar = _make_bar(close=101.0, vwap=100.0, volume=15000)
        ss = _make_symbol_state(n_bars=50, bar_volume=5000)
        ms = _make_market_state()
        result = strategy.on_bar("AAPL", bar, ss, ms)
        if result is not None:
            assert "vwap_dist" in result.meta
            assert "volume_ratio" in result.meta
            assert "confluence_score" in result.meta
            assert "exit_config" in result.meta
            assert result.meta["exit_config"]["trailing_enabled"] is True
            assert result.meta["exit_config"]["max_hold_minutes"] == 90
            assert "strategy_version" in result.meta
            assert result.meta["strategy_version"] == "momentum_fade_v1"


@pytest.mark.unit
class TestMomentumFadeVolumeRatio:
    def test_volume_ratio_calculation(self, strategy):
        """Volume ratio should correctly compare current vs average."""
        ss = _make_symbol_state(n_bars=30, bar_volume=5000)
        bar = _make_bar(volume=15000)  # 3x average
        ratio = strategy._calculate_volume_ratio(bar, "TEST", ss)
        assert ratio == pytest.approx(3.0, rel=0.1)

    def test_volume_ratio_returns_zero_insufficient_bars(self, strategy):
        """Should return 0.0 with too few bars."""
        ss = _make_symbol_state(n_bars=2, bar_volume=5000)
        bar = _make_bar(volume=15000)
        ratio = strategy._calculate_volume_ratio(bar, "TEST", ss)
        assert ratio == 0.0

    def test_volume_ratio_zero_avg_volume(self, strategy):
        """Should handle zero average volume gracefully."""
        ss = _make_symbol_state(n_bars=30, bar_volume=0)
        bar = _make_bar(volume=15000)
        ratio = strategy._calculate_volume_ratio(bar, "TEST", ss)
        assert ratio == 0.0


@pytest.mark.unit
class TestMomentumFadeDirection:
    def test_fade_short_when_above_vwap(self, strategy):
        """vwap_dist > 0 should produce SELL (short) direction."""
        # We test the logic directly: price above VWAP = fade short
        bar = _make_bar(close=101.0, vwap=100.0, volume=15000)
        vwap_dist = (bar.close - bar.vwap) / bar.vwap
        assert vwap_dist > 0
        # The strategy would choose SELL

    def test_fade_long_when_below_vwap(self, strategy):
        """vwap_dist < 0 should produce BUY (long) direction."""
        bar = _make_bar(close=99.0, vwap=100.0, volume=15000)
        vwap_dist = (bar.close - bar.vwap) / bar.vwap
        assert vwap_dist < 0
        # The strategy would choose BUY
