"""Unit tests for RsiBounce strategy."""

from collections import deque
from datetime import datetime, timezone
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
    Signal,
    SymbolState,
    TrendRegime,
    VolRegime,
)
from src.strategies.rsi_bounce import RsiBounceStrategy


def _make_bar(
    close: float = 100.0,
    volume: float = 10000,
    minutes_offset: int = 0,
    high: float | None = None,
    low: float | None = None,
    vwap: float | None = None,
) -> Bar:
    """Create a test bar with sensible defaults.

    Time is set to 15:00 UTC = 10:00 ET (within default trading window).
    """
    return Bar(
        symbol="AAPL",
        time=datetime(2025, 1, 15, 15, 0 + minutes_offset, tzinfo=timezone.utc),
        open=close - 0.10,
        high=high if high is not None else close + 0.50,
        low=low if low is not None else close - 0.50,
        close=close,
        volume=volume,
        vwap=vwap if vwap is not None else close,
    )


def _make_symbol_state(n_bars: int = 50, base_price: float = 100.0) -> SymbolState:
    """Create a SymbolState with enough bars for most strategies."""
    bars_1m = deque([_make_bar(close=base_price + i * 0.1, minutes_offset=i) for i in range(n_bars)])
    # Build 5m bars (every 5 bars aggregated)
    bars_5m: deque[Bar] = deque()
    for i in range(0, n_bars, 5):
        chunk = list(bars_1m)[i : i + 5]
        if len(chunk) >= 5:
            bars_5m.append(
                Bar(
                    symbol="AAPL",
                    time=chunk[-1].time,
                    open=chunk[0].open,
                    high=max(b.high for b in chunk),
                    low=min(b.low for b in chunk),
                    close=chunk[-1].close,
                    volume=sum(b.volume for b in chunk),
                    vwap=chunk[-1].close,
                )
            )

    return SymbolState(
        symbol="AAPL",
        bars_1m=bars_1m,
        bars_5m=bars_5m,
        bars_15m=deque(),
        bars_1d=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["rsi_bounce"],
        meta={},
    )


def _make_market_state(
    vol: VolRegime = VolRegime.NORMAL,
    session: SessionRegime = SessionRegime.MIDDAY,
    trend: TrendRegime = TrendRegime.FLAT,
) -> MarketState:
    """Create a MarketState with regime snapshot."""
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


def _make_market_state_no_snapshot() -> MarketState:
    """Create a MarketState without regime snapshot."""
    return MagicMock(spec=MarketState, regime_snapshot=None)


@pytest.fixture
def config() -> dict:
    return {
        "time_window_start": "09:35",
        "time_window_end": "15:50",
        "min_bars": 5,
        "cooldown_bars": 3,
        "higher_tf_alignment": False,
        "rsi_len": 14,
        "rsi_oversold": 10.0,
        "rsi_overbought": 90.0,
        "bb_period": 20,
        "band_sigma": 2.0,
        "band_tolerance": 0.5,
        "stop_atr_mult": 1.5,
        "target_atr_mult": 3.0,
        "max_hold_minutes": 60,
        "confluence_threshold": 60.0,
    }


@pytest.fixture
def strategy(config):
    logger = MagicMock()
    return RsiBounceStrategy(config=config, logger=logger)


@pytest.mark.unit
class TestRsiBounceInit:
    def test_name(self, strategy):
        assert strategy.name == "rsi_bounce"

    def test_params_from_config(self, strategy, config):
        assert strategy.min_bars == config["min_bars"]
        assert strategy.rsi_len == config["rsi_len"]
        assert strategy.rsi_oversold == config["rsi_oversold"]
        assert strategy.rsi_overbought == config["rsi_overbought"]
        assert strategy.bb_period == config["bb_period"]
        assert strategy.band_sigma == config["band_sigma"]
        assert strategy.band_tolerance == config["band_tolerance"]
        assert strategy.stop_atr_mult == config["stop_atr_mult"]
        assert strategy.target_atr_mult == config["target_atr_mult"]
        assert strategy.max_hold_minutes == config["max_hold_minutes"]

    def test_tf_alignment_mode_is_mean_reversion(self, strategy):
        assert strategy.tf_alignment_mode == "mean_reversion"

    def test_optuna_overrides(self):
        """Optuna overrides should take precedence over base config."""
        config = {
            "rsi_len": 14,
            "band_tolerance": 0.5,
            "higher_tf_alignment": False,
            "_optuna_overrides": {
                "rsi_len": 7,
                "band_tolerance": 1.0,
            },
        }
        logger = MagicMock()
        strat = RsiBounceStrategy(config=config, logger=logger)
        assert strat.rsi_len == 7
        assert strat.band_tolerance == 1.0


@pytest.mark.unit
class TestRsiBounceOnBar:
    def test_returns_none_on_cooldown(self, strategy):
        bar = _make_bar()
        ss = _make_symbol_state()
        ms = _make_market_state_no_snapshot()
        # Trigger cooldown
        strategy.last_signal_time["AAPL"] = bar.time
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_returns_none_insufficient_bars(self, strategy):
        bar = _make_bar()
        ss = _make_symbol_state(n_bars=1)
        ms = _make_market_state_no_snapshot()
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_returns_none_outside_time_window(self):
        """Signal should be None when bar time is outside the trading window."""
        config = {
            "time_window_start": "10:00",
            "time_window_end": "10:30",
            "min_bars": 5,
            "cooldown_bars": 0,
            "higher_tf_alignment": False,
        }
        logger = MagicMock()
        strat = RsiBounceStrategy(config=config, logger=logger)
        # 20:00 UTC = 15:00 ET -> outside 10:00-10:30 window
        bar = Bar(
            symbol="AAPL",
            time=datetime(2025, 1, 15, 20, 0, tzinfo=timezone.utc),
            open=99.9,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=10000,
            vwap=100.0,
        )
        ss = _make_symbol_state()
        ms = _make_market_state_no_snapshot()
        result = strat.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_returns_none_on_vol_shock(self, strategy):
        """Should skip VolRegime.SHOCK."""
        bar = _make_bar()
        ss = _make_symbol_state()
        ms = _make_market_state(vol=VolRegime.SHOCK)
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_returns_none_on_premarket(self, strategy):
        """Should skip SessionRegime.PREMARKET."""
        bar = _make_bar()
        ss = _make_symbol_state()
        ms = _make_market_state(session=SessionRegime.PREMARKET)
        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_generates_buy_signal_on_oversold_rsi_near_lower_band(self, strategy):
        """When RSI < oversold and price near lower BB, generate BUY signal."""
        bar = _make_bar(close=95.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        # Mock the MTF analyzer to return extreme RSI and appropriate BB/ATR
        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 5.0  # Extremely oversold
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_mean_reversion_alignment.return_value = 0.8

            # Set up BB cache: mean=100, std=2.5 -> lower_band = 95.0
            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is not None
        assert isinstance(signal, Signal)
        assert signal.side == OrderSide.BUY
        assert signal.strategy == "rsi_bounce"
        assert signal.stop_price < signal.entry_price
        assert signal.target_price > signal.entry_price
        assert "rsi_5m" in signal.meta
        assert "confluence_score" in signal.meta

    def test_generates_sell_signal_on_overbought_rsi_near_upper_band(self, strategy):
        """When RSI > overbought and price near upper BB, generate SELL signal."""
        bar = _make_bar(close=105.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 95.0  # Extremely overbought
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_mean_reversion_alignment.return_value = 0.8

            # Set up BB cache: mean=100, std=2.5 -> upper_band = 105.0
            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is not None
        assert isinstance(signal, Signal)
        assert signal.side == OrderSide.SELL
        assert signal.strategy == "rsi_bounce"
        assert signal.stop_price > signal.entry_price
        assert signal.target_price < signal.entry_price

    def test_stop_below_entry_for_buy(self, strategy):
        """For BUY signals, stop must be below entry price."""
        bar = _make_bar(close=95.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 5.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_mean_reversion_alignment.return_value = 0.8

            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is not None
        assert signal.stop_price < signal.entry_price

    def test_stop_above_entry_for_sell(self, strategy):
        """For SELL signals, stop must be above entry price."""
        bar = _make_bar(close=105.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 95.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_mean_reversion_alignment.return_value = 0.8

            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is not None
        assert signal.stop_price > signal.entry_price

    def test_meta_contains_exit_config(self, strategy):
        """Signal meta should include exit_config for the execution layer."""
        bar = _make_bar(close=95.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 5.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_mean_reversion_alignment.return_value = 0.8

            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is not None
        assert "exit_config" in signal.meta
        assert signal.meta["exit_config"]["trailing_enabled"] is True
        assert signal.meta["exit_config"]["max_hold_minutes"] == 60

    def test_returns_none_when_rsi_not_extreme(self, strategy):
        """No signal when RSI is in the normal range (neither oversold nor overbought)."""
        bar = _make_bar(close=100.0)
        ss = _make_symbol_state(n_bars=50)
        ms = _make_market_state()

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 50.0  # Normal RSI — no extreme

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is None

    def test_returns_none_when_price_not_near_band(self, strategy):
        """No signal when RSI is extreme but price is far from band."""
        bar = _make_bar(close=100.0)  # Price at mean, not near lower band
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 5.0  # Oversold

            # BB: mean=100, std=2.5 -> lower=95, upper=105
            # Price at 100 is nowhere near lower band
            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is None

    def test_returns_none_when_rsi_is_none(self, strategy):
        """No signal when RSI cannot be computed."""
        bar = _make_bar()
        ss = _make_symbol_state(n_bars=50)
        ms = _make_market_state()

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = None

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is None

    def test_confluence_factors_in_meta(self, strategy):
        """Signal meta should contain confluence factors detail."""
        bar = _make_bar(close=95.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 3.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_mean_reversion_alignment.return_value = 1.0

            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is not None
        assert "factors" in signal.meta
        factor_names = set(signal.meta["factors"].keys())
        assert "rsi_extremity" in factor_names
        assert "band_proximity" in factor_names
        assert "mr_alignment" in factor_names
