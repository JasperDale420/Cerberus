"""Unit tests for RsiBounce strategy (v2 — 6-factor mean reversion)."""

from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.ou_estimator import OUResult
from src.analysis.variance_ratio import VarianceRatioResult
from src.analysis.vpin import VPINResult
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


def _mock_analysis_objects(strategy: RsiBounceStrategy, symbol: str = "AAPL") -> None:
    """Pre-populate the strategy's per-symbol analysis objects with mocks
    that return favorable results for signal generation.
    """
    strategy._ensure_symbol_state(symbol)

    # OU estimator returns a valid half-life within acceptable range
    mock_ou = MagicMock()
    mock_ou.update.return_value = OUResult(
        theta=0.05, mu=0.0, sigma=0.01, half_life=10.0, scaling_factor=1.0
    )
    strategy._ou_estimators[symbol] = mock_ou

    # VR calculator returns mean-reverting result
    mock_vr = MagicMock()
    mock_vr.update.return_value = VarianceRatioResult(
        vr=0.75, z_score=-2.5, p_value_two_sided=0.01,
        is_mean_reverting=True, is_trending=False, period=5, n_observations=120,
    )
    strategy._vr_calculators[symbol] = mock_vr

    # VPIN calculator returns non-toxic flow
    mock_vpin = MagicMock()
    mock_vpin.update.return_value = VPINResult(
        vpin=0.3, buy_volume=5000, sell_volume=5000,
        bucket_count=10, is_toxic=False,
    )
    strategy._vpin_calculators[symbol] = mock_vpin

    # Pre-fill price/volume/rsi/roc history with enough data
    strategy._price_history[symbol] = deque([100.0 + i * 0.01 for i in range(60)], maxlen=60)
    strategy._volume_history[symbol] = deque([10000.0] * 20, maxlen=20)
    strategy._rsi_history[symbol] = deque([50.0] * 100, maxlen=100)
    strategy._roc_history[symbol] = deque([0.001] * 10, maxlen=10)


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
        "zscore_entry": 2.0,
        "zscore_lookback": 60,
        "ou_lookback": 60,
        "ou_min_obs": 30,
        "max_half_life_bars": 20.0,
        "min_half_life_bars": 1.0,
        "vr_lookback": 120,
        "vr_period": 5,
        "vpin_n_buckets": 50,
        "vpin_toxicity_threshold": 0.7,
        "volume_lookback": 20,
        "volume_climax_mult": 1.5,
        "roc_lookback": 10,
        "rsi_pctile_lookback": 100,
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

    def test_v2_params_from_config(self, strategy, config):
        """V2-specific parameters should be parsed from config."""
        assert strategy.zscore_entry == config["zscore_entry"]
        assert strategy.zscore_lookback == config["zscore_lookback"]
        assert strategy.ou_lookback == config["ou_lookback"]
        assert strategy.max_half_life_bars == config["max_half_life_bars"]
        assert strategy.vr_lookback == config["vr_lookback"]
        assert strategy.vr_period == config["vr_period"]
        assert strategy.vpin_toxicity_threshold == config["vpin_toxicity_threshold"]

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

    def test_optuna_overrides_v2_params(self):
        """Optuna overrides should work for v2 parameters too."""
        config = {
            "higher_tf_alignment": False,
            "zscore_entry": 2.0,
            "max_half_life_bars": 20.0,
            "_optuna_overrides": {
                "zscore_entry": 1.5,
                "max_half_life_bars": 30.0,
            },
        }
        logger = MagicMock()
        strat = RsiBounceStrategy(config=config, logger=logger)
        assert strat.zscore_entry == 1.5
        assert strat.max_half_life_bars == 30.0


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

        _mock_analysis_objects(strategy, "AAPL")

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 5.0  # Extremely oversold
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_vwap_distance.return_value = -0.05

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
        assert signal.meta["strategy_version"] == "rsi_bounce_v2"

    def test_generates_sell_signal_on_overbought_rsi_near_upper_band(self, strategy):
        """When RSI > overbought and price near upper BB, generate SELL signal."""
        bar = _make_bar(close=105.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        _mock_analysis_objects(strategy, "AAPL")

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 95.0  # Extremely overbought
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_vwap_distance.return_value = 0.05

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

        _mock_analysis_objects(strategy, "AAPL")

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 5.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_vwap_distance.return_value = -0.05

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

        _mock_analysis_objects(strategy, "AAPL")

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 95.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_vwap_distance.return_value = 0.05

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

        _mock_analysis_objects(strategy, "AAPL")

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 5.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_vwap_distance.return_value = -0.05

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
        """Signal meta should contain v2 confluence factors detail."""
        bar = _make_bar(close=95.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        _mock_analysis_objects(strategy, "AAPL")

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 3.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_vwap_distance.return_value = -0.05

            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is not None
        assert "factors" in signal.meta
        factor_names = set(signal.meta["factors"].keys())
        # v2 factor names
        assert "z_score_extremity" in factor_names
        assert "half_life_validity" in factor_names
        assert "rsi_percentile_rank" in factor_names
        assert "volume_climax" in factor_names
        assert "momentum_deceleration" in factor_names
        assert "variance_ratio" in factor_names


@pytest.mark.unit
class TestRsiBounceV2Gates:
    """Tests for the new v2 gates: VPIN toxicity, half-life, variance ratio."""

    def test_vpin_toxic_rejects_signal(self, strategy):
        """VPIN toxicity should prevent signal generation."""
        bar = _make_bar(close=95.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        _mock_analysis_objects(strategy, "AAPL")
        # Override VPIN to return toxic
        mock_vpin = MagicMock()
        mock_vpin.update.return_value = VPINResult(
            vpin=0.85, buy_volume=8000, sell_volume=2000,
            bucket_count=10, is_toxic=True,
        )
        strategy._vpin_calculators["AAPL"] = mock_vpin

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 5.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_vwap_distance.return_value = -0.05

            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is None

    def test_half_life_too_long_rejects_signal(self, strategy):
        """Half-life exceeding max_hold should prevent signal generation."""
        bar = _make_bar(close=95.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        _mock_analysis_objects(strategy, "AAPL")
        # Override OU to return a very long half-life
        mock_ou = MagicMock()
        mock_ou.update.return_value = OUResult(
            theta=0.001, mu=0.0, sigma=0.01, half_life=100.0, scaling_factor=1.0
        )
        strategy._ou_estimators["AAPL"] = mock_ou

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 5.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_vwap_distance.return_value = -0.05

            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is None

    def test_variance_ratio_trending_rejects_signal(self, strategy):
        """Statistically significant trending VR should prevent signal generation."""
        bar = _make_bar(close=95.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        _mock_analysis_objects(strategy, "AAPL")
        # Override VR to return trending
        mock_vr = MagicMock()
        mock_vr.update.return_value = VarianceRatioResult(
            vr=1.5, z_score=3.0, p_value_two_sided=0.002,
            is_mean_reverting=False, is_trending=True, period=5, n_observations=120,
        )
        strategy._vr_calculators["AAPL"] = mock_vr

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 5.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_vwap_distance.return_value = -0.05

            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is None

    def test_meta_contains_v2_fields(self, strategy):
        """Signal meta should contain v2 diagnostic fields."""
        bar = _make_bar(close=95.0)
        ss = _make_symbol_state(n_bars=50, base_price=100.0)
        ms = _make_market_state()

        _mock_analysis_objects(strategy, "AAPL")

        with (
            patch.object(strategy, "_check_cooldown", return_value=True),
            patch.object(strategy, "_require_min_bars", return_value=True),
            patch.object(strategy, "_check_higher_tf_alignment", return_value=True),
            patch("src.strategies.rsi_bounce.MultiTimeframeAnalyzer") as MockMTF,
        ):
            mock_mtf = MockMTF.return_value
            mock_mtf.get_rsi.return_value = 5.0
            mock_mtf.get_atr.return_value = 1.0
            mock_mtf.get_vwap_distance.return_value = -0.05

            mock_cache = MagicMock()
            mock_cache.bb = {20: (100.0, 2.5)}
            mock_mtf._cache = {"5m": mock_cache}
            mock_mtf._ensure_cache = MagicMock()

            signal = strategy.on_bar("AAPL", bar, ss, ms)

        assert signal is not None
        # V2-specific metadata
        assert "half_life" in signal.meta
        assert "ou_theta" in signal.meta
        assert "variance_ratio" in signal.meta
        assert "vr_z_score" in signal.meta
        assert "vpin" in signal.meta
        assert "volume_ratio" in signal.meta
        assert "z_score" in signal.meta
        assert signal.meta["strategy_version"] == "rsi_bounce_v2"

    def test_per_symbol_state_isolation(self, strategy):
        """Each symbol should have independent analysis state."""
        strategy._ensure_symbol_state("AAPL")
        strategy._ensure_symbol_state("MSFT")

        assert strategy._ou_estimators["AAPL"] is not strategy._ou_estimators["MSFT"]
        assert strategy._vr_calculators["AAPL"] is not strategy._vr_calculators["MSFT"]
        assert strategy._vpin_calculators["AAPL"] is not strategy._vpin_calculators["MSFT"]
