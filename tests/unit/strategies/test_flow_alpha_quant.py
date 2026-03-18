"""Tests for flow_alpha quant upgrades: IC-weighted signals, Granger, VPIN gate, GARCH."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.analysis.vpin import VPINCalculator, VPINResult
from src.core.domain import Bar, MarketState, SymbolState
from src.quant.validation import InformationCoefficientTracker
from src.quant.volatility import GARCHForecaster
from src.strategies.flow_alpha import FlowAlphaStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ET = ZoneInfo("US/Eastern")


class _MockLogger:
    """Minimal mock matching StructuredLogger interface."""

    def __init__(self):
        self.messages: list[tuple[str, str, dict]] = []

    def debug(self, msg: str, **kw: Any) -> None:
        self.messages.append(("debug", msg, kw))

    def info(self, msg: str, **kw: Any) -> None:
        self.messages.append(("info", msg, kw))

    def warning(self, msg: str, **kw: Any) -> None:
        self.messages.append(("warning", msg, kw))

    def error(self, msg: str, **kw: Any) -> None:
        self.messages.append(("error", msg, kw))

    def critical(self, msg: str, **kw: Any) -> None:
        self.messages.append(("critical", msg, kw))


def _make_bar(symbol: str, time: datetime, close: float, volume: int = 1000) -> Bar:
    return Bar(
        symbol=symbol,
        time=time,
        open=close - 0.05,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=volume,
    )


def _make_market_state(time: datetime) -> MarketState:
    return MagicMock(spec=MarketState, regime_snapshot=None)


def _make_symbol_state(symbol: str, bars: list[Bar], meta: dict | None = None) -> SymbolState:
    return SymbolState(
        symbol=symbol,
        bars=deque(bars),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta=meta or {},
    )


def _base_time() -> datetime:
    return datetime(2025, 6, 15, 10, 0, tzinfo=ET)


@pytest.fixture
def strategy() -> FlowAlphaStrategy:
    config: dict[str, Any] = {
        "confluence_threshold": 65.0,
        "min_bars": 5,
    }
    return FlowAlphaStrategy(config, _MockLogger())


# ---------------------------------------------------------------------------
# Test 1: Strategy initializes quant components
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQuantInitialization:
    def test_garch_dict_initialized(self, strategy: FlowAlphaStrategy):
        assert isinstance(strategy._garch, dict)
        assert len(strategy._garch) == 0

    def test_vpin_dict_initialized(self, strategy: FlowAlphaStrategy):
        assert isinstance(strategy._vpin, dict)
        assert len(strategy._vpin) == 0

    def test_ic_trackers_initialized(self, strategy: FlowAlphaStrategy):
        assert isinstance(strategy._ic_trackers, dict)
        assert len(strategy._ic_trackers) == 4
        for name in ("flow_zscore", "tfi", "dof_score", "ofi"):
            assert name in strategy._ic_trackers
            assert isinstance(strategy._ic_trackers[name], InformationCoefficientTracker)

    def test_granger_state_initialized(self, strategy: FlowAlphaStrategy):
        assert isinstance(strategy._granger_valid, dict)
        assert len(strategy._granger_valid) == 0
        assert isinstance(strategy._granger_bar_count, dict)

    def test_vpin_threshold_default(self, strategy: FlowAlphaStrategy):
        assert strategy._vpin_threshold == 0.7

    def test_vpin_threshold_configurable(self):
        config: dict[str, Any] = {"vpin_threshold": 0.5}
        s = FlowAlphaStrategy(config, _MockLogger())
        assert s._vpin_threshold == 0.5

    def test_granger_retest_interval_default(self, strategy: FlowAlphaStrategy):
        assert strategy._granger_retest_interval == 500

    def test_granger_retest_interval_configurable(self):
        config: dict[str, Any] = {"granger_retest_interval": 200}
        s = FlowAlphaStrategy(config, _MockLogger())
        assert s._granger_retest_interval == 200


# ---------------------------------------------------------------------------
# Test 2: VPIN gate rejects when toxic
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVpinGate:
    def test_vpin_gate_rejects_toxic(self, strategy: FlowAlphaStrategy):
        """When VPIN reports is_toxic=True, on_bar should return None."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(25, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars, meta={"flow_zscore": 2.5, "tfi": 0.8})
        ms = _make_market_state(t)

        # Pre-set a mock VPIN that returns toxic
        mock_vpin = MagicMock(spec=VPINCalculator)
        mock_vpin.update.return_value = VPINResult(
            vpin=0.85,
            buy_volume=800.0,
            sell_volume=200.0,
            bucket_count=50,
            is_toxic=True,
        )
        strategy._vpin["AAPL"] = mock_vpin

        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

        # Verify rejection was logged
        logger = strategy.logger
        toxic_logs = [m for m in logger.messages if "vpin_toxic" in m[1]]
        assert len(toxic_logs) == 1

    def test_vpin_gate_allows_non_toxic(self, strategy: FlowAlphaStrategy):
        """When VPIN reports is_toxic=False, the gate should NOT reject."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(25, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars, meta={"flow_zscore": 2.5, "tfi": 0.8})
        ms = _make_market_state(t)

        mock_vpin = MagicMock(spec=VPINCalculator)
        mock_vpin.update.return_value = VPINResult(
            vpin=0.3,
            buy_volume=600.0,
            sell_volume=400.0,
            bucket_count=50,
            is_toxic=False,
        )
        strategy._vpin["AAPL"] = mock_vpin

        # The signal may still be None due to other gates, but VPIN should not block
        strategy.on_bar("AAPL", bar, ss, ms)

        logger = strategy.logger
        toxic_logs = [m for m in logger.messages if "vpin_toxic" in m[1]]
        assert len(toxic_logs) == 0

    def test_vpin_creates_calculator_on_first_bar(self, strategy: FlowAlphaStrategy):
        """VPINCalculator should be lazily initialized for a symbol."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(25, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t)

        strategy.on_bar("AAPL", bar, ss, ms)

        assert "AAPL" in strategy._vpin
        assert isinstance(strategy._vpin["AAPL"], VPINCalculator)


# ---------------------------------------------------------------------------
# Test 3: IC trackers are present and functional
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestICTrackers:
    def test_ic_trackers_have_correct_keys(self, strategy: FlowAlphaStrategy):
        expected = {"flow_zscore", "tfi", "dof_score", "ofi"}
        assert set(strategy._ic_trackers.keys()) == expected

    def test_static_weights_fallback(self, strategy: FlowAlphaStrategy):
        """When IC trackers have no data, _get_ic_weights returns static weights."""
        weights = strategy._get_ic_weights()
        assert weights == (0.35, 0.25, 0.20, 0.20)

    def test_ic_weights_use_positive_ics(self, strategy: FlowAlphaStrategy):
        """When IC trackers have data, weights are proportional to max(0, ic)."""
        # Manually set current_ic values via mock
        for name, ic_val in [("flow_zscore", 0.4), ("tfi", 0.2), ("dof_score", -0.1), ("ofi", 0.1)]:
            tracker = MagicMock(spec=InformationCoefficientTracker)
            tracker.current_ic = ic_val
            strategy._ic_trackers[name] = tracker

        weights = strategy._get_ic_weights()

        # dof_score IC is negative => clipped to 0
        # Total positive = 0.4 + 0.2 + 0 + 0.1 = 0.7
        assert abs(weights[0] - 0.4 / 0.7) < 1e-6  # flow_zscore
        assert abs(weights[1] - 0.2 / 0.7) < 1e-6  # tfi
        assert abs(weights[2] - 0.0) < 1e-6  # dof_score (clipped)
        assert abs(weights[3] - 0.1 / 0.7) < 1e-6  # ofi
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_ic_weights_all_zero_falls_back(self, strategy: FlowAlphaStrategy):
        """When all ICs are zero or negative, fall back to static weights."""
        for name in strategy._FLOW_SIGNAL_NAMES:
            tracker = MagicMock(spec=InformationCoefficientTracker)
            tracker.current_ic = -0.1
            strategy._ic_trackers[name] = tracker

        weights = strategy._get_ic_weights()
        assert weights == strategy._STATIC_WEIGHTS


# ---------------------------------------------------------------------------
# Test 4: Granger test runs periodically
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGrangerPeriodic:
    def test_granger_runs_after_interval(self, strategy: FlowAlphaStrategy):
        """Granger test should run after granger_retest_interval bars."""
        import numpy as np

        strategy._granger_retest_interval = 5

        # Generate realistic noisy data that Granger can process
        rng = np.random.default_rng(42)
        n = 150
        flow = rng.normal(0.0, 1.0, n).tolist()
        # Returns with some dependence on lagged flow + noise
        returns = [0.0] * n
        for i in range(2, n):
            returns[i] = 0.3 * flow[i - 1] + rng.normal(0.0, 0.5)

        strategy._return_history["AAPL"] = returns
        strategy._flow_history["AAPL"] = flow
        strategy._granger_bar_count["AAPL"] = 4  # next call will be 5 => triggers

        strategy._maybe_run_granger("AAPL")

        # After running, bar count should reset
        assert strategy._granger_bar_count["AAPL"] == 0
        # Result should be stored
        assert "AAPL" in strategy._granger_valid

    def test_granger_does_not_run_early(self, strategy: FlowAlphaStrategy):
        """Granger should NOT run before interval bars."""
        strategy._granger_retest_interval = 500
        strategy._granger_bar_count["AAPL"] = 10

        strategy._maybe_run_granger("AAPL")

        assert strategy._granger_bar_count["AAPL"] == 11
        assert "AAPL" not in strategy._granger_valid

    def test_granger_skips_insufficient_data(self, strategy: FlowAlphaStrategy):
        """Granger should skip if not enough history."""
        strategy._granger_retest_interval = 5
        strategy._granger_bar_count["AAPL"] = 4
        strategy._return_history["AAPL"] = [0.001] * 10  # too few
        strategy._flow_history["AAPL"] = [0.5] * 10

        strategy._maybe_run_granger("AAPL")

        # bar count should update but no result stored
        assert strategy._granger_bar_count["AAPL"] == 5
        assert "AAPL" not in strategy._granger_valid


# ---------------------------------------------------------------------------
# Test 5: GARCH is fed prices on each bar call
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGarchFedOnBar:
    def test_garch_created_on_first_bar(self, strategy: FlowAlphaStrategy):
        """GARCH estimator should be lazily initialized for a symbol on its first bar."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(25, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t)

        strategy.on_bar("AAPL", bar, ss, ms)

        assert "AAPL" in strategy._garch
        assert isinstance(strategy._garch["AAPL"], GARCHForecaster)

    def test_garch_update_called_each_bar(self, strategy: FlowAlphaStrategy):
        """GARCH.update should receive the bar's close price."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0 + i * 0.01) for i in range(25, 0, -1)]

        mock_garch = MagicMock(spec=GARCHForecaster)
        mock_garch.update.return_value = None
        strategy._garch["AAPL"] = mock_garch

        bar = _make_bar("AAPL", t, 151.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t)

        strategy.on_bar("AAPL", bar, ss, ms)

        mock_garch.update.assert_called_once_with(151.0)
