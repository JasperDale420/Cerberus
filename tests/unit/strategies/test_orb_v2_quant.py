"""Unit tests for ORBV2Strategy quant upgrades (CUSUM, GARCH, Variance Ratio, BOCPD)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.analysis.variance_ratio import VarianceRatioCalculator
from src.core.domain import Bar, SymbolState
from src.core.logger import StructuredLogger
from src.quant.statistics import CUSUMDetector
from src.quant.volatility import GARCHForecaster
from src.strategies.orb_v2 import ORBV2Strategy


class MockLogger(StructuredLogger):
    def __init__(self):
        pass

    def info(self, msg, **kwargs):
        pass

    def error(self, msg, **kwargs):
        pass

    def warning(self, msg, **kwargs):
        pass

    def debug(self, msg, **kwargs):
        pass


def _make_bar(symbol: str, time: datetime, price: float, volume: int = 1000) -> Bar:
    return Bar(
        symbol=symbol,
        time=time,
        open=price,
        high=price + 0.5,
        low=price - 0.5,
        close=price,
        volume=volume,
    )


@pytest.fixture
def strategy():
    config = {
        "confluence_threshold": 65,
        "min_bars": 5,
        "vol_gate_mult": 1.2,
        "target_range_mult": 2.0,
    }
    return ORBV2Strategy(config, MockLogger())


# ---------------------------------------------------------------------------
# Test 1: Strategy initializes quant components
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQuantComponentInitialization:
    def test_quant_dicts_empty_on_init(self, strategy):
        """Quant dicts are empty before any bar is processed."""
        assert strategy._cusum == {}
        assert strategy._garch == {}
        assert strategy._vr == {}

    def test_ensure_quant_state_creates_components(self, strategy):
        """_ensure_quant_state lazily creates all three quant components for a symbol."""
        strategy._ensure_quant_state("AAPL")

        assert "AAPL" in strategy._cusum
        assert "AAPL" in strategy._garch
        assert "AAPL" in strategy._vr

        assert isinstance(strategy._cusum["AAPL"], CUSUMDetector)
        assert isinstance(strategy._garch["AAPL"], GARCHForecaster)
        assert isinstance(strategy._vr["AAPL"], VarianceRatioCalculator)

    def test_ensure_quant_state_idempotent(self, strategy):
        """Calling _ensure_quant_state twice does not replace existing instances."""
        strategy._ensure_quant_state("AAPL")
        cusum_ref = strategy._cusum["AAPL"]
        strategy._ensure_quant_state("AAPL")
        assert strategy._cusum["AAPL"] is cusum_ref

    def test_per_symbol_isolation(self, strategy):
        """Different symbols get independent quant components."""
        strategy._ensure_quant_state("AAPL")
        strategy._ensure_quant_state("MSFT")

        assert strategy._cusum["AAPL"] is not strategy._cusum["MSFT"]
        assert strategy._garch["AAPL"] is not strategy._garch["MSFT"]
        assert strategy._vr["AAPL"] is not strategy._vr["MSFT"]


# ---------------------------------------------------------------------------
# Test 2: CUSUM detector is initialized and updated per symbol
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCUSUMIntegration:
    def test_cusum_initialized_during_range_build(self, strategy):
        """CUSUM detector is initialised when _update_range is called."""
        eastern = ZoneInfo("America/New_York")
        base_time = datetime(2025, 6, 10, 9, 30, 0, tzinfo=eastern)

        state = strategy._empty_state()
        state["date"] = base_time.date()

        bar = _make_bar("TEST", base_time, 100.0)
        strategy._update_range("TEST", state, bar)

        assert "TEST" in strategy._cusum
        assert isinstance(strategy._cusum["TEST"], CUSUMDetector)

    def test_cusum_fed_during_range_period(self, strategy):
        """CUSUM receives updates during the opening range build."""
        eastern = ZoneInfo("America/New_York")
        base_time = datetime(2025, 6, 10, 9, 30, 0, tzinfo=eastern)

        state = strategy._empty_state()
        state["date"] = base_time.date()

        # Feed several bars to build up CUSUM internal state
        for i in range(5):
            bar = _make_bar("TEST", base_time + timedelta(minutes=i), 100.0 + i * 0.1)
            strategy._update_range("TEST", state, bar)

        # CUSUM should have received 5 updates
        assert strategy._cusum["TEST"]._count == 5

    def test_cusum_feeds_post_range(self, strategy):
        """CUSUM continues to receive updates during the breakout evaluation phase."""
        eastern = ZoneInfo("America/New_York")

        bars = deque()
        for i in range(10):
            bars.append(_make_bar("TEST", datetime(2025, 6, 10, 10, 0 + i, 0, tzinfo=eastern), 100.0 + i * 0.1))

        symbol_state = SymbolState(
            symbol="TEST",
            bars=bars,
            indicators={},
            position=None,
            open_orders={},
            allowed_strategies=[],
            meta={},
        )
        market_state = MagicMock()
        market_state.regime_snapshot = None

        # Force range complete state
        state = strategy._get_state("TEST")
        state["date"] = datetime(2025, 6, 10, tzinfo=eastern).date()
        state["range_complete"] = True
        state["range_high"] = 99.0
        state["range_low"] = 98.0
        state["range_bars"] = 5

        strategy._ensure_quant_state("TEST")
        initial_count = strategy._cusum["TEST"]._count

        # Process a post-range bar — should feed CUSUM
        bar = bars[-1]
        strategy.on_bar("TEST", bar, symbol_state, market_state)

        assert strategy._cusum["TEST"]._count > initial_count


# ---------------------------------------------------------------------------
# Test 3: Variance ratio gate is checked
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVarianceRatioGate:
    def test_vr_initialized_during_range(self, strategy):
        """VR calculator is initialized when range bars are processed."""
        eastern = ZoneInfo("America/New_York")
        base_time = datetime(2025, 6, 10, 9, 30, 0, tzinfo=eastern)

        state = strategy._empty_state()
        state["date"] = base_time.date()

        bar = _make_bar("TEST", base_time, 100.0)
        strategy._update_range("TEST", state, bar)

        assert "TEST" in strategy._vr
        assert isinstance(strategy._vr["TEST"], VarianceRatioCalculator)

    def test_vr_fed_during_range_period(self, strategy):
        """VR calculator receives close prices during range building."""
        eastern = ZoneInfo("America/New_York")
        base_time = datetime(2025, 6, 10, 9, 30, 0, tzinfo=eastern)

        state = strategy._empty_state()
        state["date"] = base_time.date()

        for i in range(10):
            bar = _make_bar("TEST", base_time + timedelta(minutes=i), 100.0 + i * 0.1)
            strategy._update_range("TEST", state, bar)

        # VR internal prices buffer should have 10 entries
        assert len(strategy._vr["TEST"]._prices) == 10

    def test_vr_mean_reverting_suppresses_breakout(self, strategy):
        """When VR signals mean-reversion, breakout is suppressed."""
        eastern = ZoneInfo("America/New_York")

        bars = deque()
        for i in range(10):
            bars.append(_make_bar("TEST", datetime(2025, 6, 10, 10, 0 + i, 0, tzinfo=eastern), 105.0, volume=5000))

        symbol_state = SymbolState(
            symbol="TEST",
            bars=bars,
            indicators={"sma_vol_1m:20": 1000.0},
            position=None,
            open_orders={},
            allowed_strategies=[],
            meta={},
        )
        market_state = MagicMock()
        market_state.regime_snapshot = None

        # Set up range state where price is above range_high
        state = strategy._get_state("TEST")
        state["date"] = datetime(2025, 6, 10, tzinfo=eastern).date()
        state["range_complete"] = True
        state["range_high"] = 100.0
        state["range_low"] = 99.0
        state["range_bars"] = 5

        strategy._ensure_quant_state("TEST")

        # Mock VR to return mean-reverting result
        mock_vr_result = MagicMock()
        mock_vr_result.is_mean_reverting = True
        mock_vr_result.vr = 0.6
        strategy._vr["TEST"] = MagicMock()
        strategy._vr["TEST"].update = MagicMock(return_value=mock_vr_result)

        bar = bars[-1]
        result = strategy.on_bar("TEST", bar, symbol_state, market_state)
        # Should be suppressed by VR gate (or earlier gates) — result is None
        assert result is None
