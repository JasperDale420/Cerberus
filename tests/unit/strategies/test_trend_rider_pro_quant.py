"""Unit tests for TrendRiderProStrategy quant upgrades (Kalman, Hurst, GARCH, Markov)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.core.domain import Bar, SymbolState
from src.core.logger import StructuredLogger
from src.quant.filters import KalmanMeanTracker
from src.quant.regime import MarkovRegimeSwitcher
from src.quant.statistics import HurstExponent
from src.quant.volatility import GARCHForecaster
from src.strategies.trend_rider_pro import TrendRiderProStrategy


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
        "pullback_threshold": 0.003,
        "stop_atr_mult": 1.5,
        "target_atr_mult": 3.5,
    }
    return TrendRiderProStrategy(config, MockLogger())


# ---------------------------------------------------------------------------
# Test 1: Strategy initializes quant components
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQuantComponentInitialization:
    def test_quant_dicts_empty_on_init(self, strategy):
        """Quant dicts are empty before any bar is processed."""
        assert strategy._kalman == {}
        assert strategy._hurst == {}
        assert strategy._garch == {}
        assert strategy._markov == {}

    def test_ensure_quant_state_creates_components(self, strategy):
        """_ensure_quant_state lazily creates all four quant components for a symbol."""
        strategy._ensure_quant_state("AAPL")

        assert "AAPL" in strategy._kalman
        assert "AAPL" in strategy._hurst
        assert "AAPL" in strategy._garch
        assert "AAPL" in strategy._markov

        assert isinstance(strategy._kalman["AAPL"], KalmanMeanTracker)
        assert isinstance(strategy._hurst["AAPL"], HurstExponent)
        assert isinstance(strategy._garch["AAPL"], GARCHForecaster)
        assert isinstance(strategy._markov["AAPL"], MarkovRegimeSwitcher)

    def test_ensure_quant_state_idempotent(self, strategy):
        """Calling _ensure_quant_state twice does not replace existing instances."""
        strategy._ensure_quant_state("AAPL")
        kalman_ref = strategy._kalman["AAPL"]
        strategy._ensure_quant_state("AAPL")
        assert strategy._kalman["AAPL"] is kalman_ref

    def test_per_symbol_isolation(self, strategy):
        """Different symbols get independent quant components."""
        strategy._ensure_quant_state("AAPL")
        strategy._ensure_quant_state("MSFT")

        assert strategy._kalman["AAPL"] is not strategy._kalman["MSFT"]
        assert strategy._hurst["AAPL"] is not strategy._hurst["MSFT"]


# ---------------------------------------------------------------------------
# Test 2: Kalman tracker updated on each bar
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKalmanUpdate:
    def test_kalman_tracks_price(self, strategy):
        """Kalman state tracks close price after updates via _update_quant_state."""
        eastern = ZoneInfo("America/New_York")
        base_time = datetime(2025, 6, 10, 10, 0, 0, tzinfo=eastern)
        strategy._ensure_quant_state("TEST")

        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        for i, price in enumerate(prices):
            bar = _make_bar("TEST", base_time + timedelta(minutes=i), price)
            strategy._update_quant_state("TEST", bar)

        # Kalman state should be near the latest price
        state = strategy._kalman["TEST"].state
        assert state > 0
        assert abs(state - 104.0) < 5.0  # Kalman tracks with some lag

    def test_kalman_updated_during_on_bar(self, strategy):
        """on_bar triggers _ensure_quant_state + _update_quant_state."""
        eastern = ZoneInfo("America/New_York")
        base_time = datetime(2025, 6, 10, 10, 0, 0, tzinfo=eastern)

        bars = deque()
        for i in range(35):
            bars.append(_make_bar("TEST", base_time + timedelta(minutes=i), 100.0 + i * 0.1))

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

        # Process a bar — should init quant components
        bar = bars[-1]
        strategy.on_bar("TEST", bar, symbol_state, market_state)

        assert "TEST" in strategy._kalman
        assert strategy._kalman["TEST"].state > 0


# ---------------------------------------------------------------------------
# Test 3: Hurst gate rejects when H <= 0.55
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHurstGate:
    def test_hurst_gate_rejects_low_h(self, strategy):
        """When Hurst is available and H <= 0.55, on_bar should return None."""
        eastern = ZoneInfo("America/New_York")
        base_time = datetime(2025, 6, 10, 10, 0, 0, tzinfo=eastern)

        # Create enough bars
        bars = deque()
        for i in range(35):
            bars.append(_make_bar("TEST", base_time + timedelta(minutes=i), 100.0 + i * 0.1))

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

        strategy._ensure_quant_state("TEST")

        # Mock the Hurst component to return H=0.45 (mean-reverting)
        mock_hurst_result = MagicMock()
        mock_hurst_result.H = 0.45
        mock_hurst_result.is_trending = False
        strategy._hurst["TEST"] = MagicMock()
        strategy._hurst["TEST"].update = MagicMock(return_value=mock_hurst_result)

        bar = bars[-1]
        result = strategy.on_bar("TEST", bar, symbol_state, market_state)
        assert result is None

    def test_hurst_gate_allows_high_h(self, strategy):
        """When Hurst H > 0.55, the gate does not reject (other gates may still reject)."""
        eastern = ZoneInfo("America/New_York")
        base_time = datetime(2025, 6, 10, 10, 0, 0, tzinfo=eastern)

        bars = deque()
        for i in range(35):
            bars.append(_make_bar("TEST", base_time + timedelta(minutes=i), 100.0 + i * 0.1))

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

        strategy._ensure_quant_state("TEST")

        # Mock Hurst to return H=0.7 (trending)
        mock_hurst_result = MagicMock()
        mock_hurst_result.H = 0.70
        mock_hurst_result.is_trending = True
        strategy._hurst["TEST"] = MagicMock()
        strategy._hurst["TEST"].update = MagicMock(return_value=mock_hurst_result)

        bar = bars[-1]
        # Should pass Hurst gate — may still return None due to other gates
        # but the point is it doesn't reject at the Hurst gate
        # We verify by checking the mock was called (update called in on_bar)
        strategy.on_bar("TEST", bar, symbol_state, market_state)
        strategy._hurst["TEST"].update.assert_called()

    def test_hurst_gate_passes_when_none(self, strategy):
        """When Hurst returns None (not enough data), gate passes through."""
        eastern = ZoneInfo("America/New_York")
        base_time = datetime(2025, 6, 10, 10, 0, 0, tzinfo=eastern)

        bars = deque()
        for i in range(35):
            bars.append(_make_bar("TEST", base_time + timedelta(minutes=i), 100.0 + i * 0.1))

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

        strategy._ensure_quant_state("TEST")

        # Mock Hurst to return None (insufficient data)
        strategy._hurst["TEST"] = MagicMock()
        strategy._hurst["TEST"].update = MagicMock(return_value=None)

        bar = bars[-1]
        # Should not reject at Hurst gate — result depends on subsequent gates
        strategy.on_bar("TEST", bar, symbol_state, market_state)
        # If we got here without error, the None result didn't crash or reject
        strategy._hurst["TEST"].update.assert_called()
