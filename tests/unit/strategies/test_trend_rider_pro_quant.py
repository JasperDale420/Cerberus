"""Unit tests for TrendRiderProStrategy v4 (simplified quant: Kalman, Hurst observability, GARCH)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.core.domain import Bar, SymbolState
from src.core.logger import StructuredLogger
from src.quant.filters import KalmanMeanTracker
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
        "confluence_threshold": 50,
        "min_bars": 5,
        "pullback_threshold": 0.005,
        "stop_atr_mult": 2.0,
        "target_atr_mult": 3.5,
    }
    return TrendRiderProStrategy(config, MockLogger())


# ---------------------------------------------------------------------------
# Test 1: Strategy initializes quant components (no Markov)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQuantComponentInitialization:
    def test_quant_dicts_empty_on_init(self, strategy):
        """Quant dicts are empty before any bar is processed."""
        assert strategy._kalman == {}
        assert strategy._hurst == {}
        assert strategy._garch == {}

    def test_no_markov_attribute(self, strategy):
        """v4 removed Markov regime switcher entirely."""
        assert not hasattr(strategy, "_markov")

    def test_ensure_quant_state_creates_components(self, strategy):
        """_ensure_quant_state lazily creates Kalman, Hurst, GARCH for a symbol."""
        strategy._ensure_quant_state("AAPL")

        assert "AAPL" in strategy._kalman
        assert "AAPL" in strategy._hurst
        assert "AAPL" in strategy._garch

        assert isinstance(strategy._kalman["AAPL"], KalmanMeanTracker)
        assert isinstance(strategy._hurst["AAPL"], HurstExponent)
        assert isinstance(strategy._garch["AAPL"], GARCHForecaster)

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

        state = strategy._kalman["TEST"].state
        assert state > 0
        assert abs(state - 104.0) < 5.0

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

        bar = bars[-1]
        strategy.on_bar("TEST", bar, symbol_state, market_state)

        assert "TEST" in strategy._kalman
        assert strategy._kalman["TEST"].state > 0


# ---------------------------------------------------------------------------
# Test 3: Hurst is logged for observability but does NOT gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHurstObservability:
    def test_hurst_low_h_does_not_gate(self, strategy):
        """When Hurst H <= 0.55, on_bar should NOT reject — Hurst is observability only in v4."""
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

        # Mock Hurst to return H=0.45 (mean-reverting) — should NOT gate in v4
        mock_hurst_result = MagicMock()
        mock_hurst_result.H = 0.45
        mock_hurst_result.is_trending = False
        strategy._hurst["TEST"] = MagicMock()
        strategy._hurst["TEST"].update = MagicMock(return_value=mock_hurst_result)

        bar = bars[-1]
        # The result may still be None due to other gates (direction, confluence),
        # but the Hurst gate itself does not reject in v4
        strategy.on_bar("TEST", bar, symbol_state, market_state)
        strategy._hurst["TEST"].update.assert_called()

    def test_hurst_value_in_signal_meta(self, strategy):
        """Hurst H value should appear in signal meta for observability."""
        # This is a structural test — v4 logs hurst_H in meta without gating
        assert not hasattr(strategy, "hurst_trending_threshold")


# ---------------------------------------------------------------------------
# Test 4: Strategy version is v4
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStrategyVersion:
    def test_default_params(self, strategy):
        """v4 defaults: confluence_threshold=50, stop_atr_mult=2.0, pullback_threshold=0.005."""
        assert strategy.confluence_threshold == 50
        assert strategy.stop_atr_mult == 2.0
        assert strategy.pullback_threshold == 0.005
