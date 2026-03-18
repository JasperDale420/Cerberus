"""Tests for mean_reversion_pro quant upgrades: GARCH, Hurst, AdaptiveThreshold."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.core.domain import Bar, MarketState, SymbolState
from src.quant.regime import AdaptiveThresholdEngine
from src.quant.statistics import HurstExponent, HurstResult
from src.quant.volatility import GARCHForecaster
from src.strategies.mean_reversion_pro import MeanReversionProStrategy

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
        open=close,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=volume,
    )


def _make_market_state(time: datetime) -> MarketState:
    return MagicMock(spec=MarketState, regime_snapshot=None)


def _make_symbol_state(symbol: str, bars: list[Bar]) -> SymbolState:
    return SymbolState(
        symbol=symbol,
        bars=deque(bars),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )


def _base_time() -> datetime:
    return datetime(2025, 6, 15, 10, 0, tzinfo=ET)


@pytest.fixture
def strategy() -> MeanReversionProStrategy:
    config: dict[str, Any] = {
        "confluence_threshold": 65.0,
        "min_bars": 5,
    }
    return MeanReversionProStrategy(config, _MockLogger())


# ---------------------------------------------------------------------------
# Test 1: Strategy initializes quant components
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQuantInitialization:
    def test_garch_dict_initialized(self, strategy: MeanReversionProStrategy):
        assert isinstance(strategy._garch, dict)
        assert len(strategy._garch) == 0

    def test_hurst_dict_initialized(self, strategy: MeanReversionProStrategy):
        assert isinstance(strategy._hurst, dict)
        assert len(strategy._hurst) == 0

    def test_threshold_engine_initialized(self, strategy: MeanReversionProStrategy):
        assert isinstance(strategy._threshold_engine, AdaptiveThresholdEngine)

    def test_hurst_gate_threshold_default(self, strategy: MeanReversionProStrategy):
        assert strategy._hurst_gate_threshold == 0.45

    def test_hurst_gate_threshold_configurable(self):
        config = {"hurst_gate_threshold": 0.40}
        s = MeanReversionProStrategy(config, _MockLogger())
        assert s._hurst_gate_threshold == 0.40


# ---------------------------------------------------------------------------
# Test 2: GARCH is fed prices on each bar call
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGarchFedOnBar:
    def test_garch_created_on_first_bar(self, strategy: MeanReversionProStrategy):
        """GARCH estimator should be lazily initialized for a symbol on its first bar."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(35, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t)

        # on_bar will return None for many reasons (no VWAP data, etc.),
        # but GARCH should still be initialized
        strategy.on_bar("AAPL", bar, ss, ms)

        assert "AAPL" in strategy._garch
        assert isinstance(strategy._garch["AAPL"], GARCHForecaster)

    def test_garch_update_called_each_bar(self, strategy: MeanReversionProStrategy):
        """GARCH.update should receive the bar's close price."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0 + i * 0.01) for i in range(35, 0, -1)]

        # Manually set up a mock GARCH to track calls
        mock_garch = MagicMock(spec=GARCHForecaster)
        mock_garch.update.return_value = None
        strategy._garch["AAPL"] = mock_garch

        bar = _make_bar("AAPL", t, 151.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t)

        strategy.on_bar("AAPL", bar, ss, ms)

        mock_garch.update.assert_called_once_with(151.0)

    def test_hurst_created_on_first_bar(self, strategy: MeanReversionProStrategy):
        """Hurst estimator should be lazily initialized for a symbol."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(35, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t)

        strategy.on_bar("AAPL", bar, ss, ms)

        assert "AAPL" in strategy._hurst
        assert isinstance(strategy._hurst["AAPL"], HurstExponent)


# ---------------------------------------------------------------------------
# Test 3: Hurst gate rejects when H >= 0.45
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHurstGate:
    def test_hurst_gate_rejects_trending(self, strategy: MeanReversionProStrategy):
        """When Hurst H >= 0.45, on_bar should return None and log rejection."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(35, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t)

        # Mock GARCH to return None (not enough data)
        mock_garch = MagicMock(spec=GARCHForecaster)
        mock_garch.update.return_value = None
        strategy._garch["AAPL"] = mock_garch

        # Mock Hurst to return trending result (H=0.65)
        mock_hurst = MagicMock(spec=HurstExponent)
        mock_hurst.update.return_value = HurstResult(
            H=0.65,
            is_mean_reverting=False,
            is_trending=True,
            n_observations=200,
        )
        strategy._hurst["AAPL"] = mock_hurst

        result = strategy.on_bar("AAPL", bar, ss, ms)

        assert result is None
        # Check that hurst_gate_rejected was logged
        logger = strategy.logger
        hurst_logs = [m for m in logger.messages if m[1] == "hurst_gate_rejected"]
        assert len(hurst_logs) == 1
        assert hurst_logs[0][2]["hurst_h"] == 0.65

    def test_hurst_gate_rejects_at_boundary(self, strategy: MeanReversionProStrategy):
        """When Hurst H == 0.45 (boundary), it should reject."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(35, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t)

        mock_garch = MagicMock(spec=GARCHForecaster)
        mock_garch.update.return_value = None
        strategy._garch["AAPL"] = mock_garch

        mock_hurst = MagicMock(spec=HurstExponent)
        mock_hurst.update.return_value = HurstResult(
            H=0.45,
            is_mean_reverting=False,
            is_trending=False,
            n_observations=200,
        )
        strategy._hurst["AAPL"] = mock_hurst

        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_hurst_gate_allows_mean_reverting(self, strategy: MeanReversionProStrategy):
        """When Hurst H < 0.45, the gate should NOT reject (other gates may still reject)."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(35, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t)

        mock_garch = MagicMock(spec=GARCHForecaster)
        mock_garch.update.return_value = None
        strategy._garch["AAPL"] = mock_garch

        mock_hurst = MagicMock(spec=HurstExponent)
        mock_hurst.update.return_value = HurstResult(
            H=0.35,
            is_mean_reverting=True,
            is_trending=False,
            n_observations=200,
        )
        strategy._hurst["AAPL"] = mock_hurst

        # The signal will still be None (no VWAP data etc.), but the Hurst
        # gate log should NOT appear
        strategy.on_bar("AAPL", bar, ss, ms)

        logger = strategy.logger
        hurst_logs = [m for m in logger.messages if m[1] == "hurst_gate_rejected"]
        assert len(hurst_logs) == 0

    def test_hurst_none_does_not_reject(self, strategy: MeanReversionProStrategy):
        """When Hurst returns None (not enough data), the gate should pass."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(35, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t)

        mock_garch = MagicMock(spec=GARCHForecaster)
        mock_garch.update.return_value = None
        strategy._garch["AAPL"] = mock_garch

        mock_hurst = MagicMock(spec=HurstExponent)
        mock_hurst.update.return_value = None
        strategy._hurst["AAPL"] = mock_hurst

        strategy.on_bar("AAPL", bar, ss, ms)

        logger = strategy.logger
        hurst_logs = [m for m in logger.messages if m[1] == "hurst_gate_rejected"]
        assert len(hurst_logs) == 0
