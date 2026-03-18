"""Tests for momentum_fade quant upgrades: GARCH z-score, exhaustion model, Hurst gate, entropy filter."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.core.domain import Bar, MarketState, SymbolState
from src.quant.statistics import HurstExponent, HurstResult
from src.quant.volatility import GARCHForecaster
from src.strategies.momentum_fade import MomentumFadeStrategy

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


def _make_bar(symbol: str, time: datetime, close: float, volume: int = 1000, vwap: float | None = None) -> Bar:
    return Bar(
        symbol=symbol,
        time=time,
        open=close,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=volume,
        vwap=vwap,
    )


def _make_market_state(time: datetime, entropy_score: float = 0.0) -> MagicMock:
    snapshot = MagicMock()
    snapshot.vol = None
    snapshot.session = None
    snapshot.trend = None
    snapshot.entropy_score = entropy_score
    snapshot.regime_tags = {}
    snapshot.confidence = {}
    ms = MagicMock(spec=MarketState)
    ms.regime_snapshot = snapshot
    return ms


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
def strategy() -> MomentumFadeStrategy:
    config: dict[str, Any] = {
        "confluence_threshold": 60.0,
        "min_bars": 5,
    }
    return MomentumFadeStrategy(config, _MockLogger())


# ---------------------------------------------------------------------------
# Test 1: Strategy initializes GARCH and Hurst components
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQuantInitialization:
    def test_garch_dict_initialized(self, strategy: MomentumFadeStrategy):
        assert isinstance(strategy._garch, dict)
        assert len(strategy._garch) == 0

    def test_hurst_dict_initialized(self, strategy: MomentumFadeStrategy):
        assert isinstance(strategy._hurst, dict)
        assert len(strategy._hurst) == 0

    def test_velocity_history_initialized(self, strategy: MomentumFadeStrategy):
        assert isinstance(strategy._velocity_history, dict)
        assert len(strategy._velocity_history) == 0

    def test_acceleration_history_initialized(self, strategy: MomentumFadeStrategy):
        assert isinstance(strategy._acceleration_history, dict)
        assert len(strategy._acceleration_history) == 0

    def test_garch_created_on_first_bar(self, strategy: MomentumFadeStrategy):
        """GARCH estimator should be lazily initialized for a symbol on its first bar."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(35, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t)

        strategy.on_bar("AAPL", bar, ss, ms)

        assert "AAPL" in strategy._garch
        assert isinstance(strategy._garch["AAPL"], GARCHForecaster)

    def test_hurst_created_on_first_bar(self, strategy: MomentumFadeStrategy):
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

    def test_garch_zscore_threshold_default(self, strategy: MomentumFadeStrategy):
        assert strategy.garch_zscore_threshold == 2.0

    def test_garch_zscore_threshold_configurable(self):
        config: dict[str, Any] = {"garch_zscore_threshold": 2.5}
        s = MomentumFadeStrategy(config, _MockLogger())
        assert s.garch_zscore_threshold == 2.5


# ---------------------------------------------------------------------------
# Test 2: Momentum exhaustion model computes velocity and acceleration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExhaustionModel:
    def test_insufficient_data_returns_no_exhaustion(self, strategy: MomentumFadeStrategy):
        """With insufficient close history, exhaustion should not flag."""
        strategy._ensure_symbol_state("AAPL")
        # Feed only 3 prices — not enough for velocity_lookback=5 + acceleration_lookback=3
        for price in [100.0, 101.0, 102.0]:
            velocity, acceleration, is_exhausting = strategy._compute_exhaustion("AAPL", price)
        assert velocity == 0.0
        assert not is_exhausting

    def test_velocity_computed_correctly(self, strategy: MomentumFadeStrategy):
        """Velocity should be (close - close_n_ago) / close_n_ago."""
        strategy._ensure_symbol_state("AAPL")
        # Need velocity_lookback(5) + acceleration_lookback(3) + 1 = 9 prices
        prices = [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0]
        for p in prices:
            velocity, acceleration, is_exhausting = strategy._compute_exhaustion("AAPL", p)

        # Last velocity: (104.0 - 102.0) / 102.0 = ~0.0196
        expected_velocity = (104.0 - prices[-1 - strategy.velocity_lookback]) / prices[-1 - strategy.velocity_lookback]
        assert abs(velocity - expected_velocity) < 1e-9

    def test_exhaustion_detected_on_deceleration(self, strategy: MomentumFadeStrategy):
        """When velocity is high and acceleration opposes it, exhaustion should be True."""
        strategy._ensure_symbol_state("TEST")
        # Need enough data for multiple velocity samples to compute acceleration.
        # velocity_lookback=5, acceleration_lookback=3: need 5+3+1=9 prices for first
        # velocity, then 3 more bars of velocity before acceleration kicks in.
        # Total: ~12+ prices. Create strong upward move that decelerates at the end.
        prices = [
            100.0,
            101.0,
            103.0,
            106.0,
            110.0,
            115.0,  # strong rally
            120.0,
            124.0,
            126.0,
            127.0,
            127.3,
            127.4,  # decelerating
        ]
        for p in prices:
            velocity, acceleration, is_exhausting = strategy._compute_exhaustion("TEST", p)

        # Velocity should be positive (price went up), acceleration negative (decelerating)
        assert velocity > 0, f"Expected positive velocity, got {velocity}"
        assert acceleration < 0, f"Expected negative acceleration (deceleration), got {acceleration}"
        assert is_exhausting, "Expected exhaustion to be detected"

    def test_no_exhaustion_when_accelerating(self, strategy: MomentumFadeStrategy):
        """When velocity is high and acceleration has same sign, no exhaustion."""
        strategy._ensure_symbol_state("TEST")
        # Steadily accelerating upward — each velocity reading is larger than the last
        prices = [
            100.0,
            100.1,
            100.3,
            100.6,
            101.0,
            101.5,  # slow start
            102.5,
            104.0,
            106.0,
            109.0,
            113.0,
            118.0,  # accelerating
        ]
        for p in prices:
            velocity, acceleration, is_exhausting = strategy._compute_exhaustion("TEST", p)

        # Velocity positive, acceleration positive — not exhausting
        assert velocity > 0
        assert acceleration > 0
        assert not is_exhausting


# ---------------------------------------------------------------------------
# Test 3: Hurst gate rejects when H >= 0.5
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHurstGate:
    def test_hurst_gate_rejects_trending(self, strategy: MomentumFadeStrategy):
        """When Hurst H >= 0.5, on_bar should return None and log rejection."""
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
            H=0.65,
            is_mean_reverting=False,
            is_trending=True,
            n_observations=200,
        )
        strategy._hurst["AAPL"] = mock_hurst

        result = strategy.on_bar("AAPL", bar, ss, ms)

        assert result is None
        logger = strategy.logger
        hurst_logs = [m for m in logger.messages if m[1] == "hurst_gate_rejected"]
        assert len(hurst_logs) == 1
        assert hurst_logs[0][2]["hurst_h"] == 0.65

    def test_hurst_gate_rejects_at_boundary(self, strategy: MomentumFadeStrategy):
        """When Hurst H == 0.5 (boundary), it should reject."""
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
            H=0.5,
            is_mean_reverting=False,
            is_trending=False,
            n_observations=200,
        )
        strategy._hurst["AAPL"] = mock_hurst

        result = strategy.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_hurst_gate_allows_mean_reverting(self, strategy: MomentumFadeStrategy):
        """When Hurst H < 0.5, the gate should NOT reject."""
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

        strategy.on_bar("AAPL", bar, ss, ms)

        logger = strategy.logger
        hurst_logs = [m for m in logger.messages if m[1] == "hurst_gate_rejected"]
        assert len(hurst_logs) == 0

    def test_hurst_none_does_not_reject(self, strategy: MomentumFadeStrategy):
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


# ---------------------------------------------------------------------------
# Test 4: Entropy filter rejects when entropy > 0.8
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntropyFilter:
    def test_entropy_filter_rejects_high_entropy(self, strategy: MomentumFadeStrategy):
        """When entropy_score > 0.8, on_bar should return None and log rejection."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(35, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t, entropy_score=0.95)

        # Mock quant to bypass Hurst gate
        mock_garch = MagicMock(spec=GARCHForecaster)
        mock_garch.update.return_value = None
        strategy._garch["AAPL"] = mock_garch

        mock_hurst = MagicMock(spec=HurstExponent)
        mock_hurst.update.return_value = HurstResult(
            H=0.35, is_mean_reverting=True, is_trending=False, n_observations=200
        )
        strategy._hurst["AAPL"] = mock_hurst

        result = strategy.on_bar("AAPL", bar, ss, ms)

        assert result is None
        logger = strategy.logger
        entropy_logs = [m for m in logger.messages if m[1] == "entropy_filter_rejected"]
        assert len(entropy_logs) == 1
        assert entropy_logs[0][2]["entropy_score"] == 0.95

    def test_entropy_filter_allows_low_entropy(self, strategy: MomentumFadeStrategy):
        """When entropy_score <= 0.8, the filter should not reject."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(35, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t, entropy_score=0.5)

        mock_garch = MagicMock(spec=GARCHForecaster)
        mock_garch.update.return_value = None
        strategy._garch["AAPL"] = mock_garch

        mock_hurst = MagicMock(spec=HurstExponent)
        mock_hurst.update.return_value = HurstResult(
            H=0.35, is_mean_reverting=True, is_trending=False, n_observations=200
        )
        strategy._hurst["AAPL"] = mock_hurst

        # Will likely return None for other reasons, but should NOT log entropy rejection
        strategy.on_bar("AAPL", bar, ss, ms)

        logger = strategy.logger
        entropy_logs = [m for m in logger.messages if m[1] == "entropy_filter_rejected"]
        assert len(entropy_logs) == 0

    def test_entropy_at_boundary_allows(self, strategy: MomentumFadeStrategy):
        """When entropy_score == 0.8 (boundary), it should NOT reject."""
        t = _base_time()
        bars = [_make_bar("AAPL", t - timedelta(minutes=i), 150.0) for i in range(35, 0, -1)]
        bar = _make_bar("AAPL", t, 150.0)
        bars.append(bar)
        ss = _make_symbol_state("AAPL", bars)
        ms = _make_market_state(t, entropy_score=0.8)

        mock_garch = MagicMock(spec=GARCHForecaster)
        mock_garch.update.return_value = None
        strategy._garch["AAPL"] = mock_garch

        mock_hurst = MagicMock(spec=HurstExponent)
        mock_hurst.update.return_value = HurstResult(
            H=0.35, is_mean_reverting=True, is_trending=False, n_observations=200
        )
        strategy._hurst["AAPL"] = mock_hurst

        strategy.on_bar("AAPL", bar, ss, ms)

        logger = strategy.logger
        entropy_logs = [m for m in logger.messages if m[1] == "entropy_filter_rejected"]
        assert len(entropy_logs) == 0

    def test_entropy_filter_threshold_configurable(self):
        """Entropy threshold should be configurable via config."""
        config: dict[str, Any] = {"entropy_threshold": 0.7}
        s = MomentumFadeStrategy(config, _MockLogger())
        assert s.entropy_threshold == 0.7
