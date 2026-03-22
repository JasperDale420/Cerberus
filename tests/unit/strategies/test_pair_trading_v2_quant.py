"""Tests for pair_trading_v2 quant upgrades: Kalman, cointegration, GARCH, OU."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

import pytest

from src.analysis.ou_estimator import OUEstimator
from src.core.domain import (
    Bar,
    MarketState,
    Regime,
    SymbolState,
)
from src.core.logger import StructuredLogger
from src.quant.cointegration import RollingCointegrationMonitor
from src.quant.filters import KalmanHedgeRatio
from src.strategies.pair_trading_v2 import PairTradingV2Strategy, _pair_key

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAIR_CONFIG = {
    "pairs": [{"leg_a": "AAPL", "leg_b": "MSFT"}],
    "entry_z_threshold": 2.0,
    "stop_z_threshold": 3.5,
    "spread_lookback": 90,
    "min_bars": 10,
    "max_hold_days": 15,
}


@pytest.fixture
def logger():
    return StructuredLogger("test_pair_v2_quant")


@pytest.fixture
def strategy(logger):
    return PairTradingV2Strategy(PAIR_CONFIG, logger)


def _make_bar(symbol: str, close: float, t: datetime) -> Bar:
    return Bar(
        symbol=symbol,
        time=t,
        open=close - 0.5,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=10000,
    )


def _make_symbol_state(symbol: str) -> SymbolState:
    return SymbolState(
        symbol=symbol,
        bars=deque(),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["pair_trading_v2"],
        meta={"avg_volume_20": 50000, "atr": 2.0},
    )


def _make_market_state(t: datetime) -> MarketState:
    return MarketState(
        time=t,
        regime=Regime.CHOP,
        meta={},
    )


# ---------------------------------------------------------------------------
# Attribute / initialization tests
# ---------------------------------------------------------------------------


class TestQuantAttributes:
    """Verify the strategy has all new quant attributes after construction."""

    def test_has_kalman_per_pair(self, strategy):
        key = _pair_key("AAPL", "MSFT")
        ps = strategy._pair_state[key]
        assert isinstance(ps["kalman"], KalmanHedgeRatio)

    def test_no_ema_hedge_in_pair_state(self, strategy):
        key = _pair_key("AAPL", "MSFT")
        ps = strategy._pair_state[key]
        assert "hedge_ema" not in ps

    def test_has_cointegration_monitors(self, strategy):
        key = _pair_key("AAPL", "MSFT")
        assert key in strategy._coint_monitors
        assert isinstance(strategy._coint_monitors[key], RollingCointegrationMonitor)

    def test_has_ou_estimators(self, strategy):
        key = _pair_key("AAPL", "MSFT")
        assert key in strategy._ou_estimators
        assert isinstance(strategy._ou_estimators[key], OUEstimator)

    def test_has_correlation_deques(self, strategy):
        key = _pair_key("AAPL", "MSFT")
        ps = strategy._pair_state[key]
        assert "prices_a" in ps
        assert "prices_b" in ps
        assert isinstance(ps["prices_a"], deque)

    def test_max_hold_days_config(self, strategy):
        assert strategy.max_hold_days == 15

    def test_min_correlation_config(self):
        logger = StructuredLogger("test")
        config = {**PAIR_CONFIG, "min_correlation": 0.7}
        s = PairTradingV2Strategy(config, logger)
        assert s.min_correlation == 0.7


# ---------------------------------------------------------------------------
# Integration: feed bars and verify behaviour
# ---------------------------------------------------------------------------


class TestBarFeedIntegration:
    """Feed bars through the strategy and verify quant gates engage."""

    def test_warmup_returns_none(self, strategy):
        """During warm-up phase (bar_count < min_bars), no signal."""
        t = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
        ms = _make_market_state(t)

        for i in range(PAIR_CONFIG["min_bars"] - 1):
            bar_a = _make_bar("AAPL", 150.0 + i * 0.01, t + timedelta(minutes=i))
            bar_b = _make_bar("MSFT", 300.0 + i * 0.01, t + timedelta(minutes=i))
            ss_a = _make_symbol_state("AAPL")
            ss_b = _make_symbol_state("MSFT")
            assert strategy.on_bar("AAPL", bar_a, ss_a, ms) is None
            assert strategy.on_bar("MSFT", bar_b, ss_b, ms) is None

    def test_kalman_updates_on_bars(self, strategy):
        """Kalman hedge ratio updates as bars flow in."""
        key = _pair_key("AAPL", "MSFT")
        t = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
        ms = _make_market_state(t)

        for i in range(20):
            bar_a = _make_bar("AAPL", 150.0 + i * 0.1, t + timedelta(minutes=i))
            bar_b = _make_bar("MSFT", 300.0 + i * 0.2, t + timedelta(minutes=i))
            ss_a = _make_symbol_state("AAPL")
            ss_b = _make_symbol_state("MSFT")
            strategy.on_bar("AAPL", bar_a, ss_a, ms)
            strategy.on_bar("MSFT", bar_b, ss_b, ms)

        kalman = strategy._pair_state[key]["kalman"]
        # Hedge ratio should have moved from 0 towards ~0.5 (AAPL/MSFT price ratio)
        assert kalman.hedge_ratio != 0.0

    def test_cointegration_monitor_updates(self, strategy):
        """Cointegration monitor accumulates observations on each bar."""
        key = _pair_key("AAPL", "MSFT")
        t = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
        ms = _make_market_state(t)

        for i in range(30):
            bar_a = _make_bar("AAPL", 150.0 + i * 0.05, t + timedelta(minutes=i))
            bar_b = _make_bar("MSFT", 300.0 + i * 0.1, t + timedelta(minutes=i))
            ss_a = _make_symbol_state("AAPL")
            ss_b = _make_symbol_state("MSFT")
            strategy.on_bar("AAPL", bar_a, ss_a, ms)
            strategy.on_bar("MSFT", bar_b, ss_b, ms)

        monitor = strategy._coint_monitors[key]
        # Monitor receives an update on each on_bar call where both prices > 0
        # (both AAPL and MSFT bars trigger updates), so count >= 30
        assert len(monitor._x) >= 30

    def test_rolling_correlation_populates(self, strategy):
        """Price deques for correlation tracking fill as bars flow."""
        key = _pair_key("AAPL", "MSFT")
        t = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
        ms = _make_market_state(t)

        for i in range(25):
            bar_a = _make_bar("AAPL", 150.0 + i * 0.1, t + timedelta(minutes=i))
            bar_b = _make_bar("MSFT", 300.0 + i * 0.2, t + timedelta(minutes=i))
            ss_a = _make_symbol_state("AAPL")
            ss_b = _make_symbol_state("MSFT")
            strategy.on_bar("AAPL", bar_a, ss_a, ms)
            strategy.on_bar("MSFT", bar_b, ss_b, ms)

        ps = strategy._pair_state[key]
        assert len(ps["prices_a"]) >= 20
        corr = strategy._rolling_correlation(ps)
        assert corr is not None
        # Linearly increasing prices should have high correlation
        assert corr > 0.9

    def test_pair_key_canonical(self):
        """Pair key is always alphabetically sorted."""
        assert _pair_key("MSFT", "AAPL") == "AAPL:MSFT"
        assert _pair_key("AAPL", "MSFT") == "AAPL:MSFT"

    def test_mark_pair_inactive(self, strategy):
        """mark_pair_inactive resets signal_active flag."""
        key = _pair_key("AAPL", "MSFT")
        strategy._pair_state[key]["signal_active"] = True
        strategy.mark_pair_inactive(key)
        assert strategy._pair_state[key]["signal_active"] is False

    def test_multiple_pairs_independent(self, logger):
        """Each pair gets independent quant monitors."""
        config = {
            **PAIR_CONFIG,
            "pairs": [
                {"leg_a": "AAPL", "leg_b": "MSFT"},
                {"leg_a": "GOOG", "leg_b": "META"},
            ],
        }
        s = PairTradingV2Strategy(config, logger)
        k1 = _pair_key("AAPL", "MSFT")
        k2 = _pair_key("GOOG", "META")
        assert k1 in s._coint_monitors
        assert k2 in s._coint_monitors
        assert s._coint_monitors[k1] is not s._coint_monitors[k2]
        assert s._ou_estimators[k1] is not s._ou_estimators[k2]
