"""Unit tests for the Auction Imbalance Strategy."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from datetime import time as time_type
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
from src.strategies.auction_imbalance import AuctionImbalanceConfig, AuctionImbalanceStrategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_logger() -> MagicMock:
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.bind = MagicMock(return_value=logger)
    return logger


def _make_bar(
    symbol: str = "AAPL",
    hour: int = 15,
    minute: int = 55,
    close: float = 150.0,
    dt: datetime | None = None,
) -> Bar:
    if dt is None:
        dt = datetime(2026, 3, 12, hour, minute, 0, tzinfo=timezone.utc)
    return Bar(
        symbol=symbol,
        time=dt,
        open=close - 0.5,
        high=close + 0.5,
        low=close - 1.0,
        close=close,
        volume=100_000.0,
    )


def _make_market_state(dt: datetime | None = None) -> MarketState:
    if dt is None:
        dt = datetime(2026, 3, 12, 15, 55, 0, tzinfo=timezone.utc)
    snapshot = MarketRegimeSnapshot(
        time=dt,
        index_symbol="SPY",
        vol_symbol="VIX",
        trend=TrendRegime.UP,
        vol=VolRegime.NORMAL,
        liquidity=LiquidityRegime.GOOD,
        risk=RiskRegime.NEUTRAL,
        session=SessionRegime.CLOSE,
    )
    return MarketState(time=dt, regime=Regime.BULL, regime_snapshot=snapshot)


def _make_symbol_state(
    symbol: str = "AAPL",
    auction_imbalance: float | None = None,
    atr: float = 1.5,
) -> SymbolState:
    meta: dict = {"atr": atr}
    if auction_imbalance is not None:
        meta["auction_imbalance"] = auction_imbalance
    return SymbolState(symbol=symbol, meta=meta)


def _make_strategy(
    entry_sigma: float = 3.0,
    lookback_days: int = 20,
    signal_time_minutes_before_close: int = 10,
    mechanical_rebal_dates: list[str] | None = None,
    skip_mechanical: bool = True,
    enabled: bool = True,
) -> AuctionImbalanceStrategy:
    config: dict = {
        "enabled": enabled,
        "entry_sigma": entry_sigma,
        "lookback_days": lookback_days,
        "signal_time_minutes_before_close": signal_time_minutes_before_close,
        "skip_mechanical": skip_mechanical,
    }
    if mechanical_rebal_dates is not None:
        config["mechanical_rebal_dates"] = mechanical_rebal_dates
    return AuctionImbalanceStrategy(config, _make_logger())


def _populate_history(strategy: AuctionImbalanceStrategy, symbol: str, values: list[float]) -> None:
    """Record a list of historical imbalance values."""
    for v in values:
        strategy.record_imbalance(symbol, v)


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAuctionImbalanceConfig:
    def test_defaults(self):
        cfg = AuctionImbalanceConfig()
        assert cfg.enabled is True
        assert cfg.entry_sigma == 3.0
        assert cfg.lookback_days == 20
        assert cfg.signal_time_minutes_before_close == 10
        assert cfg.skip_mechanical is True
        assert cfg.mechanical_rebal_dates == []

    def test_custom_values(self):
        cfg = AuctionImbalanceConfig(entry_sigma=2.5, lookback_days=30, skip_mechanical=False)
        assert cfg.entry_sigma == 2.5
        assert cfg.lookback_days == 30
        assert cfg.skip_mechanical is False


# ---------------------------------------------------------------------------
# Rolling Statistics Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRollingStats:
    def test_empty_history_returns_zero(self):
        strat = _make_strategy()
        mean, std = strat.get_rolling_stats("AAPL")
        assert mean == 0.0
        assert std == 0.0

    def test_single_observation_returns_zero(self):
        strat = _make_strategy()
        strat.record_imbalance("AAPL", 1000.0)
        mean, std = strat.get_rolling_stats("AAPL")
        assert mean == 0.0
        assert std == 0.0

    def test_uniform_history_zero_std(self):
        strat = _make_strategy()
        _populate_history(strat, "AAPL", [500.0] * 5)
        mean, std = strat.get_rolling_stats("AAPL")
        assert mean == 500.0
        assert std == 0.0

    def test_known_values(self):
        strat = _make_strategy(lookback_days=5)
        values = [100.0, 200.0, 300.0, 400.0, 500.0]
        _populate_history(strat, "AAPL", values)
        mean, std = strat.get_rolling_stats("AAPL")
        # abs values are same as values (all positive)
        expected_mean = 300.0
        assert abs(mean - expected_mean) < 1e-9
        # sample std of [100, 200, 300, 400, 500]
        variance = sum((v - 300) ** 2 for v in values) / 4  # n-1
        expected_std = math.sqrt(variance)
        assert abs(std - expected_std) < 1e-9

    def test_negative_values_use_absolute(self):
        strat = _make_strategy(lookback_days=5)
        _populate_history(strat, "AAPL", [-100.0, 200.0, -300.0, 400.0, -500.0])
        mean, std = strat.get_rolling_stats("AAPL")
        # abs: [100, 200, 300, 400, 500]
        assert abs(mean - 300.0) < 1e-9

    def test_lookback_window_capped(self):
        strat = _make_strategy(lookback_days=3)
        _populate_history(strat, "AAPL", [100.0, 200.0, 300.0, 400.0, 500.0])
        # Only last 3 should remain: 300, 400, 500
        mean, std = strat.get_rolling_stats("AAPL")
        assert abs(mean - 400.0) < 1e-9


# ---------------------------------------------------------------------------
# Time Gate Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimeGate:
    def test_signal_time_computation(self):
        strat = _make_strategy(signal_time_minutes_before_close=10)
        assert strat._signal_time == time_type(15, 50)

    def test_before_signal_time_no_signal(self):
        strat = _make_strategy()
        _populate_history(strat, "AAPL", [100.0] * 5 + [10000.0])

        bar = _make_bar(hour=15, minute=45)  # Before 15:50
        ss = _make_symbol_state(auction_imbalance=10000.0)
        ms = _make_market_state()

        result = strat.on_bar("AAPL", bar, ss, ms)
        assert result is None

    def test_at_signal_time_generates_signal(self):
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=50)  # Exactly at 15:50
        ss = _make_symbol_state(auction_imbalance=5000.0)
        ms = _make_market_state()

        result = strat.on_bar("AAPL", bar, ss, ms)
        assert result is not None

    def test_after_signal_time_generates_signal(self):
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=5000.0)
        ms = _make_market_state()

        result = strat.on_bar("AAPL", bar, ss, ms)
        assert result is not None

    def test_custom_signal_time(self):
        strat = _make_strategy(signal_time_minutes_before_close=5)
        assert strat._signal_time == time_type(15, 55)


# ---------------------------------------------------------------------------
# Signal Generation Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSignalGeneration:
    def _setup_strategy_with_history(self, entry_sigma: float = 3.0) -> AuctionImbalanceStrategy:
        strat = _make_strategy(entry_sigma=entry_sigma)
        # Mean abs = 100, std ≈ 0 (all same), so any deviation triggers
        # Use varied values to get nonzero std
        _populate_history(strat, "AAPL", [80.0, 90.0, 100.0, 110.0, 120.0] * 2)
        return strat

    def test_buy_signal_on_positive_imbalance(self):
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55, close=150.0)
        ss = _make_symbol_state(auction_imbalance=5000.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is not None
        assert signal.side == OrderSide.BUY
        assert signal.symbol == "AAPL"
        assert signal.strategy == "auction_imbalance"
        assert signal.meta["classification"] == "informational"
        assert signal.meta["imbalance_size"] == 5000.0

    def test_sell_signal_on_negative_imbalance(self):
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55, close=150.0)
        ss = _make_symbol_state(auction_imbalance=-5000.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is not None
        assert signal.side == OrderSide.SELL

    def test_no_signal_below_threshold(self):
        strat = _make_strategy(entry_sigma=3.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)
        _, std = strat.get_rolling_stats("AAPL")
        mean, _ = strat.get_rolling_stats("AAPL")

        # Use an imbalance that is well within 3 sigma
        small_imbalance = mean + std * 1.0  # Only 1 sigma
        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=small_imbalance)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is None

    def test_no_signal_zero_imbalance(self):
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0] * 5)

        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=0.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is None

    def test_no_signal_missing_imbalance(self):
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0] * 5)

        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=None)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is None

    def test_no_signal_insufficient_history(self):
        strat = _make_strategy(entry_sigma=1.0)
        # Only 1 observation -> std = 0 -> skip
        strat.record_imbalance("AAPL", 100.0)

        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=5000.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is None

    def test_no_signal_when_disabled(self):
        strat = _make_strategy(enabled=False, entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0] * 5)

        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=5000.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is None

    def test_stop_and_target_prices_buy(self):
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55, close=150.0)
        ss = _make_symbol_state(auction_imbalance=5000.0, atr=2.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is not None
        # stop_distance = atr(2.0) * stop_atr_multiplier(1.5) * vol_mult(1.0) = 3.0
        assert signal.stop_price == pytest.approx(147.0, abs=0.01)
        # target = close + stop_distance * risk_reward(2.0) = 150 + 6.0
        assert signal.target_price == pytest.approx(156.0, abs=0.01)

    def test_stop_and_target_prices_sell(self):
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55, close=150.0)
        ss = _make_symbol_state(auction_imbalance=-5000.0, atr=2.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is not None
        assert signal.stop_price == pytest.approx(153.0, abs=0.01)
        assert signal.target_price == pytest.approx(144.0, abs=0.01)

    def test_z_score_in_meta(self):
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=5000.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is not None
        assert "z_score" in signal.meta
        assert signal.meta["z_score"] > 1.0


# ---------------------------------------------------------------------------
# Mechanical vs Informational Classification Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMechanicalClassification:
    def test_informational_on_normal_date(self):
        strat = _make_strategy(mechanical_rebal_dates=["2026-03-20"])
        assert strat.classify_imbalance(date(2026, 3, 12)) == "informational"

    def test_mechanical_on_rebalance_date(self):
        strat = _make_strategy(mechanical_rebal_dates=["2026-03-12"])
        assert strat.classify_imbalance(date(2026, 3, 12)) == "mechanical"

    def test_skip_mechanical_no_signal(self):
        strat = _make_strategy(
            entry_sigma=1.0,
            mechanical_rebal_dates=["2026-03-12"],
            skip_mechanical=True,
        )
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=5000.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is None

    def test_mechanical_fade_when_not_skipped(self):
        """On mechanical dates with skip_mechanical=False, fade the imbalance."""
        strat = _make_strategy(
            entry_sigma=1.0,
            mechanical_rebal_dates=["2026-03-12"],
            skip_mechanical=False,
        )
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        # Positive imbalance on mechanical date -> SELL (fade)
        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=5000.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is not None
        assert signal.side == OrderSide.SELL
        assert signal.meta["classification"] == "mechanical"

    def test_mechanical_fade_negative_imbalance(self):
        """Negative imbalance on mechanical date -> BUY (fade)."""
        strat = _make_strategy(
            entry_sigma=1.0,
            mechanical_rebal_dates=["2026-03-12"],
            skip_mechanical=False,
        )
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=-5000.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is not None
        assert signal.side == OrderSide.BUY
        assert signal.meta["classification"] == "mechanical"


# ---------------------------------------------------------------------------
# Duplicate Signal Prevention Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDuplicatePrevention:
    def test_only_one_signal_per_symbol_per_day(self):
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=5000.0)
        ms = _make_market_state()

        signal1 = strat.on_bar("AAPL", bar, ss, ms)
        assert signal1 is not None

        # Second call same day -> None
        signal2 = strat.on_bar("AAPL", bar, ss, ms)
        assert signal2 is None

    def test_different_symbols_can_signal_same_day(self):
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)
        _populate_history(strat, "MSFT", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55)
        ss_aapl = _make_symbol_state(symbol="AAPL", auction_imbalance=5000.0)
        ss_msft = _make_symbol_state(symbol="MSFT", auction_imbalance=-5000.0)
        ms = _make_market_state()

        sig1 = strat.on_bar("AAPL", bar, ss_aapl, ms)
        sig2 = strat.on_bar("MSFT", bar, ss_msft, ms)
        assert sig1 is not None
        assert sig2 is not None


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    def test_zero_std_no_signal(self):
        """All identical history values -> std=0 -> no signal."""
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0] * 10)

        bar = _make_bar(hour=15, minute=55)
        ss = _make_symbol_state(auction_imbalance=5000.0)
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is None

    def test_atr_fallback_when_missing(self):
        """When ATR not in meta, falls back to 1% of close."""
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55, close=200.0)
        ss = SymbolState(symbol="AAPL", meta={"auction_imbalance": 5000.0})
        ms = _make_market_state()

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is not None
        # atr fallback = 200 * 0.01 = 2.0
        # stop_distance = 2.0 * 1.5 = 3.0
        assert signal.stop_price == pytest.approx(197.0, abs=0.01)

    def test_regime_volatility_multiplier_applied(self):
        """High vol regime should widen stops."""
        strat = _make_strategy(entry_sigma=1.0)
        _populate_history(strat, "AAPL", [100.0, 120.0, 80.0, 110.0, 90.0] * 2)

        bar = _make_bar(hour=15, minute=55, close=150.0)
        ss = _make_symbol_state(auction_imbalance=5000.0, atr=2.0)
        dt = datetime(2026, 3, 12, 15, 55, 0, tzinfo=timezone.utc)
        snapshot = MarketRegimeSnapshot(
            time=dt,
            index_symbol="SPY",
            vol_symbol="VIX",
            trend=TrendRegime.UP,
            vol=VolRegime.HIGH,  # 1.2x multiplier
            liquidity=LiquidityRegime.GOOD,
            risk=RiskRegime.NEUTRAL,
            session=SessionRegime.CLOSE,
        )
        ms = MarketState(time=dt, regime=Regime.BULL, regime_snapshot=snapshot)

        signal = strat.on_bar("AAPL", bar, ss, ms)
        assert signal is not None
        # stop_distance = atr(2.0) * 1.5 * vol_mult(1.2) = 3.6
        assert signal.stop_price == pytest.approx(146.4, abs=0.01)

    def test_invalid_mechanical_date_ignored(self):
        """Invalid date strings in config are logged and ignored."""
        strat = _make_strategy(mechanical_rebal_dates=["not-a-date", "2026-03-20"])
        assert date(2026, 3, 20) in strat._mechanical_dates
        assert len(strat._mechanical_dates) == 1

    def test_record_imbalance_persists(self):
        strat = _make_strategy()
        strat.record_imbalance("AAPL", 100.0)
        strat.record_imbalance("AAPL", 200.0)
        strat.record_imbalance("AAPL", 300.0)
        mean, std = strat.get_rolling_stats("AAPL")
        assert mean > 0

    def test_strategy_name(self):
        strat = _make_strategy()
        assert strat.name == "auction_imbalance"
