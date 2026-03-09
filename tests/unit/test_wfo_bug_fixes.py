"""Unit tests for WFO harness bug fixes.

Tests cover:
1. Strategy attribution in _build_trade_records (correlation_id → Signal.strategy)
2. Hold-time filtering of _flatten trades in BacktestReportCard
3. TRP _regime_allows only accepting confirmed trend directions
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.backtest.backtest_report import BacktestReportCard, TradeRecord
from src.core.domain import OrderSide, TrendRegime

# ---------------------------------------------------------------------------
# Bug 1: Strategy attribution
# ---------------------------------------------------------------------------
# The actual DB-level test for _build_trade_records would require a full
# SQLite setup with Order+Signal tables. Instead we verify that the
# TradeRecord.strategy field propagates correctly through BacktestReportCard.


@pytest.mark.unit
def test_trade_record_strategy_propagates_to_per_strategy():
    """Verify per_strategy breakdown uses TradeRecord.strategy correctly."""
    now = datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
    later = datetime(2024, 6, 1, 14, 5, tzinfo=timezone.utc)
    much_later = datetime(2024, 6, 1, 20, 0, tzinfo=timezone.utc)

    trades = [
        TradeRecord(
            symbol="AAPL",
            side="buy",
            qty=10,
            entry_price=150.0,
            exit_price=151.0,
            entry_time=now,
            exit_time=later,
            pnl=10.0,
            strategy="trend_rider_pro",
        ),
        TradeRecord(
            symbol="AAPL",
            side="buy",
            qty=10,
            entry_price=150.0,
            exit_price=149.0,
            entry_time=now,
            exit_time=much_later,
            pnl=-10.0,
            strategy="_flatten",
        ),
    ]

    report = BacktestReportCard(
        trades=trades,
        equity_curve=[(now, 100_000.0), (much_later, 100_000.0)],
        initial_capital=100_000.0,
        start_date=now,
        end_date=much_later,
    )

    strat_metrics = report.strategy_metrics
    assert "trend_rider_pro" in strat_metrics
    assert "_flatten" in strat_metrics
    assert strat_metrics["trend_rider_pro"]["n_trades"] == 1


@pytest.mark.unit
def test_unknown_strategy_when_empty():
    """TradeRecord with empty strategy should appear under 'unknown'."""
    now = datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
    later = datetime(2024, 6, 1, 14, 5, tzinfo=timezone.utc)

    trades = [
        TradeRecord(
            symbol="SPY",
            side="buy",
            qty=5,
            entry_price=450.0,
            exit_price=451.0,
            entry_time=now,
            exit_time=later,
            pnl=5.0,
            strategy="",
        ),
    ]

    report = BacktestReportCard(
        trades=trades,
        equity_curve=[(now, 100_000.0), (later, 100_005.0)],
        initial_capital=100_000.0,
    )

    assert "unknown" in report.strategy_metrics


# ---------------------------------------------------------------------------
# Bug 2: Hold-time excludes _flatten trades
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hold_time_excludes_flatten_trades():
    """avg/median hold times should exclude _flatten trades but include their PnL."""
    now = datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
    normal_exit = datetime(2024, 6, 1, 14, 2, tzinfo=timezone.utc)  # 2 min hold
    flatten_exit = datetime(2024, 6, 1, 20, 0, tzinfo=timezone.utc)  # 6 hr hold

    trades = [
        TradeRecord(
            symbol="AAPL",
            side="buy",
            qty=10,
            entry_price=150.0,
            exit_price=151.0,
            entry_time=now,
            exit_time=normal_exit,
            pnl=10.0,
            strategy="trend_rider_pro",
        ),
        TradeRecord(
            symbol="TSLA",
            side="buy",
            qty=5,
            entry_price=200.0,
            exit_price=199.0,
            entry_time=now,
            exit_time=flatten_exit,
            pnl=-5.0,
            strategy="_flatten",
        ),
    ]

    report = BacktestReportCard(
        trades=trades,
        equity_curve=[(now, 100_000.0), (flatten_exit, 100_005.0)],
        initial_capital=100_000.0,
    )

    m = report.metrics

    # PnL should include both trades
    assert m.net_pnl == pytest.approx(5.0)

    # Hold time should only reflect the normal trade (2 min = 120 sec)
    assert m.avg_hold_seconds == pytest.approx(120.0)
    assert m.median_hold_seconds == pytest.approx(120.0)


@pytest.mark.unit
def test_per_strategy_hold_time_excludes_flatten():
    """Per-strategy hold time for _flatten group should be 0 (no qualifying trades)."""
    now = datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
    flatten_exit = datetime(2024, 6, 1, 20, 0, tzinfo=timezone.utc)

    trades = [
        TradeRecord(
            symbol="AAPL",
            side="buy",
            qty=10,
            entry_price=150.0,
            exit_price=149.0,
            entry_time=now,
            exit_time=flatten_exit,
            pnl=-10.0,
            strategy="_flatten",
        ),
    ]

    report = BacktestReportCard(
        trades=trades,
        equity_curve=[(now, 100_000.0), (flatten_exit, 99_990.0)],
        initial_capital=100_000.0,
    )

    # _flatten trades should have 0 hold time since they're excluded from hold calc
    flatten_metrics = report.strategy_metrics.get("_flatten", {})
    assert flatten_metrics.get("avg_hold_min", 0.0) == 0.0
    assert flatten_metrics.get("median_hold_min", 0.0) == 0.0


# ---------------------------------------------------------------------------
# Bug 3: TRP _regime_allows only accepts confirmed directional trends
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trp_regime_allows_buy_only_in_uptrend():
    """BUY should only be allowed when trend is UP."""
    from unittest.mock import MagicMock

    from src.strategies.trend_rider_pro import TrendRiderProStrategy

    mock_mtf = MagicMock()
    mock_mtf.get_adx.return_value = 10.0  # Low ADX (previously allowed FLAT)

    # UP trend → allowed
    up_snapshot = MagicMock()
    up_snapshot.trend = TrendRegime.UP
    assert TrendRiderProStrategy._regime_allows(OrderSide.BUY, up_snapshot, mock_mtf) is True

    # DOWN trend → blocked
    down_snapshot = MagicMock()
    down_snapshot.trend = TrendRegime.DOWN
    assert TrendRiderProStrategy._regime_allows(OrderSide.BUY, down_snapshot, mock_mtf) is False

    # FLAT trend → blocked (was previously allowed with low ADX)
    flat_snapshot = MagicMock()
    flat_snapshot.trend = TrendRegime.FLAT
    assert TrendRiderProStrategy._regime_allows(OrderSide.BUY, flat_snapshot, mock_mtf) is False


@pytest.mark.unit
def test_trp_regime_allows_sell_only_in_downtrend():
    """SELL should only be allowed when trend is DOWN."""
    from unittest.mock import MagicMock

    from src.strategies.trend_rider_pro import TrendRiderProStrategy

    mock_mtf = MagicMock()
    mock_mtf.get_adx.return_value = 5.0  # Very low ADX (previously allowed FLAT)

    # DOWN trend → allowed
    down_snapshot = MagicMock()
    down_snapshot.trend = TrendRegime.DOWN
    assert TrendRiderProStrategy._regime_allows(OrderSide.SELL, down_snapshot, mock_mtf) is True

    # UP trend → blocked
    up_snapshot = MagicMock()
    up_snapshot.trend = TrendRegime.UP
    assert TrendRiderProStrategy._regime_allows(OrderSide.SELL, up_snapshot, mock_mtf) is False

    # FLAT trend → blocked
    flat_snapshot = MagicMock()
    flat_snapshot.trend = TrendRegime.FLAT
    assert TrendRiderProStrategy._regime_allows(OrderSide.SELL, flat_snapshot, mock_mtf) is False


@pytest.mark.unit
def test_trp_regime_allows_when_no_snapshot():
    """With no snapshot, regime check should pass (backward compat)."""
    from unittest.mock import MagicMock

    from src.strategies.trend_rider_pro import TrendRiderProStrategy

    mock_mtf = MagicMock()

    assert TrendRiderProStrategy._regime_allows(OrderSide.BUY, None, mock_mtf) is True
    assert TrendRiderProStrategy._regime_allows(OrderSide.SELL, None, mock_mtf) is True
