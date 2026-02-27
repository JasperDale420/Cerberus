"""Unit tests for new profit-maximization metrics in AnalyticsEngine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.analysis.analytics import AnalyticsEngine
from src.analysis.schema import Trade as DbTrade
from src.core.logger import StructuredLogger


@dataclass
class _DbStub:
    config: Dict[str, Any]

    def get_session(self):  # pragma: no cover
        raise NotImplementedError


def _make_trade(
    strategy: str = "StratA",
    regime: str = "bull",
    pnl_net: float = 0.0,
    pnl_r: Optional[float] = None,
    holding_period_seconds: Optional[float] = None,
) -> DbTrade:
    return DbTrade(
        strategy=strategy,
        regime_at_entry=regime,
        pnl_net=pnl_net,
        pnl_r=pnl_r,
        holding_period_seconds=holding_period_seconds,
        exit_time=datetime(2025, 1, 1, 15, 0, tzinfo=timezone.utc),
        entry_time=datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc),
    )


def _run_aggregation(trades: List[DbTrade]) -> Any:
    """Run aggregation with mock DB, return the added StrategyStatsDaily objects."""
    db = MagicMock()
    db.config = {"timezone": "UTC"}
    logger = StructuredLogger("test_metrics", level="INFO")
    mock_session = MagicMock()

    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = None
    db.get_session.return_value = mock_cm

    mock_session.execute.return_value.scalars.return_value.all.return_value = trades

    engine = AnalyticsEngine(db, logger)
    engine.run_daily_aggregation(date(2025, 1, 1))

    added_objs = [call[0][0] for call in mock_session.add.call_args_list]
    return added_objs


@pytest.mark.unit
def test_profit_factor_basic(tmp_path, monkeypatch) -> None:
    """2 wins ($100, $50) + 1 loss (-$20) → PF = 150/20 = 7.5"""
    monkeypatch.chdir(tmp_path)
    trades = [
        _make_trade(pnl_net=100.0, pnl_r=2.0),
        _make_trade(pnl_net=50.0, pnl_r=1.0),
        _make_trade(pnl_net=-20.0, pnl_r=-0.4),
    ]
    stats_list = _run_aggregation(trades)
    assert len(stats_list) == 1
    stats = stats_list[0]
    assert stats.profit_factor == pytest.approx(7.5)


@pytest.mark.unit
def test_avg_win_loss_ratio(tmp_path, monkeypatch) -> None:
    """avg_win=75, avg_loss=20 → ratio=3.75"""
    monkeypatch.chdir(tmp_path)
    trades = [
        _make_trade(pnl_net=100.0, pnl_r=2.0),
        _make_trade(pnl_net=50.0, pnl_r=1.0),
        _make_trade(pnl_net=-20.0, pnl_r=-0.4),
    ]
    stats_list = _run_aggregation(trades)
    stats = stats_list[0]
    assert stats.avg_win_loss_ratio == pytest.approx(3.75)


@pytest.mark.unit
def test_recovery_factor(tmp_path, monkeypatch) -> None:
    """net PnL $130, pnl_r series [2.0, 1.0, -0.4] → cumR peaks at 3.0, drops to 2.6 → DD=0.4R
    recovery = net_pnl / max_dd_r = 130 / 0.4 = 325.0"""
    monkeypatch.chdir(tmp_path)
    trades = [
        _make_trade(pnl_net=100.0, pnl_r=2.0),
        _make_trade(pnl_net=50.0, pnl_r=1.0),
        _make_trade(pnl_net=-20.0, pnl_r=-0.4),
    ]
    stats_list = _run_aggregation(trades)
    stats = stats_list[0]
    assert stats.recovery_factor == pytest.approx(325.0)


@pytest.mark.unit
def test_ev_per_trade(tmp_path, monkeypatch) -> None:
    """$130 / 3 = $43.33"""
    monkeypatch.chdir(tmp_path)
    trades = [
        _make_trade(pnl_net=100.0, pnl_r=2.0),
        _make_trade(pnl_net=50.0, pnl_r=1.0),
        _make_trade(pnl_net=-20.0, pnl_r=-0.4),
    ]
    stats_list = _run_aggregation(trades)
    stats = stats_list[0]
    assert stats.ev_per_trade == pytest.approx(43.333, abs=0.01)


@pytest.mark.unit
def test_avg_hold_seconds(tmp_path, monkeypatch) -> None:
    """3 trades with hold times 60, 120, 180 → avg = 120"""
    monkeypatch.chdir(tmp_path)
    trades = [
        _make_trade(pnl_net=10.0, pnl_r=0.5, holding_period_seconds=60.0),
        _make_trade(pnl_net=20.0, pnl_r=1.0, holding_period_seconds=120.0),
        _make_trade(pnl_net=-5.0, pnl_r=-0.25, holding_period_seconds=180.0),
    ]
    stats_list = _run_aggregation(trades)
    stats = stats_list[0]
    assert stats.avg_hold_seconds == pytest.approx(120.0)


@pytest.mark.unit
def test_all_winners_profit_factor_capped(tmp_path, monkeypatch) -> None:
    """All winners → PF capped at 999.0 (no division by zero)"""
    monkeypatch.chdir(tmp_path)
    trades = [
        _make_trade(pnl_net=100.0, pnl_r=2.0),
        _make_trade(pnl_net=50.0, pnl_r=1.0),
    ]
    stats_list = _run_aggregation(trades)
    stats = stats_list[0]
    assert stats.profit_factor == 999.0
    assert stats.avg_win_loss_ratio == 999.0


@pytest.mark.unit
def test_all_losers_profit_factor_zero(tmp_path, monkeypatch) -> None:
    """All losers → PF = 0.0"""
    monkeypatch.chdir(tmp_path)
    trades = [
        _make_trade(pnl_net=-30.0, pnl_r=-1.5),
        _make_trade(pnl_net=-20.0, pnl_r=-1.0),
    ]
    stats_list = _run_aggregation(trades)
    stats = stats_list[0]
    assert stats.profit_factor == 0.0
    assert stats.avg_win_loss_ratio == 0.0


@pytest.mark.unit
def test_stage2_metrics_include_profit_factor() -> None:
    """Stage2Metrics dataclass includes profit_factor and win_rate fields."""
    from src.agent.stage2 import Stage2Metrics

    m = Stage2Metrics(
        expectancy=0.5,
        max_drawdown_r=1.0,
        n_trades=10,
        profit_factor=2.5,
        win_rate=0.6,
    )
    assert m.profit_factor == 2.5
    assert m.win_rate == 0.6
