"""Tests for post-backtest diagnostics engine."""

import pytest


@pytest.mark.unit
def test_strategy_ranking_by_pnl():
    from src.analytics.diagnostics import run_diagnostics

    trades = [
        {
            "strategy": "orb",
            "pnl": 500,
            "regime_trend": "UP",
            "entry_hour": 10,
            "hold_minutes": 30,
            "exit_type": "target",
        },
        {
            "strategy": "orb",
            "pnl": -200,
            "regime_trend": "UP",
            "entry_hour": 10,
            "hold_minutes": 45,
            "exit_type": "stop",
        },
        {
            "strategy": "orb",
            "pnl": 300,
            "regime_trend": "FLAT",
            "entry_hour": 11,
            "hold_minutes": 20,
            "exit_type": "target",
        },
        {
            "strategy": "mean_rev",
            "pnl": -100,
            "regime_trend": "UP",
            "entry_hour": 14,
            "hold_minutes": 15,
            "exit_type": "stop",
        },
        {
            "strategy": "mean_rev",
            "pnl": -150,
            "regime_trend": "DOWN",
            "entry_hour": 14,
            "hold_minutes": 25,
            "exit_type": "time_limit",
        },
    ]
    report = run_diagnostics(trades, min_trades=2)
    assert report.strategy_rankings[0].strategy == "orb"
    assert report.strategy_rankings[1].strategy == "mean_rev"


@pytest.mark.unit
def test_regime_mismatch_detection():
    from src.analytics.diagnostics import run_diagnostics

    trades = [
        {
            "strategy": "mean_rev",
            "pnl": -200,
            "regime_trend": "UP",
            "entry_hour": 10,
            "hold_minutes": 20,
            "exit_type": "stop",
        },
        {
            "strategy": "mean_rev",
            "pnl": -300,
            "regime_trend": "UP",
            "entry_hour": 11,
            "hold_minutes": 25,
            "exit_type": "stop",
        },
        {
            "strategy": "mean_rev",
            "pnl": -150,
            "regime_trend": "UP",
            "entry_hour": 12,
            "hold_minutes": 30,
            "exit_type": "stop",
        },
        {
            "strategy": "mean_rev",
            "pnl": 400,
            "regime_trend": "FLAT",
            "entry_hour": 10,
            "hold_minutes": 15,
            "exit_type": "target",
        },
        {
            "strategy": "mean_rev",
            "pnl": 300,
            "regime_trend": "FLAT",
            "entry_hour": 11,
            "hold_minutes": 20,
            "exit_type": "target",
        },
    ]
    report = run_diagnostics(trades, min_trades=2)
    mismatches = [m for m in report.regime_mismatches if m.strategy == "mean_rev" and m.regime_value == "UP"]
    assert len(mismatches) == 1
    assert mismatches[0].avg_pnl < 0


@pytest.mark.unit
def test_time_edge_analysis():
    from src.analytics.diagnostics import run_diagnostics

    trades = [
        {
            "strategy": "orb",
            "pnl": 500,
            "regime_trend": "UP",
            "entry_hour": 10,
            "hold_minutes": 30,
            "exit_type": "target",
        },
        {
            "strategy": "orb",
            "pnl": 400,
            "regime_trend": "UP",
            "entry_hour": 10,
            "hold_minutes": 25,
            "exit_type": "target",
        },
        {
            "strategy": "orb",
            "pnl": -300,
            "regime_trend": "FLAT",
            "entry_hour": 14,
            "hold_minutes": 40,
            "exit_type": "stop",
        },
        {
            "strategy": "orb",
            "pnl": -250,
            "regime_trend": "FLAT",
            "entry_hour": 14,
            "hold_minutes": 35,
            "exit_type": "stop",
        },
    ]
    report = run_diagnostics(trades, min_trades=2)
    time_map = report.time_edge_map["orb"]
    morning = next(t for t in time_map if t.hour == 10)
    afternoon = next(t for t in time_map if t.hour == 14)
    assert morning.avg_pnl > 0
    assert afternoon.avg_pnl < 0


@pytest.mark.unit
def test_diagnostics_summary_not_empty():
    from src.analytics.diagnostics import run_diagnostics

    trades = [
        {
            "strategy": "orb",
            "pnl": 100,
            "regime_trend": "UP",
            "entry_hour": 10,
            "hold_minutes": 30,
            "exit_type": "target",
        },
    ] * 25
    report = run_diagnostics(trades, min_trades=5)
    assert len(report.summary) > 0


@pytest.mark.unit
def test_hold_analysis():
    from src.analytics.diagnostics import run_diagnostics

    trades = [
        {
            "strategy": "orb",
            "pnl": 500,
            "regime_trend": "UP",
            "entry_hour": 10,
            "hold_minutes": 30,
            "exit_type": "target",
        },
        {
            "strategy": "orb",
            "pnl": 300,
            "regime_trend": "UP",
            "entry_hour": 10,
            "hold_minutes": 25,
            "exit_type": "target",
        },
        {
            "strategy": "orb",
            "pnl": -200,
            "regime_trend": "FLAT",
            "entry_hour": 14,
            "hold_minutes": 40,
            "exit_type": "stop",
        },
        {
            "strategy": "orb",
            "pnl": -100,
            "regime_trend": "FLAT",
            "entry_hour": 14,
            "hold_minutes": 35,
            "exit_type": "time_limit",
        },
    ]
    report = run_diagnostics(trades, min_trades=2)
    hold = report.hold_analysis["orb"]
    target_exit = next(h for h in hold if h.exit_type == "target")
    stop_exit = next(h for h in hold if h.exit_type == "stop")
    assert target_exit.avg_pnl > 0
    assert stop_exit.avg_pnl < 0
