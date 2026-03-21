"""Tests for the comprehensive analytics report card (Cerberus dict format)."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.analytics.report_card import (
    analyze_mae_mfe,
    analyze_pnl_distribution,
    analyze_trade_clustering,
    catalog_drawdowns,
    compute_cost_sensitivity,
    compute_entry_exit_efficiency,
    compute_risk_metrics,
    compute_rolling_metrics,
    compute_statistical_tests,
    compute_time_breakdowns,
    generate_report,
    run_monte_carlo,
    save_report,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_trades() -> list[dict]:
    """Cerberus-style trade dicts."""
    return [
        {
            "entry_bar": 0,
            "exit_bar": 10,
            "entry_price": 100.0,
            "exit_price": 105.0,
            "pnl_r": 2.5,
            "mae_r": -1.5,
            "mfe_r": 4.0,
            "exit_reason": "signal",
        },
        {
            "entry_bar": 20,
            "exit_bar": 30,
            "entry_price": 100.0,
            "exit_price": 97.0,
            "pnl_r": -1.0,
            "mae_r": -2.5,
            "mfe_r": 1.0,
            "exit_reason": "stop_loss",
        },
        {
            "entry_bar": 40,
            "exit_bar": 50,
            "entry_price": 100.0,
            "exit_price": 110.0,
            "pnl_r": 5.0,
            "mae_r": -0.5,
            "mfe_r": 6.0,
            "exit_reason": "signal",
        },
    ]


def _make_equity_curve() -> list[float]:
    return [10000, 10250, 10100, 10500, 10300, 10800, 10600, 11000]


def _make_returns() -> list[float]:
    ec = _make_equity_curve()
    return [ec[i] / ec[i - 1] - 1.0 for i in range(1, len(ec))]


# ---------------------------------------------------------------------------
# Tier 1 tests
# ---------------------------------------------------------------------------


class TestAnalyzeMaeMfe:
    def test_basic(self):
        trades = _make_trades()
        result = analyze_mae_mfe(trades)
        assert "avg_mae" in result
        assert "avg_mfe" in result
        assert "edge_ratio" in result
        assert "avg_capture_efficiency" in result
        assert len(result["per_trade"]) == 3
        # worst MAE is -2.5
        assert result["worst_mae"] == pytest.approx(-2.5)
        # best MFE is 6.0
        assert result["best_mfe"] == pytest.approx(6.0)

    def test_empty(self):
        result = analyze_mae_mfe([])
        assert result["avg_mae"] == 0.0
        assert result["per_trade"] == []

    def test_custom_keys(self):
        trades = [{"my_mae": -1.0, "my_mfe": 2.0, "my_pnl": 1.0}]
        result = analyze_mae_mfe(trades, mae_key="my_mae", mfe_key="my_mfe", pnl_key="my_pnl")
        assert result["avg_mae"] == pytest.approx(-1.0)
        assert result["avg_mfe"] == pytest.approx(2.0)


class TestStatisticalTests:
    def test_basic(self):
        returns = _make_returns()
        result = compute_statistical_tests(returns, n_bootstrap=100, n_permutations=100, seed=42)
        assert "t_stat" in result
        assert "t_pvalue" in result
        assert "bootstrap_ci_95" in result
        assert len(result["bootstrap_ci_95"]) == 2
        assert "mean_return" in result

    def test_short_series(self):
        result = compute_statistical_tests([0.01, 0.02])
        assert result["t_pvalue"] == 1.0

    def test_empty(self):
        result = compute_statistical_tests([])
        assert result["mean_return"] == 0.0


class TestMonteCarlo:
    def test_basic(self):
        returns = _make_returns()
        result = run_monte_carlo(returns, n_simulations=100, seed=42)
        assert "ruin_probability" in result
        assert "median_final_equity" in result
        assert len(result["max_drawdown_distribution"]) == 3
        assert result["median_final_equity"] > 0

    def test_empty(self):
        result = run_monte_carlo([])
        assert result["median_final_equity"] == 10000.0

    def test_ruin(self):
        # Large negative R-multiples should trigger ruin (additive: each -20R = -$2000)
        returns = [-20.0, -20.0, -20.0, -20.0, -20.0]
        result = run_monte_carlo(returns, n_simulations=200, seed=42, ruin_threshold=0.5)
        assert result["ruin_probability"] > 0


class TestPnlDistribution:
    def test_basic(self):
        result = analyze_pnl_distribution([2.5, -1.0, 5.0])
        assert result["mean"] == pytest.approx(2.1666666, rel=1e-4)
        assert result["median"] == pytest.approx(2.5)
        assert result["std"] > 0

    def test_empty(self):
        result = analyze_pnl_distribution([])
        assert result["mean"] == 0.0


# ---------------------------------------------------------------------------
# Tier 2 tests
# ---------------------------------------------------------------------------


class TestRiskMetrics:
    def test_basic(self):
        returns = _make_returns()
        result = compute_risk_metrics(returns)
        assert "omega_ratio" in result
        assert "var_95" in result
        assert "cvar_95" in result
        assert "ulcer_index" in result
        assert result["omega_ratio"] > 0

    def test_empty(self):
        result = compute_risk_metrics([])
        assert result["omega_ratio"] == 0.0

    def test_single(self):
        result = compute_risk_metrics([0.01])
        assert result["omega_ratio"] == 0.0


class TestRollingMetrics:
    def test_basic(self):
        returns = list(np.random.default_rng(42).normal(0.001, 0.01, 100))
        result = compute_rolling_metrics(returns, window=20, bars_per_year=98280.0)
        assert len(result["rolling_sharpe"]) == 81
        assert len(result["rolling_win_rate"]) == 81

    def test_short(self):
        result = compute_rolling_metrics([0.01, 0.02], window=50)
        assert result["rolling_sharpe"] == []


class TestCatalogDrawdowns:
    def test_basic(self):
        ec = _make_equity_curve()
        result = catalog_drawdowns(ec, top_n=3)
        assert len(result) > 0
        # Sorted by depth descending
        depths = [d["depth_pct"] for d in result]
        assert depths == sorted(depths, reverse=True)

    def test_empty(self):
        assert catalog_drawdowns([]) == []
        assert catalog_drawdowns([100.0]) == []

    def test_monotonic_up(self):
        assert catalog_drawdowns([100, 110, 120, 130]) == []


class TestTradeClustering:
    def test_basic(self):
        trades = _make_trades()
        result = analyze_trade_clustering(trades)
        # Win-loss-win pattern
        assert result["win_streak_max"] == 1
        assert result["loss_streak_max"] == 1
        assert result["avg_gap_bars"] == pytest.approx(10.0)

    def test_empty(self):
        result = analyze_trade_clustering([])
        assert result["win_streak_max"] == 0

    def test_all_wins(self):
        trades = [{"pnl_r": 1.0, "entry_bar": i * 10, "exit_bar": i * 10 + 5} for i in range(5)]
        result = analyze_trade_clustering(trades)
        assert result["win_streak_max"] == 5
        assert result["loss_streak_max"] == 0


class TestCostSensitivity:
    def test_basic(self):
        trade_pnls = [2.5, -1.0, 5.0]
        result = compute_cost_sensitivity(trade_pnls, n_trades=3, base_cost_bps=20.0)
        assert "break_even_cost_bps" in result
        assert len(result["cost_sweep"]) == 9
        # At 0 cost, net_pnl should equal total pnl
        assert result["cost_sweep"][0]["net_pnl_pct"] == pytest.approx(6.5)

    def test_empty(self):
        result = compute_cost_sensitivity([], n_trades=0)
        assert result["break_even_cost_bps"] == 0.0
        assert result["cost_sweep"] == []


# ---------------------------------------------------------------------------
# Tier 3 tests
# ---------------------------------------------------------------------------


class TestTimeBreakdowns:
    def test_basic(self):
        trades = _make_trades()
        # Create a bar index with enough entries
        dates = pd.date_range("2025-01-02 09:30", periods=60, freq="1min")
        bar_index = pd.DatetimeIndex(dates)
        result = compute_time_breakdowns(trades, bar_index)
        assert "by_hour" in result
        assert "by_day_of_week" in result
        assert 9 in result["by_hour"]  # 09:30 start

    def test_empty(self):
        result = compute_time_breakdowns([], pd.DatetimeIndex([]))
        assert result["by_hour"] == {}


class TestEntryExitEfficiency:
    def test_basic(self):
        trades = _make_trades()
        result = compute_entry_exit_efficiency(trades)
        assert 0.0 <= result["avg_entry_efficiency"] <= 1.0
        assert 0.0 <= result["avg_exit_efficiency"] <= 1.0

    def test_empty(self):
        result = compute_entry_exit_efficiency([])
        assert result["avg_entry_efficiency"] == 0.0
        assert result["avg_exit_efficiency"] == 0.0


# ---------------------------------------------------------------------------
# Orchestrator + persistence tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_all_sections_present(self):
        ec = _make_equity_curve()
        trades = _make_trades()
        bar_data = pd.DataFrame(index=pd.date_range("2025-01-02 09:30", periods=60, freq="1min"))
        report = generate_report(
            ec,
            trades,
            bar_data,
            save=False,
            mc_simulations=50,
            rolling_window=3,
        )
        expected_keys = [
            "mae_mfe",
            "statistical_tests",
            "monte_carlo",
            "pnl_distribution",
            "risk_metrics",
            "rolling_metrics",
            "drawdown_catalog",
            "trade_clustering",
            "cost_sensitivity",
            "time_breakdowns",
            "entry_exit_efficiency",
        ]
        for key in expected_keys:
            assert key in report, f"Missing section: {key}"

    def test_empty_inputs(self):
        report = generate_report(
            [],
            [],
            pd.DataFrame(),
            save=False,
        )
        assert report["mae_mfe"]["avg_mae"] == 0.0

    def test_custom_pnl_key(self):
        trades = [
            {
                "entry_bar": 0,
                "exit_bar": 5,
                "entry_price": 100.0,
                "exit_price": 105.0,
                "ret": 0.05,
                "my_mae": -0.02,
                "my_mfe": 0.06,
            },
        ]
        ec = [10000, 10500]
        report = generate_report(
            ec,
            trades,
            pd.DataFrame(),
            pnl_key="ret",
            mae_key="my_mae",
            mfe_key="my_mfe",
            save=False,
        )
        assert report["mae_mfe"]["avg_mfe"] == pytest.approx(0.06)


class TestSaveReport:
    def test_save_and_load(self, tmp_path):
        report = {"mae_mfe": {"avg_mae": -1.0}, "risk_metrics": {"omega_ratio": 1.5}}
        filepath = save_report(
            report,
            strategy_name="test_strat",
            symbol="AAPL",
            output_dir=str(tmp_path),
            run_type="backtest",
        )
        assert os.path.exists(filepath)
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["metadata"]["repo"] == "cerberus"
        assert loaded["metadata"]["strategy"] == "test_strat"
        assert loaded["metadata"]["symbol"] == "AAPL"
        assert loaded["mae_mfe"]["avg_mae"] == -1.0

    def test_save_with_window(self, tmp_path):
        report = {"test": True}
        filepath = save_report(
            report,
            strategy_name="wfo_strat",
            symbol="SPY",
            output_dir=str(tmp_path),
            run_type="wfo_window",
            window_idx=3,
        )
        assert "window3" in filepath

    def test_json_serializable(self, tmp_path):
        """Ensure all numpy types are serialized properly."""
        report = {
            "value": float(np.float64(1.5)),
            "array": [float(x) for x in np.array([1.0, 2.0])],
        }
        filepath = save_report(report, strategy_name="s", symbol="X", output_dir=str(tmp_path))
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["value"] == 1.5


class TestGenerateReportSave:
    def test_auto_save(self, tmp_path):
        ec = _make_equity_curve()
        trades = _make_trades()
        bar_data = pd.DataFrame(index=pd.date_range("2025-01-02 09:30", periods=60, freq="1min"))
        report = generate_report(
            ec,
            trades,
            bar_data,
            save=True,
            strategy_name="test_strat",
            symbol="AAPL",
            output_dir=str(tmp_path),
            mc_simulations=50,
            rolling_window=3,
        )
        assert "_saved_to" in report
        assert os.path.exists(report["_saved_to"])
