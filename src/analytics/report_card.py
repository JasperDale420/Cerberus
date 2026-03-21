"""Comprehensive analytics report card for backtest diagnostics.

Produces institutional-grade analysis across three tiers:
- Tier 1: MAE/MFE, statistical significance, Monte Carlo, PnL distribution
- Tier 2: Risk metrics, rolling metrics, drawdown catalog, trade clustering, cost sensitivity
- Tier 3: Time-based breakdowns, entry/exit efficiency

Reports are saved to artifacts/reports/analytics/ organized by strategy and timestamp.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.core.logger import StructuredLogger

logger = StructuredLogger("report_card")

# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "reports", "analytics")


def save_report(
    report: dict[str, Any],
    strategy_name: str,
    symbol: str,
    output_dir: str | None = None,
    run_type: str = "backtest",
    window_idx: int | None = None,
) -> str:
    """Save an analytics report to an organized folder structure.

    Directory layout:
        artifacts/reports/analytics/{strategy}/{symbol}/
            {YYYY-MM-DD}_{HHMMSS}_{run_type}[_window{N}].json

    Args:
        report: The analytics report dict from generate_report().
        strategy_name: e.g. "momentum_breakout"
        symbol: e.g. "AAPL"
        output_dir: Override base directory (default: artifacts/reports/analytics/)
        run_type: "backtest", "wfo_window", or "wfo_aggregate"
        window_idx: WFO window index (if run_type is wfo_window)

    Returns:
        Absolute path to the saved JSON file.
    """
    base_dir = os.path.abspath(output_dir or REPORTS_DIR)
    safe_symbol = symbol.replace("/", "_")
    strategy_dir = os.path.join(base_dir, strategy_name, safe_symbol)
    os.makedirs(strategy_dir, exist_ok=True)

    ts = datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y-%m-%d_%H%M%S")

    if window_idx is not None:
        filename = f"{ts_str}_{run_type}_window{window_idx}.json"
    else:
        filename = f"{ts_str}_{run_type}.json"

    filepath = os.path.join(strategy_dir, filename)

    # Add metadata to report
    report_with_meta = {
        "metadata": {
            "repo": "cerberus",
            "strategy": strategy_name,
            "symbol": symbol,
            "run_type": run_type,
            "window_idx": window_idx,
            "timestamp": ts.isoformat(),
        },
        **report,
    }

    with open(filepath, "w") as f:
        json.dump(report_with_meta, f, indent=2, default=str)

    logger.info("report_saved", path=filepath, strategy=strategy_name, symbol=symbol)
    return filepath


# ---------------------------------------------------------------------------
# Tier 1 analytics
# ---------------------------------------------------------------------------


def analyze_mae_mfe(
    trades: list[dict],
    mae_key: str = "mae_r",
    mfe_key: str = "mfe_r",
    pnl_key: str = "pnl_r",
) -> dict[str, Any]:
    """Analyze Maximum Adverse/Favorable Excursion across all trades.

    Returns aggregate MAE/MFE statistics, edge ratio, capture efficiency,
    and per-trade detail records.
    """
    if not trades:
        return {
            "avg_mae": 0.0,
            "avg_mfe": 0.0,
            "worst_mae": 0.0,
            "best_mfe": 0.0,
            "edge_ratio": 0.0,
            "avg_capture_efficiency": 0.0,
            "per_trade": [],
        }

    mae_vals = [t[mae_key] for t in trades]
    mfe_vals = [t[mfe_key] for t in trades]

    avg_mae = sum(mae_vals) / len(mae_vals)
    avg_mfe = sum(mfe_vals) / len(mfe_vals)
    worst_mae = min(mae_vals)
    best_mfe = max(mfe_vals)

    edge_ratio = avg_mfe / abs(avg_mae) if abs(avg_mae) > 1e-12 else 0.0

    per_trade = []
    capture_efficiencies = []
    for t in trades:
        mfe_val = t[mfe_key]
        pnl_val = t[pnl_key]
        ce = (pnl_val / mfe_val) if abs(mfe_val) > 1e-12 else 0.0
        capture_efficiencies.append(ce)
        per_trade.append(
            {
                mae_key: float(t[mae_key]),
                mfe_key: float(mfe_val),
                pnl_key: float(pnl_val),
                "capture_efficiency": float(ce),
            }
        )

    avg_capture = sum(capture_efficiencies) / len(capture_efficiencies) if capture_efficiencies else 0.0

    return {
        "avg_mae": float(avg_mae),
        "avg_mfe": float(avg_mfe),
        "worst_mae": float(worst_mae),
        "best_mfe": float(best_mfe),
        "edge_ratio": float(edge_ratio),
        "avg_capture_efficiency": float(avg_capture),
        "per_trade": per_trade,
    }


def compute_statistical_tests(
    returns: list[float],
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run statistical significance tests on a return series.

    Tests: t-test (mean != 0), Ljung-Box (autocorrelation), Jarque-Bera (normality),
    runs test (randomness), bootstrap confidence interval, permutation test.
    """
    defaults = {
        "t_stat": 0.0,
        "t_pvalue": 1.0,
        "ljung_box_pvalue": 1.0,
        "jarque_bera_pvalue": 1.0,
        "runs_test_pvalue": 1.0,
        "bootstrap_ci_95": [0.0, 0.0],
        "permutation_pvalue": 1.0,
        "mean_return": 0.0,
    }

    if len(returns) < 3:
        return defaults

    arr = np.array(returns, dtype=np.float64)
    mean_ret = float(np.mean(arr))
    result: dict[str, Any] = {"mean_return": mean_ret}

    # T-test: H0 = mean is zero
    t_stat, t_pval = stats.ttest_1samp(arr, 0.0)
    result["t_stat"] = float(t_stat)
    result["t_pvalue"] = float(t_pval)

    # Ljung-Box: H0 = no autocorrelation
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        n = len(arr)
        lags = min(10, n // 5)
        if lags < 1:
            lags = 1
        lb_result = acorr_ljungbox(arr, lags=[lags], return_df=True)
        result["ljung_box_pvalue"] = float(lb_result["lb_pvalue"].iloc[0])
    except Exception:
        result["ljung_box_pvalue"] = 1.0

    # Jarque-Bera: H0 = normally distributed
    jb_stat, jb_pval = stats.jarque_bera(arr)
    result["jarque_bera_pvalue"] = float(jb_pval)

    # Runs test: H0 = randomly ordered
    median_val = float(np.median(arr))
    above = arr > median_val
    n_above = int(np.sum(above))
    n_below = len(arr) - n_above

    if n_above > 0 and n_below > 0:
        runs = 1
        for i in range(1, len(above)):
            if above[i] != above[i - 1]:
                runs += 1

        n_total = n_above + n_below
        expected_runs = 1.0 + (2.0 * n_above * n_below) / n_total
        var_runs = (2.0 * n_above * n_below * (2.0 * n_above * n_below - n_total)) / (n_total**2 * (n_total - 1.0))

        if var_runs > 0:
            z_runs = (runs - expected_runs) / math.sqrt(var_runs)
            result["runs_test_pvalue"] = float(2.0 * (1.0 - stats.norm.cdf(abs(z_runs))))
        else:
            result["runs_test_pvalue"] = 1.0
    else:
        result["runs_test_pvalue"] = 1.0

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[i] = np.mean(sample)
    result["bootstrap_ci_95"] = [
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    ]

    # Sign-flip permutation test (shuffling preserves mean, so use random signs)
    observed_abs_mean = abs(mean_ret)
    count_ge = 0
    perm_rng = np.random.default_rng(seed + 1)
    for _ in range(n_permutations):
        signs = perm_rng.choice([-1.0, 1.0], size=len(arr))
        if abs(float(np.mean(arr * signs))) >= observed_abs_mean:
            count_ge += 1
    result["permutation_pvalue"] = float(count_ge / n_permutations)

    # Ensure all keys present with finite values
    for key in defaults:
        if key not in result:
            result[key] = defaults[key]
        elif isinstance(result[key], float) and not math.isfinite(result[key]):
            result[key] = defaults[key]

    return result


def run_monte_carlo(
    returns: list[float],
    n_simulations: int = 1000,
    seed: int = 42,
    starting_equity: float = 10000.0,
    ruin_threshold: float = 0.5,
) -> dict[str, Any]:
    """Monte Carlo simulation by reshuffling trade returns.

    For each simulation, shuffles the returns and applies them sequentially
    to starting_equity, tracking final equity, max drawdown, and ruin events.
    """
    if not returns:
        return {
            "ruin_probability": 0.0,
            "median_final_equity": starting_equity,
            "percentile_5": starting_equity,
            "percentile_95": starting_equity,
            "max_drawdown_distribution": [0.0, 0.0, 0.0],
        }

    ruin_level = starting_equity * ruin_threshold
    arr = np.array(returns, dtype=np.float64)
    rng = np.random.default_rng(seed)

    final_equities = np.empty(n_simulations, dtype=np.float64)
    max_drawdowns = np.empty(n_simulations, dtype=np.float64)
    ruin_count = 0

    for i in range(n_simulations):
        shuffled = arr.copy()
        rng.shuffle(shuffled)

        equity = starting_equity
        peak = equity
        worst_dd = 0.0
        hit_ruin = False

        for ret in shuffled:
            equity += ret * starting_equity * 0.01  # additive: treat each unit as 1% of capital
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > worst_dd:
                worst_dd = dd
            if equity <= ruin_level:
                hit_ruin = True

        final_equities[i] = equity
        max_drawdowns[i] = worst_dd
        if hit_ruin:
            ruin_count += 1

    return {
        "ruin_probability": float(ruin_count / n_simulations),
        "median_final_equity": float(np.median(final_equities)),
        "percentile_5": float(np.percentile(final_equities, 5)),
        "percentile_95": float(np.percentile(final_equities, 95)),
        "max_drawdown_distribution": [
            float(np.percentile(max_drawdowns, 5)),
            float(np.median(max_drawdowns)),
            float(np.percentile(max_drawdowns, 95)),
        ],
    }


def analyze_pnl_distribution(pnls: list[float]) -> dict[str, Any]:
    """Analyze PnL distribution: central tendency, dispersion, shape, and percentiles."""
    if not pnls:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "percentile_5": 0.0,
            "percentile_25": 0.0,
            "percentile_75": 0.0,
            "percentile_95": 0.0,
        }

    arr = np.array(pnls, dtype=np.float64)

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "skew": float(stats.skew(arr, bias=False)) if len(arr) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(arr, bias=False, fisher=True)) if len(arr) > 3 else 0.0,
        "percentile_5": float(np.percentile(arr, 5)),
        "percentile_25": float(np.percentile(arr, 25)),
        "percentile_75": float(np.percentile(arr, 75)),
        "percentile_95": float(np.percentile(arr, 95)),
    }


# ---------------------------------------------------------------------------
# Tier 2 analytics
# ---------------------------------------------------------------------------


def compute_risk_metrics(returns: list[float], threshold: float = 0.0) -> dict[str, Any]:
    """Advanced risk metrics from a return series.

    Computes Omega ratio, tail ratio, VaR/CVaR at 95%, Ulcer Index, and
    gain-to-pain ratio.
    """
    zeros: dict[str, Any] = {
        "omega_ratio": 0.0,
        "tail_ratio": 0.0,
        "var_95": 0.0,
        "cvar_95": 0.0,
        "ulcer_index": 0.0,
        "gain_to_pain": 0.0,
    }

    if len(returns) < 2:
        return zeros

    arr = np.array(returns, dtype=np.float64)

    # Omega ratio
    gains = np.sum(np.maximum(arr - threshold, 0.0))
    losses = np.sum(np.maximum(threshold - arr, 0.0))
    omega = min(float(gains / losses), 999.0) if losses > 1e-12 else 999.0

    # Tail ratio
    p95 = float(np.percentile(arr, 95))
    p5 = float(np.percentile(arr, 5))
    tail_ratio = float(p95 / abs(p5)) if abs(p5) > 1e-12 else 0.0

    # VaR 95% and CVaR 95%
    var_95 = float(p5)
    mask = arr <= var_95
    cvar_95 = float(np.mean(arr[mask])) if np.any(mask) else var_95

    # Ulcer Index: sqrt(mean(drawdown_pct^2)) from running peak equity
    equity = np.concatenate(([1.0], np.cumprod(1.0 + arr)))
    running_peak = np.maximum.accumulate(equity)
    drawdown_pct = (running_peak - equity) / running_peak
    ulcer_index = float(np.sqrt(np.mean(drawdown_pct**2)))

    # Gain-to-pain ratio
    neg_returns = arr[arr < 0]
    sum_abs_neg = float(np.sum(np.abs(neg_returns)))
    gain_to_pain = float(np.sum(arr) / sum_abs_neg) if sum_abs_neg > 1e-12 else 0.0

    return {
        "omega_ratio": omega,
        "tail_ratio": tail_ratio,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "ulcer_index": ulcer_index,
        "gain_to_pain": gain_to_pain,
    }


def compute_rolling_metrics(
    returns: list[float],
    window: int = 50,
    bars_per_year: float = 98280.0,
) -> dict[str, Any]:
    """Rolling Sharpe ratio and win rate over a sliding window."""
    n = len(returns)
    length = max(0, n - window + 1)

    if length <= 0:
        return {"rolling_sharpe": [], "rolling_win_rate": []}

    arr = np.array(returns, dtype=np.float64)
    rolling_sharpe: list[float] = []
    rolling_win_rate: list[float] = []

    for i in range(length):
        chunk = arr[i : i + window]
        mean_r = float(np.mean(chunk))
        std_r = float(np.std(chunk, ddof=1))
        sharpe = (mean_r / std_r * math.sqrt(bars_per_year)) if std_r > 1e-12 else 0.0
        rolling_sharpe.append(float(sharpe))

        wr = float(np.sum(chunk > 0) / window)
        rolling_win_rate.append(wr)

    return {"rolling_sharpe": rolling_sharpe, "rolling_win_rate": rolling_win_rate}


def catalog_drawdowns(equity_curve: list[float], top_n: int = 5) -> list[dict[str, Any]]:
    """Identify top N drawdowns by depth from an equity curve.

    Each entry has: depth_pct, start_bar, trough_bar, recovery_bar, duration_bars.
    """
    if len(equity_curve) < 2:
        return []

    drawdowns: list[dict[str, Any]] = []
    peak = equity_curve[0]
    peak_bar = 0
    trough = peak
    trough_bar = 0
    in_dd = False

    for i in range(1, len(equity_curve)):
        val = equity_curve[i]

        if val >= peak:
            # Recovered or new high
            if in_dd:
                depth = (peak - trough) / peak if peak > 0 else 0.0
                if depth > 1e-12:
                    drawdowns.append(
                        {
                            "depth_pct": float(depth),
                            "start_bar": peak_bar,
                            "trough_bar": trough_bar,
                            "recovery_bar": i,
                            "duration_bars": i - peak_bar,
                        }
                    )
                in_dd = False
            peak = val
            peak_bar = i
            trough = val
            trough_bar = i
        else:
            in_dd = True
            if val < trough:
                trough = val
                trough_bar = i

    # If still in drawdown at end
    if in_dd:
        depth = (peak - trough) / peak if peak > 0 else 0.0
        if depth > 1e-12:
            drawdowns.append(
                {
                    "depth_pct": float(depth),
                    "start_bar": peak_bar,
                    "trough_bar": trough_bar,
                    "recovery_bar": None,
                    "duration_bars": len(equity_curve) - 1 - peak_bar,
                }
            )

    drawdowns.sort(key=lambda d: d["depth_pct"], reverse=True)
    return drawdowns[:top_n]


def analyze_trade_clustering(
    trades: list[dict],
    pnl_key: str = "pnl_r",
) -> dict[str, Any]:
    """Analyze trade clustering: streaks, conditional win rates, and inter-trade gaps."""
    zeros: dict[str, Any] = {
        "win_streak_max": 0,
        "loss_streak_max": 0,
        "win_after_win_rate": 0.0,
        "win_after_loss_rate": 0.0,
        "avg_gap_bars": 0.0,
    }

    if not trades:
        return zeros

    wins = [t[pnl_key] > 0 for t in trades]

    # Streaks
    win_streak = 0
    loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for w in wins:
        if w:
            win_streak += 1
            loss_streak = 0
            max_win_streak = max(max_win_streak, win_streak)
        else:
            loss_streak += 1
            win_streak = 0
            max_loss_streak = max(max_loss_streak, loss_streak)

    # Conditional win rates
    win_after_win_count = 0
    win_after_win_total = 0
    win_after_loss_count = 0
    win_after_loss_total = 0

    for i in range(1, len(wins)):
        if wins[i - 1]:
            win_after_win_total += 1
            if wins[i]:
                win_after_win_count += 1
        else:
            win_after_loss_total += 1
            if wins[i]:
                win_after_loss_count += 1

    waw = float(win_after_win_count / win_after_win_total) if win_after_win_total > 0 else 0.0
    wal = float(win_after_loss_count / win_after_loss_total) if win_after_loss_total > 0 else 0.0

    # Average gap bars between trades
    gaps: list[int] = []
    for i in range(1, len(trades)):
        entry_bar = _get_entry_bar(trades[i])
        prev_exit_bar = _get_exit_bar(trades[i - 1])
        if entry_bar is not None and prev_exit_bar is not None:
            gaps.append(entry_bar - prev_exit_bar)

    avg_gap = float(sum(gaps) / len(gaps)) if gaps else 0.0

    return {
        "win_streak_max": max_win_streak,
        "loss_streak_max": max_loss_streak,
        "win_after_win_rate": waw,
        "win_after_loss_rate": wal,
        "avg_gap_bars": avg_gap,
    }


def compute_cost_sensitivity(
    trade_pnls: list[float],
    n_trades: int,
    base_cost_bps: float = 20.0,
) -> dict[str, Any]:
    """Cost sensitivity sweep: break-even cost and net PnL at various cost levels."""
    if not trade_pnls or n_trades <= 0:
        return {"break_even_cost_bps": 0.0, "cost_sweep": []}

    total_pnl = sum(trade_pnls)

    # Each additional bps of cost reduces total PnL by n_trades * 0.02 pct
    cost_per_bps_val = n_trades * 0.02

    if total_pnl <= 0:
        break_even = 0.0
    else:
        break_even = float(base_cost_bps + total_pnl / cost_per_bps_val)

    # Cost sweep at multipliers of base_cost
    multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    cost_sweep: list[dict[str, float]] = []
    for m in multipliers:
        cost_bps = base_cost_bps * m
        cost_impact = cost_bps * cost_per_bps_val
        net_pnl = total_pnl - cost_impact
        cost_sweep.append({"cost_bps": float(cost_bps), "net_pnl_pct": float(net_pnl)})

    return {"break_even_cost_bps": float(break_even), "cost_sweep": cost_sweep}


# ---------------------------------------------------------------------------
# Tier 3 analytics
# ---------------------------------------------------------------------------


def _get_entry_bar(trade: dict) -> int | None:
    """Extract entry bar from a trade dict, supporting both bar index and datetime."""
    if "entry_bar" in trade:
        return trade["entry_bar"]
    return None


def _get_exit_bar(trade: dict) -> int | None:
    """Extract exit bar from a trade dict, supporting both bar index and datetime."""
    if "exit_bar" in trade:
        return trade["exit_bar"]
    return None


def compute_time_breakdowns(
    trades: list[dict],
    bar_index: pd.DatetimeIndex,
    pnl_key: str = "pnl_r",
    entry_bar_key: str = "entry_bar",
) -> dict[str, Any]:
    """Break down trade performance by hour-of-day and day-of-week."""
    if not trades or len(bar_index) == 0:
        return {"by_hour": {}, "by_day_of_week": {}}

    by_hour: dict[int, list[float]] = {}
    by_day: dict[str, list[float]] = {}

    for t in trades:
        entry_bar = t.get(entry_bar_key)
        if entry_bar is None or entry_bar >= len(bar_index):
            continue
        ts = bar_index[entry_bar]
        hour = ts.hour
        day_name = ts.day_name()

        pnl_val = t[pnl_key]
        by_hour.setdefault(hour, []).append(pnl_val)
        by_day.setdefault(day_name, []).append(pnl_val)

    hour_result: dict[int, dict[str, float]] = {}
    for hour, pnls in sorted(by_hour.items()):
        n = len(pnls)
        hour_result[hour] = {
            "n_trades": n,
            "avg_pnl": float(sum(pnls) / n),
            "win_rate": float(sum(1 for p in pnls if p > 0) / n),
        }

    day_result: dict[str, dict[str, float]] = {}
    for day_name, pnls in by_day.items():
        n = len(pnls)
        day_result[day_name] = {
            "n_trades": n,
            "avg_pnl": float(sum(pnls) / n),
            "win_rate": float(sum(1 for p in pnls if p > 0) / n),
        }

    return {"by_hour": hour_result, "by_day_of_week": day_result}


def compute_entry_exit_efficiency(
    trades: list[dict],
    mae_key: str = "mae_r",
    mfe_key: str = "mfe_r",
) -> dict[str, Any]:
    """Compute entry and exit efficiency using MAE/MFE.

    Entry efficiency: mfe / (mfe - mae) — how much of the range was above entry.
    Exit efficiency: (pnl_from_entry - mae) / (mfe - mae) — how much captured from the bottom.
    Both clamped to [0, 1]. Trades with range < 0.0001 are skipped.
    """
    if not trades:
        return {"avg_entry_efficiency": 0.0, "avg_exit_efficiency": 0.0}

    entry_effs: list[float] = []
    exit_effs: list[float] = []

    for t in trades:
        mfe_val = t[mfe_key]
        mae_val = t[mae_key]
        range_val = mfe_val - mae_val
        if range_val < 0.0001:
            continue

        entry_eff = mfe_val / range_val
        entry_eff = max(0.0, min(1.0, entry_eff))
        entry_effs.append(entry_eff)

        entry_price = t.get("entry_price", 0.0)
        exit_price = t.get("exit_price", 0.0)
        if entry_price > 0:
            pnl_from_entry = exit_price / entry_price - 1.0
        else:
            pnl_from_entry = 0.0
        exit_eff = (pnl_from_entry - mae_val) / range_val
        exit_eff = max(0.0, min(1.0, exit_eff))
        exit_effs.append(exit_eff)

    avg_entry = float(sum(entry_effs) / len(entry_effs)) if entry_effs else 0.0
    avg_exit = float(sum(exit_effs) / len(exit_effs)) if exit_effs else 0.0

    return {"avg_entry_efficiency": avg_entry, "avg_exit_efficiency": avg_exit}


# ---------------------------------------------------------------------------
# Report orchestrator
# ---------------------------------------------------------------------------


def generate_report(
    equity_curve: list[float],
    trades: list[dict],
    bar_data: pd.DataFrame,
    bars_per_year: float = 98280.0,
    mc_simulations: int = 1000,
    mc_seed: int = 42,
    rolling_window: int = 50,
    base_cost_bps: float = 20.0,
    pnl_key: str = "pnl_r",
    mae_key: str = "mae_r",
    mfe_key: str = "mfe_r",
    strategy_name: str | None = None,
    symbol: str | None = None,
    run_type: str = "backtest",
    window_idx: int | None = None,
    save: bool = True,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Generate comprehensive analytics report. Called automatically after every backtest.

    Reports are saved to artifacts/reports/analytics/{strategy}/{symbol}/ by default.
    Set save=False to skip persistence (e.g. during unit tests or optimization trials).
    """
    # Compute returns from equity curve
    returns: list[float] = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            returns.append(equity_curve[i] / equity_curve[i - 1] - 1.0)
        else:
            returns.append(0.0)

    trade_pnls = [t[pnl_key] for t in trades]

    # Extract DatetimeIndex
    bar_index = bar_data.index if isinstance(bar_data.index, pd.DatetimeIndex) else pd.DatetimeIndex([])

    report: dict[str, Any] = {}

    # Tier 1
    mae_mfe = analyze_mae_mfe(trades, mae_key=mae_key, mfe_key=mfe_key, pnl_key=pnl_key)
    mae_mfe.pop("per_trade", None)  # too large for serialization
    report["mae_mfe"] = mae_mfe
    report["statistical_tests"] = compute_statistical_tests(
        returns, n_bootstrap=min(1000, max(100, len(returns) * 2)), seed=mc_seed
    )
    report["monte_carlo"] = run_monte_carlo(
        trade_pnls if trade_pnls else returns,
        n_simulations=mc_simulations,
        seed=mc_seed,
    )
    report["pnl_distribution"] = analyze_pnl_distribution(trade_pnls)

    # Tier 2
    report["risk_metrics"] = compute_risk_metrics(returns)
    report["rolling_metrics"] = compute_rolling_metrics(returns, window=rolling_window, bars_per_year=252.0)
    report["drawdown_catalog"] = catalog_drawdowns(equity_curve, top_n=5)
    report["trade_clustering"] = analyze_trade_clustering(trades, pnl_key=pnl_key)
    report["cost_sensitivity"] = compute_cost_sensitivity(trade_pnls, len(trades), base_cost_bps=base_cost_bps)

    # Tier 3
    report["time_breakdowns"] = compute_time_breakdowns(trades, bar_index, pnl_key=pnl_key)
    report["entry_exit_efficiency"] = compute_entry_exit_efficiency(trades, mae_key=mae_key, mfe_key=mfe_key)

    # Auto-save to organized folder
    if save and strategy_name:
        try:
            saved_path = save_report(
                report,
                strategy_name=strategy_name,
                symbol=symbol or "unknown",
                output_dir=output_dir,
                run_type=run_type,
                window_idx=window_idx,
            )
            report["_saved_to"] = saved_path
        except Exception:
            logger.warning("report_save_failed", strategy=strategy_name, symbol=symbol)

    return report
