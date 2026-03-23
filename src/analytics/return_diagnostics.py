"""Return diagnostics — autocorrelation and turnover analysis.

Provides tools for detecting serial correlation in equity returns
and analyzing portfolio turnover costs.
"""

from __future__ import annotations

import math

import numpy as np


def compute_return_autocorrelation(daily_returns: np.ndarray, max_lag: int = 5) -> dict:
    """Compute autocorrelation of daily equity returns at multiple lags.

    Args:
        daily_returns: Array of daily equity returns.
        max_lag: Maximum lag to compute (default 5).

    Returns:
        Dict with lag entries (correlation + significance) and ``any_significant`` flag.
        Significance threshold: |corr| > 2 / sqrt(N).
    """
    n = len(daily_returns)
    threshold = 2.0 / math.sqrt(n)

    # Demean
    x = daily_returns - np.mean(daily_returns)
    var = np.dot(x, x)

    any_significant = False
    result: dict = {}

    for lag in range(1, max_lag + 1):
        if lag >= n:
            corr = 0.0
        else:
            corr = float(np.dot(x[lag:], x[:-lag]) / var)
        significant = abs(corr) > threshold
        if significant:
            any_significant = True
        result[f"lag_{lag}"] = {"correlation": corr, "significant": significant}

    result["any_significant"] = any_significant
    return result


def compute_turnover_analysis(trades: list[dict], initial_capital: float, n_trading_days: int) -> dict:
    """Analyze portfolio turnover and cost drag.

    Args:
        trades: List of trade dicts with keys: qty, entry_price, exit_price, commission, slippage.
        initial_capital: Starting capital.
        n_trading_days: Number of trading days in the period.

    Returns:
        Dict with gross_volume, annualized_turnover, total_cost, cost_drag_pct, avg_cost_per_trade.
    """
    gross_volume = 0.0
    total_cost = 0.0
    gross_pnl = 0.0

    for t in trades:
        qty = t["qty"]
        entry_price = t["entry_price"]
        exit_price = t["exit_price"]
        commission = t.get("commission", 0.0)
        slippage = t.get("slippage", 0.0)

        gross_volume += qty * entry_price + qty * exit_price
        total_cost += commission + slippage
        gross_pnl += qty * (exit_price - entry_price)

    annualized_turnover = (gross_volume / initial_capital) * (252 / n_trading_days) if n_trading_days > 0 else 0.0
    cost_drag_pct = (total_cost / gross_pnl * 100) if gross_pnl > 0 else 0.0
    avg_cost_per_trade = total_cost / len(trades) if trades else 0.0

    return {
        "gross_volume": gross_volume,
        "annualized_turnover": annualized_turnover,
        "total_cost": total_cost,
        "cost_drag_pct": cost_drag_pct,
        "avg_cost_per_trade": avg_cost_per_trade,
    }
