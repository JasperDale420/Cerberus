"""Fama-French 4-factor attribution for backtest returns.

Runs OLS regression of strategy excess returns against market factors:
    strategy_excess = alpha + beta1*MktRF + beta2*SMB + beta3*HML + beta4*UMD + epsilon

Uses numpy lstsq to avoid a statsmodels dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class FactorAttribution:
    """Result of Fama-French factor attribution regression."""

    alpha: float  # annualized alpha (intercept * 252)
    alpha_tstat: float  # t-stat of alpha
    alpha_significant: bool  # |t-stat| > 2.0
    factor_loadings: dict[str, float]  # beta coefficients: Mkt-RF, SMB, HML, UMD
    factor_tstats: dict[str, float]  # t-stats per factor
    r_squared: float  # regression R-squared
    adj_r_squared: float  # adjusted R-squared
    residual_vol: float  # annualized residual volatility
    n_observations: int


def compute_factor_attribution(
    strategy_excess_returns: np.ndarray,
    factor_returns: dict[str, np.ndarray],
) -> FactorAttribution:
    """Run OLS factor attribution regression.

    Parameters
    ----------
    strategy_excess_returns:
        1-D array of daily strategy excess returns (already minus risk-free rate).
    factor_returns:
        Dict mapping factor names (e.g. "Mkt-RF", "SMB", "HML", "UMD") to
        1-D arrays of daily factor returns.  Missing factors are simply omitted
        from the regression.

    Returns
    -------
    FactorAttribution with regression coefficients, t-stats, and fit metrics.
    """
    y = np.asarray(strategy_excess_returns, dtype=np.float64)
    n = len(y)

    # Build design matrix: column of ones (intercept) + factor columns
    factor_names = list(factor_returns.keys())
    k = len(factor_names)  # number of factors (excludes intercept)

    X = np.ones((n, k + 1), dtype=np.float64)
    for i, name in enumerate(factor_names):
        X[:, i + 1] = np.asarray(factor_returns[name], dtype=np.float64)

    # OLS via least-squares: y = X @ beta + epsilon
    beta, residuals_ss, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    # Residuals and sigma squared
    y_hat = X @ beta
    residuals = y - y_hat
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    # Degrees of freedom
    dof = n - k - 1  # n - (k factors + 1 intercept)

    # Residual variance (sigma squared)
    sigma2 = ss_res / max(dof, 1)

    # Standard errors: SE = sqrt(diag(sigma2 * (X'X)^-1))
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse for singular matrices
        XtX_inv = np.linalg.pinv(X.T @ X)

    se = np.sqrt(np.diag(sigma2 * XtX_inv))

    # t-stats: t = beta / SE(beta)
    tstats = beta / np.where(se > 0, se, 1e-12)

    # Extract results
    alpha_daily = float(beta[0])
    alpha_annualized = alpha_daily * 252
    alpha_tstat = float(tstats[0])

    factor_loadings = {name: float(beta[i + 1]) for i, name in enumerate(factor_names)}
    factor_tstats_dict = {name: float(tstats[i + 1]) for i, name in enumerate(factor_names)}

    # R-squared
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))

    # Adjusted R-squared
    if n - k - 1 > 0:
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - k - 1)
    else:
        adj_r_squared = r_squared

    # Annualized residual volatility
    residual_vol = float(np.std(residuals, ddof=1)) * np.sqrt(252)

    return FactorAttribution(
        alpha=alpha_annualized,
        alpha_tstat=alpha_tstat,
        alpha_significant=abs(alpha_tstat) > 2.0,
        factor_loadings=factor_loadings,
        factor_tstats=factor_tstats_dict,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        residual_vol=residual_vol,
        n_observations=n,
    )


def load_fama_french_factors(start_date: str, end_date: str) -> dict[str, np.ndarray] | None:
    """Load Fama-French factor data from local cache.

    Looks for a parquet file at ``artifacts/factor_data/ff_factors.parquet``.
    If not found, returns None so the caller can handle gracefully (e.g. skip
    factor attribution or warn the user).

    Parameters
    ----------
    start_date:
        ISO date string for the start of the period (inclusive).
    end_date:
        ISO date string for the end of the period (inclusive).

    Returns
    -------
    Dict with keys "Mkt-RF", "SMB", "HML", "UMD", "RF", "dates" as numpy
    arrays, or None if the cache file is missing.
    """
    cache_path = Path("artifacts/factor_data/ff_factors.parquet")
    if not cache_path.exists():
        return None

    try:
        import pyarrow.parquet as pq

        table = pq.read_table(cache_path)
        df_dict = {col: table.column(col).to_numpy() for col in table.column_names}

        # Filter by date range if a 'date' column exists
        if "date" in df_dict:
            dates = df_dict["date"].astype(str)
            mask = (dates >= start_date) & (dates <= end_date)
            df_dict = {k: v[mask] for k, v in df_dict.items()}

        result: dict[str, np.ndarray] = {}
        for key in ("Mkt-RF", "SMB", "HML", "UMD", "RF"):
            if key in df_dict:
                result[key] = df_dict[key].astype(np.float64)
        if "date" in df_dict:
            result["dates"] = df_dict["date"]

        return result if result else None
    except Exception:
        return None
