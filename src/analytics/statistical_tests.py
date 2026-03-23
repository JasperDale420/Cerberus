"""Probabilistic Sharpe Ratio (PSR) and Minimum Backtest Length (MinBTL).

Implements statistical tests from Bailey & Lopez de Prado (2014):
"The Deflated Sharpe Ratio: Correcting for Selection Bias,
 Backtest Overfitting, and Non-Normality"

These tests answer two questions:
1. PSR: What is the probability that the observed Sharpe exceeds a benchmark?
2. MinBTL: How many observations do we need for the Sharpe to be significant?
"""

from __future__ import annotations

import math

from scipy.stats import norm


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_returns: int,
    skew: float,
    kurtosis: float,
) -> dict:
    """Compute the Probabilistic Sharpe Ratio (PSR).

    PSR = Phi((SR_obs - SR_bench) * sqrt(n-1) / sqrt(1 - skew*SR_obs + ((kurtosis-1)/4)*SR_obs^2))

    Args:
        observed_sharpe: Observed (annualized) Sharpe ratio.
        benchmark_sharpe: Benchmark Sharpe ratio to test against.
        n_returns: Number of return observations.
        skew: Skewness of the return distribution.
        kurtosis: Excess kurtosis + 3 (i.e., normal = 3).

    Returns:
        Dict with "psr" (probability) and "significant" (bool, True if psr > 0.95).
    """
    if n_returns < 2:
        return {"psr": 0.0, "significant": False}

    sr = observed_sharpe
    sr_diff = sr - benchmark_sharpe

    # Variance of the Sharpe ratio estimator (non-normal adjustment)
    variance = 1.0 - skew * sr + ((kurtosis - 1) / 4.0) * sr**2

    if variance <= 0:
        # Degenerate case — treat as non-significant
        return {"psr": 0.0, "significant": False}

    z = sr_diff * math.sqrt(n_returns - 1) / math.sqrt(variance)
    psr = float(norm.cdf(z))

    return {"psr": psr, "significant": psr > 0.95}


def minimum_backtest_length(
    observed_sharpe: float,
    skew: float,
    kurtosis: float,
    frequency: int = 252,
    confidence: float = 0.95,
    n_observations: int = 0,
) -> dict:
    """Compute the Minimum Backtest Length (MinBTL).

    MinBTL = 1 + (1 - skew*SR + ((kurtosis-1)/4)*SR^2) * (z_alpha / SR)^2

    Args:
        observed_sharpe: Observed (annualized) Sharpe ratio.
        skew: Skewness of the return distribution.
        kurtosis: Excess kurtosis + 3 (i.e., normal = 3).
        frequency: Trading days per year (default 252).
        confidence: Confidence level for significance (default 0.95).
        n_observations: Actual number of observations available.

    Returns:
        Dict with "min_observations", "min_years", "have_enough", "n_observations".
    """
    sr = observed_sharpe
    z_alpha = float(norm.ppf(confidence))

    if abs(sr) < 1e-12:
        # Sharpe is effectively zero — would need infinite observations
        return {
            "min_observations": math.inf,
            "min_years": math.inf,
            "have_enough": False,
            "n_observations": n_observations,
        }

    # Non-normality adjustment factor
    adjustment = 1.0 - skew * sr + ((kurtosis - 1) / 4.0) * sr**2

    min_obs_raw = 1.0 + adjustment * (z_alpha / sr) ** 2
    min_obs = int(math.ceil(min_obs_raw))
    min_years = min_obs / frequency

    return {
        "min_observations": min_obs,
        "min_years": min_years,
        "have_enough": n_observations >= min_obs,
        "n_observations": n_observations,
    }
