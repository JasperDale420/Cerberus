"""Causal Inference Signal Filter for strategy validation.

Uses Granger causality testing to determine whether a strategy's signals
have genuine predictive power over subsequent returns, filtering out
strategies that appear profitable only due to spurious correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import f as f_dist

from src.core.logger import StructuredLogger


@dataclass
class CausalScore:
    """Result of Granger causality analysis for a single strategy."""

    strategy: str
    causal_strength: float  # 0 to 1, how strong the causal link is
    granger_f_stat: float  # F-statistic from Granger test
    granger_p_value: float  # p-value
    optimal_lag: int  # best lag by BIC
    n_observations: int  # sample size used
    last_updated: Optional[datetime] = None


class CausalSignalFilter:
    """Filters strategy signals based on Granger causality testing.

    Runs Granger causality tests to determine whether each strategy's signal
    series genuinely predicts subsequent returns. Strategies without a
    statistically significant causal link can be suppressed.

    Parameters
    ----------
    min_observations : int
        Minimum data points required before running causality test.
    max_lag : int
        Maximum lag order to evaluate during BIC-based lag selection.
    significance : float
        P-value threshold for declaring a causal relationship.
    default_strength : float
        Causal strength assigned to unknown/new strategies (1.0 = permissive).
    logger : StructuredLogger, optional
        Logger instance; a default is created if not provided.
    """

    def __init__(
        self,
        min_observations: int = 200,
        max_lag: int = 10,
        significance: float = 0.05,
        default_strength: float = 1.0,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self.min_observations = min_observations
        self.max_lag = max_lag
        self.significance = significance
        self.default_strength = default_strength
        self._logger = logger or StructuredLogger("causal_filter")
        self._scores: Dict[str, CausalScore] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_scores(self, strategy_trade_history: Dict[str, List[Tuple[float, float]]]) -> Dict[str, CausalScore]:
        """Run Granger causality for each strategy and update internal scores.

        Parameters
        ----------
        strategy_trade_history : dict
            ``{strategy_name: [(signal_value, subsequent_return), ...]}``.

        Returns
        -------
        dict[str, CausalScore]
            Updated causal scores for every strategy that had enough data.
        """
        updated: Dict[str, CausalScore] = {}
        for strategy, pairs in strategy_trade_history.items():
            if len(pairs) < self.min_observations:
                self._logger.debug(
                    "insufficient_observations",
                    strategy=strategy,
                    n=len(pairs),
                    required=self.min_observations,
                )
                continue

            signals = np.array([p[0] for p in pairs], dtype=np.float64)
            returns = np.array([p[1] for p in pairs], dtype=np.float64)

            score = self._compute_granger(strategy, signals, returns)
            if score is not None:
                self._scores[strategy] = score
                updated[strategy] = score
                self._logger.info(
                    "causal_score_updated",
                    strategy=strategy,
                    causal_strength=round(score.causal_strength, 4),
                    f_stat=round(score.granger_f_stat, 4),
                    p_value=round(score.granger_p_value, 6),
                    optimal_lag=score.optimal_lag,
                    n_observations=score.n_observations,
                )

        return updated

    def get_causal_strength(self, strategy_name: str) -> float:
        """Return causal strength for *strategy_name*, or ``default_strength`` if unknown."""
        if strategy_name in self._scores:
            return self._scores[strategy_name].causal_strength
        return self.default_strength

    def should_allow(self, strategy_name: str, threshold: float = 0.3) -> bool:
        """Return ``True`` if the strategy's causal strength meets *threshold*."""
        return self.get_causal_strength(strategy_name) >= threshold

    def get_all_scores(self) -> Dict[str, CausalScore]:
        """Return a copy of all computed causal scores."""
        return dict(self._scores)

    # ------------------------------------------------------------------
    # Internals — Granger causality from scratch
    # ------------------------------------------------------------------

    def _compute_granger(self, strategy: str, x: np.ndarray, y: np.ndarray) -> Optional[CausalScore]:
        """Run Granger causality of *x* (signal) on *y* (return).

        1. Select optimal lag via BIC on the unrestricted model.
        2. Compute F-test at optimal lag.
        3. Map p-value to causal strength.
        """
        # Degenerate data guards
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            self._logger.warning("constant_series", strategy=strategy)
            return CausalScore(
                strategy=strategy,
                causal_strength=0.0,
                granger_f_stat=0.0,
                granger_p_value=1.0,
                optimal_lag=1,
                n_observations=len(y),
                last_updated=datetime.utcnow(),
            )

        n_total = len(y)
        effective_max_lag = min(self.max_lag, (n_total // 3) - 1)
        if effective_max_lag < 1:
            self._logger.warning("series_too_short_for_lag", strategy=strategy, n=n_total)
            return None

        # --- Step 1: BIC-based lag selection on the unrestricted model ---
        best_lag = 1
        best_bic = np.inf

        for k in range(1, effective_max_lag + 1):
            bic = self._unrestricted_bic(x, y, k)
            if bic is not None and bic < best_bic:
                best_bic = bic
                best_lag = k

        # --- Step 2: F-test at the optimal lag ---
        f_stat, p_value, n_used = self._granger_f_test(x, y, best_lag)
        if f_stat is None:
            return None

        causal_strength = max(0.0, min(1.0, 1.0 - p_value))

        return CausalScore(
            strategy=strategy,
            causal_strength=causal_strength,
            granger_f_stat=f_stat,
            granger_p_value=p_value,
            optimal_lag=best_lag,
            n_observations=n_used,
            last_updated=datetime.utcnow(),
        )

    # ---- helpers ----

    def _build_matrices(
        self, x: np.ndarray, y: np.ndarray, k: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Build the dependent vector and design matrices for lag *k*.

        Returns ``(y_dep, Z_restricted, Z_unrestricted)`` or ``None`` if
        degrees-of-freedom are insufficient.
        """
        n = len(y)
        if n <= k:
            return None

        y_dep = y[k:]  # shape (n-k,)
        t = len(y_dep)

        # Restricted: intercept + k lags of y
        cols_r = [np.ones(t)]
        for j in range(1, k + 1):
            cols_r.append(y[k - j : n - j])
        Z_r = np.column_stack(cols_r)

        # Unrestricted: restricted + k lags of x
        cols_u = list(cols_r)
        for j in range(1, k + 1):
            cols_u.append(x[k - j : n - j])
        Z_u = np.column_stack(cols_u)

        # Check sufficient degrees of freedom: n_obs > n_params
        if t <= Z_u.shape[1]:
            return None

        return y_dep, Z_r, Z_u

    @staticmethod
    def _ols_rss(Z: np.ndarray, y: np.ndarray) -> Optional[float]:
        """OLS residual sum of squares via ``np.linalg.lstsq``."""
        with np.errstate(all="ignore"):
            result = np.linalg.lstsq(Z, y, rcond=None)
            coeffs = result[0]
            if not np.all(np.isfinite(coeffs)):
                return None
            residuals = y - Z @ coeffs
            if not np.all(np.isfinite(residuals)):
                return None
        rss = float(np.sum(residuals**2))
        if rss < 1e-30:
            return 1e-30  # prevent log(0)
        return rss

    def _unrestricted_bic(self, x: np.ndarray, y: np.ndarray, k: int) -> Optional[float]:
        """BIC of the unrestricted model at lag *k*."""
        built = self._build_matrices(x, y, k)
        if built is None:
            return None
        y_dep, _Z_r, Z_u = built
        rss = self._ols_rss(Z_u, y_dep)
        if rss is None:
            return None
        n = len(y_dep)
        p = Z_u.shape[1]
        bic = n * np.log(rss / n) + p * np.log(n)
        return float(bic)

    def _granger_f_test(self, x: np.ndarray, y: np.ndarray, k: int) -> Tuple[Optional[float], float, int]:
        """Compute the Granger F-statistic and p-value at lag *k*.

        Returns ``(F, p_value, n_observations)``.  On failure returns
        ``(None, 1.0, 0)``.
        """
        built = self._build_matrices(x, y, k)
        if built is None:
            return None, 1.0, 0

        y_dep, Z_r, Z_u = built
        n = len(y_dep)

        rss_r = self._ols_rss(Z_r, y_dep)
        rss_u = self._ols_rss(Z_u, y_dep)
        if rss_r is None or rss_u is None:
            return None, 1.0, 0

        df_num = k
        df_den = n - 2 * k - 1
        if df_den <= 0:
            return None, 1.0, 0

        if rss_u <= 0:
            return None, 1.0, 0

        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
        f_stat = max(f_stat, 0.0)

        p_value = float(f_dist.sf(f_stat, df_num, df_den))

        return f_stat, p_value, n
