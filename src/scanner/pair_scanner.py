import signal
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from src.config.models import PairTradingConfig
from src.core.logger import StructuredLogger


class _TimeoutError(Exception):
    """Raised when cointegration test times out."""

    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Cointegration test timed out")


class PairScanner:
    """
    Scans a universe of symbols for cointegrated pairs.
    Uses Engle-Granger two-step method.
    """

    # Per-pair timeout in seconds to prevent hangs on pathological data
    COINT_TIMEOUT_SECONDS = 5

    def __init__(self, config: PairTradingConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger

    def _is_valid_series(self, series: pd.Series) -> bool:
        """Check if series is suitable for cointegration testing."""
        if series.isna().any():
            return False
        if len(series) < 30:
            return False
        # Skip constant or near-constant series (would cause numerical issues)
        if series.std() < 1e-8:
            return False
        return True

    def find_pairs(self, price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Finds cointegrated pairs in the provided price data.
        price_data: DataFrame with symbols as columns and timestamps as index.
        """
        if not self.config.enabled:
            return []

        # Aggregate 1-minute bars to daily closes when index is time-like.
        # If callers already pass daily data with a non-datetime index, use it as-is.
        if isinstance(price_data.index, pd.DatetimeIndex):
            daily_data = price_data.resample("D").last().dropna(how="all")
        else:
            daily_data = price_data.dropna(how="all")

        if len(daily_data) < 30:
            self.logger.warning(
                "Insufficient daily data for pair discovery",
                rows=len(daily_data),
                min_required=30,
            )
            return []

        # Pre-filter symbols with valid data
        valid_symbols = [col for col in daily_data.columns if self._is_valid_series(daily_data[col])]

        if len(valid_symbols) < 2:
            self.logger.warning(
                "Insufficient valid symbols for pair discovery",
                valid_count=len(valid_symbols),
                total_count=len(daily_data.columns),
            )
            return []

        daily_data = daily_data[valid_symbols]
        symbols = daily_data.columns
        n = len(symbols)
        pairs = []
        skipped_timeout = 0

        # 1. Filter by correlation first (faster than cointegration)
        corr_matrix = daily_data.corr()

        for i in range(n):
            for j in range(i + 1, n):
                s1, s2 = symbols[i], symbols[j]

                corr_raw = corr_matrix.loc[s1, s2]
                if pd.isna(corr_raw):
                    continue
                corr = float(cast(Any, corr_raw))
                if abs(corr) < self.config.min_correlation:
                    continue

                # 2. Cointegration Test with timeout protection
                try:
                    signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(self.COINT_TIMEOUT_SECONDS)
                    try:
                        _, pvalue, _ = coint(daily_data[s1], daily_data[s2])
                    finally:
                        signal.alarm(0)  # Cancel alarm
                except _TimeoutError:
                    skipped_timeout += 1
                    continue
                except Exception as e:
                    self.logger.warning(
                        "Cointegration test failed",
                        symbol_a=s1,
                        symbol_b=s2,
                        error=str(e),
                    )
                    continue

                if pvalue < self.config.max_eg_pvalue:
                    res = self._calculate_pair_stats(daily_data[s1], daily_data[s2])
                    if res:
                        res.update(
                            {
                                "symbol_a": s1,
                                "symbol_b": s2,
                                "correlation": corr,
                                "pvalue": pvalue,
                            }
                        )
                        pairs.append(res)
                        self.logger.info(
                            "Cointegrated pair found",
                            symbol_a=s1,
                            symbol_b=s2,
                            pvalue=pvalue,
                            corr=corr,
                        )

        if skipped_timeout > 0:
            self.logger.warning("Skipped pairs due to timeout", count=skipped_timeout)

        return pairs

    def _calculate_pair_stats(self, y: pd.Series, x: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Calculates hedge ratio and spread statistics using least squares.
        """
        y_arr = y.to_numpy(dtype=float)
        x_arr = x.to_numpy(dtype=float)
        if y_arr.size == 0 or x_arr.size == 0 or y_arr.size != x_arr.size:
            return None

        x_design = np.column_stack([np.ones_like(x_arr), x_arr])
        coeffs, *_ = np.linalg.lstsq(x_design, y_arr, rcond=None)
        alpha = float(coeffs[0])
        hedge_ratio = float(coeffs[1])
        spread = y_arr - (hedge_ratio * x_arr) - alpha

        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread))

        if abs(spread_std) < 1e-9:
            return None

        # Half-life calculation for mean reversion speed
        half_life = self._calculate_half_life(pd.Series(spread))

        if half_life < self.config.min_half_life or half_life > self.config.max_half_life:
            return None

        return {
            "hedge_ratio": float(hedge_ratio),
            "alpha": float(alpha),
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "half_life": half_life,
            "last_spread": float(spread[-1]),
        }

    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculates the half-life of mean reversion using Ornstein-Uhlenbeck process.
        """
        spread_lag = spread.shift(1).dropna().to_numpy(dtype=float)
        spread_diff = (spread - spread.shift(1)).dropna().to_numpy(dtype=float)
        if spread_lag.size == 0 or spread_diff.size == 0:
            return 999.0

        # OLS of diff on lag with constant: diff = a + lambda * lag
        x_design = np.column_stack([np.ones_like(spread_lag), spread_lag])
        coeffs, *_ = np.linalg.lstsq(x_design, spread_diff, rcond=None)
        lambda_val = float(coeffs[1])

        if lambda_val >= 0:
            return 999.0  # Not mean reverting

        return float(-np.log(2) / lambda_val)
