"""Pre-compute pair-invariant Kalman hedge ratios and spreads for WFO.

The Kalman filter (filterpy 2×2 matrix ops per bar) is the most expensive
per-bar computation in PairTradingV2Strategy and depends ONLY on bar data,
not on any tunable strategy parameter.  Precomputing once per WFO window
and injecting into the strategy eliminates redundant Kalman runs across trials.

Usage in optimization harness::

    cache = PairFactorCache(pairs_config, bars_df)
    cache.precompute()
    config["strategies"]["pair_trading_v2"]["_factor_cache"] = cache

    # Each trial reads from cache instead of running Kalman online
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.quant.filters import KalmanHedgeRatio


@dataclass(frozen=True, slots=True)
class PairFactorsAtBar:
    """Precomputed Kalman factors for a single pair at a single bar index."""

    hedge_ratio: float
    intercept: float
    spread: float
    hedge_ratio_uncertainty: float


class PairFactorCache:
    """Precompute Kalman hedge ratios and spreads for all pairs across all bars.

    These are parameter-independent (depend only on price series) and identical
    across all Optuna trials within a WFO window.

    The strategy reads factors via ``get(pair_key, bar_idx)`` instead of
    maintaining its own KalmanHedgeRatio objects.
    """

    def __init__(
        self,
        pairs_config: list[dict[str, str]],
        bars_df: pd.DataFrame,
    ) -> None:
        self._pairs = pairs_config
        self._bars_df = bars_df

        # Keyed by canonical pair key → numpy arrays of length N_bars
        self._hedge_ratios: dict[str, np.ndarray] = {}
        self._intercepts: dict[str, np.ndarray] = {}
        self._spreads: dict[str, np.ndarray] = {}
        self._uncertainties: dict[str, np.ndarray] = {}

        # Mapping from (pair_key, timestamp) → bar_idx for O(1) lookup
        self._timestamp_maps: dict[str, dict[Any, int]] = {}
        self._precomputed = False

    def _pair_key(self, leg_a: str, leg_b: str) -> str:
        a, b = sorted((leg_a, leg_b))
        return f"{a}:{b}"

    @property
    def is_precomputed(self) -> bool:
        return self._precomputed

    def precompute(self) -> int:
        """Run Kalman forward pass for all pairs. Returns count of pairs computed."""
        computed = 0

        for pair_cfg in self._pairs:
            leg_a, leg_b = pair_cfg["leg_a"], pair_cfg["leg_b"]
            key = self._pair_key(leg_a, leg_b)

            # Get sorted price series for both legs
            prices_a_df = self._bars_df[self._bars_df["symbol"] == leg_a].sort_values("timestamp")
            prices_b_df = self._bars_df[self._bars_df["symbol"] == leg_b].sort_values("timestamp")

            if prices_a_df.empty or prices_b_df.empty:
                continue

            # Merge on timestamp to get synchronized bars
            merged = pd.merge(
                prices_a_df[["timestamp", "close"]].rename(columns={"close": "price_a"}),
                prices_b_df[["timestamp", "close"]].rename(columns={"close": "price_b"}),
                on="timestamp",
                how="inner",
            ).sort_values("timestamp")

            if merged.empty:
                continue

            n = len(merged)
            hr_arr = np.zeros(n, dtype=np.float64)
            ic_arr = np.zeros(n, dtype=np.float64)
            sp_arr = np.zeros(n, dtype=np.float64)
            uc_arr = np.zeros(n, dtype=np.float64)

            # Forward Kalman pass
            kalman = KalmanHedgeRatio()
            price_a_vals = merged["price_a"].values
            price_b_vals = merged["price_b"].values

            for i in range(n):
                # Kalman models y = intercept + slope * x
                # Strategy calls kalman.update(price_b, price_a)
                spread = kalman.update(float(price_b_vals[i]), float(price_a_vals[i]))
                hr_arr[i] = kalman.hedge_ratio
                ic_arr[i] = kalman.intercept
                sp_arr[i] = spread
                uc_arr[i] = kalman.hedge_ratio_uncertainty

            self._hedge_ratios[key] = hr_arr
            self._intercepts[key] = ic_arr
            self._spreads[key] = sp_arr
            self._uncertainties[key] = uc_arr

            # Build timestamp → index map
            ts_map: dict[Any, int] = {}
            timestamps = merged["timestamp"].values
            for i in range(n):
                ts_map[timestamps[i]] = i
            self._timestamp_maps[key] = ts_map

            computed += 1

        self._precomputed = True
        return computed

    @property
    def pair_keys(self) -> list[str]:
        return list(self._hedge_ratios.keys())

    def get_bar_idx(self, pair_key: str, timestamp: Any) -> int | None:
        """Look up the bar index for a given pair and timestamp."""
        ts_map = self._timestamp_maps.get(pair_key)
        if ts_map is None:
            return None
        return ts_map.get(timestamp)

    def get(self, pair_key: str, bar_idx: int) -> PairFactorsAtBar | None:
        """O(1) lookup of precomputed Kalman factors at a specific bar index."""
        hr_arr = self._hedge_ratios.get(pair_key)
        if hr_arr is None or bar_idx < 0 or bar_idx >= len(hr_arr):
            return None
        return PairFactorsAtBar(
            hedge_ratio=float(hr_arr[bar_idx]),
            intercept=float(self._intercepts[pair_key][bar_idx]),
            spread=float(self._spreads[pair_key][bar_idx]),
            hedge_ratio_uncertainty=float(self._uncertainties[pair_key][bar_idx]),
        )

    def get_spread_array(self, pair_key: str) -> np.ndarray | None:
        """Return the full spread array for z-score computation."""
        return self._spreads.get(pair_key)

    def bar_count(self, pair_key: str) -> int:
        """Return the number of precomputed bars for a pair."""
        arr = self._hedge_ratios.get(pair_key)
        return len(arr) if arr is not None else 0
