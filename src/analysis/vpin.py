"""VPIN (Volume-Synchronized Probability of Informed Trading) calculator.

Implements the Easley, Lopez de Prado, O'Hara (2012) VPIN metric using
Bulk Volume Classification (BVC) for trade direction inference.

VPIN measures order flow toxicity — values near 1.0 indicate heavy
informed trading (toxic flow), values near 0.0 indicate balanced flow.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

from src.core.domain import Bar
from src.core.logger import StructuredLogger


@dataclass
class VPINResult:
    """Result of a VPIN calculation after bucket completion."""

    vpin: float  # VPIN value in [0, 1]
    buy_volume: float  # estimated buy volume in current bucket
    sell_volume: float  # estimated sell volume in current bucket
    bucket_count: int  # number of completed buckets in window
    is_toxic: bool  # True if vpin > toxicity_threshold


class VPINCalculator:
    """Computes VPIN using Bulk Volume Classification on streaming bars.

    Feed bars via update(). Returns a VPINResult once at least one volume
    bucket has been completed. Maintains a rolling window of n_buckets
    order imbalances to compute the running VPIN metric.
    """

    def __init__(
        self,
        n_buckets: int = 50,
        bucket_volume: Optional[float] = None,
        daily_volume_estimate: float = 1_000_000,
        toxicity_threshold: float = 0.7,
        sigma_window: int = 20,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self.n_buckets = n_buckets
        self.toxicity_threshold = toxicity_threshold
        self.sigma_window = sigma_window
        self._logger = logger

        # Bucket volume: explicit or derived from daily estimate
        if bucket_volume is not None:
            self.bucket_volume = bucket_volume
        else:
            self.bucket_volume = daily_volume_estimate / n_buckets

        # Rolling window of (close - open) values for sigma estimation
        self._price_changes: deque[float] = deque(maxlen=sigma_window)

        # Current bucket accumulators
        self._bucket_buy: float = 0.0
        self._bucket_sell: float = 0.0
        self._bucket_total: float = 0.0

        # Completed bucket imbalances: |buy - sell| for each bucket
        self._imbalances: deque[float] = deque(maxlen=n_buckets)

    def update(self, bar: Bar) -> Optional[VPINResult]:
        """Feed one bar and return a VPINResult if any buckets are complete.

        Skips zero-volume bars. A single bar may fill multiple buckets if
        its volume exceeds the remaining capacity — volume is split
        proportionally across bucket boundaries.
        """
        if bar.volume <= 0:
            return None

        # --- BVC: estimate buy/sell fraction ---
        price_change = bar.close - bar.open
        self._price_changes.append(price_change)

        sigma = float(np.std(self._price_changes)) if len(self._price_changes) >= 2 else abs(price_change)
        z = price_change / (sigma + 1e-10)
        buy_fraction = float(norm.cdf(z))

        bar_buy = bar.volume * buy_fraction
        bar_sell = bar.volume * (1.0 - buy_fraction)

        # --- Volume bucketing with spillover ---
        remaining_buy = bar_buy
        remaining_sell = bar_sell
        remaining_vol = bar.volume

        while remaining_vol > 0:
            space_left = self.bucket_volume - self._bucket_total

            if remaining_vol < space_left:
                # Bar fits entirely in current bucket
                self._bucket_buy += remaining_buy
                self._bucket_sell += remaining_sell
                self._bucket_total += remaining_vol
                remaining_vol = 0.0
            else:
                # Fill current bucket, spill remainder into next
                fill_fraction = space_left / remaining_vol if remaining_vol > 0 else 0.0
                self._bucket_buy += remaining_buy * fill_fraction
                self._bucket_sell += remaining_sell * fill_fraction
                self._bucket_total += space_left

                # Record completed bucket imbalance
                imbalance = abs(self._bucket_buy - self._bucket_sell)
                self._imbalances.append(imbalance)

                # Reset bucket
                self._bucket_buy = 0.0
                self._bucket_sell = 0.0
                self._bucket_total = 0.0

                # Reduce remaining
                remaining_buy -= remaining_buy * fill_fraction
                remaining_sell -= remaining_sell * fill_fraction
                remaining_vol -= space_left

        # --- Compute VPIN if we have completed buckets ---
        if len(self._imbalances) == 0:
            return None

        vpin_value = float(np.mean(list(self._imbalances))) / self.bucket_volume
        # Clamp to [0, 1]
        vpin_value = max(0.0, min(1.0, vpin_value))

        is_toxic = vpin_value > self.toxicity_threshold

        if is_toxic and self._logger:
            self._logger.warning(
                "vpin_toxic_flow_detected",
                vpin=round(vpin_value, 4),
                threshold=self.toxicity_threshold,
                bucket_count=len(self._imbalances),
                symbol=bar.symbol,
            )

        return VPINResult(
            vpin=vpin_value,
            buy_volume=self._bucket_buy,
            sell_volume=self._bucket_sell,
            bucket_count=len(self._imbalances),
            is_toxic=is_toxic,
        )

    def reset(self) -> None:
        """Clear all internal state."""
        self._price_changes.clear()
        self._bucket_buy = 0.0
        self._bucket_sell = 0.0
        self._bucket_total = 0.0
        self._imbalances.clear()

    def set_daily_volume(self, volume: float) -> None:
        """Dynamically update bucket size based on new daily volume estimate.

        Call at the start of each trading day to recalibrate bucket boundaries.
        """
        self.bucket_volume = volume / self.n_buckets
