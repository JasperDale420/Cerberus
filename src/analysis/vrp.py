from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VRPResult:
    vrp_raw: float  # IV_proxy - RV
    vrp_zscore: float  # z-score normalized VRP
    iv_proxy: float  # (VXX/10)²
    realized_vol: float  # from EWMA


class VRPCalculator:
    """Variance Risk Premium calculator for risk axis classification.

    VRP = IV² - RV² where IV is proxied from VXX price and RV from EWMA realized vol.
    Academic basis: Bollerslev, Tauchen & Zhou (2009).
    """

    # Annualization factor for 1-minute bar returns: sqrt(252 trading days * 390 min/day)
    _ANNUALIZATION_FACTOR = np.sqrt(252 * 390)

    def __init__(
        self,
        window: int = 60,
        zscore_window: int = 1200,
        logger=None,
    ):
        self._window = window
        self._zscore_window = zscore_window
        self._logger = logger
        self._vrp_history: deque[float] = deque(maxlen=zscore_window)

    def update(self, vxx_price: float, realized_vol: float) -> Optional[VRPResult]:
        """Compute VRP from VXX price and realized volatility.

        Args:
            vxx_price: Current VXX price
            realized_vol: Current realized volatility (per-bar EWMA from regime.py)

        Returns:
            VRPResult or None if insufficient data for z-score
        """
        if vxx_price <= 0:
            return None

        # IV proxy: VXX/10 approximates VIX/100, giving annualized implied vol as a decimal.
        # Squaring gives annualized implied variance.
        iv_proxy = (vxx_price / 10.0) ** 2
        # Annualize per-bar realized vol before squaring to match IV proxy scale
        rv = (realized_vol * self._ANNUALIZATION_FACTOR) ** 2
        vrp_raw = iv_proxy - rv

        self._vrp_history.append(vrp_raw)

        if len(self._vrp_history) < self._window:
            return None

        # Z-score normalization
        vrp_array = np.array(self._vrp_history)
        vrp_mean = float(np.mean(vrp_array))
        vrp_std = float(np.std(vrp_array, ddof=1))

        if vrp_std < 1e-12:
            vrp_zscore = 0.0
        else:
            vrp_zscore = (vrp_raw - vrp_mean) / vrp_std

        return VRPResult(
            vrp_raw=vrp_raw,
            vrp_zscore=vrp_zscore,
            iv_proxy=iv_proxy,
            realized_vol=realized_vol,
        )
