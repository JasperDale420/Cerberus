"""IV Surface Dynamics analyzer for options implied volatility anomaly detection.

Detects skew steepening, term structure inversion, and risk-neutral density
bimodality from options chain data. Uses Breeden-Litzenberger (1978) for
risk-neutral density extraction via numerical second derivative of call prices.

Data source: UnusualWhales IV term structure and full chain greeks.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.core.logger import StructuredLogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IVSurfaceConfig:
    """Configuration for IV surface analysis."""

    enabled: bool = False
    skew_lookback: int = 20
    skew_sigma_threshold: float = 2.0
    min_strikes_for_rnd: int = 10
    peak_prominence: float = 0.1


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IVSurfaceResult:
    """Composite result from IV surface analysis.

    Attributes:
        skew_zscore: Z-score of current 25-delta risk reversal vs history.
        term_slope: ATM IV slope (far minus near). Positive = normal contango.
        is_inverted: True when near-term ATM IV exceeds far-term.
        is_bimodal: True when risk-neutral density has two distinct peaks.
        rnd_peaks: List of (strike, density) tuples for detected peaks.
        signal_strength: Composite signal from -1 (bearish) to +1 (bullish).
    """

    skew_zscore: float = 0.0
    term_slope: float = 0.0
    is_inverted: bool = False
    is_bimodal: bool = False
    rnd_peaks: list[tuple[float, float]] = field(default_factory=list)
    signal_strength: float = 0.0


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class IVSurfaceAnalyzer:
    """Analyzes options IV surface for anomalies that signal directional risk.

    Combines three orthogonal views of the options surface:
    1. Skew (25-delta risk reversal) -- crash fear gauge
    2. Term structure slope -- event risk detection
    3. Risk-neutral density -- binary outcome pricing

    Args:
        config: IVSurfaceConfig with tuning parameters.
        logger: Optional structured logger instance.
    """

    def __init__(
        self,
        config: Optional[IVSurfaceConfig] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self._config = config or IVSurfaceConfig()
        self._logger = logger
        self._skew_history: deque[float] = deque(maxlen=self._config.skew_lookback)

    # ------------------------------------------------------------------
    # Skew
    # ------------------------------------------------------------------

    def compute_skew(self, chain: list[dict], target_delta: float = 0.25) -> float:
        """Compute 25-delta risk reversal (put IV minus call IV).

        Finds the option closest to +target_delta for calls and -target_delta
        for puts, then returns put_iv - call_iv.  A larger positive value
        indicates heavier put skew (crash protection demand).

        Args:
            chain: List of option rows, each with keys ``delta`` and ``iv``.
                   Calls have positive delta, puts have negative delta.
            target_delta: Absolute delta target (default 0.25).

        Returns:
            Risk reversal value.  Returns 0.0 if suitable strikes not found.
        """
        if not chain:
            return 0.0

        best_call_iv: Optional[float] = None
        best_call_dist = float("inf")
        best_put_iv: Optional[float] = None
        best_put_dist = float("inf")

        for row in chain:
            delta = float(row.get("delta", 0))
            iv = float(row.get("iv", 0))
            if iv <= 0:
                continue

            # Calls: delta > 0, find closest to +target_delta
            if delta > 0:
                dist = abs(delta - target_delta)
                if dist < best_call_dist:
                    best_call_dist = dist
                    best_call_iv = iv

            # Puts: delta < 0, find closest to -target_delta
            if delta < 0:
                dist = abs(delta + target_delta)
                if dist < best_put_dist:
                    best_put_dist = dist
                    best_put_iv = iv

        if best_put_iv is None or best_call_iv is None:
            return 0.0

        skew = best_put_iv - best_call_iv
        self._skew_history.append(skew)
        return skew

    def skew_zscore(self) -> float:
        """Return z-score of the latest skew relative to rolling history.

        Requires at least 3 observations.  Returns 0.0 if insufficient data.
        """
        if len(self._skew_history) < 3:
            return 0.0

        arr = np.array(self._skew_history)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std < 1e-12:
            return 0.0

        return (self._skew_history[-1] - mean) / std

    def is_skew_steepening(self) -> bool:
        """True if current skew z-score exceeds the configured sigma threshold."""
        return self.skew_zscore() > self._config.skew_sigma_threshold

    # ------------------------------------------------------------------
    # Term Structure
    # ------------------------------------------------------------------

    def compute_term_structure(
        self,
        term_data: list[dict],
    ) -> tuple[float, bool]:
        """Evaluate ATM IV term structure slope.

        Args:
            term_data: List of dicts with keys ``dte`` (days to expiry) and
                ``iv`` (ATM implied volatility).  Must contain at least two
                expirations sorted by dte ascending.

        Returns:
            (term_slope, is_inverted) where term_slope = far_iv - near_iv.
            Positive slope is normal (contango); negative is inverted.
        """
        if not term_data or len(term_data) < 2:
            return 0.0, False

        # Sort by dte to ensure near/far ordering
        sorted_data = sorted(term_data, key=lambda d: float(d.get("dte", 0)))

        near_iv = float(sorted_data[0].get("iv", 0))
        far_iv = float(sorted_data[-1].get("iv", 0))

        if near_iv <= 0 or far_iv <= 0:
            return 0.0, False

        slope = far_iv - near_iv
        inverted = slope < 0

        if self._logger:
            self._logger.debug(
                "Term structure computed",
                near_iv=near_iv,
                far_iv=far_iv,
                slope=slope,
                inverted=inverted,
            )

        return slope, inverted

    # ------------------------------------------------------------------
    # Risk-Neutral Density (Breeden-Litzenberger)
    # ------------------------------------------------------------------

    def compute_rnd(
        self,
        strikes: list[float],
        call_prices: list[float],
        rate: float = 0.05,
        tte: float = 30 / 365,
    ) -> tuple[list[float], list[float]]:
        """Extract risk-neutral density via Breeden-Litzenberger.

        f(K) = e^{r*T} * d2C/dK2

        Uses central finite differences for the second derivative.

        Args:
            strikes: Monotonically increasing strike prices.
            call_prices: Corresponding call option mid-prices.
            rate: Risk-free rate (annualized).
            tte: Time to expiry in years.

        Returns:
            (density_strikes, density_values) — strikes[1:-1] and their
            corresponding density values.  Empty lists if fewer than
            ``min_strikes_for_rnd`` strikes provided.
        """
        n = len(strikes)
        if n < self._config.min_strikes_for_rnd or n != len(call_prices):
            return [], []

        discount = math.exp(rate * tte)
        density_strikes: list[float] = []
        density_values: list[float] = []

        for i in range(1, n - 1):
            dk_left = strikes[i] - strikes[i - 1]
            dk_right = strikes[i + 1] - strikes[i]

            if dk_left <= 0 or dk_right <= 0:
                continue

            # Central difference second derivative
            dk_avg = 0.5 * (dk_left + dk_right)
            d2c = (call_prices[i + 1] - 2 * call_prices[i] + call_prices[i - 1]) / (dk_avg * dk_avg)

            density = discount * d2c
            # Density must be non-negative; negative values indicate arbitrage / noise
            density = max(density, 0.0)

            density_strikes.append(strikes[i])
            density_values.append(density)

        return density_strikes, density_values

    def detect_bimodality(
        self,
        density_values: list[float],
    ) -> tuple[bool, list[int]]:
        """Detect bimodality in the risk-neutral density via simple peak finding.

        A peak is a local maximum whose value exceeds ``peak_prominence`` times
        the maximum density.

        Args:
            density_values: Density array from ``compute_rnd``.

        Returns:
            (is_bimodal, peak_indices) — True if >= 2 peaks found.
        """
        if len(density_values) < 3:
            return False, []

        max_density = max(density_values)
        if max_density < 1e-15:
            return False, []

        threshold = self._config.peak_prominence * max_density
        peaks: list[int] = []

        for i in range(1, len(density_values) - 1):
            if (
                density_values[i] > density_values[i - 1]
                and density_values[i] > density_values[i + 1]
                and density_values[i] > threshold
            ):
                peaks.append(i)

        return len(peaks) >= 2, peaks

    # ------------------------------------------------------------------
    # Composite Signal
    # ------------------------------------------------------------------

    def analyze(
        self,
        chain: list[dict],
        term_data: list[dict],
        strikes: list[float],
        call_prices: list[float],
        rate: float = 0.05,
        tte: float = 30 / 365,
    ) -> IVSurfaceResult:
        """Run full IV surface analysis and return composite result.

        Combines skew, term structure, and RND into a single
        ``IVSurfaceResult`` with a directional signal.

        Args:
            chain: Options chain rows with ``delta`` and ``iv`` per option.
            term_data: ATM IV term structure rows with ``dte`` and ``iv``.
            strikes: Strike prices for RND computation.
            call_prices: Call mid-prices aligned with ``strikes``.
            rate: Risk-free rate (annualized).
            tte: Time to expiry in years.

        Returns:
            IVSurfaceResult with all sub-signals and composite strength.
        """
        # Skew
        self.compute_skew(chain)
        szs = self.skew_zscore()

        # Term structure
        term_slope, inverted = self.compute_term_structure(term_data)

        # RND
        rnd_strikes, rnd_densities = self.compute_rnd(strikes, call_prices, rate, tte)
        is_bimodal, peak_indices = self.detect_bimodality(rnd_densities)

        rnd_peaks: list[tuple[float, float]] = []
        for idx in peak_indices:
            if idx < len(rnd_strikes):
                rnd_peaks.append((rnd_strikes[idx], rnd_densities[idx]))

        # Composite signal: negative = bearish, positive = bullish
        # Steep skew (high z-score) is bearish
        skew_component = -np.clip(szs / 3.0, -1.0, 1.0)
        # Inversion is bearish (near-term fear)
        inversion_component = -0.3 if inverted else 0.0
        # Bimodality is uncertainty / slightly bearish
        bimodal_component = -0.2 if is_bimodal else 0.0

        raw_signal = float(skew_component) + inversion_component + bimodal_component
        signal_strength = float(np.clip(raw_signal, -1.0, 1.0))

        result = IVSurfaceResult(
            skew_zscore=szs,
            term_slope=term_slope,
            is_inverted=inverted,
            is_bimodal=is_bimodal,
            rnd_peaks=rnd_peaks,
            signal_strength=signal_strength,
        )

        if self._logger:
            self._logger.debug(
                "IV surface analysis complete",
                skew_zscore=szs,
                term_slope=term_slope,
                inverted=inverted,
                is_bimodal=is_bimodal,
                signal_strength=signal_strength,
            )

        return result
