"""Permutation Entropy and Lempel-Ziv Complexity analyzer.

Provides real-time complexity regime detection for market return series.
Permutation entropy captures ordinal pattern randomness; Lempel-Ziv complexity
measures compressibility of the sign sequence. Together they classify the
current market micro-regime as STRUCTURED, NORMAL, or RANDOM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import factorial, log
from typing import Optional

import numpy as np

from src.core.domain import ComplexityRegime
from src.core.logger import StructuredLogger


@dataclass(frozen=True, slots=True)
class EntropyResult:
    """Result of a single entropy/complexity update."""

    pe_normalized: float  # permutation entropy normalized to [0, 1]
    lzc_normalized: float  # Lempel-Ziv complexity normalized to [0, 1]
    complexity_zscore: float  # z-score of PE vs trailing average
    regime: ComplexityRegime  # classified regime


class EntropyAnalyzer:
    """Streaming permutation entropy and Lempel-Ziv complexity analyzer.

    Feed one return value at a time via ``update()``. Once the internal window
    is full the analyzer emits an ``EntropyResult`` on every subsequent call.

    Parameters
    ----------
    pe_order:
        Embedding dimension for ordinal patterns (default 5).
    pe_delay:
        Time delay between elements of each ordinal pattern (default 1).
    window:
        Number of return values kept in the rolling buffer (default 60).
    zscore_window:
        Trailing window of PE values used for z-score computation (default 1200).
    logger:
        Optional ``StructuredLogger`` instance.
    """

    def __init__(
        self,
        pe_order: int = 5,
        pe_delay: int = 1,
        window: int = 60,
        zscore_window: int = 1200,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self.pe_order = pe_order
        self.pe_delay = pe_delay
        self.window = window
        self.zscore_window = zscore_window
        self._logger = logger

        # Rolling buffer of raw return values
        self._buffer: deque[float] = deque(maxlen=window)

        # Trailing PE values for z-score
        self._pe_history: deque[float] = deque(maxlen=zscore_window)

        # Pre-compute max PE for normalization: ln(order!)
        self._max_pe: float = log(factorial(pe_order))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, value: float) -> Optional[EntropyResult]:
        """Feed a single return value and optionally receive an entropy result.

        Returns ``None`` until the buffer contains at least ``window`` values.
        """
        self._buffer.append(value)

        if len(self._buffer) < self.window:
            return None

        series = np.array(self._buffer, dtype=np.float64)

        pe_norm = self._compute_pe(series)
        lzc_norm = self._compute_lzc(series)
        zscore = self._compute_zscore(pe_norm)
        regime = self._classify_regime(zscore)

        result = EntropyResult(
            pe_normalized=pe_norm,
            lzc_normalized=lzc_norm,
            complexity_zscore=zscore,
            regime=regime,
        )

        if self._logger:
            self._logger.debug(
                "entropy_update",
                pe_normalized=round(pe_norm, 4),
                lzc_normalized=round(lzc_norm, 4),
                complexity_zscore=round(zscore, 4),
                regime=regime.value,
            )

        return result

    def reset(self) -> None:
        """Clear all internal state."""
        self._buffer.clear()
        self._pe_history.clear()

    # ------------------------------------------------------------------
    # Permutation Entropy
    # ------------------------------------------------------------------

    def _compute_pe(self, series: np.ndarray) -> float:
        """Compute normalised permutation entropy of *series*."""
        order = self.pe_order
        delay = self.pe_delay
        n = len(series)

        # Minimum length required to form at least one pattern
        min_length = (order - 1) * delay + 1
        if n < min_length:
            return 1.0  # degenerate — treat as maximum entropy

        pattern_counts: dict[tuple[int, ...], int] = {}
        total = 0

        for i in range(n - (order - 1) * delay):
            # Build the subsequence with the given delay
            subseq = tuple(series[i + j * delay] for j in range(order))
            # Ordinal pattern = argsort rank
            pattern = tuple(int(x) for x in np.argsort(subseq))
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            total += 1

        if total == 0:
            return 1.0

        # Shannon entropy of the pattern distribution
        entropy = 0.0
        for count in pattern_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * log(p)

        # Normalise to [0, 1]
        if self._max_pe == 0:
            return 0.0
        return entropy / self._max_pe

    # ------------------------------------------------------------------
    # Lempel-Ziv Complexity
    # ------------------------------------------------------------------

    def _compute_lzc(self, series: np.ndarray) -> float:
        """Compute normalised Lempel-Ziv complexity of the sign sequence."""
        n = len(series)
        if n == 0:
            return 0.0

        # Convert to binary string: 1 if return > 0, else 0
        binary = "".join("1" if v > 0 else "0" for v in series)

        complexity = self._lempel_ziv_count(binary)

        # Normalise: upper bound for random binary ≈ n / ln(n)
        if n <= 1:
            return 0.0
        normalizer = n / log(n)
        return min(complexity / normalizer, 1.0)

    @staticmethod
    def _lempel_ziv_count(s: str) -> int:
        """Count the number of distinct words in the Lempel-Ziv decomposition."""
        n = len(s)
        if n == 0:
            return 0

        complexity = 1
        prefix_len = 1  # length of the current prefix S
        component_len = 1  # length of the current component Q

        while prefix_len + component_len <= n:
            # Check if current component Q is a substring of S·Q_π
            # (S·Q_π is the string formed by S concatenated with Q minus its last char)
            sq_pi = s[: prefix_len + component_len - 1]
            q = s[prefix_len : prefix_len + component_len]

            if q in sq_pi:
                component_len += 1
            else:
                complexity += 1
                prefix_len += component_len
                component_len = 1

        # Account for final component if not already counted
        if component_len > 1:
            complexity += 1

        return complexity

    # ------------------------------------------------------------------
    # Z-score & regime classification
    # ------------------------------------------------------------------

    def _compute_zscore(self, pe_norm: float) -> float:
        """Compute z-score of the current PE against the trailing history."""
        self._pe_history.append(pe_norm)

        if len(self._pe_history) < 2:
            return 0.0

        arr = np.array(self._pe_history, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))

        if std < 1e-12:
            return 0.0
        return (pe_norm - mean) / std

    @staticmethod
    def _classify_regime(zscore: float) -> ComplexityRegime:
        """Map a PE z-score to a ``ComplexityRegime``."""
        if zscore < -2.0:
            return ComplexityRegime.STRUCTURED
        if zscore > 2.0:
            return ComplexityRegime.RANDOM
        return ComplexityRegime.NORMAL
