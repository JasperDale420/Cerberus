"""Cross-Asset ETF Flow Propagation analyzer.

Detects divergences between ETF-level options flow and constituent stock flow.
Based on the empirical finding that ETF flow leads constituent returns by 5-15
minutes. A SPY put surge combined with individual stock call buying indicates
sector rotation (money leaving the index, entering individual names).

Data source: UnusualWhales ETF tide and individual stock flow.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.core.logger import StructuredLogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ETFFlowConfig:
    """Configuration for the ETF flow propagation analyzer."""

    enabled: bool = False
    lookback_bars: int = 60  # rolling window size for z-score computation
    min_bars: int = 20  # minimum observations before producing signals
    rotation_threshold: float = 1.5  # ETF put/call ratio threshold for rotation
    divergence_sigma: float = 2.0  # divergence significance threshold
    etf_symbols: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])


# ---------------------------------------------------------------------------
# Default ETF-constituent mapping
# ---------------------------------------------------------------------------

DEFAULT_ETF_CONSTITUENTS: dict[str, list[str]] = {
    "SPY": [],  # broad market -- applies to all stocks
    "QQQ": [],  # tech-heavy -- applies to tech stocks
    "IWM": [],  # small cap
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ETFFlowResult:
    """Composite result from ETF flow analysis for a single stock."""

    etf_symbol: str
    stock_symbol: str
    divergence_score: float
    rotation_detected: bool
    lead_signal: float  # -1 to 1
    lead_confidence: float
    etf_put_call_ratio: float
    stock_put_call_ratio: float


# ---------------------------------------------------------------------------
# Internal per-symbol flow state
# ---------------------------------------------------------------------------


class _FlowState:
    """Rolling window of net premium and put/call ratio for one symbol."""

    __slots__ = ("net_premiums", "put_call_ratios")

    def __init__(self, maxlen: int) -> None:
        self.net_premiums: deque[float] = deque(maxlen=maxlen)
        self.put_call_ratios: deque[float] = deque(maxlen=maxlen)

    def update(self, net_premium: float, put_call_ratio: float) -> None:
        self.net_premiums.append(net_premium)
        self.put_call_ratios.append(put_call_ratio)

    @property
    def count(self) -> int:
        return len(self.net_premiums)


# ---------------------------------------------------------------------------
# ETFFlowAnalyzer
# ---------------------------------------------------------------------------


class ETFFlowAnalyzer:
    """Analyzes cross-asset ETF flow to detect divergences and rotation signals.

    Feed ETF-level and stock-level options flow data via ``update_etf_flow``
    and ``update_stock_flow``. Then call ``analyze`` for a composite result
    or use the individual methods (``compute_divergence``, ``detect_rotation``,
    ``compute_lead_signal``) for granular access.

    Args:
        config: ETFFlowConfig with tuning parameters.
        logger: Optional structured logger instance.
    """

    def __init__(
        self,
        config: ETFFlowConfig | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._config = config or ETFFlowConfig()
        self._logger = logger

        # Per-symbol flow state keyed by uppercase symbol
        self._etf_states: dict[str, _FlowState] = {}
        self._stock_states: dict[str, _FlowState] = {}

        # Time-series of ETF net premium changes for lead-lag detection
        self._etf_flow_changes: dict[str, deque[float]] = {}

    # ------------------------------------------------------------------
    # Flow ingestion
    # ------------------------------------------------------------------

    def update_etf_flow(self, data: dict) -> None:
        """Ingest one ETF flow observation from UnusualWhales.

        Args:
            data: Dict with keys ``symbol``, ``call_premium``, ``put_premium``,
                  ``net_premium``, ``timestamp``.
        """
        symbol = str(data.get("symbol", "")).upper()
        if not symbol:
            return

        call_premium = float(data.get("call_premium", 0.0))
        put_premium = float(data.get("put_premium", 0.0))
        net_premium = float(data.get("net_premium", 0.0))
        put_call_ratio = put_premium / max(call_premium, 1.0)

        state = self._etf_states.get(symbol)
        if state is None:
            state = _FlowState(maxlen=self._config.lookback_bars)
            self._etf_states[symbol] = state

        # Track flow changes for lead-lag
        prev_net = state.net_premiums[-1] if state.count > 0 else 0.0
        change = net_premium - prev_net

        changes_dq = self._etf_flow_changes.get(symbol)
        if changes_dq is None:
            changes_dq = deque(maxlen=self._config.lookback_bars)
            self._etf_flow_changes[symbol] = changes_dq
        changes_dq.append(change)

        state.update(net_premium, put_call_ratio)

        if self._logger:
            self._logger.debug(
                "ETF flow updated",
                symbol=symbol,
                net_premium=net_premium,
                put_call_ratio=round(put_call_ratio, 4),
            )

    def update_stock_flow(self, data: dict) -> None:
        """Ingest one stock flow observation from UnusualWhales.

        Args:
            data: Dict with keys ``symbol``, ``call_premium``, ``put_premium``,
                  ``net_premium``, ``timestamp``.
        """
        symbol = str(data.get("symbol", "")).upper()
        if not symbol:
            return

        call_premium = float(data.get("call_premium", 0.0))
        put_premium = float(data.get("put_premium", 0.0))
        net_premium = float(data.get("net_premium", 0.0))
        put_call_ratio = put_premium / max(call_premium, 1.0)

        state = self._stock_states.get(symbol)
        if state is None:
            state = _FlowState(maxlen=self._config.lookback_bars)
            self._stock_states[symbol] = state

        state.update(net_premium, put_call_ratio)

        if self._logger:
            self._logger.debug(
                "Stock flow updated",
                symbol=symbol,
                net_premium=net_premium,
                put_call_ratio=round(put_call_ratio, 4),
            )

    # ------------------------------------------------------------------
    # Divergence detection
    # ------------------------------------------------------------------

    def compute_divergence(self, stock_symbol: str, etf_symbol: str) -> Optional[float]:
        """Compute flow divergence between a stock and its parent ETF.

        Divergence = stock_net_flow_zscore - etf_net_flow_zscore.

        A positive value means the stock is relatively more bullish than the
        ETF, a negative value means relatively more bearish.

        Returns:
            Divergence score, or None if insufficient data.
        """
        stock_symbol = stock_symbol.upper()
        etf_symbol = etf_symbol.upper()

        stock_state = self._stock_states.get(stock_symbol)
        etf_state = self._etf_states.get(etf_symbol)

        if stock_state is None or etf_state is None:
            return None
        if stock_state.count < self._config.min_bars or etf_state.count < self._config.min_bars:
            return None

        stock_z = self._zscore(stock_state.net_premiums)
        etf_z = self._zscore(etf_state.net_premiums)

        if stock_z is None or etf_z is None:
            return None

        return stock_z - etf_z

    # ------------------------------------------------------------------
    # Sector rotation signal
    # ------------------------------------------------------------------

    def detect_rotation(self, stock_symbol: str, etf_symbol: str = "SPY") -> bool:
        """Detect sector rotation: money leaving the index, entering individual names.

        Conditions:
        - ETF put/call ratio > rotation_threshold (put surge on index)
        - Stock put/call ratio < 0.5 (call surge on individual name)

        Returns:
            True if rotation signal is present.
        """
        etf_symbol = etf_symbol.upper()
        stock_symbol = stock_symbol.upper()

        etf_state = self._etf_states.get(etf_symbol)
        stock_state = self._stock_states.get(stock_symbol)

        if etf_state is None or stock_state is None:
            return False
        if etf_state.count == 0 or stock_state.count == 0:
            return False

        etf_pcr = etf_state.put_call_ratios[-1]
        stock_pcr = stock_state.put_call_ratios[-1]

        return etf_pcr > self._config.rotation_threshold and stock_pcr < 0.5

    # ------------------------------------------------------------------
    # Lead-lag signal
    # ------------------------------------------------------------------

    def compute_lead_signal(self, etf_symbol: str) -> tuple[float, float, int]:
        """Compute a lead signal based on ETF flow direction changes.

        When ETF flow changes direction sharply (z-score of recent change is
        significant), constituents are expected to follow in 5-15 minutes.

        Returns:
            Tuple of (lead_signal, lead_confidence, estimated_lag_minutes).
            lead_signal is in [-1, 1]: negative = bearish lead, positive = bullish.
            lead_confidence is in [0, 1].
            estimated_lag_minutes is always 10 (midpoint of 5-15 range).
        """
        etf_symbol = etf_symbol.upper()
        changes = self._etf_flow_changes.get(etf_symbol)

        if changes is None or len(changes) < self._config.min_bars:
            return 0.0, 0.0, 10

        changes_arr = np.array(changes, dtype=np.float64)
        std = float(np.std(changes_arr))

        if std < 1e-12:
            return 0.0, 0.0, 10

        latest_change = changes_arr[-1]
        z = latest_change / std

        # Clip signal to [-1, 1]
        lead_signal = float(np.clip(z / self._config.divergence_sigma, -1.0, 1.0))

        # Confidence: how extreme is the z-score relative to the significance threshold
        lead_confidence = float(np.clip(abs(z) / self._config.divergence_sigma, 0.0, 1.0))

        return lead_signal, lead_confidence, 10

    # ------------------------------------------------------------------
    # Composite analysis
    # ------------------------------------------------------------------

    def analyze(self, stock_symbol: str, etf_symbol: str = "SPY") -> Optional[ETFFlowResult]:
        """Run full ETF flow analysis for a stock against its parent ETF.

        Combines divergence, rotation, and lead-lag signals into a single
        frozen dataclass result.

        Args:
            stock_symbol: Individual stock ticker.
            etf_symbol: Parent ETF ticker (default SPY).

        Returns:
            ETFFlowResult, or None if insufficient data for any component.
        """
        stock_symbol = stock_symbol.upper()
        etf_symbol = etf_symbol.upper()

        etf_state = self._etf_states.get(etf_symbol)
        stock_state = self._stock_states.get(stock_symbol)

        if etf_state is None or stock_state is None:
            return None
        if etf_state.count < self._config.min_bars or stock_state.count < self._config.min_bars:
            return None

        divergence = self.compute_divergence(stock_symbol, etf_symbol)
        if divergence is None:
            return None

        rotation = self.detect_rotation(stock_symbol, etf_symbol)
        lead_signal, lead_confidence, _ = self.compute_lead_signal(etf_symbol)

        etf_pcr = etf_state.put_call_ratios[-1]
        stock_pcr = stock_state.put_call_ratios[-1]

        result = ETFFlowResult(
            etf_symbol=etf_symbol,
            stock_symbol=stock_symbol,
            divergence_score=divergence,
            rotation_detected=rotation,
            lead_signal=lead_signal,
            lead_confidence=lead_confidence,
            etf_put_call_ratio=etf_pcr,
            stock_put_call_ratio=stock_pcr,
        )

        if self._logger:
            self._logger.debug(
                "ETF flow analysis complete",
                etf_symbol=etf_symbol,
                stock_symbol=stock_symbol,
                divergence_score=round(divergence, 4),
                rotation_detected=rotation,
                lead_signal=round(lead_signal, 4),
            )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zscore(values: deque[float]) -> Optional[float]:
        """Compute z-score of the latest value in a deque.

        Returns None if the series has fewer than 2 elements or zero variance.
        """
        if len(values) < 2:
            return None

        arr = np.array(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        if std < 1e-12:
            return 0.0

        return (arr[-1] - mean) / std
