"""Conviction-based and correlation-aware adaptive position sizing."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.domain import Position, Signal


@dataclass
class SizingConfig:
    """Configuration for adaptive position sizing."""

    # Base risk as percentage of account equity
    base_risk_pct: float = 0.020  # 2.0% default (up from 1.5%)
    max_risk_pct: float = 0.030  # 3.0% absolute max (up from 2.0%)
    min_risk_pct: float = 0.005  # 0.5% minimum (prevents tiny positions)

    # Conviction scaling (from confluence score)
    conviction_enabled: bool = True
    conviction_floor: float = 0.5  # Minimum multiplier at threshold
    conviction_ceiling: float = 1.0  # Maximum multiplier at score=100

    # Correlation reduction
    correlation_enabled: bool = True
    correlation_penalty: float = 0.15  # Reduce by 15% per correlated position (down from 30%)
    max_correlation_reduction: float = 0.60  # Never reduce by more than 60%

    # Sector mapping for correlation detection
    sector_map: dict[str, str] = field(default_factory=dict)

    # Strategy-specific risk overrides
    strategy_risk_overrides: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default sector mapping for symbols commonly traded by Cerberus
# ---------------------------------------------------------------------------
_DEFAULT_SECTORS: dict[str, str] = {
    # Semiconductors
    "NVDA": "semiconductor",
    "AMD": "semiconductor",
    "INTC": "semiconductor",
    "AVGO": "semiconductor",
    "MU": "semiconductor",
    "MRVL": "semiconductor",
    # Mega-cap tech
    "AAPL": "mega_tech",
    "MSFT": "mega_tech",
    "GOOG": "mega_tech",
    "GOOGL": "mega_tech",
    "META": "mega_tech",
    "AMZN": "mega_tech",
    # EV
    "TSLA": "ev",
    "RIVN": "ev",
    "LCID": "ev",
    "NIO": "ev",
    # AI software
    "AI": "ai_software",
    "PLTR": "ai_software",
    "SNOW": "ai_software",
    # Index ETFs
    "SPY": "index",
    "QQQ": "index",
    "IWM": "index",
    # Banks
    "JPM": "banks",
    "GS": "banks",
    "BAC": "banks",
    "MS": "banks",
    # Energy
    "XOM": "energy",
    "CVX": "energy",
    "OXY": "energy",
}


class AdaptiveSizer:
    """
    Conviction-based position sizing with correlation awareness.

    Sits alongside the existing RiskManager -- provides a dollar risk amount
    that the risk manager uses in place of the static ``max_risk_per_trade``.

    Flow:
        1. Start with ``base_risk_pct`` of account equity.
        2. Apply strategy-specific override if configured.
        3. Scale by conviction multiplier (from confluence score).
        4. Reduce for correlated open positions.
        5. Clamp to ``[min_risk_pct, max_risk_pct]``.
        6. Return risk amount in dollars.

    Usage::

        sizer = AdaptiveSizer(config)
        risk_dollars = sizer.calculate_risk(
            signal=signal,
            account_equity=75_000.0,
            open_positions=[pos1, pos2],
            confluence_score=82.5,
            confluence_threshold=65.0,
        )
    """

    def __init__(self, config: SizingConfig | None = None) -> None:
        self.config = config or SizingConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_risk(
        self,
        signal: Signal,
        account_equity: float,
        open_positions: list[Position] | None = None,
        confluence_score: float | None = None,
        confluence_threshold: float = 65.0,
    ) -> float:
        """Calculate the risk amount in dollars for a trade.

        Args:
            signal: The trade signal.
            account_equity: Current account equity in dollars.
            open_positions: Currently open positions (for correlation check).
            confluence_score: Confluence score 0-100 (``None`` skips conviction
                scaling and uses the base risk percentage directly).
            confluence_threshold: Minimum score required to take a trade.

        Returns:
            Risk amount in dollars.  ``0.0`` if the trade should not be taken
            (e.g. confluence score below threshold or non-positive equity).
        """
        if account_equity <= 0:
            return 0.0

        # 1. Base risk percentage (with strategy override)
        risk_pct = self._get_base_risk_pct(signal.strategy)

        # 2. Conviction multiplier
        conviction_mult = self._get_conviction_multiplier(
            confluence_score, confluence_threshold
        )
        if conviction_mult <= 0.0:
            return 0.0  # Below threshold -- do not trade
        risk_pct *= conviction_mult

        # 3. Correlation reduction
        corr_mult = self._get_correlation_reduction(signal, open_positions)
        risk_pct *= corr_mult

        # 4. Clamp to configured bounds
        risk_pct = max(self.config.min_risk_pct, min(risk_pct, self.config.max_risk_pct))

        return round(risk_pct * account_equity, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_base_risk_pct(self, strategy: str) -> float:
        """Return base risk percentage, applying a strategy override if one exists."""
        override = self.config.strategy_risk_overrides.get(strategy)
        if override is not None:
            return float(override)
        return self.config.base_risk_pct

    def _get_conviction_multiplier(
        self, score: float | None, threshold: float
    ) -> float:
        """Map a confluence score to a conviction multiplier.

        Linear interpolation between ``conviction_floor`` (at *threshold*) and
        ``conviction_ceiling`` (at score == 100).

        Returns:
            * ``1.0`` when conviction scaling is disabled or *score* is ``None``
              (caller did not supply a confluence score -- use base risk).
            * ``0.0`` when *score* < *threshold* (do not trade).
            * A value in ``[conviction_floor, conviction_ceiling]`` otherwise.
        """
        if not self.config.conviction_enabled or score is None:
            return 1.0

        if score < threshold:
            return 0.0

        floor = self.config.conviction_floor
        ceiling = self.config.conviction_ceiling

        # Avoid division by zero when threshold == 100
        score_range = 100.0 - threshold
        if score_range <= 0:
            return ceiling

        t = (score - threshold) / score_range  # 0.0 .. 1.0
        return floor + t * (ceiling - floor)

    def _get_correlation_reduction(
        self,
        signal: Signal,
        open_positions: list[Position] | None,
    ) -> float:
        """Calculate a multiplier (0.4 -- 1.0) based on correlated open positions.

        Each correlated position applies ``correlation_penalty`` (e.g. 0.30),
        capped at ``max_correlation_reduction`` (e.g. 0.60).

        Returns ``1.0`` (no reduction) when correlation checking is disabled or
        there are no open positions.
        """
        if not self.config.correlation_enabled or not open_positions:
            return 1.0

        correlated_count = 0
        for pos in open_positions:
            if self._are_correlated(
                signal.symbol, signal.strategy, pos.symbol, pos.strategy
            ):
                correlated_count += 1

        if correlated_count == 0:
            return 1.0

        raw_reduction = correlated_count * self.config.correlation_penalty
        capped_reduction = min(raw_reduction, self.config.max_correlation_reduction)
        return 1.0 - capped_reduction

    def _get_sector(self, symbol: str) -> str:
        """Return sector for *symbol*.  Config overrides take precedence."""
        sector = self.config.sector_map.get(symbol)
        if sector is not None:
            return sector
        return _DEFAULT_SECTORS.get(symbol, "unknown")

    def _are_correlated(
        self,
        symbol1: str,
        strategy1: str,
        symbol2: str,
        strategy2: str,
    ) -> bool:
        """Determine if two positions are correlated.

        Two positions are considered correlated when **any** of the following
        conditions hold:

        * Same symbol.
        * Same sector (excluding ``"unknown"``).
        * Same strategy on symbols in different but known sectors.
        * Both are index ETFs (SPY, QQQ, IWM).
        """
        if symbol1 == symbol2:
            return True

        sector1 = self._get_sector(symbol1)
        sector2 = self._get_sector(symbol2)

        # Same known sector
        if sector1 == sector2 and sector1 != "unknown":
            return True

        # Both index ETFs (always highly correlated)
        _index_etfs = {"SPY", "QQQ", "IWM"}
        if symbol1 in _index_etfs and symbol2 in _index_etfs:
            return True

        # Same strategy on two known (non-unknown) symbols
        return strategy1 == strategy2 and sector1 != "unknown" and sector2 != "unknown"
