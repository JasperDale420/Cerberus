from typing import Optional

from src.core.domain import Signal, TechnicalFeatures


class MetaLabeler:
    """
    Vets trading signals using secondary alpha features.
    Reduces false positives by requiring favorable probabilistic context.
    """

    def __init__(self, logger):
        self.logger = logger

    def vet_signal(self, signal: Signal) -> bool:
        """
        Returns True if the signal is statistically favorable.

        Currently implements a heuristic 'Alpha Filter':
        - Longs: Hurst > 0.5 (Trending) AND TFI > 0 (Buying Pressure)
        - Shorts: Hurst > 0.5 (Trending) AND TFI < 0 (Selling Pressure)
        - GEX filter: If net_gex is extreme, avoid trading against it.
        """
        features: Optional[TechnicalFeatures] = signal.feature_snapshot
        if not features:
            # If no snapshot, we can't vet, so we allow it but log a warning
            self.logger.warning(
                "Signal missing feature snapshot, skipping meta-labeling",
                strategy=signal.strategy,
                symbol=signal.symbol,
            )
            return True

        # Heuristic 1: Hurst Exponent (Trend Persistence)
        # We prefer trading in trending regimes for momentum strategies
        is_trending = features.hurst_exponent > 0.5

        # Heuristic 2: Trade Flow Imbalance (Microstructure)
        # TFI should align with signal direction
        tfi_ok = True
        if signal.side == "buy" and features.tfi < -0.1:
            tfi_ok = False
        elif signal.side == "sell" and features.tfi > 0.1:
            tfi_ok = False

        # Heuristic 3: Net GEX (Market Magnetism)
        # Extreme negative GEX can lead to high volatility/slippage
        gex_ok = True
        # Simple threshold: if GEX is deeply negative, be cautious with longs
        if (
            signal.side == "buy" and features.net_gex < -1000000
        ):  # Placeholder threshold
            gex_ok = False

        # Vetting Decision
        passed = is_trending and tfi_ok and gex_ok

        if not passed:
            self.logger.info(
                "Signal rejected by Meta-Labeler",
                strategy=signal.strategy,
                symbol=signal.symbol,
                hurst=features.hurst_exponent,
                tfi=features.tfi,
                net_gex=features.net_gex,
                reason=f"Hurst:{is_trending}, TFI:{tfi_ok}, GEX:{gex_ok}",
            )

        return passed
