from typing import List

import numpy as np

from src.core.domain import SymbolFeatures
from src.core.logger import StructuredLogger


class RankingEngine:
    """
    Ranks the universe of symbols by a composite AlphaScore.
    Applies cross-sectional Z-score normalization to features.
    """

    def __init__(self, logger: StructuredLogger = None):
        self.logger = logger or StructuredLogger("RankingEngine")

    def rank_symbols(self, symbols: List[SymbolFeatures]) -> List[SymbolFeatures]:
        """
        Computes composite AlphaScore and ranks symbols cross-sectionally.
        """
        if not symbols:
            return []

        # 1. Extract raw factors
        factors = {
            "ma_dist_200": np.array([s.ma_dist_200 for s in symbols]),
            "ma_dist_50": np.array([s.ma_dist_50 for s in symbols]),
            "relative_strength": np.array([s.relative_strength for s in symbols]),
        }

        # 2. Compute Z-Scores cross-sectionally
        z_scores = {}
        for name, values in factors.items():
            mean = np.mean(values)
            std = np.std(values)
            if std > 1e-6:
                z_scores[name] = (values - mean) / std
            else:
                z_scores[name] = np.zeros_like(values)

        # 3. Compute AlphaScore (V1: Morning Mean Reversion)
        # Weights based on Audit ICs: ma_dist_200 (0.5), ma_dist_50 (0.3), RS (0.2)
        # We multiply by -1 because distance from MA has negative IC (Low factor -> High return)
        alpha_scores = -1.0 * (
            0.5 * z_scores["ma_dist_200"]
            + 0.3 * z_scores["ma_dist_50"]
            + 0.2 * z_scores["relative_strength"]
        )

        # 4. Map back to features
        for i, sym_feat in enumerate(symbols):
            sym_feat.alpha_score = float(alpha_scores[i])

        # 5. Assign Ranks
        # Sort symbols by AlphaScore descending
        symbols.sort(key=lambda s: s.alpha_score, reverse=True)

        for rank, sym_feat in enumerate(symbols, 1):
            sym_feat.alpha_rank = rank

        self.logger.info(
            "Ranking complete",
            top_symbol=symbols[0].symbol if symbols else None,
            top_score=symbols[0].alpha_score if symbols else None,
            symbols_ranked=len(symbols),
        )

        return symbols
