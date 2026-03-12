"""Atlas factor reader — consumes validated factors from Heber Gold.

Reads FactorCatalogEntry and FactorScore parquet files published by
Atlas's AtlasFactorPublisher.  Provides per-symbol scores and a
composite signal for integration into Cerberus's FeaturePipeline.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import structlog

from src.data.atlas_schema import FactorCatalogEntry

logger = structlog.get_logger(__name__)

_DEFAULT_FACTOR_DIR = Path("/Volumes/heber/data/gold/atlas_factors")


class AtlasFactorReader:
    """Reads Atlas factor scores from Heber Gold.

    Designed for graceful degradation — never raises, returns empty
    results when factors are unavailable or stale.

    Usage in Cerberus FeaturePipeline Stage 3::

        reader = AtlasFactorReader(factor_dir=Path("/Volumes/heber/data/gold/atlas_factors"))
        scores = reader.get_scores("AAPL", date.today())
        composite = reader.get_composite_score("AAPL", date.today())
    """

    def __init__(
        self,
        factor_dir: Path | None = None,
        max_staleness_days: int = 3,
        min_soft_score: float = 0.3,
    ) -> None:
        self.factor_dir = factor_dir or _DEFAULT_FACTOR_DIR
        self.max_staleness_days = max_staleness_days
        self.min_soft_score = min_soft_score

        # Daily cache
        self._catalog_cache: list[FactorCatalogEntry] | None = None
        self._catalog_cache_date: date | None = None
        self._scores_cache: dict[str, pd.DataFrame] = {}
        self._scores_cache_date: date | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_catalog(self) -> list[FactorCatalogEntry]:
        """Load and filter the factor catalog.

        Returns active (non-expired, quality-filtered) catalog entries.
        Results are cached for the current day.
        """
        today = date.today()
        if self._catalog_cache is not None and self._catalog_cache_date == today:
            return self._catalog_cache

        try:
            entries = self._read_catalog()
        except Exception:
            logger.warning("atlas_catalog_load_failed", exc_info=True)
            self._catalog_cache = []
            self._catalog_cache_date = today
            return []

        # Filter: quality threshold + not expired
        active = [e for e in entries if e.soft_score >= self.min_soft_score and not e.is_expired]

        self._catalog_cache = active
        self._catalog_cache_date = today

        logger.debug(
            "atlas_catalog_loaded",
            total_entries=len(entries),
            active_entries=len(active),
        )
        return active

    def get_scores(self, symbol: str, as_of: date | None = None) -> dict[str, float]:
        """Get per-factor scores for a symbol.

        Returns a dict like::

            {
                "atlas_momentum_HYP_abc123": 0.72,
                "atlas_flow_momentum_HYP_def456": -0.31,
            }

        Returns empty dict if no factors are available.
        """
        as_of = as_of or date.today()
        catalog = self.load_catalog()
        if not catalog:
            return {}

        result: dict[str, float] = {}
        for entry in catalog:
            score = self._get_factor_score(entry, symbol, as_of)
            if score is not None:
                key = f"atlas_{entry.family}_{entry.hypothesis_id}"
                result[key] = score

        return result

    def get_composite_score(self, symbol: str, as_of: date | None = None) -> float:
        """Compute soft-score-weighted average across all active factors.

        This is the value that maps to ``ml_prediction`` for
        MLDirectionalStrategy.  Returns 0.0 if no factors contribute.
        """
        as_of = as_of or date.today()
        catalog = self.load_catalog()
        if not catalog:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0

        for entry in catalog:
            score = self._get_factor_score(entry, symbol, as_of)
            if score is not None:
                weighted_sum += score * entry.soft_score
                weight_total += entry.soft_score

        if abs(weight_total) < 1e-9:
            return 0.0

        return weighted_sum / weight_total

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_catalog(self) -> list[FactorCatalogEntry]:
        """Read catalog.parquet and parse into FactorCatalogEntry objects."""
        catalog_path = self.factor_dir / "catalog.parquet"
        if not catalog_path.exists():
            return []

        df = pd.read_parquet(catalog_path)
        entries: list[FactorCatalogEntry] = []

        for _, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                # Handle pandas Timestamp → datetime conversion
                if isinstance(row_dict.get("published_at"), pd.Timestamp):
                    ts = row_dict["published_at"]
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    row_dict["published_at"] = ts.to_pydatetime()
                # Handle list conversion for symbols
                if isinstance(row_dict.get("symbols"), str):
                    import json

                    try:
                        row_dict["symbols"] = json.loads(row_dict["symbols"])
                    except (json.JSONDecodeError, TypeError):
                        row_dict["symbols"] = []
                entries.append(FactorCatalogEntry(**row_dict))
            except Exception:
                logger.debug("atlas_catalog_entry_parse_failed", row=row.to_dict(), exc_info=True)
                continue

        return entries

    def _get_factor_score(
        self,
        entry: FactorCatalogEntry,
        symbol: str,
        as_of: date,
    ) -> float | None:
        """Get the score for a specific factor + symbol + date.

        Searches for the exact date first, then falls back to the
        most recent score within max_staleness_days.
        """
        scores_dir = self.factor_dir / "scores" / entry.hypothesis_id
        if not scores_dir.exists():
            return None

        # Try exact date first
        score = self._read_score_file(scores_dir, entry.hypothesis_id, symbol, as_of)
        if score is not None:
            return score

        # Fall back to recent scores within staleness window
        for days_back in range(1, self.max_staleness_days + 1):
            check_date = date.fromordinal(as_of.toordinal() - days_back)
            score = self._read_score_file(scores_dir, entry.hypothesis_id, symbol, check_date)
            if score is not None:
                return score

        return None

    def _read_score_file(
        self,
        scores_dir: Path,
        hypothesis_id: str,
        symbol: str,
        score_date: date,
    ) -> float | None:
        """Read a single score parquet file and extract the symbol's score."""
        # Check daily cache
        cache_key = f"{hypothesis_id}:{score_date.isoformat()}"
        today = date.today()

        if self._scores_cache_date != today:
            self._scores_cache = {}
            self._scores_cache_date = today

        if cache_key not in self._scores_cache:
            score_path = scores_dir / f"{score_date.isoformat()}.parquet"
            if not score_path.exists():
                return None
            try:
                self._scores_cache[cache_key] = pd.read_parquet(score_path)
            except Exception:
                logger.debug(
                    "atlas_score_read_failed",
                    path=str(score_path),
                    exc_info=True,
                )
                return None

        df = self._scores_cache[cache_key]
        match = df[df["symbol"] == symbol]
        if match.empty:
            return None

        return float(match.iloc[0]["score"])
