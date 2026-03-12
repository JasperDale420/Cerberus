"""Reader-side bridge schema for Atlas factor consumption.

These models mirror the Atlas publisher schema
(atlas.bridge.schema.FactorCatalogEntry / FactorScore) but live in
Cerberus's codebase so there is no cross-repo Python import.

The parquet file format is the actual contract — these models are
just convenience parsers for the reader.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field


class FactorCatalogEntry(BaseModel):
    """Metadata for one validated Atlas factor (reader-side copy)."""

    hypothesis_id: str
    family: str
    claim: str = ""
    feature_definition: str = ""
    symbols: list[str] = Field(default_factory=list)
    soft_score: float
    ic: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    gate_version: str = "v1"
    model_version: str = ""
    published_at: datetime
    ttl_days: int = 30

    @property
    def is_expired(self) -> bool:
        """Check if this catalog entry has exceeded its TTL."""
        from datetime import timezone

        age_days = (datetime.now(timezone.utc) - self.published_at).days
        return age_days > self.ttl_days


class FactorScore(BaseModel):
    """Per-symbol daily score from a validated factor (reader-side copy)."""

    hypothesis_id: str
    symbol: str
    score_date: date
    score: float
    confidence: float = 0.0
    family: str = ""
