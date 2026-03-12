"""Tests for Cerberus Atlas factor reader."""

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data.atlas_reader import AtlasFactorReader
from src.data.atlas_schema import FactorCatalogEntry, FactorScore


@pytest.fixture()
def factor_dir(tmp_path: Path) -> Path:
    return tmp_path / "atlas_factors"


def _write_catalog(factor_dir: Path, entries: list[FactorCatalogEntry]) -> None:
    """Helper to write a catalog.parquet from entries."""
    factor_dir.mkdir(parents=True, exist_ok=True)
    rows = [e.model_dump() for e in entries]
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df.to_parquet(factor_dir / "catalog.parquet", index=False)


def _write_scores(
    factor_dir: Path,
    hypothesis_id: str,
    score_date: date,
    scores: list[FactorScore],
) -> None:
    """Helper to write a scores parquet."""
    scores_dir = factor_dir / "scores" / hypothesis_id
    scores_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([s.model_dump() for s in scores])
    df.to_parquet(scores_dir / f"{score_date.isoformat()}.parquet", index=False)


def _make_entry(
    hypothesis_id: str = "HYP_abc123",
    family: str = "momentum",
    soft_score: float = 0.7,
    ttl_days: int = 30,
    days_ago: int = 0,
) -> FactorCatalogEntry:
    """Create a test catalog entry."""
    from datetime import timedelta

    published_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return FactorCatalogEntry(
        hypothesis_id=hypothesis_id,
        family=family,
        claim="Test hypothesis",
        soft_score=soft_score,
        published_at=published_at,
        ttl_days=ttl_days,
    )


def _make_score(
    hypothesis_id: str = "HYP_abc123",
    symbol: str = "AAPL",
    score_date: date | None = None,
    score: float = 0.5,
    family: str = "momentum",
) -> FactorScore:
    return FactorScore(
        hypothesis_id=hypothesis_id,
        symbol=symbol,
        score_date=score_date or date.today(),
        score=score,
        confidence=0.7,
        family=family,
    )


class TestLoadCatalog:
    def test_empty_dir(self, factor_dir: Path) -> None:
        reader = AtlasFactorReader(factor_dir=factor_dir)
        assert reader.load_catalog() == []

    def test_missing_dir(self, tmp_path: Path) -> None:
        reader = AtlasFactorReader(factor_dir=tmp_path / "nonexistent")
        assert reader.load_catalog() == []

    def test_loads_valid_entries(self, factor_dir: Path) -> None:
        entry = _make_entry()
        _write_catalog(factor_dir, [entry])
        reader = AtlasFactorReader(factor_dir=factor_dir)
        catalog = reader.load_catalog()
        assert len(catalog) == 1
        assert catalog[0].hypothesis_id == "HYP_abc123"

    def test_filters_expired(self, factor_dir: Path) -> None:
        fresh = _make_entry(hypothesis_id="HYP_fresh", days_ago=5)
        expired = _make_entry(hypothesis_id="HYP_old", days_ago=60, ttl_days=30)
        _write_catalog(factor_dir, [fresh, expired])

        reader = AtlasFactorReader(factor_dir=factor_dir)
        catalog = reader.load_catalog()
        assert len(catalog) == 1
        assert catalog[0].hypothesis_id == "HYP_fresh"

    def test_filters_low_soft_score(self, factor_dir: Path) -> None:
        good = _make_entry(hypothesis_id="HYP_good", soft_score=0.7)
        bad = _make_entry(hypothesis_id="HYP_bad", soft_score=0.1)
        _write_catalog(factor_dir, [good, bad])

        reader = AtlasFactorReader(factor_dir=factor_dir, min_soft_score=0.3)
        catalog = reader.load_catalog()
        assert len(catalog) == 1
        assert catalog[0].hypothesis_id == "HYP_good"

    def test_catalog_cached_daily(self, factor_dir: Path) -> None:
        entry = _make_entry()
        _write_catalog(factor_dir, [entry])
        reader = AtlasFactorReader(factor_dir=factor_dir)

        catalog1 = reader.load_catalog()
        catalog2 = reader.load_catalog()
        assert catalog1 is catalog2  # Same object — cached


class TestGetScores:
    def test_no_catalog(self, factor_dir: Path) -> None:
        reader = AtlasFactorReader(factor_dir=factor_dir)
        assert reader.get_scores("AAPL") == {}

    def test_returns_factor_scores(self, factor_dir: Path) -> None:
        today = date.today()
        entry = _make_entry()
        _write_catalog(factor_dir, [entry])
        _write_scores(
            factor_dir,
            "HYP_abc123",
            today,
            [_make_score(symbol="AAPL", score=0.72, score_date=today)],
        )

        reader = AtlasFactorReader(factor_dir=factor_dir)
        scores = reader.get_scores("AAPL", today)
        assert "atlas_momentum_HYP_abc123" in scores
        assert scores["atlas_momentum_HYP_abc123"] == 0.72

    def test_missing_symbol(self, factor_dir: Path) -> None:
        today = date.today()
        entry = _make_entry()
        _write_catalog(factor_dir, [entry])
        _write_scores(
            factor_dir,
            "HYP_abc123",
            today,
            [_make_score(symbol="AAPL", score=0.5, score_date=today)],
        )

        reader = AtlasFactorReader(factor_dir=factor_dir)
        scores = reader.get_scores("TSLA", today)
        assert scores == {}

    def test_staleness_fallback(self, factor_dir: Path) -> None:
        """Scores from a few days ago should still be found."""
        today = date.today()
        yesterday = date.fromordinal(today.toordinal() - 1)

        entry = _make_entry()
        _write_catalog(factor_dir, [entry])
        # Only scores from yesterday, not today
        _write_scores(
            factor_dir,
            "HYP_abc123",
            yesterday,
            [_make_score(symbol="AAPL", score=0.3, score_date=yesterday)],
        )

        reader = AtlasFactorReader(factor_dir=factor_dir, max_staleness_days=3)
        scores = reader.get_scores("AAPL", today)
        assert "atlas_momentum_HYP_abc123" in scores
        assert scores["atlas_momentum_HYP_abc123"] == 0.3

    def test_multiple_factors(self, factor_dir: Path) -> None:
        today = date.today()
        e1 = _make_entry(hypothesis_id="HYP_mom", family="momentum")
        e2 = _make_entry(hypothesis_id="HYP_flow", family="flow_momentum")
        _write_catalog(factor_dir, [e1, e2])

        _write_scores(
            factor_dir,
            "HYP_mom",
            today,
            [_make_score(hypothesis_id="HYP_mom", symbol="AAPL", score=0.5, score_date=today)],
        )
        _write_scores(
            factor_dir,
            "HYP_flow",
            today,
            [
                _make_score(
                    hypothesis_id="HYP_flow", symbol="AAPL", score=-0.3, family="flow_momentum", score_date=today
                )
            ],
        )

        reader = AtlasFactorReader(factor_dir=factor_dir)
        scores = reader.get_scores("AAPL", today)
        assert len(scores) == 2
        assert "atlas_momentum_HYP_mom" in scores
        assert "atlas_flow_momentum_HYP_flow" in scores


class TestGetCompositeScore:
    def test_no_factors(self, factor_dir: Path) -> None:
        reader = AtlasFactorReader(factor_dir=factor_dir)
        assert reader.get_composite_score("AAPL") == 0.0

    def test_single_factor(self, factor_dir: Path) -> None:
        today = date.today()
        entry = _make_entry(soft_score=0.8)
        _write_catalog(factor_dir, [entry])
        _write_scores(
            factor_dir,
            "HYP_abc123",
            today,
            [_make_score(symbol="AAPL", score=0.6, score_date=today)],
        )

        reader = AtlasFactorReader(factor_dir=factor_dir)
        composite = reader.get_composite_score("AAPL", today)
        # Single factor: (0.6 * 0.8) / 0.8 = 0.6
        assert composite == pytest.approx(0.6)

    def test_weighted_average(self, factor_dir: Path) -> None:
        today = date.today()
        e1 = _make_entry(hypothesis_id="HYP_a", family="mom", soft_score=0.8)
        e2 = _make_entry(hypothesis_id="HYP_b", family="flow", soft_score=0.4)
        _write_catalog(factor_dir, [e1, e2])

        _write_scores(
            factor_dir,
            "HYP_a",
            today,
            [_make_score(hypothesis_id="HYP_a", symbol="AAPL", score=0.6, score_date=today)],
        )
        _write_scores(
            factor_dir,
            "HYP_b",
            today,
            [_make_score(hypothesis_id="HYP_b", symbol="AAPL", score=-0.2, family="flow", score_date=today)],
        )

        reader = AtlasFactorReader(factor_dir=factor_dir)
        composite = reader.get_composite_score("AAPL", today)
        # Weighted: (0.6 * 0.8 + (-0.2) * 0.4) / (0.8 + 0.4) = (0.48 - 0.08) / 1.2 = 0.333...
        assert composite == pytest.approx(1 / 3, abs=0.01)


class TestResilience:
    def test_corrupt_catalog_parquet(self, factor_dir: Path) -> None:
        factor_dir.mkdir(parents=True, exist_ok=True)
        (factor_dir / "catalog.parquet").write_bytes(b"not a parquet file")

        reader = AtlasFactorReader(factor_dir=factor_dir)
        assert reader.load_catalog() == []

    def test_corrupt_score_parquet(self, factor_dir: Path) -> None:
        today = date.today()
        entry = _make_entry()
        _write_catalog(factor_dir, [entry])

        scores_dir = factor_dir / "scores" / "HYP_abc123"
        scores_dir.mkdir(parents=True, exist_ok=True)
        (scores_dir / f"{today.isoformat()}.parquet").write_bytes(b"corrupt")

        reader = AtlasFactorReader(factor_dir=factor_dir)
        scores = reader.get_scores("AAPL", today)
        assert scores == {}
