"""Tests for Atlas factor integration in Cerberus FeaturePipeline."""

from datetime import date, datetime, timezone
from typing import Dict
from unittest.mock import MagicMock

from src.core.domain import SymbolFeatures


class FakeAtlasReader:
    """Minimal fake for AtlasFactorReader."""

    def __init__(self, scores: Dict[str, Dict[str, float]] | None = None, composites: Dict[str, float] | None = None):
        self._scores = scores or {}
        self._composites = composites or {}

    def get_scores(self, symbol: str, as_of: date | None = None) -> Dict[str, float]:
        return dict(self._scores.get(symbol, {}))

    def get_composite_score(self, symbol: str, as_of: date | None = None) -> float:
        return self._composites.get(symbol, 0.0)


def _make_features(symbol: str = "AAPL") -> SymbolFeatures:
    """Build a minimal SymbolFeatures with defaults."""
    return SymbolFeatures(
        symbol=symbol,
        price=150.0,
        atr_pct=1.5,
        avg_volume=50_000_000.0,
        intraday_range_pct=2.0,
        gap_pct=0.5,
        ema20_slope=0.002,
        ema_trend_strength=0.6,
        distance_from_vwap=0.1,
        premarket_volume=1_000_000.0,
        adx=25.0,
        distance_from_ema20=0.3,
        prior_day_high=152.0,
        prior_day_low=148.0,
        bb_upper=155.0,
        bb_lower=145.0,
        price_zscore=0.2,
        flow_zscore=0.5,
        call_put_ratio=1.1,
        large_sweeps_count=3,
        aggressive_flow_share=0.4,
        last_updated=datetime.now(timezone.utc),
    )


class TestAppendAtlasFactors:
    """Test the Stage 3 _append_atlas_factors method."""

    def test_atlas_scores_injected(self) -> None:
        """Atlas factor scores should appear in SymbolFeatures.extra."""
        from src.data.pipeline import FeaturePipeline

        reader = FakeAtlasReader(
            scores={"AAPL": {"atlas_momentum_HYP_a": 0.72, "atlas_flow_HYP_b": -0.3}},
            composites={"AAPL": 0.5},
        )
        pipeline = FeaturePipeline(
            alpaca_client=None,
            unusual_whales_client=MagicMock(),
            logger=MagicMock(),
            atlas_reader=reader,
        )

        feat = _make_features("AAPL")
        _features_map = {"AAPL": feat}
        pipeline._append_atlas_factors(_features_map)

        assert feat.extra["atlas_momentum_HYP_a"] == 0.72
        assert feat.extra["atlas_flow_HYP_b"] == -0.3
        assert feat.extra["atlas_composite"] == 0.5

    def test_no_atlas_reader_skips(self) -> None:
        """When atlas_reader is None, Stage 3 should be skipped entirely."""
        from src.data.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(
            alpaca_client=None,
            unusual_whales_client=MagicMock(),
            logger=MagicMock(),
            atlas_reader=None,
        )

        feat = _make_features("AAPL")

        # _append_atlas_factors shouldn't exist in the flow without reader
        assert pipeline.atlas_reader is None
        # Extra should remain empty
        assert feat.extra == {} or "atlas_composite" not in feat.extra

    def test_atlas_reader_failure_graceful(self) -> None:
        """If atlas_reader.get_scores raises, the symbol should still have features."""
        from src.data.pipeline import FeaturePipeline

        broken_reader = MagicMock()
        broken_reader.get_scores.side_effect = RuntimeError("bad disk")
        pipeline = FeaturePipeline(
            alpaca_client=None,
            unusual_whales_client=MagicMock(),
            logger=MagicMock(),
            atlas_reader=broken_reader,
        )

        feat = _make_features("AAPL")
        features_map = {"AAPL": feat}
        # Should not raise
        pipeline._append_atlas_factors(features_map)

        # No atlas keys should be present
        assert "atlas_composite" not in feat.extra

    def test_zero_composite_with_no_scores(self) -> None:
        """If there are no scores for a symbol, no atlas keys should be written."""
        from src.data.pipeline import FeaturePipeline

        reader = FakeAtlasReader(scores={}, composites={})
        pipeline = FeaturePipeline(
            alpaca_client=None,
            unusual_whales_client=MagicMock(),
            logger=MagicMock(),
            atlas_reader=reader,
        )

        feat = _make_features("AAPL")
        features_map = {"AAPL": feat}
        pipeline._append_atlas_factors(features_map)

        # No scores, no composite
        assert "atlas_composite" not in feat.extra

    def test_multiple_symbols(self) -> None:
        """Each symbol gets its own atlas scores."""
        from src.data.pipeline import FeaturePipeline

        reader = FakeAtlasReader(
            scores={
                "AAPL": {"atlas_mom_HYP_x": 0.5},
                "TSLA": {"atlas_mom_HYP_x": -0.8},
            },
            composites={"AAPL": 0.5, "TSLA": -0.8},
        )
        pipeline = FeaturePipeline(
            alpaca_client=None,
            unusual_whales_client=MagicMock(),
            logger=MagicMock(),
            atlas_reader=reader,
        )

        aapl = _make_features("AAPL")
        tsla = _make_features("TSLA")
        features_map = {"AAPL": aapl, "TSLA": tsla}
        pipeline._append_atlas_factors(features_map)

        assert aapl.extra["atlas_mom_HYP_x"] == 0.5
        assert tsla.extra["atlas_mom_HYP_x"] == -0.8
        assert aapl.extra["atlas_composite"] == 0.5
        assert tsla.extra["atlas_composite"] == -0.8
