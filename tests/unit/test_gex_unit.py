"""Unit tests for GEX (Gamma Exposure) analyzer and regime integration."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.analysis.gex import GEXAnalyzer
from src.core.domain import Bar, MarketRegimeSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain(
    strikes: list[float],
    call_ois: list[float],
    put_ois: list[float],
    call_gammas: list[float],
    put_gammas: list[float],
) -> list[dict]:
    """Build a chain_data list from parallel arrays."""
    return [
        {
            "strike": s,
            "call_oi": co,
            "put_oi": po,
            "call_gamma": cg,
            "put_gamma": pg,
        }
        for s, co, po, cg, pg in zip(strikes, call_ois, put_ois, call_gammas, put_gammas, strict=False)
    ]


def _make_bar(price: float = 100.0, volume: float = 1_000_000.0) -> Bar:
    return Bar(
        symbol="SPY",
        time=datetime(2026, 3, 12, 10, 30, tzinfo=timezone.utc),
        open=price,
        high=price * 1.005,
        low=price * 0.995,
        close=price,
        volume=volume,
    )


# ---------------------------------------------------------------------------
# GEXAnalyzer — core computation tests
# ---------------------------------------------------------------------------


class TestPositiveGEXRegime:
    """Heavy call OI should produce positive GEX -> 'positive' regime."""

    def test_positive_gex_regime(self) -> None:
        analyzer = GEXAnalyzer()
        chain = _make_chain(
            strikes=[95, 100, 105],
            call_ois=[5000, 10000, 3000],
            put_ois=[500, 200, 100],
            call_gammas=[0.04, 0.05, 0.03],
            put_gammas=[0.04, 0.05, 0.03],
        )
        result = analyzer.update(current_price=100.0, chain_data=chain)

        assert result is not None
        assert result.gex_regime == "positive"
        assert result.net_gex > 0
        assert analyzer.get_regime() == "positive"


class TestNegativeGEXRegime:
    """Heavy put OI should produce negative GEX -> 'negative' regime."""

    def test_negative_gex_regime(self) -> None:
        analyzer = GEXAnalyzer()
        chain = _make_chain(
            strikes=[95, 100, 105],
            call_ois=[100, 200, 50],
            put_ois=[8000, 12000, 5000],
            call_gammas=[0.04, 0.05, 0.03],
            put_gammas=[0.04, 0.05, 0.03],
        )
        result = analyzer.update(current_price=100.0, chain_data=chain)

        assert result is not None
        assert result.gex_regime == "negative"
        assert result.net_gex < 0


class TestNeutralGEXSmallOI:
    """When OI is small/balanced, regime should be 'neutral'."""

    def test_neutral_gex_small_oi(self) -> None:
        analyzer = GEXAnalyzer()
        # Balanced call/put OI with identical gammas → near-zero net GEX
        chain = _make_chain(
            strikes=[95, 100, 105],
            call_ois=[1000, 1000, 1000],
            put_ois=[1000, 1000, 1000],
            call_gammas=[0.04, 0.05, 0.03],
            put_gammas=[0.04, 0.05, 0.03],
        )
        result = analyzer.update(current_price=100.0, chain_data=chain)

        assert result is not None
        assert result.gex_regime == "neutral"
        assert abs(result.net_gex) < 1e-6


class TestGammaFlipLevel:
    """Verify gamma flip level calculation with mixed strikes."""

    def test_gamma_flip_level(self) -> None:
        analyzer = GEXAnalyzer()
        # Construct chain where lower strikes have negative net GEX (put heavy)
        # and upper strikes have positive net GEX (call heavy) to force a flip
        chain = _make_chain(
            strikes=[90, 95, 100, 105, 110],
            call_ois=[100, 200, 500, 5000, 8000],
            put_ois=[6000, 4000, 500, 100, 50],
            call_gammas=[0.02, 0.03, 0.05, 0.04, 0.02],
            put_gammas=[0.02, 0.03, 0.05, 0.04, 0.02],
        )
        result = analyzer.update(current_price=100.0, chain_data=chain)

        assert result is not None
        # The flip should occur somewhere between 95 and 105
        # (cumulative GEX starts negative, then goes positive)
        assert result.gamma_flip_distance != 0.0


class TestGammaFlipDistance:
    """Verify % distance from current price to gamma flip level."""

    def test_gamma_flip_distance_positive(self) -> None:
        analyzer = GEXAnalyzer()
        # Put-heavy at lower strikes, call-heavy at higher strikes
        chain = _make_chain(
            strikes=[90, 100, 110],
            call_ois=[10, 10, 10000],
            put_ois=[10000, 10, 10],
            call_gammas=[0.05, 0.05, 0.05],
            put_gammas=[0.05, 0.05, 0.05],
        )
        result = analyzer.update(current_price=100.0, chain_data=chain)
        assert result is not None
        # Flip should be between 90 and 110, distance is relative to 100
        # A positive distance means flip is above current price
        # A negative distance means flip is below current price
        assert isinstance(result.gamma_flip_distance, float)


class TestUpdateFromNormalized:
    """Pre-computed GEX data path via Data-Gateway."""

    def test_update_from_normalized_positive(self) -> None:
        analyzer = GEXAnalyzer()
        gex_data = {"call_gamma": 500.0, "put_gamma": 100.0}
        result = analyzer.update_from_normalized(current_price=100.0, gex_data=gex_data)

        assert result is not None
        assert result.gex_regime == "positive"
        assert result.net_gex > 0
        # No per-strike data, so no flip distance or top strikes
        assert result.gamma_flip_distance == 0.0
        assert result.top_strikes == []

    def test_update_from_normalized_negative(self) -> None:
        analyzer = GEXAnalyzer()
        gex_data = {"call_gamma": 50.0, "put_gamma": 500.0}
        result = analyzer.update_from_normalized(current_price=100.0, gex_data=gex_data)

        assert result is not None
        assert result.gex_regime == "negative"
        assert result.net_gex < 0

    def test_update_from_normalized_zero_gamma(self) -> None:
        analyzer = GEXAnalyzer()
        gex_data = {"call_gamma": 0.0, "put_gamma": 0.0}
        result = analyzer.update_from_normalized(current_price=100.0, gex_data=gex_data)

        assert result is None

    def test_update_from_normalized_empty(self) -> None:
        analyzer = GEXAnalyzer()
        result = analyzer.update_from_normalized(current_price=100.0, gex_data={})

        assert result is None


class TestInsufficientData:
    """Empty chain or invalid price returns None."""

    def test_empty_chain(self) -> None:
        analyzer = GEXAnalyzer()
        result = analyzer.update(current_price=100.0, chain_data=[])
        assert result is None

    def test_zero_price(self) -> None:
        analyzer = GEXAnalyzer()
        chain = _make_chain([100], [1000], [1000], [0.05], [0.05])
        result = analyzer.update(current_price=0.0, chain_data=chain)
        assert result is None

    def test_negative_price(self) -> None:
        analyzer = GEXAnalyzer()
        chain = _make_chain([100], [1000], [1000], [0.05], [0.05])
        result = analyzer.update(current_price=-10.0, chain_data=chain)
        assert result is None

    def test_get_regime_before_update(self) -> None:
        analyzer = GEXAnalyzer()
        assert analyzer.get_regime() is None


class TestPutCallGEXRatio:
    """Verify put/call GEX ratio computation."""

    def test_ratio_call_dominant(self) -> None:
        analyzer = GEXAnalyzer()
        chain = _make_chain(
            strikes=[100],
            call_ois=[10000],
            put_ois=[1000],
            call_gammas=[0.05],
            put_gammas=[0.05],
        )
        result = analyzer.update(current_price=100.0, chain_data=chain)
        assert result is not None
        assert result.put_call_gex_ratio == pytest.approx(0.1, rel=1e-6)


class TestTopStrikes:
    """Verify top strikes are returned sorted by absolute GEX."""

    def test_top_strikes_sorted(self) -> None:
        analyzer = GEXAnalyzer()
        chain = _make_chain(
            strikes=[90, 95, 100, 105, 110, 115],
            call_ois=[100, 200, 10000, 500, 300, 50],
            put_ois=[100, 200, 100, 500, 300, 50],
            call_gammas=[0.02, 0.03, 0.05, 0.04, 0.03, 0.01],
            put_gammas=[0.02, 0.03, 0.05, 0.04, 0.03, 0.01],
        )
        result = analyzer.update(current_price=100.0, chain_data=chain)
        assert result is not None
        assert len(result.top_strikes) == 5
        # First entry should have the largest absolute GEX
        abs_gex_values = [abs(g) for _, g in result.top_strikes]
        assert abs_gex_values == sorted(abs_gex_values, reverse=True)


class TestRollingHistory:
    """Verify rolling history is maintained."""

    def test_history_accumulates(self) -> None:
        analyzer = GEXAnalyzer()
        chain = _make_chain([100], [5000], [500], [0.05], [0.05])
        for _ in range(5):
            analyzer.update(current_price=100.0, chain_data=chain)
        assert len(analyzer._history) == 5

    def test_history_max_length(self) -> None:
        analyzer = GEXAnalyzer()
        chain = _make_chain([100], [5000], [500], [0.05], [0.05])
        for _ in range(25):
            analyzer.update(current_price=100.0, chain_data=chain)
        assert len(analyzer._history) == 20  # maxlen=20


# ---------------------------------------------------------------------------
# MarketRegimeSnapshot integration
# ---------------------------------------------------------------------------


class TestGEXInRegimeSnapshot:
    """Verify GEX fields appear in MarketRegimeSnapshot."""

    def test_gex_fields_exist(self) -> None:
        from src.core.domain import (
            LiquidityRegime,
            RiskRegime,
            SessionRegime,
            TrendRegime,
            VolRegime,
        )

        snapshot = MarketRegimeSnapshot(
            time=datetime(2026, 3, 12, 10, 30, tzinfo=timezone.utc),
            index_symbol="SPY",
            vol_symbol="VXX",
            trend=TrendRegime.UP,
            vol=VolRegime.NORMAL,
            liquidity=LiquidityRegime.GOOD,
            risk=RiskRegime.NEUTRAL,
            session=SessionRegime.MIDDAY,
            gex_regime="positive",
            net_gex=1_500_000.0,
            gamma_flip_distance=0.02,
        )

        assert snapshot.gex_regime == "positive"
        assert snapshot.net_gex == 1_500_000.0
        assert snapshot.gamma_flip_distance == 0.02
        assert snapshot.model_version == "2.2"

    def test_gex_fields_default_none(self) -> None:
        from src.core.domain import (
            LiquidityRegime,
            RiskRegime,
            SessionRegime,
            TrendRegime,
            VolRegime,
        )

        snapshot = MarketRegimeSnapshot(
            time=datetime(2026, 3, 12, 10, 30, tzinfo=timezone.utc),
            index_symbol="SPY",
            vol_symbol=None,
            trend=TrendRegime.FLAT,
            vol=VolRegime.NORMAL,
            liquidity=LiquidityRegime.GOOD,
            risk=RiskRegime.NEUTRAL,
            session=SessionRegime.MIDDAY,
        )

        assert snapshot.gex_regime is None
        assert snapshot.net_gex == 0.0
        assert snapshot.gamma_flip_distance == 0.0


# ---------------------------------------------------------------------------
# MarketContextService integration
# ---------------------------------------------------------------------------


class TestRegimeIntegration:
    """MarketContextService.update_gex flows through to snapshot."""

    def _build_service(self):
        from src.analysis.regime import MarketContextService

        return MarketContextService(window=30, min_bars=5)

    def _feed_bars(self, svc, n: int = 25, price: float = 100.0):
        """Feed enough bars so the service produces a real snapshot."""
        for i in range(n):
            bar = Bar(
                symbol="SPY",
                time=datetime(2026, 3, 12, 10, 30 + i, tzinfo=timezone.utc),
                open=price,
                high=price * 1.002,
                low=price * 0.998,
                close=price + i * 0.01,
                volume=1_000_000.0,
            )
            svc.update(bar)

    def test_update_gex_flows_to_snapshot(self) -> None:
        svc = self._build_service()
        self._feed_bars(svc)

        # Update GEX with call-heavy chain
        chain = _make_chain(
            strikes=[95, 100, 105],
            call_ois=[5000, 10000, 3000],
            put_ois=[500, 200, 100],
            call_gammas=[0.04, 0.05, 0.03],
            put_gammas=[0.04, 0.05, 0.03],
        )
        svc.update_gex(current_price=100.0, chain_data=chain)

        # Trigger a new snapshot
        bar = Bar(
            symbol="SPY",
            time=datetime(2026, 3, 12, 11, 0, tzinfo=timezone.utc),
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.25,
            volume=1_000_000.0,
        )
        snapshot = svc.update(bar)

        assert snapshot.gex_regime == "positive"
        assert snapshot.net_gex > 0
        assert isinstance(snapshot.gamma_flip_distance, float)

    def test_update_gex_normalized_flows_to_snapshot(self) -> None:
        svc = self._build_service()
        self._feed_bars(svc)

        gex_data = {"call_gamma": 50.0, "put_gamma": 500.0}
        svc.update_gex_normalized(current_price=100.0, gex_data=gex_data)

        bar = Bar(
            symbol="SPY",
            time=datetime(2026, 3, 12, 11, 0, tzinfo=timezone.utc),
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.25,
            volume=1_000_000.0,
        )
        snapshot = svc.update(bar)

        assert snapshot.gex_regime == "negative"
        assert snapshot.net_gex < 0

    def test_no_gex_update_defaults(self) -> None:
        """Without calling update_gex, snapshot GEX fields should be defaults."""
        svc = self._build_service()
        self._feed_bars(svc)

        bar = Bar(
            symbol="SPY",
            time=datetime(2026, 3, 12, 11, 0, tzinfo=timezone.utc),
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.25,
            volume=1_000_000.0,
        )
        snapshot = svc.update(bar)

        assert snapshot.gex_regime is None
        assert snapshot.net_gex == 0.0
        assert snapshot.gamma_flip_distance == 0.0
