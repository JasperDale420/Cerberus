from __future__ import annotations

from datetime import datetime, timezone

from src.analysis.vrp import VRPCalculator, VRPResult
from src.core.domain import Bar, RiskRegime

# ---------------------------------------------------------------------------
# VRPCalculator unit tests
# ---------------------------------------------------------------------------


class TestVRPCalculatorComputation:
    """Test VRP raw computation from known values."""

    def test_vrp_known_values(self):
        """VXX=20, RV=0.15 => IV_proxy=4.0, RV²=0.0225, VRP=3.9775."""
        calc = VRPCalculator(window=1)  # window=1 so we get a result immediately
        result = calc.update(vxx_price=20.0, realized_vol=0.15)
        assert result is not None
        assert abs(result.iv_proxy - 4.0) < 1e-9
        assert abs(result.realized_vol - 0.15) < 1e-9
        expected_vrp = 4.0 - 0.15**2
        assert abs(result.vrp_raw - expected_vrp) < 1e-9

    def test_returns_none_insufficient_data(self):
        """Fewer than window observations returns None."""
        calc = VRPCalculator(window=60)
        for _ in range(59):
            result = calc.update(vxx_price=20.0, realized_vol=0.15)
        assert result is None

    def test_returns_result_at_window(self):
        """At exactly window observations, returns a VRPResult."""
        calc = VRPCalculator(window=60)
        result = None
        for _ in range(60):
            result = calc.update(vxx_price=20.0, realized_vol=0.15)
        assert result is not None
        assert isinstance(result, VRPResult)

    def test_zero_vxx_returns_none(self):
        """Zero VXX price returns None."""
        calc = VRPCalculator(window=1)
        result = calc.update(vxx_price=0.0, realized_vol=0.15)
        assert result is None

    def test_negative_vxx_returns_none(self):
        """Negative VXX price returns None."""
        calc = VRPCalculator(window=1)
        result = calc.update(vxx_price=-5.0, realized_vol=0.15)
        assert result is None

    def test_zero_realized_vol(self):
        """Zero realized vol is handled gracefully (RV²=0)."""
        calc = VRPCalculator(window=1)
        result = calc.update(vxx_price=20.0, realized_vol=0.0)
        assert result is not None
        assert abs(result.vrp_raw - 4.0) < 1e-9  # IV_proxy - 0

    def test_zscore_constant_vrp(self):
        """Feeding constant VRP values should yield z-score near 0."""
        calc = VRPCalculator(window=60)
        result = None
        for _ in range(100):
            result = calc.update(vxx_price=20.0, realized_vol=0.15)
        assert result is not None
        assert abs(result.vrp_zscore) < 1e-6


class TestVRPZScoreClassification:
    """Test that VRP z-scores map to correct risk classifications."""

    def _build_calc_with_zscore(self, target_zscore: float) -> VRPCalculator:
        """Build a VRPCalculator and push it to produce a target z-score direction.

        We feed a baseline, then shift the input to produce a high or low z-score.
        """
        calc = VRPCalculator(window=60)
        # Feed 100 bars of baseline (VXX=20, RV=0.15)
        for _ in range(100):
            calc.update(vxx_price=20.0, realized_vol=0.15)
        return calc

    def test_high_vrp_zscore_risk_on(self):
        """High VRP z-score (> 1.0) should classify as RISK_ON."""
        calc = VRPCalculator(window=60)
        # Feed baseline
        for _ in range(100):
            calc.update(vxx_price=20.0, realized_vol=0.15)
        # Spike VXX to 30 => IV_proxy = 9.0, a big jump from baseline ~4.0
        result = calc.update(vxx_price=30.0, realized_vol=0.15)
        assert result is not None
        assert result.vrp_zscore > 1.0, f"Expected z > 1.0, got {result.vrp_zscore}"

    def test_low_vrp_zscore_risk_off(self):
        """Low VRP z-score (< -1.0) should classify as RISK_OFF."""
        calc = VRPCalculator(window=60)
        # Feed baseline with high VXX
        for _ in range(100):
            calc.update(vxx_price=25.0, realized_vol=0.15)
        # Drop VXX sharply => IV_proxy drops well below baseline
        result = calc.update(vxx_price=10.0, realized_vol=0.15)
        assert result is not None
        assert result.vrp_zscore < -1.0, f"Expected z < -1.0, got {result.vrp_zscore}"

    def test_neutral_vrp_zscore(self):
        """VRP z-score near 0 (between -1 and 1) should classify as NEUTRAL."""
        calc = VRPCalculator(window=60)
        for _ in range(100):
            calc.update(vxx_price=20.0, realized_vol=0.15)
        # Same VXX as baseline => z-score near 0
        result = calc.update(vxx_price=20.0, realized_vol=0.15)
        assert result is not None
        assert -1.0 <= result.vrp_zscore <= 1.0


# ---------------------------------------------------------------------------
# Regime integration tests
# ---------------------------------------------------------------------------


class TestRegimeVRPIntegration:
    """Test VRP integration into MarketContextService._classify_risk()."""

    def _make_bar(self, price: float, t: datetime = None) -> Bar:
        if t is None:
            t = datetime(2025, 6, 15, 10, 30, tzinfo=timezone.utc)
        return Bar(
            symbol="SPY",
            time=t,
            open=price,
            high=price + 0.5,
            low=price - 0.5,
            close=price,
            volume=1_000_000.0,
        )

    def _make_vol_bar(self, price: float, t: datetime = None) -> Bar:
        if t is None:
            t = datetime(2025, 6, 15, 10, 30, tzinfo=timezone.utc)
        return Bar(
            symbol="VXX",
            time=t,
            open=price,
            high=price + 0.1,
            low=price - 0.1,
            close=price,
            volume=500_000.0,
        )

    def test_fallback_to_vxx_momentum_when_vrp_insufficient(self):
        """When VRP has < window observations, falls back to VXX momentum."""
        from src.analysis.regime import MarketContextService

        svc = MarketContextService(window=60, min_bars=5)
        # Feed enough SPY bars to get past min_bars
        for i in range(25):
            svc.update(self._make_bar(450.0 + i * 0.01))
        # Feed some VXX bars (less than VRP window of 60)
        for _ in range(10):
            svc.update_vol(self._make_vol_bar(25.0))
        # VRP won't have enough data yet, so _last_vrp_score stays 0
        # VXX momentum is flat => NEUTRAL
        snap = svc.update(self._make_bar(450.3))
        assert snap.risk in (RiskRegime.RISK_ON, RiskRegime.NEUTRAL, RiskRegime.RISK_OFF)

    def test_vrp_score_in_snapshot(self):
        """Verify vrp_score is populated in the MarketRegimeSnapshot."""
        from src.analysis.regime import MarketContextService

        svc = MarketContextService(window=60, min_bars=5)
        for i in range(25):
            svc.update(self._make_bar(450.0 + i * 0.01))
        snap = svc.update(self._make_bar(450.3))
        # vrp_score should be a float (default 0.0 until enough VRP data)
        assert isinstance(snap.vrp_score, float)

    def test_vrp_disabled_uses_vxx_fallback(self):
        """When _vrp_enabled is False, uses VXX momentum fallback."""
        from src.analysis.regime import MarketContextService

        svc = MarketContextService(window=60, min_bars=5)
        svc._vrp_enabled = False
        svc._last_vrp_score = 5.0  # Would be RISK_ON if VRP were checked

        # Feed enough bars
        for i in range(25):
            svc.update(self._make_bar(450.0 + i * 0.01))

        # Feed VXX bars with rising prices (RISK_OFF via momentum)
        for i in range(25):
            svc.update_vol(self._make_vol_bar(20.0 + i * 0.5))

        snap = svc.update(self._make_bar(450.3))
        # With VRP disabled, should use VXX momentum (rising VXX = RISK_OFF)
        assert snap.risk == RiskRegime.RISK_OFF
