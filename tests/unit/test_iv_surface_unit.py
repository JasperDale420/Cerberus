"""Unit tests for IV Surface Dynamics analyzer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis.iv_surface import IVSurfaceAnalyzer, IVSurfaceConfig, IVSurfaceResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain(call_iv: float, put_iv: float, call_delta: float = 0.25, put_delta: float = -0.25) -> list[dict]:
    """Build a minimal two-option chain with target deltas."""
    return [
        {"delta": call_delta, "iv": call_iv},
        {"delta": put_delta, "iv": put_iv},
    ]


def _make_term_data(near_iv: float, far_iv: float, near_dte: int = 7, far_dte: int = 60) -> list[dict]:
    """Build a two-expiration term structure."""
    return [
        {"dte": near_dte, "iv": near_iv},
        {"dte": far_dte, "iv": far_iv},
    ]


def _parabolic_call_prices(strikes: list[float], atm: float, base_price: float = 10.0) -> list[float]:
    """Generate synthetic call prices that produce a single-peaked RND.

    Uses a Gaussian-shaped call price curve C(K) = A * exp(-0.5*((K-atm)/sigma)^2)
    which is convex (positive second derivative) in the tails and produces a
    realistic bell-shaped risk-neutral density.
    """
    sigma = (strikes[-1] - strikes[0]) / 4.0
    return [base_price * math.exp(-0.5 * ((k - atm) / sigma) ** 2) for k in strikes]


def _bimodal_call_prices(strikes: list[float], peak1: float, peak2: float) -> list[float]:
    """Generate call prices whose second derivative has two peaks.

    Constructs a piecewise curve that produces two bumps in d2C/dK2.
    """
    prices = []
    for k in strikes:
        # Sum of two inverted Gaussians → two humps in second derivative
        c1 = 5.0 * math.exp(-0.5 * ((k - peak1) / 3.0) ** 2)
        c2 = 5.0 * math.exp(-0.5 * ((k - peak2) / 3.0) ** 2)
        prices.append(max(c1 + c2 + 0.5, 0.01))
    return prices


# ===========================================================================
# Skew Tests
# ===========================================================================


@pytest.mark.unit
class TestComputeSkew:
    """Tests for skew computation and z-score."""

    def test_known_skew_value(self) -> None:
        """25-delta put IV minus call IV gives correct risk reversal."""
        analyzer = IVSurfaceAnalyzer()
        chain = _make_chain(call_iv=0.20, put_iv=0.30)
        skew = analyzer.compute_skew(chain)
        assert skew == pytest.approx(0.10, abs=1e-9)

    def test_symmetric_iv_zero_skew(self) -> None:
        """Equal put and call IV yields zero skew."""
        analyzer = IVSurfaceAnalyzer()
        chain = _make_chain(call_iv=0.25, put_iv=0.25)
        skew = analyzer.compute_skew(chain)
        assert skew == pytest.approx(0.0, abs=1e-9)

    def test_negative_skew(self) -> None:
        """Call IV > put IV gives negative risk reversal."""
        analyzer = IVSurfaceAnalyzer()
        chain = _make_chain(call_iv=0.35, put_iv=0.20)
        skew = analyzer.compute_skew(chain)
        assert skew == pytest.approx(-0.15, abs=1e-9)

    def test_empty_chain_returns_zero(self) -> None:
        """Empty chain yields 0.0 skew."""
        analyzer = IVSurfaceAnalyzer()
        assert analyzer.compute_skew([]) == 0.0

    def test_missing_put_returns_zero(self) -> None:
        """Chain with only calls returns 0.0."""
        analyzer = IVSurfaceAnalyzer()
        chain = [{"delta": 0.25, "iv": 0.20}]
        assert analyzer.compute_skew(chain) == 0.0

    def test_zero_iv_ignored(self) -> None:
        """Options with iv <= 0 are skipped."""
        analyzer = IVSurfaceAnalyzer()
        chain = [
            {"delta": 0.25, "iv": 0.0},
            {"delta": -0.25, "iv": 0.30},
        ]
        assert analyzer.compute_skew(chain) == 0.0

    def test_picks_closest_delta(self) -> None:
        """When multiple calls exist, pick the one closest to target delta."""
        analyzer = IVSurfaceAnalyzer()
        chain = [
            {"delta": 0.50, "iv": 0.18},  # ATM call
            {"delta": 0.26, "iv": 0.22},  # closer to 0.25
            {"delta": 0.10, "iv": 0.28},  # wing
            {"delta": -0.24, "iv": 0.35},  # close put
            {"delta": -0.50, "iv": 0.40},  # ATM put
        ]
        skew = analyzer.compute_skew(chain)
        # Closest call: delta=0.26, iv=0.22. Closest put: delta=-0.24, iv=0.35
        assert skew == pytest.approx(0.35 - 0.22, abs=1e-9)


@pytest.mark.unit
class TestSkewZscore:
    """Tests for skew z-score and steepening detection."""

    def test_zscore_insufficient_data(self) -> None:
        """Fewer than 4 observations returns 0.0 (need prior history for z-score)."""
        analyzer = IVSurfaceAnalyzer()
        analyzer.compute_skew(_make_chain(0.20, 0.30))
        analyzer.compute_skew(_make_chain(0.20, 0.30))
        analyzer.compute_skew(_make_chain(0.20, 0.30))
        assert analyzer.skew_zscore() == 0.0

    def test_zscore_constant_history(self) -> None:
        """Constant skew history yields 0.0 z-score (zero std in prior)."""
        analyzer = IVSurfaceAnalyzer()
        for _ in range(5):
            analyzer.compute_skew(_make_chain(0.20, 0.30))
        assert analyzer.skew_zscore() == 0.0

    def test_zscore_spike_detected(self) -> None:
        """A large spike produces a positive z-score."""
        analyzer = IVSurfaceAnalyzer()
        # 10 observations with slight variation (so prior std > 0)
        for i in range(10):
            put_iv = 0.30 + 0.001 * (i % 3)
            analyzer.compute_skew(_make_chain(0.20, put_iv))
        # One big spike: skew = 0.40
        analyzer.compute_skew(_make_chain(0.10, 0.50))
        zscore = analyzer.skew_zscore()
        assert zscore > 2.0

    def test_steepening_flag(self) -> None:
        """is_skew_steepening returns True when z-score > threshold."""
        config = IVSurfaceConfig(skew_sigma_threshold=1.5)
        analyzer = IVSurfaceAnalyzer(config=config)
        for i in range(10):
            put_iv = 0.30 + 0.001 * (i % 3)
            analyzer.compute_skew(_make_chain(0.20, put_iv))
        analyzer.compute_skew(_make_chain(0.10, 0.50))
        assert analyzer.is_skew_steepening() is True

    def test_no_steepening_when_flat(self) -> None:
        """Stable skew does not trigger steepening."""
        analyzer = IVSurfaceAnalyzer()
        for _ in range(10):
            analyzer.compute_skew(_make_chain(0.20, 0.30))
        assert analyzer.is_skew_steepening() is False


# ===========================================================================
# Term Structure Tests
# ===========================================================================


@pytest.mark.unit
class TestTermStructure:
    """Tests for term structure slope and inversion."""

    def test_normal_contango(self) -> None:
        """Far IV > near IV is normal (positive slope, not inverted)."""
        analyzer = IVSurfaceAnalyzer()
        slope, inverted = analyzer.compute_term_structure(
            _make_term_data(near_iv=0.20, far_iv=0.25),
        )
        assert slope == pytest.approx(0.05, abs=1e-9)
        assert inverted is False

    def test_inverted_term_structure(self) -> None:
        """Near IV > far IV yields negative slope and inversion flag."""
        analyzer = IVSurfaceAnalyzer()
        slope, inverted = analyzer.compute_term_structure(
            _make_term_data(near_iv=0.35, far_iv=0.20),
        )
        assert slope == pytest.approx(-0.15, abs=1e-9)
        assert inverted is True

    def test_flat_term_structure(self) -> None:
        """Equal near/far IV yields zero slope and no inversion."""
        analyzer = IVSurfaceAnalyzer()
        slope, inverted = analyzer.compute_term_structure(
            _make_term_data(near_iv=0.25, far_iv=0.25),
        )
        assert slope == pytest.approx(0.0, abs=1e-9)
        assert inverted is False

    def test_single_expiration_returns_defaults(self) -> None:
        """Fewer than 2 expirations returns (0.0, False)."""
        analyzer = IVSurfaceAnalyzer()
        slope, inverted = analyzer.compute_term_structure([{"dte": 7, "iv": 0.20}])
        assert slope == 0.0
        assert inverted is False

    def test_empty_term_data(self) -> None:
        """Empty input returns defaults."""
        analyzer = IVSurfaceAnalyzer()
        slope, inverted = analyzer.compute_term_structure([])
        assert slope == 0.0
        assert inverted is False

    def test_zero_iv_returns_defaults(self) -> None:
        """Zero IV in any expiration returns defaults."""
        analyzer = IVSurfaceAnalyzer()
        slope, inverted = analyzer.compute_term_structure(
            [{"dte": 7, "iv": 0.0}, {"dte": 30, "iv": 0.25}],
        )
        assert slope == 0.0
        assert inverted is False

    def test_unsorted_input_handled(self) -> None:
        """Data provided in reverse dte order is still correctly sorted."""
        analyzer = IVSurfaceAnalyzer()
        data = [{"dte": 60, "iv": 0.25}, {"dte": 7, "iv": 0.20}]
        slope, inverted = analyzer.compute_term_structure(data)
        assert slope == pytest.approx(0.05, abs=1e-9)
        assert inverted is False

    def test_multiple_expirations(self) -> None:
        """Uses nearest and farthest from a multi-expiry list."""
        analyzer = IVSurfaceAnalyzer()
        data = [
            {"dte": 7, "iv": 0.22},
            {"dte": 30, "iv": 0.24},
            {"dte": 90, "iv": 0.28},
        ]
        slope, inverted = analyzer.compute_term_structure(data)
        # near = 0.22 (dte 7), far = 0.28 (dte 90)
        assert slope == pytest.approx(0.06, abs=1e-9)
        assert inverted is False


# ===========================================================================
# Risk-Neutral Density Tests
# ===========================================================================


@pytest.mark.unit
class TestComputeRND:
    """Tests for Breeden-Litzenberger risk-neutral density extraction."""

    def test_parabolic_prices_produce_positive_density(self) -> None:
        """Parabolic call prices should produce non-negative density values."""
        analyzer = IVSurfaceAnalyzer(config=IVSurfaceConfig(min_strikes_for_rnd=5))
        strikes = list(np.linspace(90, 110, 15))
        prices = _parabolic_call_prices(strikes, atm=100.0, base_price=10.0)
        d_strikes, d_values = analyzer.compute_rnd(strikes, prices)

        assert len(d_strikes) > 0
        assert len(d_values) == len(d_strikes)
        assert all(v >= 0 for v in d_values)

    def test_insufficient_strikes_returns_empty(self) -> None:
        """Fewer than min_strikes_for_rnd returns empty lists."""
        config = IVSurfaceConfig(min_strikes_for_rnd=10)
        analyzer = IVSurfaceAnalyzer(config=config)
        strikes = [95.0, 100.0, 105.0]
        prices = [6.0, 3.0, 1.0]
        d_strikes, d_values = analyzer.compute_rnd(strikes, prices)
        assert d_strikes == []
        assert d_values == []

    def test_mismatched_lengths_returns_empty(self) -> None:
        """Different length strikes/prices returns empty."""
        analyzer = IVSurfaceAnalyzer(config=IVSurfaceConfig(min_strikes_for_rnd=3))
        d_strikes, d_values = analyzer.compute_rnd([1.0, 2.0, 3.0], [1.0, 2.0])
        assert d_strikes == []

    def test_negative_density_clamped_to_zero(self) -> None:
        """Concave-up call prices (negative d2C/dK2) get clamped to 0."""
        analyzer = IVSurfaceAnalyzer(config=IVSurfaceConfig(min_strikes_for_rnd=3))
        strikes = list(np.linspace(90, 110, 11))
        # Concave-up: prices accelerate upward (opposite of normal)
        prices = [(k - 90) ** 2 * 0.01 + 0.5 for k in strikes]
        d_strikes, d_values = analyzer.compute_rnd(strikes, prices)
        # Some values will be positive due to the parabolic shape, but
        # the key invariant is no negatives
        assert all(v >= 0 for v in d_values)

    def test_discount_factor_applied(self) -> None:
        """Density scales with e^{rT}."""
        analyzer = IVSurfaceAnalyzer(config=IVSurfaceConfig(min_strikes_for_rnd=5))
        strikes = list(np.linspace(90, 110, 15))
        prices = _parabolic_call_prices(strikes, atm=100.0)

        _, d_r0 = analyzer.compute_rnd(strikes, prices, rate=0.0, tte=1.0)
        _, d_r5 = analyzer.compute_rnd(strikes, prices, rate=0.05, tte=1.0)

        # Find a non-zero interior density for comparison
        assert len(d_r0) > 0 and len(d_r5) > 0
        idx = int(np.argmax(d_r0))
        assert d_r0[idx] > 1e-15, "Need a positive density value to test scaling"
        ratio = d_r5[idx] / d_r0[idx]
        assert ratio == pytest.approx(math.exp(0.05), rel=0.01)


# ===========================================================================
# Bimodality Detection Tests
# ===========================================================================


@pytest.mark.unit
class TestBimodalityDetection:
    """Tests for peak detection in the risk-neutral density."""

    def test_single_peak(self) -> None:
        """Unimodal density returns is_bimodal=False."""
        analyzer = IVSurfaceAnalyzer()
        # Single hump: [0, 0.1, 0.3, 0.5, 0.3, 0.1, 0]
        density = [0.0, 0.1, 0.3, 0.5, 0.3, 0.1, 0.0]
        is_bimodal, peaks = analyzer.detect_bimodality(density)
        assert is_bimodal is False
        assert len(peaks) == 1

    def test_two_peaks(self) -> None:
        """Bimodal density with two clear humps."""
        analyzer = IVSurfaceAnalyzer(config=IVSurfaceConfig(peak_prominence=0.1))
        density = [0.0, 0.1, 0.5, 0.2, 0.1, 0.2, 0.5, 0.1, 0.0]
        is_bimodal, peaks = analyzer.detect_bimodality(density)
        assert is_bimodal is True
        assert len(peaks) == 2

    def test_flat_density_no_peaks(self) -> None:
        """Flat density has no peaks."""
        analyzer = IVSurfaceAnalyzer()
        density = [0.1, 0.1, 0.1, 0.1, 0.1]
        is_bimodal, peaks = analyzer.detect_bimodality(density)
        assert is_bimodal is False
        assert len(peaks) == 0

    def test_zero_density_no_peaks(self) -> None:
        """All-zero density returns no peaks."""
        analyzer = IVSurfaceAnalyzer()
        density = [0.0, 0.0, 0.0, 0.0, 0.0]
        is_bimodal, peaks = analyzer.detect_bimodality(density)
        assert is_bimodal is False
        assert len(peaks) == 0

    def test_too_short_returns_false(self) -> None:
        """Fewer than 3 values can't form a peak."""
        analyzer = IVSurfaceAnalyzer()
        is_bimodal, peaks = analyzer.detect_bimodality([0.5, 0.8])
        assert is_bimodal is False
        assert peaks == []

    def test_peaks_below_prominence_ignored(self) -> None:
        """Small peaks below prominence threshold are not counted."""
        config = IVSurfaceConfig(peak_prominence=0.5)
        analyzer = IVSurfaceAnalyzer(config=config)
        # Max is 1.0; threshold = 0.5. Second peak at 0.3 is below threshold.
        density = [0.0, 0.1, 1.0, 0.1, 0.0, 0.1, 0.3, 0.1, 0.0]
        is_bimodal, peaks = analyzer.detect_bimodality(density)
        assert is_bimodal is False
        assert len(peaks) == 1


# ===========================================================================
# Composite Analyze Tests
# ===========================================================================


@pytest.mark.unit
class TestAnalyze:
    """Tests for the composite analyze() method."""

    def _build_inputs(
        self,
        call_iv: float = 0.20,
        put_iv: float = 0.30,
        near_iv: float = 0.20,
        far_iv: float = 0.25,
        n_strikes: int = 15,
    ) -> dict:
        """Build a standard set of analyze() inputs."""
        strikes = list(np.linspace(90, 110, n_strikes))
        call_prices = _parabolic_call_prices(strikes, atm=100.0)
        return {
            "chain": _make_chain(call_iv, put_iv),
            "term_data": _make_term_data(near_iv, far_iv),
            "strikes": strikes,
            "call_prices": call_prices,
        }

    def test_returns_iv_surface_result(self) -> None:
        """analyze() returns an IVSurfaceResult."""
        analyzer = IVSurfaceAnalyzer(config=IVSurfaceConfig(min_strikes_for_rnd=5))
        # Seed skew history so z-score is computable
        for _ in range(5):
            analyzer.compute_skew(_make_chain(0.20, 0.30))

        inputs = self._build_inputs()
        result = analyzer.analyze(**inputs)
        assert isinstance(result, IVSurfaceResult)

    def test_bearish_signal_on_steep_skew_and_inversion(self) -> None:
        """Steep skew + inverted term structure should produce negative signal."""
        config = IVSurfaceConfig(skew_sigma_threshold=1.5, min_strikes_for_rnd=5)
        analyzer = IVSurfaceAnalyzer(config=config)

        # Build stable skew history with slight variation (so prior std > 0)
        for i in range(15):
            put_iv = 0.30 + 0.001 * (i % 3)
            analyzer.compute_skew(_make_chain(0.20, put_iv))

        # Now spike skew + inversion
        inputs = self._build_inputs(call_iv=0.10, put_iv=0.55, near_iv=0.35, far_iv=0.20)
        result = analyzer.analyze(**inputs)

        assert result.signal_strength < 0
        assert result.is_inverted is True
        assert result.skew_zscore > 0

    def test_neutral_signal_on_normal_surface(self) -> None:
        """Normal surface conditions produce near-zero signal."""
        config = IVSurfaceConfig(min_strikes_for_rnd=5)
        analyzer = IVSurfaceAnalyzer(config=config)

        # Consistent history
        for _ in range(10):
            analyzer.compute_skew(_make_chain(0.20, 0.30))

        inputs = self._build_inputs()
        result = analyzer.analyze(**inputs)

        # Signal should be close to zero for normal conditions
        assert abs(result.signal_strength) < 0.5
        assert result.is_inverted is False

    def test_signal_strength_clamped(self) -> None:
        """Signal strength is bounded to [-1, 1]."""
        config = IVSurfaceConfig(min_strikes_for_rnd=5)
        analyzer = IVSurfaceAnalyzer(config=config)

        for _ in range(10):
            analyzer.compute_skew(_make_chain(0.20, 0.30))

        inputs = self._build_inputs()
        result = analyzer.analyze(**inputs)

        assert -1.0 <= result.signal_strength <= 1.0

    def test_rnd_peaks_populated_for_bimodal(self) -> None:
        """When call prices produce bimodal density, rnd_peaks has entries."""
        config = IVSurfaceConfig(min_strikes_for_rnd=5, peak_prominence=0.05)
        analyzer = IVSurfaceAnalyzer(config=config)

        for _ in range(5):
            analyzer.compute_skew(_make_chain(0.20, 0.30))

        strikes = list(np.linspace(80, 120, 25))
        call_prices = _bimodal_call_prices(strikes, peak1=92.0, peak2=108.0)

        result = analyzer.analyze(
            chain=_make_chain(0.20, 0.30),
            term_data=_make_term_data(0.20, 0.25),
            strikes=strikes,
            call_prices=call_prices,
        )

        assert result.is_bimodal is True
        assert len(result.rnd_peaks) >= 2

    def test_bimodal_contributes_bearish(self) -> None:
        """Bimodality adds a bearish component to signal strength."""
        config = IVSurfaceConfig(min_strikes_for_rnd=5, peak_prominence=0.05)
        analyzer = IVSurfaceAnalyzer(config=config)

        for _ in range(10):
            analyzer.compute_skew(_make_chain(0.20, 0.30))

        strikes = list(np.linspace(80, 120, 25))

        # Unimodal
        uni_prices = _parabolic_call_prices(strikes, atm=100.0)
        result_uni = analyzer.analyze(
            chain=_make_chain(0.20, 0.30),
            term_data=_make_term_data(0.20, 0.25),
            strikes=strikes,
            call_prices=uni_prices,
        )

        # Bimodal
        bi_prices = _bimodal_call_prices(strikes, peak1=90.0, peak2=110.0)
        result_bi = analyzer.analyze(
            chain=_make_chain(0.20, 0.30),
            term_data=_make_term_data(0.20, 0.25),
            strikes=strikes,
            call_prices=bi_prices,
        )

        # Bimodal result should be more bearish (lower signal)
        if result_bi.is_bimodal:
            assert result_bi.signal_strength <= result_uni.signal_strength


# ===========================================================================
# Edge Cases
# ===========================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Additional edge case coverage."""

    def test_config_defaults(self) -> None:
        """Default config values match spec."""
        config = IVSurfaceConfig()
        assert config.enabled is False
        assert config.skew_lookback == 20
        assert config.skew_sigma_threshold == 2.0
        assert config.min_strikes_for_rnd == 10
        assert config.peak_prominence == 0.1

    def test_result_frozen(self) -> None:
        """IVSurfaceResult is immutable."""
        result = IVSurfaceResult(skew_zscore=1.0)
        with pytest.raises(AttributeError):
            result.skew_zscore = 2.0  # type: ignore[misc]

    def test_analyze_with_empty_chain(self) -> None:
        """analyze() handles empty chain gracefully."""
        config = IVSurfaceConfig(min_strikes_for_rnd=5)
        analyzer = IVSurfaceAnalyzer(config=config)
        strikes = list(np.linspace(90, 110, 15))
        prices = _parabolic_call_prices(strikes, atm=100.0)

        result = analyzer.analyze(
            chain=[],
            term_data=_make_term_data(0.20, 0.25),
            strikes=strikes,
            call_prices=prices,
        )
        assert result.skew_zscore == 0.0

    def test_analyze_with_insufficient_rnd_data(self) -> None:
        """analyze() with too few strikes still returns a valid result."""
        config = IVSurfaceConfig(min_strikes_for_rnd=100)
        analyzer = IVSurfaceAnalyzer(config=config)

        result = analyzer.analyze(
            chain=_make_chain(0.20, 0.30),
            term_data=_make_term_data(0.20, 0.25),
            strikes=[100.0, 105.0],
            call_prices=[5.0, 2.0],
        )
        assert result.is_bimodal is False
        assert result.rnd_peaks == []

    def test_skew_history_respects_lookback(self) -> None:
        """Skew history deque respects configured max length."""
        config = IVSurfaceConfig(skew_lookback=5)
        analyzer = IVSurfaceAnalyzer(config=config)
        for i in range(10):
            analyzer.compute_skew(_make_chain(0.20, 0.25 + 0.01 * i))
        assert len(analyzer._skew_history) == 5

    def test_logger_does_not_crash(self) -> None:
        """Passing a logger does not cause errors."""
        from unittest.mock import MagicMock

        mock_logger = MagicMock(spec=["debug", "info", "warning", "error"])
        analyzer = IVSurfaceAnalyzer(logger=mock_logger, config=IVSurfaceConfig(min_strikes_for_rnd=5))

        for _ in range(5):
            analyzer.compute_skew(_make_chain(0.20, 0.30))

        strikes = list(np.linspace(90, 110, 15))
        prices = _parabolic_call_prices(strikes, atm=100.0)

        result = analyzer.analyze(
            chain=_make_chain(0.20, 0.30),
            term_data=_make_term_data(0.20, 0.25),
            strikes=strikes,
            call_prices=prices,
        )
        assert isinstance(result, IVSurfaceResult)
        assert mock_logger.debug.called
