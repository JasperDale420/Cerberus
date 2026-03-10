"""Test Numba batch indicators match Rolling* class outputs."""

import numpy as np
import pytest

from src.core.indicators import (
    RollingADX,
    RollingATR,
    RollingEMA,
    RollingRSI,
    RollingSMA,
    RollingStd,
)
from src.core.indicators_fast import (
    batch_adx,
    batch_atr,
    batch_bb,
    batch_ema,
    batch_rsi,
    batch_sma,
    batch_vwap_distance,
)

# Tolerance for float64 comparison
TOL = 1e-10


def _make_prices(n: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic-ish OHLCV arrays for testing."""
    rng = np.random.RandomState(seed)
    closes = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    highs = closes + rng.uniform(0.1, 1.0, n)
    lows = closes - rng.uniform(0.1, 1.0, n)
    opens = closes + rng.randn(n) * 0.2
    volumes = rng.uniform(500, 5000, n)
    return opens, highs, lows, closes, volumes


@pytest.mark.unit
class TestBatchEMA:
    def test_matches_rolling_ema(self):
        _, _, _, closes, _ = _make_prices()
        for period in (9, 20, 50, 200):
            batch_result = batch_ema(closes, period)
            rolling = RollingEMA.from_period(period)
            for i, c in enumerate(closes):
                rolling.update(c)
                assert batch_result[i] == pytest.approx(rolling.value, abs=TOL), (
                    f"EMA({period}) mismatch at index {i}"
                )

    def test_empty_array(self):
        result = batch_ema(np.array([], dtype=np.float64), 20)
        assert len(result) == 0

    def test_single_element(self):
        result = batch_ema(np.array([42.0], dtype=np.float64), 20)
        assert result[0] == pytest.approx(42.0)


@pytest.mark.unit
class TestBatchRSI:
    def test_matches_rolling_rsi(self):
        _, _, _, closes, _ = _make_prices()
        for period in (2, 14):
            batch_result = batch_rsi(closes, period)
            rolling = RollingRSI(period=period)
            for i, c in enumerate(closes):
                rv = rolling.update(c)
                if rv is None:
                    assert np.isnan(batch_result[i]), f"RSI({period}) should be NaN at index {i}"
                else:
                    assert batch_result[i] == pytest.approx(rv, abs=TOL), (
                        f"RSI({period}) mismatch at index {i}"
                    )

    def test_all_same_price_gives_100(self):
        c = np.full(20, 100.0, dtype=np.float64)
        result = batch_rsi(c, 14)
        # After first bar, no change → avg_gain=0, avg_loss=0 → RSI=100
        for i in range(1, 20):
            assert result[i] == pytest.approx(100.0)


@pytest.mark.unit
class TestBatchATR:
    def test_matches_rolling_atr(self):
        _, highs, lows, closes, _ = _make_prices()
        batch_result = batch_atr(highs, lows, closes, 14)
        rolling = RollingATR(period=14)
        for i in range(len(closes)):
            rv = rolling.update(float(highs[i]), float(lows[i]), float(closes[i]))
            if rv is None:
                assert np.isnan(batch_result[i])
            else:
                assert batch_result[i] == pytest.approx(rv, abs=TOL), (
                    f"ATR mismatch at index {i}"
                )


@pytest.mark.unit
class TestBatchADX:
    def test_matches_rolling_adx(self):
        _, highs, lows, closes, _ = _make_prices()
        batch_result = batch_adx(highs, lows, closes, 14)
        rolling = RollingADX(period=14)
        for i in range(len(closes)):
            rv = rolling.update(float(highs[i]), float(lows[i]), float(closes[i]))
            if rv is None:
                assert np.isnan(batch_result[i])
            else:
                assert batch_result[i] == pytest.approx(rv, abs=TOL), (
                    f"ADX mismatch at index {i}"
                )


@pytest.mark.unit
class TestBatchBB:
    def test_matches_rolling_std(self):
        _, _, _, closes, _ = _make_prices()
        batch_means, batch_stds = batch_bb(closes, 20)
        rolling = RollingStd.create(20)
        for i, c in enumerate(closes):
            mean, std = rolling.update(c)
            assert batch_means[i] == pytest.approx(mean, abs=TOL), (
                f"BB mean mismatch at index {i}"
            )
            assert batch_stds[i] == pytest.approx(std, abs=TOL), (
                f"BB std mismatch at index {i}"
            )


@pytest.mark.unit
class TestBatchSMA:
    def test_matches_rolling_sma(self):
        _, _, _, closes, _ = _make_prices()
        for period in (5, 20):
            batch_result = batch_sma(closes, period)
            rolling = RollingSMA.create(period)
            for i, c in enumerate(closes):
                rv = rolling.update(c)
                assert batch_result[i] == pytest.approx(rv, abs=TOL), (
                    f"SMA({period}) mismatch at index {i}"
                )


@pytest.mark.unit
class TestBatchVWAPDistance:
    def test_basic_computation(self):
        n = 50
        _, highs, lows, closes, volumes = _make_prices(n)
        vwaps = closes.copy()  # Use close as vwap
        result = batch_vwap_distance(highs, lows, closes, volumes, vwaps)
        assert len(result) == n
        # When vwap == close, distance should be small (due to volume weighting)
        # But not exactly zero since cumulative average differs from pointwise
        for i in range(n):
            assert not np.isnan(result[i])

    def test_zero_volume_gives_nan(self):
        c = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        h = c + 0.5
        low = c - 0.5
        v = np.zeros(3, dtype=np.float64)
        vw = c.copy()
        result = batch_vwap_distance(h, low, c, v, vw)
        # zero volume → NaN
        for i in range(3):
            assert np.isnan(result[i])
