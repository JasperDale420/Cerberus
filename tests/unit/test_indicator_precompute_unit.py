"""Test indicator pre-computation module."""

import numpy as np
import pytest

from src.backtest.indicator_precompute import (
    aggregate_bars,
    clear_precomputed,
    compute_tf_indicators,
    get_precomputed,
    install_precomputed,
    precompute_symbol_indicators,
)


def _make_1m_data(n_bars: int = 100, seed: int = 42):
    """Generate mock 1-minute OHLCV data."""
    rng = np.random.RandomState(seed)
    import pandas as pd

    timestamps = pd.date_range("2024-01-03 09:30", periods=n_bars, freq="1min", tz="UTC")
    closes = 100.0 + np.cumsum(rng.randn(n_bars) * 0.1)
    highs = closes + rng.uniform(0.05, 0.3, n_bars)
    lows = closes - rng.uniform(0.05, 0.3, n_bars)
    opens = closes + rng.randn(n_bars) * 0.05
    volumes = rng.uniform(500, 3000, n_bars)
    vwaps = closes + rng.randn(n_bars) * 0.02

    return timestamps.values, opens, highs, lows, closes, volumes, vwaps


@pytest.mark.unit
class TestComputeTFIndicators:
    def test_returns_all_indicators(self):
        _, _, highs, lows, closes, volumes, vwaps = _make_1m_data(100)
        result = compute_tf_indicators(highs, lows, closes, volumes, vwaps)

        expected_keys = {
            "ema_9",
            "ema_20",
            "ema_50",
            "ema_200",
            "rsi_2",
            "rsi_14",
            "atr_14",
            "adx_14",
            "bb_mean_20",
            "bb_std_20",
            "vwap_distance",
        }
        assert expected_keys == set(result.keys())

    def test_all_arrays_same_length(self):
        n = 50
        _, _, highs, lows, closes, volumes, vwaps = _make_1m_data(n)
        result = compute_tf_indicators(highs, lows, closes, volumes, vwaps)
        for key, arr in result.items():
            assert len(arr) == n, f"{key} length mismatch"

    def test_empty_input(self):
        empty = np.array([], dtype=np.float64)
        result = compute_tf_indicators(empty, empty, empty, empty, empty)
        for _key, arr in result.items():
            assert len(arr) == 0


@pytest.mark.unit
class TestAggregateBars:
    def test_5m_aggregation(self):
        ts, opens, highs, lows, closes, volumes, vwaps = _make_1m_data(100)
        agg_ts, agg_o, agg_h, agg_l, agg_c, agg_v, agg_vw = aggregate_bars(
            ts,
            opens,
            highs,
            lows,
            closes,
            volumes,
            vwaps,
            5,
        )
        # 100 1-min bars → 20 5-min bars
        assert len(agg_c) == 20
        # High of aggregated bar >= any constituent high
        assert agg_h[0] >= max(highs[:5])

    def test_15m_aggregation(self):
        ts, opens, highs, lows, closes, volumes, vwaps = _make_1m_data(150)
        agg_ts, agg_o, agg_h, agg_l, agg_c, agg_v, agg_vw = aggregate_bars(
            ts,
            opens,
            highs,
            lows,
            closes,
            volumes,
            vwaps,
            15,
        )
        assert len(agg_c) == 10  # 150 / 15


@pytest.mark.unit
class TestPrecomputeSymbolIndicators:
    def test_returns_all_timeframes(self):
        ts, opens, highs, lows, closes, volumes, vwaps = _make_1m_data(200)
        result = precompute_symbol_indicators(ts, opens, highs, lows, closes, volumes, vwaps)
        assert "1m" in result
        assert "5m" in result
        assert "15m" in result

    def test_1m_arrays_match_input_length(self):
        n = 200
        ts, opens, highs, lows, closes, volumes, vwaps = _make_1m_data(n)
        result = precompute_symbol_indicators(ts, opens, highs, lows, closes, volumes, vwaps)
        for key, arr in result["1m"].items():
            assert len(arr) == n, f"1m/{key} length mismatch"

    def test_higher_tf_has_index_mapping(self):
        ts, opens, highs, lows, closes, volumes, vwaps = _make_1m_data(200)
        result = precompute_symbol_indicators(ts, opens, highs, lows, closes, volumes, vwaps)
        assert "_1m_index" in result["5m"]
        assert "_1m_index" in result["15m"]
        # The 1m index mapping should have same length as input
        assert len(result["5m"]["_1m_index"]) == 200


@pytest.mark.unit
class TestModuleCache:
    def test_install_and_get(self):
        clear_precomputed()
        data = {"ema_20": np.array([1.0, 2.0, 3.0])}
        install_precomputed("SPY", {"1m": data})
        got = get_precomputed("SPY", "1m")
        assert got is not None
        assert np.array_equal(got["ema_20"], data["ema_20"])

    def test_get_missing_returns_none(self):
        clear_precomputed()
        assert get_precomputed("MISSING", "1m") is None

    def test_clear(self):
        install_precomputed("SPY", {"1m": {"ema_20": np.array([1.0])}})
        clear_precomputed()
        assert get_precomputed("SPY", "1m") is None
