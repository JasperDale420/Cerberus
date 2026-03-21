"""Pre-compute multi-timeframe indicators using Numba-accelerated batch functions.

Used by the backtest runner to avoid recomputing indicators from scratch
on every bar.  The pre-computed arrays are injected into a module-level
cache that ``MultiTimeframeAnalyzer`` reads before falling back to its
default ``_compute_tf()`` path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.indicators_fast import (
    batch_adx,
    batch_atr,
    batch_bb,
    batch_ema,
    batch_rsi,
    batch_vwap_distance,
)

# ---------------------------------------------------------------------------
# Public: single-timeframe batch compute
# ---------------------------------------------------------------------------


def compute_tf_indicators(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    vwaps: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute all standard indicators for one timeframe's OHLCV arrays.

    *timestamps* (datetime64[ns] or int64 UTC nanos) enables daily VWAP
    reset.  When ``None``, VWAP accumulates across all bars.

    Returns a dict keyed by indicator label → numpy array of same length.
    """
    result: dict[str, np.ndarray] = {}

    # EMAs
    for p in (9, 20, 50, 200):
        result[f"ema_{p}"] = batch_ema(closes, p)

    # RSI
    for p in (2, 14):
        result[f"rsi_{p}"] = batch_rsi(closes, p)

    # ATR & ADX
    result["atr_14"] = batch_atr(highs, lows, closes, 14)
    result["adx_14"] = batch_adx(highs, lows, closes, 14)

    # Bollinger Bands
    bb_mean, bb_std = batch_bb(closes, 20)
    result["bb_mean_20"] = bb_mean
    result["bb_std_20"] = bb_std

    # VWAP distance (resets daily when timestamps are available)
    result["vwap_distance"] = batch_vwap_distance(highs, lows, closes, volumes, vwaps, timestamps)

    return result


# ---------------------------------------------------------------------------
# Public: aggregate 1m bars into higher timeframes
# ---------------------------------------------------------------------------


def aggregate_bars(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    vwaps: np.ndarray,
    tf_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate 1-min OHLCV into higher-TF bars using pandas resample.

    Returns (timestamps, opens, highs, lows, closes, volumes, vwaps) as numpy arrays.
    Each aggregated bar corresponds to one completed TF-bin.
    Also returns a mapping from 1m bar index → aggregated bar index so
    we can look up the "current" higher-TF indicator value at each 1m bar.
    """
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "vwap": vwaps,
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    rule = f"{tf_minutes}min"
    # Compute price*volume for proper volume-weighted VWAP aggregation
    df["_pv"] = df["vwap"] * df["volume"]
    agg = (
        df.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "_pv": "sum",
            }
        )
        .dropna(subset=["open"])
    )
    agg["vwap"] = np.where(agg["volume"] > 0, agg["_pv"] / agg["volume"], agg["close"])
    agg = agg.drop(columns=["_pv"])

    return (
        agg.index.values,
        agg["open"].values.astype(np.float64),
        agg["high"].values.astype(np.float64),
        agg["low"].values.astype(np.float64),
        agg["close"].values.astype(np.float64),
        agg["volume"].values.astype(np.float64),
        agg["vwap"].values.astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Public: full pre-computation for one symbol
# ---------------------------------------------------------------------------


def precompute_symbol_indicators(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    vwaps: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Pre-compute indicators for all three timeframes for a single symbol.

    Args:
        1-min OHLCV arrays (all same length N).

    Returns:
        ``{"1m": {...}, "5m": {...}, "15m": {...}}`` where each value is
        the output of ``compute_tf_indicators()``.

    For 5m/15m, bars are aggregated first, and a ``_1m_index`` array is
    included that maps each 1-min bar index → the latest completed
    higher-TF bar index.  This allows O(1) lookup during replay.
    """
    result: dict[str, dict[str, np.ndarray]] = {}

    # 1m indicators (same length as input)
    result["1m"] = compute_tf_indicators(highs, lows, closes, volumes, vwaps, timestamps)

    # Higher TFs
    for tf_minutes, tf_label in ((5, "5m"), (15, "15m")):
        agg_ts, agg_o, agg_h, agg_l, agg_c, agg_v, agg_vw = aggregate_bars(
            timestamps,
            opens,
            highs,
            lows,
            closes,
            volumes,
            vwaps,
            tf_minutes,
        )

        if len(agg_c) == 0:
            # Not enough data for this timeframe
            result[tf_label] = {}
            continue

        tf_indicators = compute_tf_indicators(agg_h, agg_l, agg_c, agg_v, agg_vw, agg_ts)

        # Build a mapping: for each 1m bar index, what is the latest completed
        # higher-TF bar index?  We use searchsorted on the aggregated timestamps.
        ts_1m = pd.to_datetime(timestamps, utc=True)
        ts_agg = pd.DatetimeIndex(agg_ts)
        # Ensure both have the same tz-awareness
        if ts_agg.tz is None and ts_1m.tz is not None:
            ts_agg = ts_agg.tz_localize("UTC")
        elif ts_agg.tz is not None and ts_1m.tz is None:
            ts_1m = ts_1m.tz_localize("UTC")

        # searchsorted: for each 1m timestamp, find the insertion point
        # in the aggregated timestamps.  Subtract 2 to get the last
        # *fully completed* higher-TF bar:
        #   -1 because searchsorted returns the insertion point
        #   -1 more to skip the currently-forming bar
        # Example with 5m bars [9:30, 9:35, 9:40]:
        #   At 9:33 → right=1, -2 = -1 → clipped to 0 (warmup, no completed bar yet)
        #   At 9:35 → right=2, -2 = 0  → 9:30 bar (just completed) ✓
        #   At 9:38 → right=2, -2 = 0  → 9:30 bar (last completed) ✓
        #   At 9:40 → right=3, -2 = 1  → 9:35 bar (just completed) ✓
        # This eliminates look-ahead bias: we never read indicators from
        # a bar whose OHLCV includes future 1m data.
        indices = ts_agg.searchsorted(ts_1m, side="right") - 2
        # -1 means no completed higher-TF bar yet; leave as -1 so MTF returns None
        indices = np.clip(indices, -1, len(agg_c) - 1)

        tf_indicators["_1m_index"] = indices.astype(np.int64)
        result[tf_label] = tf_indicators

    return result


# ---------------------------------------------------------------------------
# Module-level cache for MTF fast-path
# ---------------------------------------------------------------------------

# Keyed by (symbol, tf_label) → dict with "bar_count", "indicators"
_precomputed_cache: dict[tuple[str, str], dict] = {}


def install_precomputed(symbol: str, precomputed: dict[str, dict[str, np.ndarray]]) -> None:
    """Install pre-computed indicator arrays for a symbol into the module cache.

    Called once per symbol before the bar replay loop.
    ``MultiTimeframeAnalyzer._compute_tf_fast()`` reads from this cache.
    """
    for tf_label, indicators in precomputed.items():
        _precomputed_cache[(symbol, tf_label)] = indicators


def get_precomputed(symbol: str, tf_label: str) -> dict[str, np.ndarray] | None:
    """Get pre-computed indicator arrays for a symbol + timeframe, or None."""
    return _precomputed_cache.get((symbol, tf_label))


def clear_precomputed() -> None:
    """Clear all pre-computed indicator caches (call between backtest runs)."""
    _precomputed_cache.clear()
