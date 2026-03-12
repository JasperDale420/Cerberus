"""Numba-compiled batch indicator functions for backtest acceleration.

Each function operates on contiguous numpy float64 arrays and returns
arrays of the same length.  Numba ``@njit`` compiles them to native
ARM64 machine code on first call; the result is cached to disk for
subsequent runs.

These are **functional equivalents** of the ``Rolling*`` classes in
``indicators.py`` — they produce identical values but in a single
vectorized pass instead of per-bar Python calls.
"""

from __future__ import annotations

import numba
import numpy as np

# ---------------------------------------------------------------------------
# EMA  (Exponential Moving Average)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def batch_ema(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA over an array of close prices.

    Uses alpha = 2 / (period + 1).  Matches ``RollingEMA.from_period()``.
    Returns array of same length; first value is seeded with ``closes[0]``.
    """
    n = closes.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    alpha = 2.0 / (period + 1.0)
    out[0] = closes[0]
    for i in range(1, n):
        out[i] = alpha * closes[i] + (1.0 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# RSI  (Relative Strength Index)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def batch_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute RSI over an array of close prices.

    Uses Wilder smoothing.  Matches ``RollingRSI(period=N)``.
    Returns array of same length; index 0 is NaN (no previous bar).
    """
    n = closes.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    out[0] = np.nan  # No previous bar for first element
    if n == 1:
        return out

    # Seed with first change
    change = closes[1] - closes[0]
    avg_gain = max(0.0, change)
    avg_loss = max(0.0, -change)

    if abs(avg_loss) < 1e-9:
        out[1] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[1] = 100.0 - 100.0 / (1.0 + rs)

    p = max(1, period)
    for i in range(2, n):
        change = closes[i] - closes[i - 1]
        gain = max(0.0, change)
        loss = max(0.0, -change)

        avg_gain = ((avg_gain * (p - 1)) + gain) / p
        avg_loss = ((avg_loss * (p - 1)) + loss) / p

        if abs(avg_loss) < 1e-9:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)

    return out


# ---------------------------------------------------------------------------
# ATR  (Average True Range)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def batch_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    """Compute ATR using Wilder smoothing.  Matches ``RollingATR``."""
    n = highs.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    out[0] = np.nan  # Need previous close for TR
    if n == 1:
        return out

    p = max(1, period)
    # First TR seeds ATR
    tr = max(
        highs[1] - lows[1],
        abs(highs[1] - closes[0]),
        abs(lows[1] - closes[0]),
    )
    atr = tr
    out[1] = atr

    for i in range(2, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        count = i  # 1-based count of TR values (i-1 changes, but count matches RollingATR)
        if count < p:
            atr = ((atr * (count - 1)) + tr) / count
        else:
            atr = ((atr * (p - 1)) + tr) / p
        out[i] = atr

    return out


# ---------------------------------------------------------------------------
# ADX  (Average Directional Index)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def batch_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    """Compute ADX using Wilder smoothing.  Matches ``RollingADX``."""
    n = highs.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    out[0] = np.nan  # Need previous bar
    if n == 1:
        return out

    p = max(1, period)

    # First bar: seed smoothed values
    up_move = highs[1] - highs[0]
    down_move = lows[0] - lows[1]

    plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
    minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

    tr = max(
        highs[1] - lows[1],
        abs(highs[1] - closes[0]),
        abs(lows[1] - closes[0]),
    )

    smoothed_tr = tr
    smoothed_plus_dm = plus_dm
    smoothed_minus_dm = minus_dm

    # Compute DI and DX
    if smoothed_tr > 0:
        plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
        minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
    else:
        plus_di = 0.0
        minus_di = 0.0

    di_sum = plus_di + minus_di
    dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0
    smoothed_dx = dx
    out[1] = smoothed_dx

    for i in range(2, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

        # Wilder smoothing
        smoothed_tr = smoothed_tr - (smoothed_tr / p) + tr
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / p) + plus_dm
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / p) + minus_dm

        if smoothed_tr > 0:
            plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
        else:
            plus_di = 0.0
            minus_di = 0.0

        di_sum = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0
        smoothed_dx = ((smoothed_dx * (p - 1)) + dx) / p
        out[i] = smoothed_dx

    return out


# ---------------------------------------------------------------------------
# Bollinger Bands  (rolling mean + std)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def batch_bb(
    closes: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling mean and population std (Bollinger Band components).

    Matches ``RollingStd.create(period)`` — population variance, not sample.
    Returns (mean_arr, std_arr) of same length as input.
    """
    n = closes.shape[0]
    means = np.empty(n, dtype=np.float64)
    stds = np.empty(n, dtype=np.float64)

    if n == 0:
        return means, stds

    running_sum = 0.0
    running_sumsq = 0.0

    for i in range(n):
        if i >= period:
            old = closes[i - period]
            running_sum -= old
            running_sumsq -= old * old

        running_sum += closes[i]
        running_sumsq += closes[i] * closes[i]

        count = min(i + 1, period)
        mean = running_sum / count
        var = max(0.0, (running_sumsq / count) - (mean * mean))
        means[i] = mean
        stds[i] = var**0.5

    return means, stds


# ---------------------------------------------------------------------------
# VWAP distance
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _vwap_core(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    vwaps: np.ndarray,
    timestamps_ns: np.ndarray,
    reset_daily: bool,
) -> np.ndarray:
    """Inner VWAP kernel called by both overloads of batch_vwap_distance."""
    n = closes.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    cum_pv = 0.0
    cum_vol = 0.0

    # Nanoseconds per day and the EST offset (UTC-5 = -5h).
    # Using EST for day boundary detection is safe during US market hours;
    # the calendar date is the same under both EST and EDT for 9:30-16:00 ET.
    ns_per_day = np.int64(86_400_000_000_000)
    est_offset_ns = np.int64(-5 * 3_600_000_000_000)
    prev_day = np.int64(-1)

    for i in range(n):
        # Reset accumulators on new trading day (Eastern Time).
        if reset_daily:
            day = (timestamps_ns[i] + est_offset_ns) // ns_per_day
            if day != prev_day:
                cum_pv = 0.0
                cum_vol = 0.0
                prev_day = day

        vol = volumes[i]
        if abs(vwaps[i]) >= 1e-9 and vol > 0:
            cum_pv += vwaps[i] * vol
        elif vol > 0:
            tp = (highs[i] + lows[i] + closes[i]) / 3.0
            cum_pv += tp * vol
        cum_vol += vol

        if cum_vol > 0:
            vwap = cum_pv / cum_vol
            if vwap > 0:
                out[i] = (closes[i] - vwap) / vwap
            else:
                out[i] = np.nan
        else:
            out[i] = np.nan

    return out


def batch_vwap_distance(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    vwaps: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """Compute cumulative VWAP and distance as (close - vwap) / vwap.

    Uses bar.vwap when available (non-zero); otherwise typical price.
    VWAP resets at each new trading day (Eastern Time) when *timestamps*
    is provided.  *timestamps* should be a datetime64[ns] or int64 array
    of UTC nanosecond epoch values, one per bar.
    Returns array of same length.
    """
    if timestamps is not None and len(timestamps) == len(closes):
        ts_ns = timestamps.astype(np.int64)
        return _vwap_core(highs, lows, closes, volumes, vwaps, ts_ns, True)
    else:
        dummy = np.empty(0, dtype=np.int64)
        return _vwap_core(highs, lows, closes, volumes, vwaps, dummy, False)


# ---------------------------------------------------------------------------
# SMA  (Simple Moving Average)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def batch_sma(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute SMA.  Matches ``RollingSMA``."""
    n = closes.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    running_sum = 0.0
    for i in range(n):
        if i >= period:
            running_sum -= closes[i - period]
        running_sum += closes[i]
        count = min(i + 1, period)
        out[i] = running_sum / count
    return out
