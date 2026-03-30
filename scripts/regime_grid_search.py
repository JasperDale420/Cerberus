"""Grid search regime detector parameters against SPY ground truth.

Tests trend, volatility, and risk axis parameters to maximize accuracy
against known market periods 2020-2025.

Usage: uv run python scripts/regime_grid_search.py
"""

import itertools
import logging
import os
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["EMPIRE_LOG_LEVEL"] = "CRITICAL"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ── Load data ──────────────────────────────────────────────────────

DATA_FILES = ["data/bars_5yr/SPY_1Min.parquet", "data/bars_2023_2025/SPY_1Min.parquet"]

frames = []
for path in DATA_FILES:
    if os.path.exists(path):
        frames.append(pd.read_parquet(path))
raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
raw["date"] = raw["timestamp"].dt.date

daily = (
    raw.groupby("date")
    .agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    .reset_index()
)
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)
closes = daily["close"].values.astype(float)
n = len(closes)

print(f"Loaded {n} daily bars: {daily['date'].iloc[0].date()} → {daily['date'].iloc[-1].date()}")

# ── Ground truth ───────────────────────────────────────────────────

GROUND_TRUTH = [
    {
        "label": "COVID Crash",
        "start": "2020-02-01",
        "end": "2020-03-31",
        "trend": ["DOWN"],
        "vol": ["HIGH", "SHOCK"],
        "risk": ["RISK_OFF"],
    },
    {
        "label": "COVID Recovery",
        "start": "2020-04-01",
        "end": "2020-08-31",
        "trend": ["UP"],
        "vol": ["HIGH", "NORMAL"],
        "risk": ["RISK_ON"],
    },
    {
        "label": "2021 Bull",
        "start": "2021-01-01",
        "end": "2021-12-31",
        "trend": ["UP"],
        "vol": ["LOW", "NORMAL"],
        "risk": ["RISK_ON"],
    },
    {
        "label": "2022 H1 Bear",
        "start": "2022-01-01",
        "end": "2022-06-30",
        "trend": ["DOWN"],
        "vol": ["HIGH", "NORMAL"],
        "risk": ["RISK_OFF"],
    },
    {
        "label": "2022 H2 Range",
        "start": "2022-07-01",
        "end": "2022-12-31",
        "trend": ["FLAT", "DOWN"],
        "vol": ["NORMAL", "HIGH"],
        "risk": ["NEUTRAL", "RISK_OFF"],
    },
    {
        "label": "2023 H1 Recovery",
        "start": "2023-01-01",
        "end": "2023-07-31",
        "trend": ["UP"],
        "vol": ["NORMAL", "LOW"],
        "risk": ["RISK_ON"],
    },
    {
        "label": "2023 Q3 Correction",
        "start": "2023-08-01",
        "end": "2023-10-31",
        "trend": ["DOWN", "FLAT"],
        "vol": ["NORMAL", "HIGH"],
        "risk": ["RISK_OFF", "NEUTRAL"],
    },
    {
        "label": "2024 Bull",
        "start": "2024-01-01",
        "end": "2024-12-31",
        "trend": ["UP"],
        "vol": ["LOW", "NORMAL"],
        "risk": ["RISK_ON"],
    },
]


def score_axis(labels: list[str], axis_name: str) -> float:
    """Score a single axis against all ground truth periods. Returns avg match %."""
    scores = []
    for gt in GROUND_TRUTH:
        mask = (daily["date"] >= gt["start"]) & (daily["date"] <= gt["end"])
        idx = daily.index[mask]
        if len(idx) == 0:
            continue
        expected = gt[axis_name]
        matched = sum(1 for i in idx if labels[i] in expected)
        scores.append(matched / len(idx) * 100)
    return np.mean(scores) if scores else 0.0


# ══════════════════════════════════════════════════════════════════
# TREND AXIS GRID SEARCH
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("TREND AXIS GRID SEARCH")
print("=" * 80)

# Methods to test:
# 1. SMA slope: price vs SMA(fast), SMA direction over N bars
# 2. Dual SMA: price > SMA(slow) AND SMA(fast) > SMA(slow) = UP
# 3. ADX + direction: ADX > threshold = trending, DI+ vs DI- for direction


def classify_trend_sma(fast: int, slow: int, flat_band: float) -> list[str]:
    """Dual SMA crossover with flat band."""
    sma_fast = pd.Series(closes).rolling(fast).mean().values
    sma_slow = pd.Series(closes).rolling(slow).mean().values
    labels = []
    for i in range(n):
        if i < slow or np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]):
            labels.append("FLAT")
            continue
        pct_above = (closes[i] - sma_slow[i]) / sma_slow[i]
        fast_above_slow = (sma_fast[i] - sma_slow[i]) / sma_slow[i]
        if pct_above > flat_band and fast_above_slow > 0:
            labels.append("UP")
        elif pct_above < -flat_band and fast_above_slow < 0:
            labels.append("DOWN")
        else:
            labels.append("FLAT")
    return labels


def classify_trend_adx(period: int, adx_thresh: float, lookback: int) -> list[str]:
    """ADX-based trend detection with directional index."""
    highs = daily["high"].values.astype(float)
    lows = daily["low"].values.astype(float)
    labels = []

    # Compute +DM, -DM, TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))

    # Smoothed with EMA
    alpha = 1.0 / period
    sm_plus = np.zeros(n)
    sm_minus = np.zeros(n)
    sm_tr = np.zeros(n)
    for i in range(1, n):
        sm_plus[i] = alpha * plus_dm[i] + (1 - alpha) * sm_plus[i - 1]
        sm_minus[i] = alpha * minus_dm[i] + (1 - alpha) * sm_minus[i - 1]
        sm_tr[i] = alpha * tr[i] + (1 - alpha) * sm_tr[i - 1]

    di_plus = np.where(sm_tr > 0, 100 * sm_plus / sm_tr, 0)
    di_minus = np.where(sm_tr > 0, 100 * sm_minus / sm_tr, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)

    adx = np.zeros(n)
    for i in range(1, n):
        adx[i] = alpha * dx[i] + (1 - alpha) * adx[i - 1]

    for i in range(n):
        if i < period * 2:
            labels.append("FLAT")
        elif adx[i] < adx_thresh:
            labels.append("FLAT")
        elif di_plus[i] > di_minus[i]:
            labels.append("UP")
        else:
            labels.append("DOWN")
    return labels


def classify_trend_return(lookback: int, up_thresh: float, down_thresh: float) -> list[str]:
    """Simple cumulative return over lookback window."""
    labels = []
    for i in range(n):
        if i < lookback:
            labels.append("FLAT")
            continue
        ret = (closes[i] / closes[i - lookback]) - 1
        if ret > up_thresh:
            labels.append("UP")
        elif ret < down_thresh:
            labels.append("DOWN")
        else:
            labels.append("FLAT")
    return labels


# Grid search trend
best_trend_score = 0
best_trend_params = {}
best_trend_labels = []
trend_results = []

# Method 1: Dual SMA
for fast, slow, band in itertools.product([10, 15, 20], [40, 50, 60, 80], [0.01, 0.02, 0.03, 0.05]):
    if fast >= slow:
        continue
    labels = classify_trend_sma(fast, slow, band)
    score = score_axis(labels, "trend")
    trend_results.append({"method": "sma", "params": f"fast={fast},slow={slow},band={band}", "score": score})
    if score > best_trend_score:
        best_trend_score = score
        best_trend_params = {"method": "sma", "fast": fast, "slow": slow, "band": band}
        best_trend_labels = labels

# Method 2: ADX
for period, thresh, lb in itertools.product([10, 14, 20], [15, 20, 25, 30], [20, 40]):
    labels = classify_trend_adx(period, thresh, lb)
    score = score_axis(labels, "trend")
    trend_results.append({"method": "adx", "params": f"period={period},thresh={thresh}", "score": score})
    if score > best_trend_score:
        best_trend_score = score
        best_trend_params = {"method": "adx", "period": period, "thresh": thresh}
        best_trend_labels = labels

# Method 3: Simple return
for lb, up, down in itertools.product([20, 30, 40, 60], [0.02, 0.03, 0.05, 0.08], [-0.02, -0.03, -0.05, -0.08]):
    labels = classify_trend_return(lb, up, down)
    score = score_axis(labels, "trend")
    trend_results.append({"method": "return", "params": f"lb={lb},up={up},down={down}", "score": score})
    if score > best_trend_score:
        best_trend_score = score
        best_trend_params = {"method": "return", "lb": lb, "up": up, "down": down}
        best_trend_labels = labels

# Top 5 trend
trend_df = pd.DataFrame(trend_results).sort_values("score", ascending=False)
print("\nTop 10 trend configurations:")
for _, row in trend_df.head(10).iterrows():
    print(f"  {row['method']:8s} {row['params']:45s} → {row['score']:.1f}%")
print(f"\n★ BEST TREND: {best_trend_params} → {best_trend_score:.1f}%")

# Per-period breakdown for best
print("\n  Per-period breakdown:")
for gt in GROUND_TRUTH:
    mask = (daily["date"] >= gt["start"]) & (daily["date"] <= gt["end"])
    idx = daily.index[mask]
    matched = sum(1 for i in idx if best_trend_labels[i] in gt["trend"])
    pct = matched / len(idx) * 100 if len(idx) > 0 else 0
    dist = {}
    for i in idx:
        label = best_trend_labels[i]
        dist[label] = dist.get(label, 0) + 1
    print(f"  {gt['label']:<25} {pct:5.1f}% (expected {gt['trend']}, got {dist})")


# ══════════════════════════════════════════════════════════════════
# VOLATILITY AXIS GRID SEARCH
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("VOLATILITY AXIS GRID SEARCH")
print("=" * 80)


def classify_vol_ewma(short_span: int, long_span: int, high_z: float, low_z: float, shock_z: float) -> list[str]:
    """EWMA variance ratio vol classification."""
    alpha_s = 2.0 / (short_span + 1)
    alpha_l = 2.0 / (long_span + 1)
    s_var = None
    l_var = None
    labels = ["NORMAL"]  # i=0
    for i in range(1, n):
        log_ret = np.log(closes[i] / closes[i - 1])
        sq = log_ret**2
        if s_var is None:
            s_var = sq
            l_var = sq
        else:
            s_var = alpha_s * sq + (1 - alpha_s) * s_var
            l_var = alpha_l * sq + (1 - alpha_l) * l_var
        if l_var < 1e-12:
            labels.append("NORMAL")
        else:
            z = np.sqrt(s_var / l_var)
            if z >= shock_z:
                labels.append("SHOCK")
            elif z >= high_z:
                labels.append("HIGH")
            elif z <= low_z:
                labels.append("LOW")
            else:
                labels.append("NORMAL")
    return labels


def classify_vol_realized(window: int, low_ann: float, high_ann: float, shock_ann: float) -> list[str]:
    """Realized vol (annualized daily std)."""
    log_rets = np.zeros(n)
    for i in range(1, n):
        log_rets[i] = np.log(closes[i] / closes[i - 1])
    labels = []
    for i in range(n):
        if i < window:
            labels.append("NORMAL")
            continue
        rvol = np.std(log_rets[i - window + 1 : i + 1]) * np.sqrt(252)
        if rvol >= shock_ann:
            labels.append("SHOCK")
        elif rvol >= high_ann:
            labels.append("HIGH")
        elif rvol <= low_ann:
            labels.append("LOW")
        else:
            labels.append("NORMAL")
    return labels


best_vol_score = 0
best_vol_params = {}
best_vol_labels = []
vol_results = []

# Method 1: EWMA ratio
for ss, ls, hz, lz, sz in itertools.product(
    [5, 10, 15], [60, 90, 120, 180], [1.3, 1.5, 1.8], [0.5, 0.6, 0.7], [2.5, 3.0, 3.5]
):
    labels = classify_vol_ewma(ss, ls, hz, lz, sz)
    score = score_axis(labels, "vol")
    vol_results.append({"method": "ewma", "params": f"ss={ss},ls={ls},hz={hz},lz={lz},sz={sz}", "score": score})
    if score > best_vol_score:
        best_vol_score = score
        best_vol_params = {
            "method": "ewma",
            "short_span": ss,
            "long_span": ls,
            "high_z": hz,
            "low_z": lz,
            "shock_z": sz,
        }
        best_vol_labels = labels

# Method 2: Realized vol
for win, low_a, high_a, shock_a in itertools.product(
    [10, 15, 20, 30], [0.08, 0.10, 0.12], [0.20, 0.25, 0.30], [0.35, 0.40, 0.50]
):
    labels = classify_vol_realized(win, low_a, high_a, shock_a)
    score = score_axis(labels, "vol")
    vol_results.append(
        {"method": "realized", "params": f"win={win},low={low_a},high={high_a},shock={shock_a}", "score": score}
    )
    if score > best_vol_score:
        best_vol_score = score
        best_vol_params = {"method": "realized", "window": win, "low": low_a, "high": high_a, "shock": shock_a}
        best_vol_labels = labels

vol_df = pd.DataFrame(vol_results).sort_values("score", ascending=False)
print("\nTop 10 vol configurations:")
for _, row in vol_df.head(10).iterrows():
    print(f"  {row['method']:10s} {row['params']:55s} → {row['score']:.1f}%")
print(f"\n★ BEST VOL: {best_vol_params} → {best_vol_score:.1f}%")

print("\n  Per-period breakdown:")
for gt in GROUND_TRUTH:
    mask = (daily["date"] >= gt["start"]) & (daily["date"] <= gt["end"])
    idx = daily.index[mask]
    matched = sum(1 for i in idx if best_vol_labels[i] in gt["vol"])
    pct = matched / len(idx) * 100 if len(idx) > 0 else 0
    dist = {}
    for i in idx:
        label = best_vol_labels[i]
        dist[label] = dist.get(label, 0) + 1
    print(f"  {gt['label']:<25} {pct:5.1f}% (expected {gt['vol']}, got {dist})")


# ══════════════════════════════════════════════════════════════════
# FINAL COMBINED ACCURACY (Trend + Vol only — Risk axis dropped)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("OPTIMIZED REGIME DETECTOR — 2-AXIS (Trend + Vol)")
print("=" * 80)

print(f"\n  Trend:  {best_trend_score:.1f}% (was 53.8%)  — {best_trend_params}")
print(f"  Vol:    {best_vol_score:.1f}% (was 84.3%)  — {best_vol_params}")
combined = (best_trend_score + best_vol_score) / 2
baseline_2axis = (53.8 + 84.3) / 2
print(f"\n  Combined: {combined:.1f}% (was {baseline_2axis:.1f}%)")
print(f"  Improvement: {combined - baseline_2axis:+.1f}%")

print(f"\n{'=' * 80}")
print("Per-period combined accuracy:")
print(f"{'=' * 80}")
for gt in GROUND_TRUTH:
    mask = (daily["date"] >= gt["start"]) & (daily["date"] <= gt["end"])
    idx = daily.index[mask]
    if len(idx) == 0:
        continue
    t_match = sum(1 for i in idx if best_trend_labels[i] in gt["trend"]) / len(idx) * 100
    v_match = sum(1 for i in idx if best_vol_labels[i] in gt["vol"]) / len(idx) * 100
    overall = (t_match + v_match) / 2
    print(f"  {gt['label']:<25} trend={t_match:5.1f}%  vol={v_match:5.1f}%  combined={overall:5.1f}%")

# Regime combination distribution
print(f"\n{'=' * 80}")
print("Regime combination distribution (optimized):")
print(f"{'=' * 80}")
combos = {}
for i in range(n):
    key = f"{best_trend_labels[i]}+{best_vol_labels[i]}"
    combos[key] = combos.get(key, 0) + 1
for k in sorted(combos, key=combos.get, reverse=True):
    print(f"  {k:<20} {combos[k]:>5} days ({combos[k] / n * 100:5.1f}%)")

# Save optimized params
import json  # noqa: E402

os.makedirs("artifacts/regime_validation", exist_ok=True)
with open("artifacts/regime_validation/optimized_params.json", "w") as f:
    json.dump(
        {
            "trend": best_trend_params,
            "trend_accuracy": best_trend_score,
            "vol": best_vol_params,
            "vol_accuracy": best_vol_score,
            "combined_accuracy": combined,
            "baseline_trend": 53.8,
            "baseline_vol": 84.3,
            "axes": ["trend", "vol"],
            "note": "Risk axis dropped — 5-day return proxy too noisy (58% accuracy)",
        },
        f,
        indent=2,
    )
print("\nOptimized params saved to artifacts/regime_validation/optimized_params.json")
