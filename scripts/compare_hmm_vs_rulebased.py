"""
Compare HMM regime detector vs optimized rule-based detector on SPY 2020-2025.
"""

import logging
import os
import warnings

# Suppress all logging and warnings
os.environ["EMPIRE_LOG_LEVEL"] = "CRITICAL"
os.environ["EMPIRE_LOG_FORMAT"] = "json"
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Load SPY daily bars from parquet (resample 1Min -> 1Day)
# ---------------------------------------------------------------------------


def load_spy_daily() -> pd.DataFrame:
    """Load and merge SPY minute bars, resample to daily OHLCV."""
    frames = []
    for path in [
        "data/bars_5yr/SPY_1Min.parquet",
        "data/bars_2023_2025/SPY_1Min.parquet",
    ]:
        if os.path.exists(path):
            df = pq.read_table(path).to_pandas()
            frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    raw = raw.set_index("timestamp")

    daily = (
        raw.resample("1D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["close"])
    )

    daily = daily[(daily.index >= "2020-01-01") & (daily.index <= "2025-12-31")]
    daily = daily.reset_index()
    daily = daily.rename(columns={"timestamp": "time"})
    daily["time"] = pd.to_datetime(daily["time"])
    return daily


# ---------------------------------------------------------------------------
# 2. HMM regime detection
# ---------------------------------------------------------------------------


def run_hmm(daily: pd.DataFrame) -> pd.DataFrame:
    """Fit HMM on daily bars and return regime labels per day."""
    from src.regime_models.hmm.adapters import PomegranateDenseHmmAdapter
    from src.regime_models.hmm.config import HmmFeatureConfig, HmmTrainingConfig
    from src.regime_models.hmm.features import build_hmm_feature_frame
    from src.regime_models.hmm.labeling import label_hidden_states

    feat_cfg = HmmFeatureConfig(
        time_column="time",
        high_column="high",
        low_column="low",
        close_column="close",
        volume_column="volume",
        return_windows=(1, 5),
        realized_vol_window=20,
        volume_window=20,
    )
    train_cfg = HmmTrainingConfig(
        n_states=3,
        max_iter=50,
        random_seed=7,
        covariance_type="diag",
        min_samples=30,
    )

    features = build_hmm_feature_frame(daily, feat_cfg)
    feature_cols = ["log_return_1", "log_return_5", "realized_vol_20", "hl_range_pct", "volume_zscore_20"]
    matrix = features[feature_cols].values

    adapter = PomegranateDenseHmmAdapter(train_cfg)
    adapter.fit(matrix)
    states = adapter.predict(matrix)

    labels = label_hidden_states(features, states)

    result = features[["time"]].copy()
    result["hmm_state"] = states
    result["hmm_label"] = result["hmm_state"].map(labels)

    # Map HMM labels to trend/vol
    trend_map = {
        "trend_up": "UP",
        "trend_down": "DOWN",
        "stress_up": "UP",
        "stress_down": "DOWN",
        "neutral": "FLAT",
    }
    vol_map = {
        "stress_up": "HIGH",
        "stress_down": "HIGH",
        "trend_up": "NORMAL",
        "trend_down": "NORMAL",
        "neutral": "NORMAL",
    }
    result["hmm_trend"] = result["hmm_label"].map(trend_map)
    result["hmm_vol"] = result["hmm_label"].map(vol_map)
    return result


# ---------------------------------------------------------------------------
# 3. Rule-based regime detection (optimized params)
# ---------------------------------------------------------------------------


def run_rulebased(daily: pd.DataFrame) -> pd.DataFrame:
    """Apply optimized rule-based regime detection."""
    # Optimized params from artifacts/regime_validation/optimized_params.json
    sma_fast = 10
    sma_slow = 40
    band = 0.01
    vol_window = 30
    vol_low = 0.08
    vol_high = 0.20
    vol_shock = 0.50

    df = daily.copy()

    # Trend: SMA dual crossover with band
    df["sma_fast"] = df["close"].rolling(sma_fast).mean()
    df["sma_slow"] = df["close"].rolling(sma_slow).mean()
    df["sma_ratio"] = df["sma_fast"] / df["sma_slow"] - 1.0

    def classify_trend(ratio):
        if pd.isna(ratio):
            return None
        if ratio > band:
            return "UP"
        elif ratio < -band:
            return "DOWN"
        else:
            return "FLAT"

    df["rb_trend"] = df["sma_ratio"].apply(classify_trend)

    # Vol: 30-day realized vol (annualized)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol"] = log_ret.rolling(vol_window).std() * np.sqrt(252)

    def classify_vol(v):
        if pd.isna(v):
            return None
        if v >= vol_shock:
            return "SHOCK"
        elif v >= vol_high:
            return "HIGH"
        elif v <= vol_low:
            return "LOW"
        else:
            return "NORMAL"

    df["rb_vol"] = df["realized_vol"].apply(classify_vol)

    result = df[["time", "rb_trend", "rb_vol"]].dropna(subset=["rb_trend", "rb_vol"])
    return result


# ---------------------------------------------------------------------------
# 4. Ground truth periods
# ---------------------------------------------------------------------------

GROUND_TRUTH = [
    ("COVID Crash", "2020-02-01", "2020-03-31", ["DOWN"], ["HIGH", "SHOCK"]),
    ("COVID Recovery", "2020-04-01", "2020-08-31", ["UP"], ["HIGH", "NORMAL"]),
    ("2021 Bull", "2021-01-01", "2021-12-31", ["UP"], ["LOW", "NORMAL"]),
    ("2022 H1 Bear", "2022-01-01", "2022-06-30", ["DOWN"], ["HIGH", "NORMAL"]),
    ("2022 H2 Range", "2022-07-01", "2022-12-31", ["FLAT", "DOWN"], ["NORMAL", "HIGH"]),
    ("2023 H1 Recovery", "2023-01-01", "2023-07-31", ["UP"], ["NORMAL", "LOW"]),
    ("2023 Q3 Correction", "2023-08-01", "2023-10-31", ["DOWN", "FLAT"], ["NORMAL", "HIGH"]),
    ("2024 Bull", "2024-01-01", "2024-12-31", ["UP"], ["LOW", "NORMAL"]),
]


def score_period(series: pd.Series, valid_labels: list[str]) -> float:
    """Fraction of days matching any of the valid labels."""
    if series.empty:
        return 0.0
    matches = series.isin(valid_labels).sum()
    return matches / len(series) * 100


# ---------------------------------------------------------------------------
# 5. Transition frequency (flips within 5-day windows)
# ---------------------------------------------------------------------------


def transition_freq(series: pd.Series, window: int = 5) -> float:
    """Average number of regime changes per 5-day rolling window."""
    changes = (series != series.shift(1)).astype(int)
    rolling_changes = changes.rolling(window).sum()
    return float(rolling_changes.mean())


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    print("=" * 90)
    print("  HMM vs Rule-Based Regime Detector  --  SPY 2020-2025")
    print("=" * 90)
    print()

    # Load data
    print("Loading SPY daily bars...")
    daily = load_spy_daily()
    print(f"  Daily bars: {len(daily)} days  ({daily['time'].min().date()} to {daily['time'].max().date()})")
    print()

    # Run HMM
    print("Fitting HMM model (3 states, pomegranate)...")
    hmm_result = run_hmm(daily)
    hmm_label_dist = hmm_result["hmm_label"].value_counts()
    print("  HMM label distribution:")
    for label, count in hmm_label_dist.items():
        print(f"    {label}: {count} days ({count / len(hmm_result) * 100:.1f}%)")
    print()

    # Run rule-based
    print("Running optimized rule-based detector (SMA 10/40, Vol 30d)...")
    rb_result = run_rulebased(daily)
    rb_trend_dist = rb_result["rb_trend"].value_counts()
    rb_vol_dist = rb_result["rb_vol"].value_counts()
    print("  Rule-based trend distribution:")
    for label, count in rb_trend_dist.items():
        print(f"    {label}: {count} days ({count / len(rb_result) * 100:.1f}%)")
    print("  Rule-based vol distribution:")
    for label, count in rb_vol_dist.items():
        print(f"    {label}: {count} days ({count / len(rb_result) * 100:.1f}%)")
    print()

    # Merge on time
    merged = hmm_result.merge(rb_result, on="time", how="inner")
    merged["time_dt"] = pd.to_datetime(merged["time"])

    # ---------------------------------------------------------------------------
    # 6. Per-period accuracy table
    # ---------------------------------------------------------------------------
    print("=" * 90)
    print("  PER-PERIOD ACCURACY COMPARISON")
    print("=" * 90)
    header = f"{'Period':<22} {'HMM Trend':>10} {'RB Trend':>10} {'HMM Vol':>10} {'RB Vol':>10} {'Winner':>12}"
    print(header)
    print("-" * 90)

    hmm_trend_scores = []
    rb_trend_scores = []
    hmm_vol_scores = []
    rb_vol_scores = []

    for name, start, end, gt_trend, gt_vol in GROUND_TRUTH:
        mask = (merged["time_dt"] >= start) & (merged["time_dt"] <= end)
        period = merged[mask]

        if period.empty:
            print(f"{name:<22}  (no data)")
            continue

        ht = score_period(period["hmm_trend"], gt_trend)
        rt = score_period(period["rb_trend"], gt_trend)
        hv = score_period(period["hmm_vol"], gt_vol)
        rv = score_period(period["rb_vol"], gt_vol)

        hmm_trend_scores.append(ht)
        rb_trend_scores.append(rt)
        hmm_vol_scores.append(hv)
        rb_vol_scores.append(rv)

        hmm_total = (ht + hv) / 2
        rb_total = (rt + rv) / 2
        if hmm_total > rb_total + 3:
            winner = "HMM"
        elif rb_total > hmm_total + 3:
            winner = "Rule-Based"
        else:
            winner = "TIE"

        print(f"{name:<22} {ht:>9.1f}% {rt:>9.1f}% {hv:>9.1f}% {rv:>9.1f}%  {winner:>11}")

    print("-" * 90)
    hmm_t_avg = np.mean(hmm_trend_scores)
    rb_t_avg = np.mean(rb_trend_scores)
    hmm_v_avg = np.mean(hmm_vol_scores)
    rb_v_avg = np.mean(rb_vol_scores)
    print(f"{'AVERAGE':<22} {hmm_t_avg:>9.1f}% {rb_t_avg:>9.1f}% {hmm_v_avg:>9.1f}% {rb_v_avg:>9.1f}%")
    print()

    # ---------------------------------------------------------------------------
    # 7. Transition frequency (stability)
    # ---------------------------------------------------------------------------
    print("=" * 90)
    print("  TRANSITION FREQUENCY (regime flips per 5-day window)")
    print("  Lower = more stable = better for trading")
    print("=" * 90)

    hmm_trend_trans = transition_freq(merged["hmm_trend"])
    rb_trend_trans = transition_freq(merged["rb_trend"])
    hmm_vol_trans = transition_freq(merged["hmm_vol"])
    rb_vol_trans = transition_freq(merged["rb_vol"])

    print(f"  {'Metric':<30} {'HMM':>10} {'Rule-Based':>12}")
    print(f"  {'-' * 55}")
    print(f"  {'Trend flips / 5d window':<30} {hmm_trend_trans:>10.3f} {rb_trend_trans:>12.3f}")
    print(f"  {'Vol flips / 5d window':<30} {hmm_vol_trans:>10.3f} {rb_vol_trans:>12.3f}")
    print(
        f"  {'Combined avg':<30} {(hmm_trend_trans + hmm_vol_trans) / 2:>10.3f} {(rb_trend_trans + rb_vol_trans) / 2:>12.3f}"
    )

    hmm_stability = (hmm_trend_trans + hmm_vol_trans) / 2
    rb_stability = (rb_trend_trans + rb_vol_trans) / 2
    if hmm_stability < rb_stability:
        print(f"\n  --> HMM is MORE STABLE ({hmm_stability:.3f} vs {rb_stability:.3f} flips/5d)")
    else:
        print(f"\n  --> Rule-based is MORE STABLE ({rb_stability:.3f} vs {hmm_stability:.3f} flips/5d)")
    print()

    # ---------------------------------------------------------------------------
    # 8. Final verdict
    # ---------------------------------------------------------------------------
    print("=" * 90)
    print("  FINAL VERDICT")
    print("=" * 90)

    hmm_overall = (hmm_t_avg + hmm_v_avg) / 2
    rb_overall = (rb_t_avg + rb_v_avg) / 2

    print("\n  Overall accuracy:")
    print(f"    HMM:        {hmm_overall:.1f}%  (trend {hmm_t_avg:.1f}%, vol {hmm_v_avg:.1f}%)")
    print(f"    Rule-based: {rb_overall:.1f}%  (trend {rb_t_avg:.1f}%, vol {rb_v_avg:.1f}%)")
    print("\n  Stability (flips/5d, lower=better):")
    print(f"    HMM:        {hmm_stability:.3f}")
    print(f"    Rule-based: {rb_stability:.3f}")

    # Weighted score: 70% accuracy, 30% stability (inverted: fewer flips = better)
    # Normalize stability to 0-100 score (0 flips = 100, 5 flips = 0)
    hmm_stab_score = max(0, 100 - hmm_stability * 20)
    rb_stab_score = max(0, 100 - rb_stability * 20)
    hmm_weighted = 0.7 * hmm_overall + 0.3 * hmm_stab_score
    rb_weighted = 0.7 * rb_overall + 0.3 * rb_stab_score

    print("\n  Weighted score (70% accuracy + 30% stability):")
    print(f"    HMM:        {hmm_weighted:.1f}")
    print(f"    Rule-based: {rb_weighted:.1f}")

    print(f"\n  {'=' * 60}")
    if hmm_weighted > rb_weighted + 2:
        print("  WINNER: HMM regime detector")
        print(f"  Advantage: +{hmm_weighted - rb_weighted:.1f} weighted points")
    elif rb_weighted > hmm_weighted + 2:
        print("  WINNER: Rule-based regime detector")
        print(f"  Advantage: +{rb_weighted - hmm_weighted:.1f} weighted points")
    else:
        print(f"  RESULT: Too close to call (delta = {abs(hmm_weighted - rb_weighted):.1f})")

    # Recommendation
    print("\n  RECOMMENDATION:")
    if rb_overall > hmm_overall and rb_stability <= hmm_stability:
        print("  Use RULE-BASED as primary — it wins on both accuracy and stability.")
        print("  Keep HMM in shadow mode as a confirmation signal.")
    elif hmm_overall > rb_overall and hmm_stability <= rb_stability:
        print("  Use HMM as primary — it wins on both accuracy and stability.")
        print("  Rule-based can serve as a fast fallback when HMM confidence is low.")
    elif rb_overall > hmm_overall:
        print(f"  Use RULE-BASED as primary (better accuracy: {rb_overall:.1f}% vs {hmm_overall:.1f}%).")
        print("  HMM is more stable — consider ENSEMBLE: rule-based for trend,")
        print("  HMM for regime-change detection (its state transitions are smoother).")
    elif hmm_overall > rb_overall:
        print(f"  Use HMM as primary (better accuracy: {hmm_overall:.1f}% vs {rb_overall:.1f}%).")
        print("  Rule-based is more stable — consider ENSEMBLE: HMM for trend,")
        print("  rule-based as a low-latency confirmation filter.")
    else:
        print("  ENSEMBLE both: rule-based for fast real-time signals,")
        print("  HMM for regime-change detection and confirmation.")
    print(f"  {'=' * 60}")


if __name__ == "__main__":
    main()
