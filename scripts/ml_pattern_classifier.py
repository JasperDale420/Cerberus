#!/usr/bin/env python3
"""
ML Pattern Classifier for Cerberus Backtest Trades.

Trains a LightGBM classifier on backtest trade data to identify
regime combinations that predict winning trades.

Usage:
    python scripts/ml_pattern_classifier.py artifacts/backtests/5yr_wide_stops
"""

import json
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split

    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("Warning: lightgbm or sklearn not installed. Install with:")
    print("  pip install lightgbm scikit-learn")


# Feature columns to use for prediction
REGIME_FEATURES = ["trend", "vol", "session", "risk", "liquidity"]
CATEGORICAL_FEATURES = ["strategy", "side"] + REGIME_FEATURES


def load_trades(path: str) -> List[Dict[str, Any]]:
    """Load trades from backtest JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("engine_trades", [])


def build_feature_df(trades: List[Dict]) -> pd.DataFrame:
    """Convert trades to feature DataFrame for ML."""
    rows = []
    for t in trades:
        regime = t.get("regime_tags_at_entry", {}) or {}
        pnl = t.get("pnl_net", 0) or 0

        row = {
            "strategy": t.get("strategy", "unknown"),
            "symbol": t.get("symbol", ""),
            "side": t.get("side", ""),
            "trend": regime.get("trend", "unknown"),
            "vol": regime.get("vol", "unknown"),
            "session": regime.get("session", "unknown"),
            "risk": regime.get("risk", "unknown"),
            "liquidity": regime.get("liquidity", "unknown"),
            "pnl": pnl,
            "is_winner": 1 if pnl > 0 else 0,
            "holding_period_sec": t.get("holding_period_seconds", 0) or 0,
            "pnl_r": t.get("pnl_r", 0) or 0,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def encode_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """One-hot encode categorical features for LightGBM."""
    encoded = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=False)
    feature_names = list(encoded.columns)
    return encoded.values, feature_names


def train_classifier(
    X: np.ndarray, y: np.ndarray, feature_names: List[str]
) -> Tuple[Any, float, float]:
    """Train LightGBM classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Train model
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
    )

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_auc = roc_auc_score(y_train, train_pred)
    test_auc = roc_auc_score(y_test, test_pred)

    return model, train_auc, test_auc


def extract_importance(model: Any, feature_names: List[str], top_k: int = 15):
    """Extract top feature importances."""
    importance = model.feature_importance(importance_type="gain")
    indices = np.argsort(importance)[::-1][:top_k]

    print("\n" + "=" * 60)
    print("TOP FEATURE IMPORTANCES (by gain)")
    print("=" * 60)
    for i in indices:
        print(f"  {feature_names[i]:40} {importance[i]:>10.1f}")


def extract_rules(df: pd.DataFrame):
    """Extract human-readable rules from data patterns."""
    print("\n" + "=" * 60)
    print("PREDICTIVE REGIME COMBINATIONS")
    print("=" * 60)

    # Group by regime combinations and calculate win rate
    for strategy in df["strategy"].unique():
        strat_df = df[df["strategy"] == strategy]
        if len(strat_df) < 50:
            continue

        print(f"\n{strategy.upper()} ({len(strat_df)} trades)")
        print("-" * 50)

        # Find best regime combinations
        groups = (
            strat_df.groupby(["trend", "session", "vol", "risk"])
            .agg({"is_winner": ["sum", "count", "mean"]})
            .reset_index()
        )
        groups.columns = [
            "trend",
            "session",
            "vol",
            "risk",
            "wins",
            "total",
            "win_rate",
        ]
        groups = groups[groups["total"] >= 20]  # Minimum sample size
        groups = groups.sort_values("win_rate", ascending=False)

        if len(groups) == 0:
            print("  No regime combinations with N >= 20")
            continue

        # Show top 3 and bottom 3
        print("  BEST:")
        for _, row in groups.head(3).iterrows():
            print(
                f"    {row['trend']:6} + {row['session']:12} + {row['vol']:6} + {row['risk']:8} "
                f"→ {row['win_rate'] * 100:5.1f}% WR (N={int(row['total'])})"
            )

        if len(groups) > 3:
            print("  WORST:")
            for _, row in groups.tail(3).iterrows():
                print(
                    f"    {row['trend']:6} + {row['session']:12} + {row['vol']:6} + {row['risk']:8} "
                    f"→ {row['win_rate'] * 100:5.1f}% WR (N={int(row['total'])})"
                )


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ml_pattern_classifier.py <backtest_json_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading trades from: {path}")

    trades = load_trades(path)
    print(f"Loaded {len(trades):,} trades")

    df = build_feature_df(trades)
    print(f"Win rate: {df['is_winner'].mean() * 100:.1f}%")

    # Extract regime-based rules (works without ML libraries)
    extract_rules(df)

    if not HAS_ML:
        print("\n[ML analysis skipped - install lightgbm and scikit-learn]")
        return

    # Train ML classifier
    print("\n" + "=" * 60)
    print("TRAINING LIGHTGBM CLASSIFIER")
    print("=" * 60)

    X, feature_names = encode_features(df)
    y = df["is_winner"].values

    model, train_auc, test_auc = train_classifier(X, y, feature_names)
    print(f"\nTrain AUC: {train_auc:.3f}")
    print(f"Test AUC:  {test_auc:.3f}")

    # Extract feature importance
    extract_importance(model, feature_names)

    print("\n✅ ML analysis complete")


if __name__ == "__main__":
    main()
