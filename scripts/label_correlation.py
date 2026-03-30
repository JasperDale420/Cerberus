"""Label correlation regime (dispersion vs herding) on a daily basis.

Computes rolling 20-day pairwise correlation among 10 liquid symbols,
labels each day as HIGH_CORR / NORMAL_CORR / LOW_CORR, and computes
per-symbol rolling SPY beta. Enriches each symbol's daily regime parquet.
"""

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import logging  # noqa: E402

logging.disable(logging.CRITICAL)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "regime_labeled"
SYMBOLS = ["AAPL", "NVDA", "TSLA", "AMD", "AMZN", "META", "JPM", "XOM", "NFLX", "BAC"]
WINDOW = 20

# --- Thresholds ---
HIGH_CORR_THRESHOLD = 0.6
LOW_CORR_THRESHOLD = 0.3


def load_daily(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_daily_regime.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    # 1. Load SPY
    spy = load_daily("SPY")[["date", "close"]].rename(columns={"close": "spy_close"})
    spy = spy.sort_values("date").drop_duplicates(subset="date")
    spy["spy_ret"] = spy["spy_close"].pct_change()

    # 2. Load symbols, build return matrix aligned to SPY dates
    returns: dict[str, pd.Series] = {}
    for sym in SYMBOLS:
        df = load_daily(sym)[["date", "close"]].sort_values("date").drop_duplicates(subset="date")
        df = df.set_index("date")["close"].pct_change()
        returns[sym] = df

    ret_df = pd.DataFrame(returns)
    # Align to SPY trading dates
    spy_dates = spy.set_index("date")
    ret_df = ret_df.reindex(spy_dates.index)

    # 3. Rolling pairwise correlation and average
    n_symbols = len(SYMBOLS)
    n_symbols * (n_symbols - 1) // 2

    avg_corrs = []
    dates = ret_df.index.tolist()

    for i in range(len(dates)):
        if i < WINDOW - 1:
            avg_corrs.append(np.nan)
            continue
        window_data = ret_df.iloc[i - WINDOW + 1 : i + 1]
        if window_data.dropna(how="all").shape[0] < WINDOW // 2:
            avg_corrs.append(np.nan)
            continue
        corr_matrix = window_data.corr()
        # Extract upper triangle (excluding diagonal)
        upper = corr_matrix.values[np.triu_indices(n_symbols, k=1)]
        avg_corrs.append(np.nanmean(upper))

    market_df = pd.DataFrame({"date": dates, "avg_correlation": avg_corrs})
    market_df["correlation_regime"] = market_df["avg_correlation"].apply(
        lambda x: "HIGH_CORR"
        if x > HIGH_CORR_THRESHOLD
        else ("LOW_CORR" if x < LOW_CORR_THRESHOLD else "NORMAL_CORR")
        if pd.notna(x)
        else None
    )

    # 4. Rolling 20-day SPY beta per symbol
    spy_ret_series = spy.set_index("date")["spy_ret"]

    for sym in SYMBOLS:
        sym_ret = returns[sym].reindex(spy_ret_series.index)
        betas = []
        for i in range(len(spy_ret_series)):
            if i < WINDOW - 1:
                betas.append(np.nan)
                continue
            s = sym_ret.iloc[i - WINDOW + 1 : i + 1]
            m = spy_ret_series.iloc[i - WINDOW + 1 : i + 1]
            valid = s.notna() & m.notna()
            if valid.sum() < WINDOW // 2:
                betas.append(np.nan)
                continue
            cov = np.cov(s[valid].values, m[valid].values)
            var_m = cov[1, 1]
            betas.append(cov[0, 1] / var_m if var_m > 0 else np.nan)
        market_df[f"beta_{sym}"] = betas

    # 5. Save market-wide file
    out_path = DATA_DIR / "MARKET_correlation_regime.parquet"
    market_df.to_parquet(out_path, index=False)
    print(f"Saved {out_path} ({len(market_df)} rows)")

    # 6. Enrich each symbol's daily regime parquet
    merge_cols = ["date", "correlation_regime"]
    beta_lookup = market_df[["date"] + [f"beta_{s}" for s in SYMBOLS]].copy()

    for sym in SYMBOLS:
        sym_path = DATA_DIR / f"{sym}_daily_regime.parquet"
        df = pd.read_parquet(sym_path)
        df["date"] = pd.to_datetime(df["date"])

        # Drop old columns if re-running
        for col in ["correlation_regime", "spy_beta"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Merge correlation regime
        df = df.merge(market_df[merge_cols], on="date", how="left")

        # Merge symbol-specific beta
        beta_col = f"beta_{sym}"
        sym_beta = beta_lookup[["date", beta_col]].rename(columns={beta_col: "spy_beta"})
        df = df.merge(sym_beta, on="date", how="left")

        df.to_parquet(sym_path, index=False)
        print(f"  Enriched {sym}: +correlation_regime, +spy_beta")

    # 7. Summary
    valid = market_df.dropna(subset=["correlation_regime"])
    print("\n=== Correlation Regime Distribution ===")
    dist = valid["correlation_regime"].value_counts()
    total = len(valid)
    for regime in ["HIGH_CORR", "NORMAL_CORR", "LOW_CORR"]:
        count = dist.get(regime, 0)
        print(f"  {regime:12s}: {count:5d} days ({count / total * 100:5.1f}%)")

    print("\n=== Average Pairwise Correlation by Year ===")
    valid_copy = valid.copy()
    valid_copy["year"] = valid_copy["date"].dt.year
    yearly = valid_copy.groupby("year")["avg_correlation"].mean()
    for year, avg in yearly.items():
        print(f"  {year}: {avg:.3f}")

    print("\n=== Top 10 Highest Correlation Days ===")
    top = valid.nlargest(10, "avg_correlation")[["date", "avg_correlation", "correlation_regime"]]
    for _, row in top.iterrows():
        print(
            f"  {row['date'].strftime('%Y-%m-%d')}: avg_corr={row['avg_correlation']:.3f} ({row['correlation_regime']})"
        )


if __name__ == "__main__":
    main()
