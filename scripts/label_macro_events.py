"""Label economic event windows (OPEX proximity, FOMC) on daily regime parquets."""

import datetime
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "regime_labeled")
DATA_DIR = os.environ.get("CERBERUS_REGIME_DIR", _DEFAULT_DATA_DIR)

# ---------------------------------------------------------------------------
# Part A: OPEX dates
# ---------------------------------------------------------------------------

QUARTERLY_MONTHS = {3, 6, 9, 12}


def third_friday(year: int, month: int) -> datetime.date:
    """Return the 3rd Friday of the given month."""
    # 1st day of month
    first = datetime.date(year, month, 1)
    # weekday: Monday=0 ... Friday=4
    # days until first Friday
    days_to_friday = (4 - first.weekday()) % 7
    first_friday = first + datetime.timedelta(days=days_to_friday)
    # 3rd Friday = first Friday + 14 days
    return first_friday + datetime.timedelta(days=14)


def generate_opex_dates(
    start_year: int = 2020, end_year: int = 2026, end_month: int = 3
) -> tuple[list[datetime.date], list[datetime.date]]:
    """Generate monthly and quarterly OPEX dates."""
    monthly = []
    quarterly = []
    for year in range(start_year, end_year + 1):
        month_end = 12 if year < end_year else end_month
        for month in range(1, month_end + 1):
            d = third_friday(year, month)
            if d > datetime.date(end_year, end_month, 31):
                break
            monthly.append(d)
            if month in QUARTERLY_MONTHS:
                quarterly.append(d)
    return monthly, quarterly


# ---------------------------------------------------------------------------
# Part B: FOMC dates (static)
# ---------------------------------------------------------------------------

FOMC_RAW = {
    2020: [(1, 29), (3, 3), (3, 15), (4, 29), (6, 10), (7, 29), (9, 16), (11, 5), (12, 16)],
    2021: [(1, 27), (3, 17), (4, 28), (6, 16), (7, 28), (9, 22), (11, 3), (12, 15)],
    2022: [(1, 26), (3, 16), (5, 4), (6, 15), (7, 27), (9, 21), (11, 2), (12, 14)],
    2023: [(2, 1), (3, 22), (5, 3), (6, 14), (7, 26), (9, 20), (11, 1), (12, 13)],
    2024: [(1, 31), (3, 20), (5, 1), (6, 12), (7, 31), (9, 18), (11, 7), (12, 18)],
    2025: [(1, 29), (3, 19), (5, 7), (6, 18), (7, 30), (9, 17), (10, 29), (12, 17)],
}


def fomc_dates() -> list[datetime.date]:
    dates = []
    for year, entries in FOMC_RAW.items():
        for month, day in entries:
            dates.append(datetime.date(year, month, day))
    return sorted(dates)


# ---------------------------------------------------------------------------
# Labeling helpers
# ---------------------------------------------------------------------------


def add_opex_labels(df: pd.DataFrame, monthly: list[datetime.date], quarterly: list[datetime.date]) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"]).dt.date.values
    # Build a set of all trading dates in this dataframe for trading-day distance
    trading_dates_set = set(dates)
    trading_dates_sorted = sorted(trading_dates_set)
    td_index = {d: i for i, d in enumerate(trading_dates_sorted)}

    monthly_arr = np.array(monthly)
    quarterly_set = set(quarterly)

    days_to_opex = np.empty(len(dates), dtype=np.int64)
    opex_week = np.empty(len(dates), dtype=bool)
    quad_witch_week = np.empty(len(dates), dtype=bool)

    for i, d in enumerate(dates):
        # days_to_opex: calendar days until next monthly OPEX (keep negative for day-after)
        future_diffs = [(m - d).days for m in monthly_arr if (m - d).days >= -1]
        days_to_opex[i] = min(future_diffs) if future_diffs else 999

        # opex_week: within 5 trading days up to and including OPEX
        # (i.e. OPEX week Mon-Fri, where OPEX is Friday)
        d_idx = td_index.get(d)
        in_opex = False
        in_quad = False
        if d_idx is not None:
            for m in monthly_arr:
                m_idx = td_index.get(m)
                if m_idx is not None:
                    # d is in OPEX week if d_idx is within [m_idx-4, m_idx]
                    # (5 trading days: Mon through Fri of OPEX week)
                    diff = m_idx - d_idx
                    if 0 <= diff <= 4:
                        in_opex = True
                        if m in quarterly_set:
                            in_quad = True
                        break

        opex_week[i] = in_opex
        quad_witch_week[i] = in_quad

    df["days_to_opex"] = days_to_opex
    df["opex_week"] = opex_week
    df["quad_witch_week"] = quad_witch_week
    return df


def add_fomc_labels(df: pd.DataFrame, fomc_list: list[datetime.date]) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"]).dt.date.values
    fomc_set = set(fomc_list)

    fomc_window = []
    near_fomc = np.empty(len(dates), dtype=bool)

    for i, d in enumerate(dates):
        if d in fomc_set:
            fomc_window.append("fomc_day")
            near_fomc[i] = True
        elif any((f - d).days in (1, 2) for f in fomc_list):
            # d is 1 or 2 calendar days BEFORE an FOMC date
            fomc_window.append("pre_2d")
            near_fomc[i] = True
        elif any((d - f).days == 1 for f in fomc_list):
            # d is 1 calendar day AFTER an FOMC date
            fomc_window.append("post_1d")
            near_fomc[i] = True
        else:
            fomc_window.append("none")
            near_fomc[i] = False

    df["fomc_window"] = fomc_window
    df["near_fomc"] = near_fomc
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    monthly_opex, quarterly_opex = generate_opex_dates()
    fomc_list = fomc_dates()

    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith("_daily_regime.parquet"))
    if not files:
        print("No regime parquet files found.")
        return

    total_rows = 0
    opex_week_count = 0
    quad_witch_count = 0
    near_fomc_count = 0
    fomc_window_dist: dict[str, int] = {"pre_2d": 0, "fomc_day": 0, "post_1d": 0, "none": 0}

    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_parquet(path)

        # Drop old columns if re-running
        for col in ["days_to_opex", "opex_week", "quad_witch_week", "fomc_window", "near_fomc"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = add_opex_labels(df, monthly_opex, quarterly_opex)
        df = add_fomc_labels(df, fomc_list)
        df.to_parquet(path, index=False)

        total_rows += len(df)
        opex_week_count += df["opex_week"].sum()
        quad_witch_count += df["quad_witch_week"].sum()
        near_fomc_count += df["near_fomc"].sum()
        for label in fomc_window_dist:
            fomc_window_dist[label] += (df["fomc_window"] == label).sum()

    normal_rows = total_rows - opex_week_count

    print("=" * 60)
    print("MACRO EVENT LABELING SUMMARY")
    print("=" * 60)
    print(f"Files processed:          {len(files)}")
    print(f"Total trading day rows:   {total_rows}")
    print()
    print("--- OPEX ---")
    print(f"Monthly OPEX dates:       {len(monthly_opex)}")
    print(f"Quarterly OPEX dates:     {len(quarterly_opex)}")
    print(f"Rows in OPEX week:        {opex_week_count}  ({opex_week_count / total_rows * 100:.1f}%)")
    print(f"Rows in quad-witch week:  {quad_witch_count}  ({quad_witch_count / total_rows * 100:.1f}%)")
    print(f"Rows NOT in OPEX week:    {normal_rows}  ({normal_rows / total_rows * 100:.1f}%)")
    print()
    print("--- FOMC ---")
    print(f"Total FOMC dates:         {len(fomc_list)}")
    print(f"Rows near FOMC:           {near_fomc_count}  ({near_fomc_count / total_rows * 100:.1f}%)")
    print(
        f"Rows NOT near FOMC:       {total_rows - near_fomc_count}  ({(total_rows - near_fomc_count) / total_rows * 100:.1f}%)"
    )
    print("FOMC window distribution:")
    for label, count in fomc_window_dist.items():
        print(f"  {label:12s}: {count:6d}  ({count / total_rows * 100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
