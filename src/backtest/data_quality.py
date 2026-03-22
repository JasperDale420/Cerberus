from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SymbolQuality:
    symbol: str
    total_bars: int
    expected_bars: int
    coverage_pct: float
    gap_count: int
    zero_volume_bars: int
    outlier_count: int
    stale_streak: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    symbols: dict[str, SymbolQuality] = field(default_factory=dict)
    excluded_symbols: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_data_quality(
    bars_by_symbol: dict[str, pd.DataFrame],
    min_coverage_pct: float = 80.0,
    exclude_below_pct: float = 50.0,
    max_gap_bars: int = 5,
    outlier_threshold: float = 0.15,
    bars_per_day: int = 390,
) -> DataQualityReport:
    report = DataQualityReport()

    for symbol, df in bars_by_symbol.items():
        if df.empty:
            report.excluded_symbols.append(symbol)
            continue

        total_bars = len(df)
        trading_days = df["timestamp"].dt.date.nunique()
        expected_bars = max(trading_days * bars_per_day, 1)
        coverage_pct = (total_bars / expected_bars) * 100.0

        # Gap detection
        ts_sorted = df["timestamp"].sort_values()
        diffs = ts_sorted.diff().dt.total_seconds().dropna()
        gap_count = int((diffs > max_gap_bars * 60).sum())

        # Zero volume
        zero_volume_bars = int((df["volume"] == 0).sum())

        # Price outliers
        closes = df["close"].values
        returns = np.diff(closes) / np.where(closes[:-1] != 0, closes[:-1], 1.0)
        outlier_count = int(np.sum(np.abs(returns) > outlier_threshold))

        # Stale prices (max consecutive identical closes)
        stale_streak = 0
        current_streak = 1
        for i in range(1, len(closes)):
            if closes[i] == closes[i - 1]:
                current_streak += 1
                stale_streak = max(stale_streak, current_streak)
            else:
                current_streak = 1

        sq = SymbolQuality(
            symbol=symbol,
            total_bars=total_bars,
            expected_bars=expected_bars,
            coverage_pct=min(coverage_pct, 100.0),
            gap_count=gap_count,
            zero_volume_bars=zero_volume_bars,
            outlier_count=outlier_count,
            stale_streak=stale_streak,
        )

        if coverage_pct < exclude_below_pct:
            report.excluded_symbols.append(symbol)
            sq.warnings.append(f"Excluded: coverage {coverage_pct:.1f}% < {exclude_below_pct}%")
        elif coverage_pct < min_coverage_pct:
            sq.warnings.append(f"Low coverage: {coverage_pct:.1f}%")

        report.symbols[symbol] = sq

    return report
