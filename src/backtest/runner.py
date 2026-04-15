import argparse
import asyncio
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Order as DbOrder
from src.analysis.schema import Signal as DbSignal
from src.backtest.backtest_report import BacktestReportCard, TradeRecord
from src.backtest.executor import SimulatedOrderExecutor
from src.backtest.indicator_precompute import (
    clear_precomputed,
    install_precomputed,
    is_installed,
    precompute_symbol_indicators,
)
from src.core.config import ConfigLoader
from src.core.domain import Bar, SymbolState
from src.core.logger import StructuredLogger
from src.data.client import UnifiedDataClient
from src.engine.execution import ExecutionEngine

_BAR_DATAFRAME_CACHE: dict[tuple[str, tuple[str, ...], str, str, int], pd.DataFrame] = {}
_BAR_CACHE_MAX_SIZE = 4  # Keep at most 4 entries (2 windows x raw + aggregated)
_ET_TZ = ZoneInfo("America/New_York")


def clear_bar_cache() -> None:
    """Explicitly clear the bar DataFrame cache to free memory between WFO windows."""
    _BAR_DATAFRAME_CACHE.clear()


def _dynamic_import_strategy_class(strategy_file: str):
    """Import a strategy module by file stem and return the BaseStrategy subclass.

    Used by autoresearch_driver.sh to verify a newly-written strategy file
    imports correctly before running WFO evaluation.

    Returns the class if found, or None if no BaseStrategy subclass exists.
    """
    import importlib
    import inspect

    from src.strategies.base import BaseStrategy

    module_path = f"src.strategies.{strategy_file}"
    try:
        module = importlib.import_module(module_path)
    except Exception:
        return None

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseStrategy) and obj is not BaseStrategy and obj.__module__ == module_path:
            return obj
    return None


def _should_flatten_position(strategy, hold_days: int) -> bool:
    """Determine if a position should be flattened at EOD."""
    if not getattr(strategy, "allow_overnight", False):
        return True
    if getattr(strategy, "max_hold_days", 0) > 0 and hold_days >= strategy.max_hold_days:
        return True
    return False


class BacktestClock:
    def __init__(self, start_time: datetime):
        self._now = start_time

    def __call__(self) -> datetime:
        return self._now

    def set_time(self, new_time: datetime):
        self._now = new_time


def _build_backtest_logger(config: dict) -> StructuredLogger:
    """Build the backtest logger honoring the configured log level."""
    level = str(config.get("log_level", "INFO"))
    return StructuredLogger("CERBERUS-BACKTEST", level=level, logging_config=config.get("logging"))


def _get_bar_resolution_minutes(config: dict) -> int:
    """Return the requested bar resolution for backtests.

    ``bar_resolution_minutes`` may be an integer (e.g. 1, 5) or the string
    ``"daily"`` which is normalised to 1440 (minutes per trading day).
    """
    backtest_cfg = config.get("backtest", {})
    if not isinstance(backtest_cfg, dict):
        return 1
    raw_value = backtest_cfg.get("bar_resolution_minutes", 1)
    if isinstance(raw_value, str) and raw_value.lower() == "daily":
        return 1440
    try:
        return max(int(raw_value), 1)
    except (TypeError, ValueError):
        return 1


def _apply_bar_resolution(bars_df: pd.DataFrame, bar_resolution_minutes: int) -> pd.DataFrame:
    """Aggregate 1-minute bars to a coarser resolution for faster optimization sweeps."""
    if bars_df.empty or bar_resolution_minutes <= 1:
        return bars_df

    frames: list[pd.DataFrame] = []

    for symbol, group in bars_df.groupby("symbol", sort=True):
        working = group.sort_values("timestamp").copy()
        ts_index = pd.DatetimeIndex(pd.to_datetime(working["timestamp"], utc=True))
        working = working.set_index(ts_index)

        if bar_resolution_minutes >= 1440:
            # Daily aggregation: group by trading date
            aggregated = working.groupby(working.index.date).agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            price_volume = (
                (working["vwap"].fillna(working["close"]) * working["volume"]).groupby(working.index.date).sum()
            )
            total_volume = working["volume"].groupby(working.index.date).sum()
        else:
            freq = f"{bar_resolution_minutes}min"
            aggregated = working.resample(freq, label="left", closed="left").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            price_volume = (
                (working["vwap"].fillna(working["close"]) * working["volume"])
                .resample(freq, label="left", closed="left")
                .sum()
            )
            total_volume = working["volume"].resample(freq, label="left", closed="left").sum()

        aggregated["vwap"] = price_volume.div(total_volume.where(total_volume != 0)).fillna(aggregated["close"])
        aggregated = aggregated.dropna(subset=["open", "high", "low", "close"]).reset_index()
        # Normalise the index column name (daily groupby produces 'index', resample produces different names)
        if aggregated.columns[0] != "timestamp":
            aggregated.rename(columns={aggregated.columns[0]: "timestamp"}, inplace=True)
        # Ensure timestamp is a proper datetime (daily groupby produces date objects)
        aggregated["timestamp"] = pd.to_datetime(aggregated["timestamp"], utc=True)
        aggregated["symbol"] = symbol
        frames.append(aggregated)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap", "symbol"])

    return pd.concat(frames, ignore_index=True)


def _prepare_bars_dataframe(bars_df: pd.DataFrame, bar_resolution_minutes: int) -> pd.DataFrame:
    """Normalize loaded bars into the replay-ready shape used by the backtest loop."""
    if bars_df.empty:
        return bars_df

    prepared = _apply_bar_resolution(bars_df, bar_resolution_minutes)
    prepared = prepared.sort_values(by="timestamp").reset_index(drop=True)

    if "vwap" not in prepared.columns:
        prepared["vwap"] = prepared["close"]
    else:
        prepared["vwap"] = prepared["vwap"].fillna(prepared["close"])

    for col in ("open", "high", "low", "close", "volume", "vwap"):
        prepared[col] = prepared[col].astype("float64")

    return prepared


def _build_bar_cache_key(
    data_dir: Path,
    symbols: set[str],
    start_dt: datetime,
    end_dt: datetime,
    bar_resolution_minutes: int,
) -> tuple[str, tuple[str, ...], str, str, int]:
    return (
        str(data_dir.resolve()),
        tuple(sorted(symbols)),
        start_dt.isoformat(),
        end_dt.isoformat(),
        int(bar_resolution_minutes),
    )


def _load_daily_bars_fast(
    data_dir: Path,
    symbols: set[str],
    start_dt: datetime,
    end_dt: datetime,
    logger: StructuredLogger,
) -> pd.DataFrame | None:
    """Load pre-aggregated daily bars if available (from precompute_daily_bars.py).

    Returns None if daily files don't exist, falling back to 1-min load + aggregation.
    Daily files are ~137× smaller than 1-min files, cutting worker RSS from ~1.9 GB to ~15 MB.
    """
    frames: list[pd.DataFrame] = []
    for symbol in sorted(symbols):
        daily_path = data_dir / f"{symbol}_1Day.parquet"
        if not daily_path.exists():
            return None  # Not all symbols have daily files — fall back to full load
        df = pd.read_parquet(daily_path)
        # Ensure timestamp column for compatibility
        if "timestamp" not in df.columns and "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        frames.append(df)

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(by="timestamp").reset_index(drop=True)
    # Ensure required float columns
    for col in ("open", "high", "low", "close", "volume"):
        if col in combined.columns:
            combined[col] = combined[col].astype("float64")
    if "vwap" not in combined.columns:
        combined["vwap"] = combined["close"]
    else:
        combined["vwap"] = combined["vwap"].fillna(combined["close"]).astype("float64")
    return combined


def _load_cached_parquet_bars(
    data_dir: Path,
    symbols: set[str],
    start_dt: datetime,
    end_dt: datetime,
    logger: StructuredLogger,
    bar_resolution_minutes: int,
) -> pd.DataFrame:
    """Load parquet bars once per worker process and reuse them across Optuna trials."""
    cache_key = _build_bar_cache_key(data_dir, symbols, start_dt, end_dt, bar_resolution_minutes)
    cached = _BAR_DATAFRAME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Fast path: load pre-aggregated daily bars if available and daily resolution requested
    if bar_resolution_minutes >= 1440:
        daily_fast = _load_daily_bars_fast(data_dir, symbols, start_dt, end_dt, logger)
        if daily_fast is not None:
            # Evict oldest entries if cache exceeds max size
            while len(_BAR_DATAFRAME_CACHE) >= _BAR_CACHE_MAX_SIZE:
                oldest_key = next(iter(_BAR_DATAFRAME_CACHE))
                del _BAR_DATAFRAME_CACHE[oldest_key]
            _BAR_DATAFRAME_CACHE[cache_key] = daily_fast
            return daily_fast

    loaded = _load_bars_from_parquet(data_dir, symbols, start_dt, end_dt, logger)
    prepared = _prepare_bars_dataframe(loaded, bar_resolution_minutes)

    # Evict oldest entries if cache exceeds max size
    while len(_BAR_DATAFRAME_CACHE) >= _BAR_CACHE_MAX_SIZE:
        oldest_key = next(iter(_BAR_DATAFRAME_CACHE))
        del _BAR_DATAFRAME_CACHE[oldest_key]

    _BAR_DATAFRAME_CACHE[cache_key] = prepared
    return prepared


def _load_bars_from_parquet(
    data_dir: Path,
    symbols: set[str],
    start_dt: datetime,
    end_dt: datetime,
    logger: StructuredLogger,
) -> pd.DataFrame:
    """Load 1-min bars from per-symbol parquet files in *data_dir*.

    Each file is expected to be ``{SYMBOL}_1Min.parquet`` with columns:
    ``[close, high, low, trade_count, open, timestamp, volume, vwap, symbol]``.

    Filters to the requested date range and returns a combined DataFrame
    with the same columns the runner expects (``o/h/l/c/v`` aliases plus
    ``timestamp`` and ``symbol``).
    """
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for symbol in sorted(symbols):
        parquet_path = data_dir / f"{symbol}_1Min.parquet"
        if not parquet_path.exists():
            missing.append(symbol)
            continue

        df = pd.read_parquet(parquet_path)

        # Ensure timezone-aware timestamps for filtering
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")

        # Filter to requested date range
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

        # Filter to Regular Trading Hours only (9:30 AM – 4:00 PM ET)
        et_times = df["timestamp"].dt.tz_convert("America/New_York")
        rth_mask = (et_times.dt.time >= pd.Timestamp("09:30").time()) & (
            et_times.dt.time < pd.Timestamp("16:00").time()
        )
        pre_filter_len = len(df)
        df = df[rth_mask]
        extended_dropped = pre_filter_len - len(df)

        if df.empty:
            logger.warning(
                "Parquet file has no bars in requested range",
                symbol=symbol,
                path=str(parquet_path),
            )
            continue

        # Ensure symbol column is set
        df["symbol"] = symbol

        frames.append(df)
        logger.info(
            "Loaded bars from parquet",
            symbol=symbol,
            bars=len(df),
            extended_hours_dropped=extended_dropped,
        )

    if missing:
        logger.warning(
            "Parquet files missing for symbols — these will be skipped",
            missing_symbols=missing,
        )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined


def _validate_survivorship(
    bars_df: pd.DataFrame,
    symbols: set[str],
    start_dt: datetime,
    end_dt: datetime,
    logger: StructuredLogger,
    bar_resolution_minutes: int = 1,
) -> None:
    """Log warnings for symbols with suspiciously few bars (survivorship bias check).

    A typical equity trades ~390 1-min bars per day.  We warn when a symbol
    has fewer than 50 % of the expected bar count for the date range.
    """
    if bars_df.empty:
        return

    trading_days = pd.bdate_range(start_dt, end_dt).size
    bars_per_day = max(int(round(390 / max(bar_resolution_minutes, 1))), 1)
    expected_bars = trading_days * bars_per_day

    bar_counts = bars_df.groupby("symbol").size()
    for symbol in sorted(symbols):
        count = bar_counts.get(symbol, 0)
        if count == 0:
            logger.warning("No bars at all for symbol — possible survivorship issue", symbol=symbol)
        elif count < expected_bars * 0.50:
            logger.warning(
                "Symbol has significantly fewer bars than expected — possible data gap",
                symbol=symbol,
                actual_bars=int(count),
                expected_bars=int(expected_bars),
                coverage_pct=round(count / expected_bars * 100, 1),
            )


_REGIME_LABEL_COLUMNS = [
    "regime_trend",
    "regime_vol",
    "regime",
    "liquidity_regime",
    "earnings_window",
    "near_earnings",
    "opex_week",
    "quad_witch_week",
    "fomc_window",
    "near_fomc",
    "correlation_regime",
    "spy_beta",
]


def _load_regime_labels(
    data_dir: Path,
    symbols: set[str],
    logger: StructuredLogger,
) -> dict[str, pd.DataFrame]:
    """Load pre-computed daily regime labels for each symbol.

    Returns a dict mapping symbol -> DataFrame indexed by ``datetime.date``.
    Missing files are logged as warnings and produce empty DataFrames.
    """
    regime_dir = data_dir / "regime_labeled"
    if not regime_dir.is_dir():
        logger.info("No regime_labeled directory found — skipping regime label loading", path=str(regime_dir))
        return {}

    result: dict[str, pd.DataFrame] = {}
    loaded = 0
    for symbol in sorted(symbols):
        parquet_path = regime_dir / f"{symbol}_daily_regime.parquet"
        if not parquet_path.exists():
            logger.debug("No daily regime labels for symbol", symbol=symbol, path=str(parquet_path))
            result[symbol] = pd.DataFrame()
            continue
        try:
            df = pd.read_parquet(parquet_path)
            # Normalize the date column to datetime.date for fast lookup
            df["_date_key"] = pd.to_datetime(df["date"]).dt.date
            df = df.set_index("_date_key")
            # Deduplicate index to prevent .loc returning DataFrame instead of Series
            df = df[~df.index.duplicated(keep="last")]
            result[symbol] = df
            loaded += 1
        except Exception as e:
            logger.warning("Failed to load regime labels", symbol=symbol, error=str(e))
            result[symbol] = pd.DataFrame()

    logger.info("Loaded regime labels", symbols_loaded=loaded, symbols_total=len(symbols))
    return result


def _load_session_labels(
    data_dir: Path,
    symbols: set[str],
    logger: StructuredLogger,
) -> dict[str, pd.DataFrame]:
    """Load pre-computed 1-minute session phase labels for each symbol.

    Returns a dict mapping symbol -> DataFrame indexed by UTC timestamp (rounded to minute).
    """
    regime_dir = data_dir / "regime_labeled"
    if not regime_dir.is_dir():
        return {}

    result: dict[str, pd.DataFrame] = {}
    loaded = 0
    for symbol in sorted(symbols):
        parquet_path = regime_dir / f"{symbol}_1m_session.parquet"
        if not parquet_path.exists():
            result[symbol] = pd.DataFrame()
            continue
        try:
            df = pd.read_parquet(parquet_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
            result[symbol] = df
            loaded += 1
        except Exception as e:
            logger.warning("Failed to load session labels", symbol=symbol, error=str(e))
            result[symbol] = pd.DataFrame()

    if loaded:
        logger.info("Loaded session labels", symbols_loaded=loaded)
    return result


def _build_trade_records(
    db: DatabaseDatabase,
    config: dict,
    regime_labels: dict[str, pd.DataFrame] | None = None,
) -> list[TradeRecord]:
    """Pair filled buy/sell orders into TradeRecord objects using FIFO matching.

    Uses a position-stack approach (same as BacktestAnalyzer._process_symbol_fills
    in stats.py) so that partial fills, scaling in/out, and multiple entries are
    handled correctly.

    When *regime_labels* is provided, the entry date's regime context is attached
    to each TradeRecord's ``meta`` dict under the ``"regime_labels"`` key.
    """
    from collections import deque

    QTY_EPSILON = 1e-7

    # Extract commission config from risk settings
    risk_cfg = config.get("risk", {})
    commission_per_share = float(risk_cfg.get("commission_per_share", 0.0) or 0.0)
    min_commission = float(risk_cfg.get("min_commission", 0.0) or 0.0)

    # Extract all data inside the session to avoid DetachedInstanceError
    with db.get_session() as session:
        fills = session.query(DbOrder).filter(DbOrder.status == "filled").order_by(DbOrder.time_placed).all()
        # Materialize to plain dicts while session is open
        fill_dicts = [
            {
                "symbol": f.symbol,
                "side": f.side,
                "qty": float(f.qty),
                "limit_price": float(f.limit_price) if f.limit_price else 0.0,
                "time_placed": f.time_placed,
                "correlation_id": f.correlation_id or "",
                "meta_json": f.meta_json if isinstance(f.meta_json, dict) else {},
            }
            for f in fills
        ]

        # Build correlation_id → strategy lookup from Signal table
        sig_rows = session.query(DbSignal.correlation_id, DbSignal.strategy).all()
        corr_to_strategy: dict[str, str] = {s.correlation_id: s.strategy for s in sig_rows}

    # Group by symbol
    by_symbol: dict[str, list[dict]] = {}
    for f in fill_dicts:
        by_symbol.setdefault(f["symbol"], []).append(f)

    trades: list[TradeRecord] = []

    for sym, sym_fills in by_symbol.items():
        # Sort chronologically within each symbol
        sym_fills.sort(key=lambda x: x["time_placed"])

        # FIFO position stack: each entry is {side, qty, price, time}
        stack: deque[dict] = deque()

        for fill in sym_fills:
            remaining = fill["qty"]
            fill_side = fill["side"]
            fill_price = fill["limit_price"]
            fill_time = fill["time_placed"]

            fill_meta = fill.get("meta_json", {})

            while remaining > QTY_EPSILON:
                if not stack:
                    # No existing position -- open a new one
                    stack.append(
                        {
                            "side": fill_side,
                            "qty": remaining,
                            "price": fill_price,
                            "time": fill_time,
                            "correlation_id": fill.get("correlation_id", ""),
                            "meta_json": fill_meta,
                        }
                    )
                    remaining = 0.0
                elif stack[0]["side"] == fill_side:
                    # Same direction -- add to position
                    stack.append(
                        {
                            "side": fill_side,
                            "qty": remaining,
                            "price": fill_price,
                            "time": fill_time,
                            "correlation_id": fill.get("correlation_id", ""),
                            "meta_json": fill_meta,
                        }
                    )
                    remaining = 0.0
                else:
                    # Opposite direction -- close/reduce the oldest entry
                    head = stack[0]
                    match_qty = min(remaining, head["qty"])

                    entry_price = head["price"]
                    exit_price = fill_price

                    if head["side"] == "buy":
                        pnl = (exit_price - entry_price) * match_qty
                    else:
                        pnl = (entry_price - exit_price) * match_qty

                    # Resolve strategy from entry order's correlation_id
                    entry_corr = head.get("correlation_id", "")
                    if entry_corr.startswith("flatten-"):
                        strategy = "_flatten"
                    else:
                        strategy = corr_to_strategy.get(entry_corr, "")
                        # Fallback: parse strategy from correlation_id format
                        # (strategy-SYMBOL-timestamp-hash). In backtest mode,
                        # signals aren't persisted to DB so the lookup is empty.
                        if not strategy and entry_corr:
                            parts = entry_corr.split("-")
                            if len(parts) >= 4:
                                # Strategy name may contain hyphens (e.g. "pair_trading_v2")
                                # but symbols and timestamps don't contain underscores at
                                # this position, so take everything before the symbol segment.
                                # The format is: {strategy}-{SYMBOL}-{epoch_ms}-{hex_hash}
                                # Find the symbol segment (all uppercase letters)
                                for i, part in enumerate(parts[1:], 1):
                                    if part.isupper() or (part.isalpha() and part == part.upper()):
                                        strategy = "-".join(parts[:i])
                                        break

                    # Commission: entry leg + exit leg
                    entry_leg_commission = max(min_commission, commission_per_share * match_qty)
                    exit_leg_commission = max(min_commission, commission_per_share * match_qty)
                    total_commission = entry_leg_commission + exit_leg_commission

                    # Make PnL net of commission
                    net_pnl = pnl - total_commission

                    # -- Regime context from order meta_json (signal-level) --
                    entry_order_meta = head.get("meta_json", {}) or {}
                    exit_order_meta = fill_meta or {}
                    entry_regime = entry_order_meta.get("regime_labels", {}) or {}
                    exit_regime = exit_order_meta.get("regime_labels", {}) or {}

                    # -- Regime context from parquet labels (fallback) --
                    trade_meta: dict[str, Any] = {}
                    parquet_regime: dict[str, Any] = {}
                    if regime_labels:
                        entry_date = head["time"].date() if hasattr(head["time"], "date") else None
                        if entry_date:
                            rl_df = regime_labels.get(sym)
                            if rl_df is not None and not rl_df.empty and entry_date in rl_df.index:
                                rl_row = rl_df.loc[entry_date]
                                parquet_regime = {
                                    col: rl_row[col] for col in _REGIME_LABEL_COLUMNS if col in rl_row.index
                                }
                                trade_meta["regime_labels"] = parquet_regime

                    # Resolve: prefer signal meta, then parquet, then "unknown"
                    def _resolve(
                        signal_val: Any,
                        parquet_key: str,
                        default: str = "unknown",
                        _pq: dict[str, Any] = parquet_regime,
                    ) -> str:
                        if signal_val and str(signal_val) != "unknown":
                            return str(signal_val)
                        pq = _pq.get(parquet_key)
                        if pq is not None and str(pq) != "":
                            return str(pq)
                        return default

                    entry_trend = _resolve(entry_regime.get("trend"), "regime_trend")
                    entry_vol = _resolve(entry_regime.get("vol"), "regime_vol")
                    entry_liq = _resolve(entry_regime.get("liquidity"), "liquidity_regime")
                    entry_session = _resolve(entry_regime.get("session"), "session_phase")
                    entry_corr_regime = _resolve(
                        entry_order_meta.get("correlation"),
                        "correlation_regime",
                    )
                    # Use 'is not None' for boolean fields — 'or' short-circuits on False
                    _om_earn = entry_order_meta.get("near_earnings")
                    entry_near_earn = bool(
                        _om_earn if _om_earn is not None else parquet_regime.get("near_earnings", False)
                    )
                    _om_fomc = entry_order_meta.get("near_fomc")
                    entry_near_fomc_val = bool(
                        _om_fomc if _om_fomc is not None else parquet_regime.get("near_fomc", False)
                    )
                    _om_opex = entry_order_meta.get("opex_week")
                    entry_opex = bool(_om_opex if _om_opex is not None else parquet_regime.get("opex_week", False))

                    # Exit regime: from exit order's meta_json regime_labels
                    exit_trend = str(exit_regime.get("trend", "unknown") or "unknown")
                    exit_vol_val = str(exit_regime.get("vol", "unknown") or "unknown")

                    trades.append(
                        TradeRecord(
                            symbol=sym,
                            side=head["side"],
                            qty=match_qty,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            entry_time=head["time"],
                            exit_time=fill_time,
                            pnl=round(net_pnl, 2),
                            commission=round(total_commission, 2),
                            strategy=strategy,
                            meta=trade_meta,
                            entry_regime_trend=entry_trend,
                            entry_regime_vol=entry_vol,
                            entry_liquidity=entry_liq,
                            entry_correlation=entry_corr_regime,
                            entry_near_earnings=entry_near_earn,
                            entry_near_fomc=entry_near_fomc_val,
                            entry_opex_week=entry_opex,
                            entry_session_phase=entry_session,
                            exit_regime_trend=exit_trend,
                            exit_regime_vol=exit_vol_val,
                        )
                    )

                    head["qty"] -= match_qty
                    if head["qty"] <= QTY_EPSILON:
                        stack.popleft()

                    remaining -= match_qty

    return trades


def _maybe_aggregate_bar(
    state: SymbolState,
    bar: Bar,
    tf_minutes: int,
    target_deque: deque[Bar],
) -> None:
    """Aggregate 1m bars into higher-TF bars and append completed bars.

    Accumulates incoming 1m bars in ``state.indicators["_agg_{tf}m"]``.
    When a new TF boundary is crossed, the accumulated bars are flushed
    into a single OHLCV bar and appended to *target_deque*.
    """
    key = f"_agg_{tf_minutes}m"
    accum = state.indicators.get(key)
    if accum is None:
        accum = {"bars": [], "boundary": None}
        state.indicators[key] = accum

    bar_time = bar.time
    minute = bar_time.minute
    boundary_minute = (minute // tf_minutes) * tf_minutes
    boundary = bar_time.replace(minute=boundary_minute, second=0, microsecond=0)

    if accum["boundary"] is not None and boundary != accum["boundary"] and accum["bars"]:
        # Emit the completed higher-TF bar
        agg_bars = accum["bars"]
        total_volume = sum(float(b.volume) for b in agg_bars)
        # Volume-weighted VWAP if available
        vwap_val = None
        if total_volume > 0:
            vwap_sum = sum(float(b.vwap or ((b.high + b.low + b.close) / 3.0)) * float(b.volume) for b in agg_bars)
            vwap_val = vwap_sum / total_volume

        agg_bar = Bar(
            symbol=bar.symbol,
            time=accum["boundary"],
            open=float(agg_bars[0].open),
            high=max(float(b.high) for b in agg_bars),
            low=min(float(b.low) for b in agg_bars),
            close=float(agg_bars[-1].close),
            volume=total_volume,
            vwap=vwap_val,
        )
        target_deque.append(agg_bar)
        accum["bars"] = [bar]
    else:
        accum["bars"].append(bar)

    accum["boundary"] = boundary


async def run_backtest(
    start_date: str,
    end_date: str,
    config_path: str,
    data_dir: str | None = None,
    skip_indicator_precompute: bool = False,
):
    import dotenv

    dotenv.load_dotenv()

    # Ensure local gateway works without docker DNS
    if not os.environ.get("CERBERUS_GATEWAY_URL"):
        os.environ["CERBERUS_GATEWAY_URL"] = "http://localhost:8080"
    elif "host.docker.internal" in os.environ.get("CERBERUS_GATEWAY_URL", ""):
        os.environ["CERBERUS_GATEWAY_URL"] = "http://localhost:8080"

    config_loader = ConfigLoader()
    config = config_loader.load_config(config_path)
    logger = _build_backtest_logger(config)

    # Use a temporary file-based DB instead of memory to avoid weird thread issues with sqlite.
    # If the config already has a per-trial DB URL (set by the optimization harness for
    # parallel execution), respect it instead of overwriting with a shared path.
    existing_url = config.get("database_url", "")
    if "optuna_dbs/" in existing_url:
        # Per-trial DB from optimization harness — use as-is
        pass
    else:
        import tempfile

        db_fd, db_path = tempfile.mkstemp(suffix=".db", prefix="cerberus_bt_")
        os.close(db_fd)
        config["database_url"] = f"sqlite:///{db_path}"

    db = DatabaseDatabase(config_loader, logger, config=config, config_path_or_dir=config_path)
    db.init_db()

    start_dt = datetime.fromisoformat(start_date)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)

    end_dt = datetime.fromisoformat(end_date)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    clock = BacktestClock(start_dt)
    bar_resolution_minutes = _get_bar_resolution_minutes(config)

    # Collect universe symbols
    symbols: set[str] = set()
    index_symbol = config.get("index_symbol", "SPY")
    vol_symbol = config.get("regime", {}).get("vol_symbol", "VXX")
    symbols.add(index_symbol)
    symbols.add(vol_symbol)

    universes = config.get("universe", {})
    if isinstance(universes, list):
        for u in universes:
            for s in u.get("symbols", []):
                symbols.add(s)
    elif isinstance(universes, dict):
        for s in universes.get("symbols", []):
            symbols.add(s)

    # ── Load bar data ────────────────────────────────────────────────
    logger.info("Fetching historical data for backtest...", start_date=start_date, end_date=end_date)

    if data_dir:
        data_path = Path(data_dir)
        if not data_path.is_dir():
            logger.error("--data-dir does not exist", path=data_dir)
            return
        logger.info("Loading bars from local parquet cache", data_dir=data_dir)
        bars_df = _load_cached_parquet_bars(
            data_path,
            symbols,
            start_dt,
            end_dt,
            logger,
            bar_resolution_minutes,
        )
    else:
        # Fetch from API (legacy path)
        gateway_url = os.environ.get("CERBERUS_GATEWAY_URL", "http://localhost:8080")
        gateway_key = os.environ.get("CERBERUS_GATEWAY_KEY", "")
        gateway_client = UnifiedDataClient(gateway_url, gateway_key)
        bars_list: list[dict] = []
        for symbol in symbols:
            try:
                logger.info("Fetching bars for symbol...", symbol=symbol)
                resp = gateway_client.get_historical_bars(symbol=symbol, start=start_dt, end=end_dt, timeframe="1Min")
                bars = resp.get("bars", [])
                for b in bars:
                    b["symbol"] = symbol
                    if isinstance(b["t"], str):
                        b["timestamp"] = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
                    else:
                        b["timestamp"] = pd.to_datetime(b["t"])
                    b["open"] = b["o"]
                    b["high"] = b["h"]
                    b["low"] = b["l"]
                    b["close"] = b["c"]
                    b["volume"] = b["v"]
                bars_list.extend(bars)
            except Exception as e:
                logger.error("Failed to fetch historical data for symbol", symbol=symbol, error=str(e))
                continue
        bars_df = pd.DataFrame(bars_list) if bars_list else pd.DataFrame()
        bars_df = _prepare_bars_dataframe(bars_df, bar_resolution_minutes)

    if bars_df.empty:
        logger.warning("No historical data found for the given dates.")
        return

    # Survivorship bias check
    _validate_survivorship(bars_df, symbols, start_dt, end_dt, logger, bar_resolution_minutes)

    # ── Data quality checks ─────────────────────────────────────────
    from src.backtest.data_quality import check_data_quality

    bars_by_symbol = {sym: grp for sym, grp in bars_df.groupby("symbol")}
    dq_cfg = config.get("analytics", {}).get("data_quality", {})
    # Adjust expected bars per day based on bar resolution
    # Daily bars = 1 bar/day, 5-min = 78 bars/day, 1-min = 390 bars/day
    if bar_resolution_minutes >= 1440:
        bars_per_day = 1
    elif bar_resolution_minutes > 1:
        bars_per_day = 390 // bar_resolution_minutes
    else:
        bars_per_day = 390
    dq_report = check_data_quality(
        bars_by_symbol,
        min_coverage_pct=dq_cfg.get("min_coverage_pct", 80.0),
        exclude_below_pct=dq_cfg.get("exclude_below_pct", 50.0),
        bars_per_day=bars_per_day,
    )
    for sym in dq_report.excluded_symbols:
        logger.warning("data_quality_excluded", symbol=sym)
        bars_by_symbol.pop(sym, None)
    for sym, sq in dq_report.symbols.items():
        for w in sq.warnings:
            logger.warning("data_quality_warning", symbol=sym, warning=w)
    if dq_report.excluded_symbols:
        bars_df = pd.concat(bars_by_symbol.values(), ignore_index=True) if bars_by_symbol else pd.DataFrame()
        if bars_df.empty:
            logger.warning("All symbols excluded by data quality checks.")
            return

    # ── Load pre-computed regime labels (optional) ──────────────────
    regime_labels: dict[str, pd.DataFrame] = {}
    session_labels: dict[str, pd.DataFrame] = {}
    regime_data_dir = Path(data_dir) if data_dir else Path("data")
    regime_labels = _load_regime_labels(regime_data_dir, symbols, logger)
    session_labels = _load_session_labels(regime_data_dir, symbols, logger)

    # ── Pre-compute multi-timeframe indicators ───────────────────────
    # V2 strategies depend on 5m/15m EMA, ATR, RSI, ADX etc. via
    # MultiTimeframeAnalyzer.  Pre-computing them in batch (Numba) and
    # installing into the module-level cache activates the fast-path
    # in _compute_tf() so strategies get valid indicator values.
    #
    # When skip_indicator_precompute is True, the caller has already
    # installed indicators (e.g., the Optuna worker pre-installs once
    # per WFO window to avoid recomputing across trials).
    if skip_indicator_precompute and any(is_installed(s) for s in symbols):
        logger.info("Skipping MTF indicator precompute — already installed")
    else:
        clear_precomputed()
        for _sym in sorted(symbols):
            _sym_df = bars_df[bars_df["symbol"] == _sym].sort_values("timestamp")
            if _sym_df.empty:
                continue
            try:
                _precomp = precompute_symbol_indicators(
                    timestamps=_sym_df["timestamp"].values,
                    opens=_sym_df["open"].values,
                    highs=_sym_df["high"].values,
                    lows=_sym_df["low"].values,
                    closes=_sym_df["close"].values,
                    volumes=_sym_df["volume"].values,
                    vwaps=_sym_df["vwap"].values,
                )
                install_precomputed(_sym, _precomp)
                logger.info(
                    "Pre-computed MTF indicators",
                    symbol=_sym,
                    tf_1m_bars=len(_sym_df),
                )
            except Exception as e:
                logger.warning(
                    "MTF indicator pre-computation failed for symbol — falling back to slow path",
                    symbol=_sym,
                    error=str(e),
                )

    engine = ExecutionEngine(
        config=config,
        logger=logger,
        db=db,
        clock=clock,
    )

    # Setup mock account for equity tracking
    class MockAccount:
        def __init__(self, initial_cash: float):
            self.cash = initial_cash
            self.equity = initial_cash
            self.positions_qty: dict[str, float] = {}

    initial_cash = config.get("initial_cash", 100_000.0)
    engine.account = MockAccount(initial_cash)

    # Inject our mock executor with backtest config for slippage
    backtest_cfg = config.get("backtest", {})
    risk_cfg = config.get("risk", config.get("risk_management", {}))

    from src.backtest.fill_models import create_fill_model

    fill_model = create_fill_model(config)
    executor = SimulatedOrderExecutor(
        logger=logger,
        db=db,
        clock=clock,
        on_trade_update=engine.on_trade_update,
        account=engine.account,
        backtest_cfg=backtest_cfg,
        risk_cfg=risk_cfg,
        fill_model=fill_model,
    )
    # Wire advanced exits config so executor delegates stop/TP to PositionManager
    executor.configure_advanced_exits(risk_cfg)
    engine.order_executor = executor
    engine.backtest_mode = True  # Enable scanner_bypass for V2 strategies
    engine.market_manager.persist_to_db = False  # Skip per-bar RegimeHistory DB writes

    # Register strategies
    from src.main import _build_strategy_registry

    registry = _build_strategy_registry()
    strategies_cfg = config.get("strategies", {})
    for name, strat_cfg in strategies_cfg.items():
        if isinstance(strat_cfg, dict) and strat_cfg.get("enabled", True):
            cls = registry.get(name)
            if cls is None:
                # Dynamic import fallback for autoresearch strategies not in the hardcoded registry
                import importlib.util as _ilu
                import sys as _sys_mod

                from src.strategies.base import BaseStrategy as _BaseStrategy

                _strat_path = Path(f"src/strategies/{name}.py")
                if _strat_path.exists():
                    try:
                        _spec = _ilu.spec_from_file_location(f"src.strategies.{name}", _strat_path)
                        if _spec and _spec.loader:
                            _mod = _ilu.module_from_spec(_spec)
                            _sys_mod.modules[_spec.name] = _mod
                            _spec.loader.exec_module(_mod)
                            for _attr_name in dir(_mod):
                                _attr = getattr(_mod, _attr_name)
                                if (
                                    isinstance(_attr, type)
                                    and issubclass(_attr, _BaseStrategy)
                                    and _attr is not _BaseStrategy
                                    and getattr(_attr, "name", None) == name
                                ):
                                    cls = _attr
                                    break
                    except Exception as _e:
                        logger.warning("Dynamic strategy import failed", strategy=name, error=str(_e))
            if cls:
                engine.register_strategy(cls(strat_cfg, logger))

    # ── Backtest-aware flatten ───────────────────────────────────────
    def _backtest_flatten_all(reason: str, symbols: list[str] | None = None) -> None:
        """Close open positions and cancel pending orders in simulation.

        Uses account.positions_qty (synchronously updated by executor) as the
        source of truth. Bypasses executor.submit/process_bar to avoid partial
        fill logic — EOD flattens must be full fills to prevent position dust
        from carrying over to the next day.

        Args:
            reason: Log reason for the flatten.
            symbols: If provided, only flatten these symbols. None = flatten all.
        """
        # 1. Cancel pending orders (for targeted symbols or all)
        cancelled = 0
        cancel_symbols = symbols if symbols is not None else list(engine.symbol_states.keys())
        for symbol in cancel_symbols:
            cancelled += executor.cancel_all_for_symbol(symbol)

        # 2. Close positions at last known price (full fill, no partials)
        closed = 0
        prices = getattr(engine, "_latest_prices", {})
        QTY_EPSILON = 1e-6  # Ignore floating-point dust
        now = clock()

        flatten_items = (
            [(s, engine.account.positions_qty.get(s, 0.0)) for s in symbols]
            if symbols is not None
            else list(engine.account.positions_qty.items())
        )

        for symbol, qty in flatten_items:
            if abs(qty) < QTY_EPSILON:
                engine.account.positions_qty[symbol] = 0.0
                continue

            last_price = prices.get(symbol)
            if not last_price or last_price <= 0:
                logger.warning(
                    "Skipping EOD flatten — invalid last price",
                    symbol=symbol,
                    qty=qty,
                    last_price=last_price,
                    reason=reason,
                )
                continue

            exit_side = "sell" if qty > 0 else "buy"
            exit_qty = abs(qty)
            fill_val = exit_qty * last_price
            commission = max(
                getattr(executor, "_min_commission", 0.0),
                getattr(executor, "_commission_per_share", 0.0) * exit_qty,
            )

            # Update cash: selling adds cash, buying subtracts
            if exit_side == "sell":
                engine.account.cash += fill_val - commission
            else:
                engine.account.cash -= fill_val + commission

            # Zero the position completely
            engine.account.positions_qty[symbol] = 0.0

            # Record fill in DB so _build_trade_records can pair it
            db_order = DbOrder(
                correlation_id=f"flatten-{symbol}-{now.isoformat()}",
                symbol=symbol,
                side=exit_side,
                qty=exit_qty,
                type="market",
                limit_price=last_price,
                status="filled",
                time_placed=now,
                time_last_update=now,
                meta_json={"reason": reason},
            )
            with db.get_session() as session:
                session.add(db_order)
                session.commit()

            closed += 1

        # 3. Reset local position/order state so strategies see clean slate
        reset_symbols = set(symbols) if symbols is not None else None
        for sym, state in engine.symbol_states.items():
            if reset_symbols is not None and sym not in reset_symbols:
                continue
            state.position = None
            state.open_orders = {}

        logger.info(
            "Backtest flatten complete",
            reason=reason,
            orders_cancelled=cancelled,
            positions_closed=closed,
        )

    logger.info("Starting backtest replay...")

    prev_day = None
    bar_count = 0
    equity_curve: list[tuple[datetime, float]] = []

    # Pre-allocate _latest_prices dict once (avoids per-bar hasattr check)
    if not hasattr(engine, "_latest_prices"):
        engine._latest_prices = {}
    latest_prices = engine._latest_prices
    positions_qty = engine.account.positions_qty

    # Use itertuples() — 10-50x faster than iterrows() because it avoids
    # creating a Series per row and returns lightweight namedtuples instead.
    for row in bars_df.itertuples(index=False):
        ts = row.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        clock.set_time(ts)

        current_day = ts.date()
        if prev_day and current_day != prev_day:
            # Flush pending async fill callbacks so symbol_states are current
            await asyncio.sleep(0)
            # Per-strategy EOD flatten: only flatten positions whose strategy
            # does NOT allow overnight holds (or has exceeded max_hold_days).
            symbols_to_flatten: list[str] = []
            for sym, qty in list(engine.account.positions_qty.items()):
                if abs(qty) < 1e-6:
                    continue
                sym_state = engine.symbol_states.get(sym)
                pos = sym_state.position if sym_state else None
                strat_name = (pos.strategy_name or pos.strategy) if pos else ""
                strat = engine.strategies.get(strat_name)
                if strat is None:
                    # Unknown strategy — flatten for safety
                    symbols_to_flatten.append(sym)
                    continue
                # Calculate hold days from entry
                hold_days = 0
                if pos and pos.entry_time:
                    entry_time = pos.entry_time
                    if isinstance(entry_time, (int, float)):
                        if entry_time > 1e15:
                            entry_time = datetime.fromtimestamp(entry_time / 1_000_000, tz=timezone.utc)
                        elif entry_time > 1e10:
                            entry_time = datetime.fromtimestamp(entry_time / 1000, tz=timezone.utc)
                        else:
                            entry_time = datetime.fromtimestamp(entry_time, tz=timezone.utc)
                    hold_days = (ts.date() - entry_time.date()).days
                if _should_flatten_position(strat, hold_days):
                    symbols_to_flatten.append(sym)
            if symbols_to_flatten:
                _backtest_flatten_all(reason="EOD Backtest Simulation", symbols=symbols_to_flatten)
            # Snapshot equity at day boundary for Sharpe/DD calculations
            equity_curve.append((ts, engine.account.equity))
        prev_day = current_day

        # Intraday flatten at 15:55 ET (5 min before close) — per-strategy overnight logic
        if backtest_cfg.get("force_flat_at_1600", False):
            et_time = ts.astimezone(_ET_TZ)
            if et_time.hour == 15 and et_time.minute >= 55:
                # Check each position individually against its strategy's overnight config
                symbols_to_flatten: list[str] = []
                for sym, qty in list(engine.account.positions_qty.items()):
                    if abs(qty) < 1e-6:
                        continue
                    sym_state = engine.symbol_states.get(sym)
                    pos = sym_state.position if sym_state else None
                    strat_name = (pos.strategy_name or pos.strategy) if pos else ""
                    strat = engine.strategies.get(strat_name)
                    if strat is None:
                        # Unknown strategy — flatten for safety
                        symbols_to_flatten.append(sym)
                        continue
                    # Calculate hold days from entry
                    hold_days = 0
                    if pos and pos.entry_time:
                        entry_time = pos.entry_time
                        if isinstance(entry_time, (int, float)):
                            # Detect epoch unit: >1e15 = microseconds, >1e10 = milliseconds, else seconds
                            if entry_time > 1e15:
                                entry_time = datetime.fromtimestamp(entry_time / 1_000_000, tz=timezone.utc)
                            elif entry_time > 1e10:
                                entry_time = datetime.fromtimestamp(entry_time / 1000, tz=timezone.utc)
                            else:
                                entry_time = datetime.fromtimestamp(entry_time, tz=timezone.utc)
                        hold_days = (ts.date() - entry_time.date()).days
                    if _should_flatten_position(strat, hold_days):
                        symbols_to_flatten.append(sym)
                if symbols_to_flatten:
                    _backtest_flatten_all(reason="Force flat before 16:00", symbols=symbols_to_flatten)

        mock_bar = Bar(
            symbol=row.symbol,
            time=ts,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
            vwap=row.vwap,
        )

        # Aggregate 1m bars into 5m/15m deques so the MTF analyzer's
        # slow-path (Rolling* Python objects) also has data to work with.
        _sym_state = engine.symbol_states.get(row.symbol)
        if _sym_state is None:
            # Force creation so we can append aggregated bars
            _sym_state = engine._get_or_create_symbol_state(row.symbol)
        _maybe_aggregate_bar(_sym_state, mock_bar, 5, _sym_state.bars_5m)
        _maybe_aggregate_bar(_sym_state, mock_bar, 15, _sym_state.bars_15m)

        # Attach regime labels to symbol_state.meta for strategy/engine access
        if regime_labels:
            _rl_df = regime_labels.get(row.symbol)
            if _rl_df is not None and not _rl_df.empty:
                _bar_date = current_day  # already computed above
                if _bar_date in _rl_df.index:
                    _rl_row = _rl_df.loc[_bar_date]
                    _sym_state.meta["regime_labels"] = {
                        col: _rl_row[col] for col in _REGIME_LABEL_COLUMNS if col in _rl_row.index
                    }
                # else: keep previous day's labels (or empty if never set)

        if session_labels:
            _sl_df = session_labels.get(row.symbol)
            if _sl_df is not None and not _sl_df.empty:
                # Round to nearest minute for lookup
                _bar_ts_rounded = ts.replace(second=0, microsecond=0)
                if _bar_ts_rounded in _sl_df.index:
                    _sym_state.meta["session_phase"] = _sl_df.loc[_bar_ts_rounded, "session_phase"]

        # Track the latest price for equity calculation
        latest_prices[row.symbol] = row.close

        # 1. Evaluate fills sequentially (must happen BEFORE equity calc
        #    so that the current bar's fills are reflected in PnL)
        had_open_orders = bool(executor.open_orders)
        # Pass regime labels to executor for regime-dependent slippage
        if had_open_orders and _sym_state is not None and hasattr(executor, "set_regime_context"):
            _rl = _sym_state.meta.get("regime_labels", {})
            if _rl:
                executor.set_regime_context(row.symbol, _rl)
        executor.process_bar(mock_bar)

        # Flush async fill callbacks only when fills may have occurred
        # (avoid yielding every bar — sleep(0) on 200K+ bars is expensive)
        if had_open_orders:
            await asyncio.sleep(0)

        # 2. Calculate updated equity AFTER fills are processed
        pos_value = sum(qty * latest_prices.get(sym, 0.0) for sym, qty in positions_qty.items() if qty != 0)
        engine.account.equity = engine.account.cash + pos_value

        # 3. Feed to execution engine
        engine.on_bar(mock_bar.symbol, mock_bar)

        bar_count += 1

        # Yield to event loop periodically (every 5000 bars instead of every bar)
        if bar_count % 5000 == 0:
            await asyncio.sleep(0)

        if bar_count % 50000 == 0:
            logger.info("Replay progress", bars_processed=bar_count, current_time=str(ts))

    # Flush pending async tasks, then flatten remaining positions
    await asyncio.sleep(0)
    _backtest_flatten_all(reason="End of Backtest")
    # Final equity snapshot
    equity_curve.append((clock(), engine.account.equity))

    logger.info("Backtest replay complete. Generating report...")

    # Build trade records from paired buy/sell orders
    trades = _build_trade_records(db, config, regime_labels=regime_labels or None)

    report = BacktestReportCard(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=initial_cash,
        start_date=start_dt,
        end_date=end_dt,
    )

    # ── Benchmark comparison ────────────────────────────────────────
    import numpy as np

    from src.analytics.benchmark import compute_benchmark_comparison

    strategy_daily_returns = np.array(report._compute_daily_returns())
    index_bars = bars_df[bars_df["symbol"] == index_symbol].sort_values("timestamp")
    if not index_bars.empty and len(strategy_daily_returns) > 1:
        # Compute benchmark daily returns from index (SPY) bars
        daily_index = index_bars.groupby(index_bars["timestamp"].dt.date)["close"].last()
        daily_index = daily_index.sort_index()
        if len(daily_index) > 1:
            bench_returns = daily_index.pct_change().dropna().values
            # Align lengths (use the shorter series)
            min_len = min(len(strategy_daily_returns), len(bench_returns))
            if min_len > 1:
                bm_result = compute_benchmark_comparison(
                    strategy_daily_returns=strategy_daily_returns[-min_len:],
                    benchmark_daily_returns=bench_returns[-min_len:],
                    benchmark_symbol=index_symbol,
                )
                report.metrics.benchmark = bm_result

    # ── Monte Carlo simulation ──────────────────────────────────────
    mc_cfg = config.get("analytics", {}).get("monte_carlo", {})
    if mc_cfg.get("enabled", False):
        from src.analytics.monte_carlo import run_monte_carlo

        trade_pnls = [t.pnl for t in trades]
        if len(trade_pnls) >= 5:  # Need minimum trades for meaningful MC
            mc_result = run_monte_carlo(
                trade_pnls,
                initial_capital=config.get("initial_cash", 100_000),
                n_simulations=mc_cfg.get("n_simulations", 10_000),
                ruin_threshold_pct=mc_cfg.get("ruin_threshold_pct", 30.0),
            )
            report.metrics.monte_carlo = mc_result

    # ── Diagnostics engine ──────────────────────────────────────────
    diag_cfg = config.get("analytics", {}).get("diagnostics", {})
    if diag_cfg.get("enabled", False):
        from src.analytics.diagnostics import run_diagnostics

        min_trades = diag_cfg.get("min_trades_for_analysis", 20)
        trades_dicts = [
            {
                "strategy": t.strategy or "unknown",
                "pnl": t.pnl,
                "regime_trend": (
                    t.entry_regime_trend.upper()
                    if t.entry_regime_trend != "unknown"
                    else t.meta.get("regime_labels", {}).get("regime_trend", "UNKNOWN")
                ),
                "entry_hour": t.entry_time.hour if t.entry_time else 10,
                "hold_minutes": (
                    (t.exit_time - t.entry_time).total_seconds() / 60.0 if t.exit_time and t.entry_time else 0.0
                ),
                "exit_type": "unknown",
            }
            for t in trades
        ]
        diag = run_diagnostics(trades_dicts, min_trades=min_trades)
        report.metrics.diagnostics = diag
        logger.info("diagnostics_summary", summary=diag.summary)

    # ── Daily Return Stats ─────────────────────────────────────────
    if len(strategy_daily_returns) > 1:
        from src.analytics.report_card import compute_daily_return_stats

        try:
            drs = compute_daily_return_stats(strategy_daily_returns)
            report.metrics.daily_return_stats = drs
        except Exception as e:
            logger.warning("daily_return_stats_failed", error=str(e))

    # ── Return Autocorrelation ─────────────────────────────────────
    if len(strategy_daily_returns) > 10:
        from src.analytics.return_diagnostics import compute_return_autocorrelation

        try:
            ac = compute_return_autocorrelation(strategy_daily_returns)
            report.metrics.return_autocorrelation = ac
        except Exception as e:
            logger.warning("return_autocorrelation_failed", error=str(e))

    # ── Turnover Analysis ──────────────────────────────────────────
    if trades:
        from src.analytics.return_diagnostics import compute_turnover_analysis

        try:
            trade_dicts_for_turnover = [
                {
                    "qty": t.qty,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "commission": t.commission,
                    "slippage": t.slippage,
                }
                for t in trades
            ]
            ta = compute_turnover_analysis(
                trade_dicts_for_turnover,
                initial_capital=initial_cash,
                n_trading_days=report.metrics.trading_days or 1,
            )
            report.metrics.turnover_analysis = ta
        except Exception as e:
            logger.warning("turnover_analysis_failed", error=str(e))

    # ── Probabilistic Sharpe Ratio (PSR) ───────────────────────────
    if len(strategy_daily_returns) > 10 and abs(report.metrics.sharpe_ratio) > 1e-12:
        from scipy.stats import kurtosis as sp_kurtosis
        from scipy.stats import skew as sp_skew

        from src.analytics.statistical_tests import minimum_backtest_length as compute_min_btl
        from src.analytics.statistical_tests import probabilistic_sharpe_ratio

        try:
            skew_val = float(sp_skew(strategy_daily_returns))
            kurt_val = float(sp_kurtosis(strategy_daily_returns, fisher=False))  # excess+3
            psr = probabilistic_sharpe_ratio(
                observed_sharpe=report.metrics.sharpe_ratio,
                benchmark_sharpe=0.0,
                n_returns=len(strategy_daily_returns),
                skew=skew_val,
                kurtosis=kurt_val,
            )
            report.metrics.psr = psr

            # ── Minimum Backtest Length (MinBTL) ───────────────────
            mbl = compute_min_btl(
                observed_sharpe=report.metrics.sharpe_ratio,
                skew=skew_val,
                kurtosis=kurt_val,
                n_observations=len(strategy_daily_returns),
            )
            report.metrics.min_backtest_length = mbl
        except Exception as e:
            logger.warning("statistical_tests_failed", error=str(e))

    # ── Factor Attribution ─────────────────────────────────────────
    fa_cfg = config.get("analytics", {}).get("factor_attribution", {})
    if fa_cfg.get("enabled", False) and len(strategy_daily_returns) > 10:
        from src.analytics.factor_attribution import compute_factor_attribution, load_fama_french_factors

        try:
            ff_factors = load_fama_french_factors(start_date, end_date)
            if ff_factors is not None:
                # Align lengths
                rf = ff_factors.pop("RF", np.zeros(len(strategy_daily_returns)))
                min_len = (
                    min(len(strategy_daily_returns), min(len(v) for v in ff_factors.values())) if ff_factors else 0
                )
                if min_len > 10:
                    excess_returns = strategy_daily_returns[-min_len:] - rf[-min_len:]
                    aligned_factors = {k: v[-min_len:] for k, v in ff_factors.items()}
                    fa_result = compute_factor_attribution(excess_returns, aligned_factors)
                    report.metrics.factor_attribution = fa_result
        except Exception as e:
            logger.warning("factor_attribution_failed", error=str(e))

    # ── Strategy Correlation ───────────────────────────────────────
    corr_cfg = config.get("analytics", {}).get("strategy_correlation", {})
    if corr_cfg.get("enabled", False) and len(trades) > 1:
        from src.analytics.correlation import build_strategy_daily_returns, compute_strategy_correlation

        try:
            trade_dicts_for_corr = [
                {
                    "strategy": t.strategy or "unknown",
                    "pnl": t.pnl,
                    "exit_time": t.exit_time,
                }
                for t in trades
            ]
            strat_returns = build_strategy_daily_returns(
                trade_dicts_for_corr,
                n_trading_days=report.metrics.trading_days or 1,
            )
            if len(strat_returns) > 1:
                corr_result = compute_strategy_correlation(strat_returns)
                report.metrics.strategy_correlation = corr_result
        except Exception as e:
            logger.warning("strategy_correlation_failed", error=str(e))

    report.print_summary(logger)

    # Save markdown report
    report_name = f"backtest_{start_date}_to_{end_date}.md"
    report_path = report.write_markdown(f"artifacts/backtest_reports/{report_name}")
    logger.info("Report saved", path=str(report_path))

    # Also log as single JSON for machine parsing
    logger.info("Report card JSON", **report.to_dict())

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory with per-symbol parquet files (e.g. data/bars_2024). Skips API fetch when set.",
    )
    args = parser.parse_args()

    asyncio.run(run_backtest(args.start, args.end, args.config, data_dir=args.data_dir))
