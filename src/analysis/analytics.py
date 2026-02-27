from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select, text

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import StrategyStatsDaily
from src.analysis.schema import Trade as DbTrade
from src.core.errors import ErrorCode
from src.core.logger import StructuredLogger


class AnalyticsEngine:
    """
    Aggregates trade data into performance statistics.
    """

    def __init__(self, db: DatabaseDatabase, logger: StructuredLogger):
        self.db = db
        self.logger = logger

    def run_daily_aggregation(self, target_date: Optional[date] = None):
        """
        Aggregates trades for a specific date and updates strategy_stats_daily.
        """
        if not target_date:
            target_date = datetime.now(timezone.utc).date()

        self.logger.info("Running daily aggregation", date=target_date)

        try:
            with self.db.get_session() as session:
                # 1. Fetch Trades for Date
                # We filter by entry_time or exit_time? Usually exit_time for realized PnL.
                start_dt = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
                end_dt = datetime.combine(target_date, datetime.max.time(), tzinfo=timezone.utc)

                stmt = select(DbTrade).where(DbTrade.exit_time >= start_dt, DbTrade.exit_time <= end_dt)
                trades = session.execute(stmt).scalars().all()

                if not trades:
                    self.logger.info("No trades found for date", date=target_date)
                    return

                # 2. DataFrame Aggregation (day-level)
                def _extract_feature(trade: DbTrade, key: str) -> Optional[float]:
                    try:
                        feats = trade.features_json or {}
                        v = feats.get(key)
                        if v is None:
                            return None
                        return float(v)
                    except Exception:
                        return None

                df = pd.DataFrame(
                    [
                        {
                            "strategy": t.strategy,
                            "regime": t.regime_at_entry,
                            "pnl": t.pnl_net,
                            "pnl_r": t.pnl_r,
                            "mae_r": t.mae_r,
                            "mfe_r": t.mfe_r,
                            "commission": t.commission,
                            "slippage": t.slippage_estimate,
                            "exit_time": t.exit_time,
                            "holding_period_seconds": t.holding_period_seconds,
                            "scanner_score": _extract_feature(t, "scanner_score"),
                            "flow_zscore": _extract_feature(t, "flow_zscore"),
                            "regime_tags": t.regime_tags_entry_json,
                            "win": 1 if (t.pnl_net or 0) > 0 else 0,
                        }
                        for t in trades
                    ]
                )

                if df.empty:
                    return

                # Avoid noisy runtime warnings from numpy/pandas aggregations when
                # optional columns are entirely missing/NaN for a day.
                for col in [
                    "pnl",
                    "pnl_r",
                    "mae_r",
                    "mfe_r",
                    "commission",
                    "slippage",
                    "holding_period_seconds",
                    "scanner_score",
                    "flow_zscore",
                ]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

                # Group by Strategy + Regime
                # PRD 8.2: StrategyStatsDaily keys: date, strategy, regime
                result = (
                    df.groupby(["strategy", "regime"])
                    .agg(
                        trade_count=("pnl", "count"),
                        win_count=("win", "sum"),
                        total_pnl=("pnl", "sum"),
                        avg_pnl=("pnl", "mean"),
                        std_dev=("pnl", "std"),
                        pnl_r_total=("pnl_r", "sum"),
                        avg_r=("pnl_r", "mean"),
                        median_r=("pnl_r", "median"),
                        std_r=("pnl_r", "std"),
                        avg_hold_seconds=("holding_period_seconds", "mean"),
                    )
                    .reset_index()
                )

                # Compute profit_factor + avg_win_loss_ratio per group
                pf_by_group: Dict[Tuple[str, str], float] = {}
                awl_by_group: Dict[Tuple[str, str], float] = {}
                for (strat, regime), grp_df in df.groupby(["strategy", "regime"]):
                    key = (str(strat), str(regime))
                    winners = grp_df[grp_df["pnl"] > 0]["pnl"]
                    losers = grp_df[grp_df["pnl"] < 0]["pnl"]
                    gross_win = float(winners.sum()) if len(winners) > 0 else 0.0
                    gross_loss = abs(float(losers.sum())) if len(losers) > 0 else 0.0
                    pf_by_group[key] = (
                        min(gross_win / gross_loss, 999.0) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
                    )
                    avg_win = float(winners.mean()) if len(winners) > 0 else 0.0
                    avg_loss = abs(float(losers.mean())) if len(losers) > 0 else 0.0
                    awl_by_group[key] = (
                        min(avg_win / avg_loss, 999.0) if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0)
                    )

                # 2b. Per-(strategy, regime) path-dependent metrics (drawdown, losers).
                dd_by_group: Dict[Tuple[str, str], float] = {}
                losers_by_group: Dict[Tuple[str, str], int] = {}
                pnl_r_by_group: Dict[Tuple[str, str], List[float]] = {}
                for t in sorted(
                    trades,
                    key=lambda x: (
                        str(x.strategy),
                        str(x.regime_at_entry),
                        x.exit_time or x.entry_time,
                    ),
                ):
                    key = (str(t.strategy), str(t.regime_at_entry))
                    r = t.pnl_r
                    if r is None:
                        continue
                    pnl_r_by_group.setdefault(key, []).append(float(r))

                def _max_drawdown_r(series: List[float]) -> float:
                    equity = 0.0
                    peak = 0.0
                    max_dd = 0.0
                    for x in series:
                        equity += float(x)
                        peak = max(peak, equity)
                        max_dd = max(max_dd, peak - equity)
                    return float(max_dd)

                def _max_consecutive_losers(series: List[float]) -> int:
                    cur = 0
                    best = 0
                    for x in series:
                        if x < 0:
                            cur += 1
                            best = max(best, cur)
                        else:
                            cur = 0
                    return int(best)

                for key, series in pnl_r_by_group.items():
                    dd_by_group[key] = _max_drawdown_r(series)
                    losers_by_group[key] = _max_consecutive_losers(series)

                # 3. Upsert to DB
                for _, row in result.iterrows():
                    key = (str(row["strategy"]), str(row["regime"]))
                    n_trades = int(row["trade_count"])
                    avg_r = float(0.0 if pd.isna(row["avg_r"]) else row["avg_r"])
                    std_r = float(0.0 if pd.isna(row["std_r"]) else row["std_r"])
                    net_pnl = float(row["total_pnl"])
                    max_dd_r = float(dd_by_group.get(key, 0.0))
                    pf = float(pf_by_group.get(key, 0.0))
                    awl = float(awl_by_group.get(key, 0.0))
                    ev = float(net_pnl / n_trades) if n_trades > 0 else 0.0
                    avg_hold = float(0.0 if pd.isna(row["avg_hold_seconds"]) else row["avg_hold_seconds"])
                    recovery = float(net_pnl / max_dd_r) if max_dd_r > 0 else (999.0 if net_pnl > 0 else 0.0)
                    stats = StrategyStatsDaily(
                        date=target_date,
                        strategy=row["strategy"],
                        regime=row["regime"],
                        net_pnl=net_pnl,
                        n_trades=n_trades,
                        winrate=(float(row["win_count"] / row["trade_count"]) if row["trade_count"] > 0 else 0.0),
                        avg_r=avg_r,
                        median_r=float(0.0 if pd.isna(row["median_r"]) else row["median_r"]),
                        std_r=std_r,
                        max_drawdown_r=max_dd_r,
                        max_consecutive_losers=int(losers_by_group.get(key, 0)),
                        std_dev_pnl=(float(row["std_dev"]) if not pd.isna(row["std_dev"]) else 0.0),
                        z_score=0.0,
                        pnl_r_total=float(0.0 if pd.isna(row["pnl_r_total"]) else row["pnl_r_total"]),
                        profit_factor=pf,
                        avg_win_loss_ratio=awl,
                        recovery_factor=recovery,
                        ev_per_trade=ev,
                        avg_hold_seconds=avg_hold,
                    )

                    # PRD 9.1: z = expectancy / se, where expectancy is avg_r.
                    if n_trades > 0 and std_r > 0:
                        se = std_r / (n_trades**0.5)
                        stats.z_score = float(avg_r / se) if se > 0 else 0.0
                    else:
                        stats.z_score = 0.0

                    # UPSERT Logic (Delete existing for date/strat/regime then Insert, or Merge)
                    # For simplicity: Check exist, delete, add
                    delete_stmt = text(
                        "DELETE FROM strategy_stats_daily WHERE date = :d AND strategy = :s AND regime = :r"
                    )
                    session.execute(
                        delete_stmt,
                        {"d": target_date, "s": row["strategy"], "r": row["regime"]},
                    )

                    session.add(stats)

                session.commit()
                self.logger.info("Aggregation complete", rows=len(result))

                # PRD 8.2: produce a human-readable daily report with extra breakdowns.
                try:
                    self._write_daily_report(target_date, df)
                except Exception as e:
                    self.logger.warning("Daily report generation failed", error=str(e))

        except Exception as e:
            self.logger.error(
                "Aggregation failed",
                error_code=ErrorCode.ANALYTICS_AGG_FAILED.value,
                error=str(e),
            )

    def _write_daily_report(self, target_date: date, df: pd.DataFrame) -> Path:
        tz_name = str((self.db.config or {}).get("timezone", "America/New_York"))
        tz = ZoneInfo(tz_name)

        report_dir = Path("artifacts") / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        out_path = report_dir / f"daily_report_{target_date.isoformat()}.md"

        df_local = df.copy()
        if "exit_time" in df_local.columns:
            df_local["exit_time"] = pd.to_datetime(df_local["exit_time"], utc=True, errors="coerce")
            df_local["exit_time_local"] = df_local["exit_time"].dt.tz_convert(tz)
            df_local["exit_hour"] = df_local["exit_time_local"].dt.hour

        def _md_table(frame: pd.DataFrame, cols: List[str]) -> str:
            if frame.empty:
                return "_(no data)_\n"
            sub = frame[cols].copy()
            for c in cols:
                sub[c] = sub[c].map(lambda v: "" if pd.isna(v) else str(v))
            rows = sub.values.tolist()
            header = "| " + " | ".join(cols) + " |\n"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
            body = ""
            for r in rows:
                body += "| " + " | ".join(r) + " |\n"
            return header + sep + body + "\n"

        lines: List[str] = []
        lines.append(f"# Cerberus Daily Report – {target_date.isoformat()}\n")

        # Summary
        total_trades = int(len(df_local))
        pnl_total = float(df_local["pnl"].fillna(0.0).sum()) if "pnl" in df_local else 0.0
        pnl_r_total = float(df_local["pnl_r"].fillna(0.0).sum()) if "pnl_r" in df_local else 0.0
        winrate = float(df_local["win"].mean()) if "win" in df_local and total_trades > 0 else 0.0
        lines.append("## Summary\n")
        lines.append(f"- Trades: {total_trades}\n")
        lines.append(f"- Net PnL: {pnl_total:.2f}\n")
        lines.append(f"- Total R: {pnl_r_total:.3f}\n")
        lines.append(f"- Winrate: {winrate:.3f}\n")

        # Strategy/Regime table
        lines.append("\n## Strategy/Regime\n")
        grp = (
            df_local.groupby(["strategy", "regime"], dropna=False)
            .agg(
                n_trades=("pnl", "count"),
                net_pnl=("pnl", "sum"),
                total_r=("pnl_r", "sum"),
                avg_r=("pnl_r", "mean"),
                avg_mae_r=("mae_r", "mean"),
                avg_mfe_r=("mfe_r", "mean"),
                commission=("commission", "sum"),
                slippage=("slippage", "sum"),
            )
            .reset_index()
            .sort_values(["n_trades", "total_r"], ascending=[False, False])
        )
        lines.append(
            _md_table(
                grp,
                [
                    "strategy",
                    "regime",
                    "n_trades",
                    "net_pnl",
                    "total_r",
                    "avg_r",
                    "avg_mae_r",
                    "avg_mfe_r",
                    "commission",
                    "slippage",
                ],
            )
        )

        # PnL by hour
        if "exit_hour" in df_local.columns:
            lines.append("\n## PnL By Exit Hour (" + tz_name + ")\n")
            by_hour = (
                df_local.groupby(["exit_hour"], dropna=True)
                .agg(
                    n_trades=("pnl", "count"),
                    net_pnl=("pnl", "sum"),
                    total_r=("pnl_r", "sum"),
                )
                .reset_index()
                .sort_values("exit_hour", ascending=True)
            )
            lines.append(_md_table(by_hour, ["exit_hour", "n_trades", "net_pnl", "total_r"]))

        # Scanner score deciles
        if "scanner_score" in df_local.columns and df_local["scanner_score"].notna().any():
            lines.append("\n## Scanner Score Deciles\n")
            scored = df_local[df_local["scanner_score"].notna()].copy()
            scored["scanner_decile"] = pd.qcut(scored["scanner_score"], 10, labels=False, duplicates="drop") + 1
            dec = (
                scored.groupby("scanner_decile")
                .agg(
                    n_trades=("pnl", "count"),
                    net_pnl=("pnl", "sum"),
                    avg_r=("pnl_r", "mean"),
                    total_r=("pnl_r", "sum"),
                )
                .reset_index()
                .sort_values("scanner_decile")
            )
            lines.append(_md_table(dec, ["scanner_decile", "n_trades", "net_pnl", "avg_r", "total_r"]))

        # Flow strength deciles (absolute z-score)
        if "flow_zscore" in df_local.columns and df_local["flow_zscore"].notna().any():
            lines.append("\n## Flow Strength Deciles (|flow_zscore|)\n")
            flowed = df_local[df_local["flow_zscore"].notna()].copy()
            flowed["flow_strength"] = flowed["flow_zscore"].abs()
            flowed["flow_decile"] = pd.qcut(flowed["flow_strength"], 10, labels=False, duplicates="drop") + 1
            fdec = (
                flowed.groupby("flow_decile")
                .agg(
                    n_trades=("pnl", "count"),
                    net_pnl=("pnl", "sum"),
                    avg_r=("pnl_r", "mean"),
                    total_r=("pnl_r", "sum"),
                )
                .reset_index()
                .sort_values("flow_decile")
            )
            lines.append(_md_table(fdec, ["flow_decile", "n_trades", "net_pnl", "avg_r", "total_r"]))

        # Regime-axis breakdowns (trend, vol, session)
        if "regime_tags" in df_local.columns:
            for axis in ["trend", "vol", "session"]:
                axis_vals = df_local["regime_tags"].apply(
                    lambda tags, _axis=axis: (tags or {}).get(_axis, "unknown") if isinstance(tags, dict) else "unknown"
                )
                if axis_vals.nunique() > 1 or (axis_vals.nunique() == 1 and axis_vals.iloc[0] != "unknown"):
                    lines.append(f"\n## PnL By {axis.title()} Regime\n")
                    df_local[f"regime_{axis}"] = axis_vals
                    by_axis = (
                        df_local.groupby(f"regime_{axis}", dropna=True)
                        .agg(
                            n_trades=("pnl", "count"),
                            net_pnl=("pnl", "sum"),
                            avg_r=("pnl_r", "mean"),
                            total_r=("pnl_r", "sum"),
                            win_rate=("win", "mean"),
                        )
                        .reset_index()
                        .sort_values("net_pnl", ascending=False)
                    )
                    lines.append(
                        _md_table(
                            by_axis,
                            [f"regime_{axis}", "n_trades", "net_pnl", "avg_r", "total_r", "win_rate"],
                        )
                    )

        # Hold time summary by strategy
        if "holding_period_seconds" in df_local.columns and df_local["holding_period_seconds"].gt(0).any():
            lines.append("\n## Avg Hold Time By Strategy\n")
            hold = (
                df_local[df_local["holding_period_seconds"] > 0]
                .groupby("strategy")
                .agg(
                    n_trades=("pnl", "count"),
                    avg_hold_min=("holding_period_seconds", lambda x: round(x.mean() / 60, 1)),
                    min_hold_min=("holding_period_seconds", lambda x: round(x.min() / 60, 1)),
                    max_hold_min=("holding_period_seconds", lambda x: round(x.max() / 60, 1)),
                )
                .reset_index()
                .sort_values("avg_hold_min")
            )
            lines.append(_md_table(hold, ["strategy", "n_trades", "avg_hold_min", "min_hold_min", "max_hold_min"]))

        out_path.write_text("".join(lines))
        self.logger.info("Daily report written", path=str(out_path))
        return out_path
