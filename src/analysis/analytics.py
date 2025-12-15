from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import select, text

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import StrategyStatsDaily
from src.analysis.schema import Trade as DbTrade
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
                start_dt = datetime.combine(
                    target_date, datetime.min.time(), tzinfo=timezone.utc
                )
                end_dt = datetime.combine(
                    target_date, datetime.max.time(), tzinfo=timezone.utc
                )

                stmt = select(DbTrade).where(
                    DbTrade.exit_time >= start_dt, DbTrade.exit_time <= end_dt
                )
                trades = session.execute(stmt).scalars().all()

                if not trades:
                    self.logger.info("No trades found for date", date=target_date)
                    return

                # 2. DataFrame Aggregation
                df = pd.DataFrame(
                    [
                        {
                            "strategy": t.strategy,
                            "regime": t.regime_at_entry,
                            "pnl": t.pnl_net,
                            "win": 1 if (t.pnl_net or 0) > 0 else 0,
                        }
                        for t in trades
                    ]
                )

                if df.empty:
                    return

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
                    )
                    .reset_index()
                )

                # 3. Upsert to DB
                for _, row in result.iterrows():
                    stats = StrategyStatsDaily(
                        date=target_date,
                        strategy=row["strategy"],
                        regime=row["regime"],
                        net_pnl=float(row["total_pnl"]),  # Mapped to net_pnl
                        n_trades=int(row["trade_count"]),  # Mapped to n_trades
                        winrate=(
                            float(row["win_count"] / row["trade_count"])
                            if row["trade_count"] > 0
                            else 0.0
                        ),
                        avg_r=0.0,
                        median_r=0.0,
                        std_r=0.0,
                        max_drawdown_r=0.0,
                        std_dev_pnl=(
                            float(row["std_dev"])
                            if not pd.isna(row["std_dev"])
                            else 0.0
                        ),
                        z_score=0.0,
                    )

                    # Compute simple Z-score approximation: Expectancy / StdDev
                    # Expected PnL per trade / StdDev of PnL
                    if (
                        row["std_dev"]
                        and not pd.isna(row["std_dev"])
                        and row["std_dev"] > 0
                    ):
                        stats.z_score = float(row["avg_pnl"] / row["std_dev"])
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

        except Exception as e:
            self.logger.error("Aggregation failed", error=str(e))
