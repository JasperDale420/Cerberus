"""Standardized backtest report card.

Mirrors the metrics from the live ``AnalyticsEngine`` / ``StrategyStatsDaily``
and adds portfolio-level metrics (Sharpe, Sortino, CAGR, Calmar, max drawdown %).

Usage::

    from src.backtest.backtest_report import BacktestReportCard

    report = BacktestReportCard(
        trades=trades_list,             # list of dicts from DB
        equity_curve=equity_snapshots,  # list of (datetime, float)
        initial_capital=10_000.0,
        start_date=start_dt,
        end_date=end_dt,
    )
    report.print_summary(logger)
    report.write_markdown("artifacts/backtest_reports/q1_2024.md")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.logger import StructuredLogger


@dataclass
class TradeRecord:
    """Lightweight trade representation for report-card calculation."""

    symbol: str
    side: str  # "buy" or "sell" (entry side)
    qty: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float = 0.0
    pnl_r: float = 0.0  # PnL in R-multiples (if available)
    commission: float = 0.0
    slippage: float = 0.0
    strategy: str = ""  # Strategy that generated this trade


@dataclass
class ReportMetrics:
    """All computed metrics for the backtest report card."""

    # Trade-level (mirrors StrategyStatsDaily)
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    winrate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_win_loss_ratio: float = 0.0
    ev_per_trade: float = 0.0
    max_drawdown_r: float = 0.0
    max_consecutive_losers: int = 0
    max_consecutive_winners: int = 0
    avg_hold_seconds: float = 0.0
    median_hold_seconds: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # R-multiple metrics
    avg_r: float = 0.0
    median_r: float = 0.0
    std_r: float = 0.0
    pnl_r_total: float = 0.0
    z_score: float = 0.0

    # Portfolio-level
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0

    # Trade-level risk metrics (more meaningful for intraday strategies)
    trade_sharpe_ratio: float = 0.0
    trade_sortino_ratio: float = 0.0

    # Equity curve stats
    best_day_pct: float = 0.0
    worst_day_pct: float = 0.0
    positive_days: int = 0
    negative_days: int = 0
    trading_days: int = 0


class BacktestReportCard:
    """Generates a standardized report card from backtest results."""

    def __init__(
        self,
        trades: list[TradeRecord],
        equity_curve: list[tuple[datetime, float]],
        initial_capital: float = 10_000.0,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ):
        self.trades = sorted(trades, key=lambda t: t.exit_time)
        self.equity_curve = sorted(equity_curve, key=lambda x: x[0])
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.metrics = self._compute_all()

    def _compute_all(self) -> ReportMetrics:
        m = ReportMetrics()
        self._compute_trade_metrics(m)
        self._compute_equity_metrics(m)
        self._compute_trade_level_sharpe(m)
        self.strategy_metrics = self._compute_per_strategy()
        return m

    def _compute_per_strategy(self) -> dict[str, dict[str, Any]]:
        """Compute key metrics grouped by strategy."""
        by_strat: dict[str, list[TradeRecord]] = {}
        for t in self.trades:
            key = t.strategy or "unknown"
            by_strat.setdefault(key, []).append(t)

        results: dict[str, dict[str, Any]] = {}
        for strat, strat_trades in sorted(by_strat.items()):
            pnls = [t.pnl for t in strat_trades]
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]
            n = len(strat_trades)
            hold_secs = [
                (t.exit_time - t.entry_time).total_seconds()
                for t in strat_trades
                if t.exit_time and t.entry_time and t.strategy != "_flatten"
            ]
            gross_profit = sum(winners) if winners else 0.0
            gross_loss = abs(sum(losers)) if losers else 0.0

            results[strat] = {
                "n_trades": n,
                "n_wins": len(winners),
                "n_losses": len(losers),
                "winrate": len(winners) / n if n > 0 else 0.0,
                "net_pnl": round(sum(pnls), 2),
                "gross_profit": round(gross_profit, 2),
                "gross_loss": round(gross_loss, 2),
                "profit_factor": round(
                    min(gross_profit / gross_loss, 999.0) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
                    2,
                ),
                "avg_pnl": round(sum(pnls) / n, 2) if n > 0 else 0.0,
                "avg_win": round(sum(winners) / len(winners), 2) if winners else 0.0,
                "avg_loss": round(abs(sum(losers)) / len(losers), 2) if losers else 0.0,
                "avg_hold_min": round(sum(hold_secs) / len(hold_secs) / 60, 1) if hold_secs else 0.0,
                "median_hold_min": round(sorted(hold_secs)[len(hold_secs) // 2] / 60, 1) if hold_secs else 0.0,
            }
        return results

    # ------------------------------------------------------------------
    # Trade-level metrics
    # ------------------------------------------------------------------
    def _compute_trade_metrics(self, m: ReportMetrics) -> None:
        trades = self.trades
        m.n_trades = len(trades)
        if m.n_trades == 0:
            return

        pnls = [t.pnl for t in trades]
        r_multiples = [t.pnl_r for t in trades if abs(t.pnl_r) > 1e-12]
        hold_times = [
            (t.exit_time - t.entry_time).total_seconds()
            for t in trades
            if t.exit_time and t.entry_time and t.strategy != "_flatten"
        ]

        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        m.n_wins = len(winners)
        m.n_losses = len(losers)
        m.winrate = m.n_wins / m.n_trades if m.n_trades > 0 else 0.0

        m.gross_profit = sum(winners) if winners else 0.0
        m.gross_loss = abs(sum(losers)) if losers else 0.0
        m.net_pnl = sum(pnls)
        m.avg_pnl = m.net_pnl / m.n_trades
        m.avg_win = (sum(winners) / len(winners)) if winners else 0.0
        m.avg_loss = (abs(sum(losers)) / len(losers)) if losers else 0.0
        m.ev_per_trade = m.avg_pnl

        # Profit factor
        if m.gross_loss > 0:
            m.profit_factor = min(m.gross_profit / m.gross_loss, 999.0)
        elif m.gross_profit > 0:
            m.profit_factor = 999.0

        # Average win/loss ratio
        if m.avg_loss > 0:
            m.avg_win_loss_ratio = min(m.avg_win / m.avg_loss, 999.0)
        elif m.avg_win > 0:
            m.avg_win_loss_ratio = 999.0

        # Consecutive winners/losers
        m.max_consecutive_losers = self._max_consecutive(pnls, negative=True)
        m.max_consecutive_winners = self._max_consecutive(pnls, negative=False)

        # Hold time
        if hold_times:
            m.avg_hold_seconds = sum(hold_times) / len(hold_times)
            sorted_ht = sorted(hold_times)
            mid = len(sorted_ht) // 2
            m.median_hold_seconds = (
                sorted_ht[mid] if len(sorted_ht) % 2 == 1 else (sorted_ht[mid - 1] + sorted_ht[mid]) / 2
            )

        # Commission and slippage
        m.total_commission = sum(t.commission for t in trades)
        m.total_slippage = sum(t.slippage for t in trades)

        # R-multiple metrics
        if r_multiples:
            m.pnl_r_total = sum(r_multiples)
            m.avg_r = m.pnl_r_total / len(r_multiples)
            sorted_r = sorted(r_multiples)
            mid = len(sorted_r) // 2
            m.median_r = sorted_r[mid] if len(sorted_r) % 2 == 1 else (sorted_r[mid - 1] + sorted_r[mid]) / 2
            if len(r_multiples) > 1:
                mean_r = m.avg_r
                m.std_r = math.sqrt(sum((x - mean_r) ** 2 for x in r_multiples) / (len(r_multiples) - 1))
            # Z-score
            if m.std_r > 0 and len(r_multiples) > 0:
                se = m.std_r / math.sqrt(len(r_multiples))
                m.z_score = m.avg_r / se if se > 0 else 0.0

        # Max drawdown in R
        m.max_drawdown_r = self._max_drawdown_series([t.pnl_r for t in trades])

    # ------------------------------------------------------------------
    # Portfolio / equity curve metrics
    # ------------------------------------------------------------------
    def _compute_equity_metrics(self, m: ReportMetrics) -> None:
        if not self.equity_curve:
            m.final_equity = self.initial_capital + m.net_pnl
            m.total_return_pct = (m.net_pnl / self.initial_capital) * 100.0 if self.initial_capital > 0 else 0.0
            return

        equities = [e for _, e in self.equity_curve]
        m.final_equity = equities[-1] if equities else self.initial_capital

        # Total return
        m.total_return_pct = (
            ((m.final_equity - self.initial_capital) / self.initial_capital) * 100.0
            if self.initial_capital > 0
            else 0.0
        )

        # Daily returns from equity curve
        daily_returns = self._compute_daily_returns()
        m.trading_days = len(daily_returns)

        if not daily_returns:
            return

        m.best_day_pct = max(daily_returns) * 100.0
        m.worst_day_pct = min(daily_returns) * 100.0
        m.positive_days = sum(1 for r in daily_returns if r > 0)
        m.negative_days = sum(1 for r in daily_returns if r < 0)

        # CAGR
        years = m.trading_days / 252.0
        if years > 0 and m.final_equity > 0 and self.initial_capital > 0:
            m.cagr = ((m.final_equity / self.initial_capital) ** (1.0 / years) - 1.0) * 100.0

        # Sharpe ratio (annualized, risk-free = 0 for simplicity)
        mean_daily = sum(daily_returns) / len(daily_returns)
        if len(daily_returns) > 1:
            variance = sum((r - mean_daily) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            std_daily = math.sqrt(variance) if variance > 0 else 0.0
            if std_daily > 0:
                m.sharpe_ratio = (mean_daily / std_daily) * math.sqrt(252)

        # Sortino ratio (annualized, only downside deviation)
        downside_returns = [r for r in daily_returns if r < 0]
        if downside_returns:
            downside_var = sum(r**2 for r in downside_returns) / len(downside_returns)
            downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.0
            if downside_std > 0:
                m.sortino_ratio = (mean_daily / downside_std) * math.sqrt(252)

        # Max drawdown (% of equity peak)
        peak = equities[0]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_start = 0
        in_drawdown = False

        for i, eq in enumerate(equities):
            if eq >= peak:
                if in_drawdown:
                    duration = i - current_dd_start
                    max_dd_duration = max(max_dd_duration, duration)
                    in_drawdown = False
                peak = eq
            else:
                if not in_drawdown:
                    current_dd_start = i
                    in_drawdown = True
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)

        if in_drawdown:
            max_dd_duration = max(max_dd_duration, len(equities) - current_dd_start)

        m.max_drawdown_pct = max_dd * 100.0
        # Approximate days from equity data points
        m.max_drawdown_duration_days = max_dd_duration

        # Calmar ratio
        if m.max_drawdown_pct > 0:
            m.calmar_ratio = m.cagr / m.max_drawdown_pct

    # ------------------------------------------------------------------
    # Trade-level Sharpe / Sortino
    # ------------------------------------------------------------------
    def _compute_trade_level_sharpe(self, m: ReportMetrics) -> None:
        """Compute Sharpe/Sortino from per-trade PnL, annualized by trade frequency.

        For intraday strategies with short holds, this is more meaningful than
        daily-return Sharpe because it captures the actual per-decision risk/reward
        without compressing many intraday trades into a single EOD snapshot.
        """
        if m.n_trades < 10:
            return

        pnls = [t.pnl for t in self.trades]
        mean_pnl = sum(pnls) / len(pnls)

        if len(pnls) > 1:
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            std_pnl = math.sqrt(variance) if variance > 0 else 0.0
        else:
            std_pnl = 0.0

        # Estimate annualization factor from trade frequency
        trading_days = m.trading_days if m.trading_days > 0 else 252
        trades_per_day = m.n_trades / max(trading_days, 1)
        trades_per_year = trades_per_day * 252
        annualization = math.sqrt(trades_per_year) if trades_per_year > 0 else 1.0

        if std_pnl > 0:
            m.trade_sharpe_ratio = (mean_pnl / std_pnl) * annualization

        # Trade-level Sortino (downside deviation only)
        downside = [p for p in pnls if p < 0]
        if downside:
            ds_var = sum(p**2 for p in downside) / len(downside)
            ds_std = math.sqrt(ds_var) if ds_var > 0 else 0.0
            if ds_std > 0:
                m.trade_sortino_ratio = (mean_pnl / ds_std) * annualization

    def _compute_daily_returns(self) -> list[float]:
        """Compute daily returns from the equity curve snapshots."""
        if len(self.equity_curve) < 2:
            return []

        # Group equity by date, taking the last value each day
        daily: dict[Any, float] = {}
        for ts, eq in self.equity_curve:
            day = ts.date() if hasattr(ts, "date") else ts
            daily[day] = eq

        sorted_days = sorted(daily.keys())
        if len(sorted_days) < 2:
            return []

        returns = []
        for i in range(1, len(sorted_days)):
            prev_eq = daily[sorted_days[i - 1]]
            curr_eq = daily[sorted_days[i]]
            if prev_eq > 0:
                returns.append((curr_eq - prev_eq) / prev_eq)

        return returns

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _max_consecutive(values: list[float], negative: bool = True) -> int:
        cur = 0
        best = 0
        for v in values:
            if (negative and v < 0) or (not negative and v > 0):
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    @staticmethod
    def _max_drawdown_series(values: list[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for v in values:
            equity += v
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        return max_dd

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def print_summary(self, logger: StructuredLogger) -> None:
        """Log a structured summary of the backtest report card."""
        m = self.metrics

        logger.info("=" * 60)
        logger.info("BACKTEST REPORT CARD")
        logger.info("=" * 60)

        # Portfolio performance
        logger.info(
            "Portfolio Performance",
            initial_capital=f"${self.initial_capital:,.2f}",
            final_equity=f"${m.final_equity:,.2f}",
            net_pnl=f"${m.net_pnl:,.2f}",
            total_return=f"{m.total_return_pct:.2f}%",
            cagr=f"{m.cagr:.2f}%",
        )

        # Risk metrics
        logger.info(
            "Risk Metrics",
            sharpe_daily=f"{m.sharpe_ratio:.3f}",
            sharpe_trade=f"{m.trade_sharpe_ratio:.3f}",
            sortino_daily=f"{m.sortino_ratio:.3f}",
            sortino_trade=f"{m.trade_sortino_ratio:.3f}",
            calmar_ratio=f"{m.calmar_ratio:.3f}",
            max_drawdown_pct=f"{m.max_drawdown_pct:.2f}%",
        )

        # Trade statistics
        logger.info(
            "Trade Statistics",
            n_trades=m.n_trades,
            winrate=f"{m.winrate:.1%}",
            profit_factor=f"{m.profit_factor:.2f}",
            avg_win_loss_ratio=f"{m.avg_win_loss_ratio:.2f}",
        )
        logger.info(
            "Trade Detail",
            avg_pnl=f"${m.avg_pnl:.2f}",
            avg_win=f"${m.avg_win:.2f}",
            avg_loss=f"${m.avg_loss:.2f}",
            ev_per_trade=f"${m.ev_per_trade:.2f}",
        )
        logger.info(
            "Streaks",
            max_consecutive_losers=m.max_consecutive_losers,
            max_consecutive_winners=m.max_consecutive_winners,
        )
        logger.info(
            "Hold Times",
            avg_hold=f"{m.avg_hold_seconds / 60:.1f} min",
            median_hold=f"{m.median_hold_seconds / 60:.1f} min",
        )
        logger.info(
            "Costs",
            total_commission=f"${m.total_commission:.2f}",
            total_slippage=f"${m.total_slippage:.2f}",
        )

        # Equity curve stats
        if m.trading_days > 0:
            logger.info(
                "Daily Stats",
                trading_days=m.trading_days,
                positive_days=m.positive_days,
                negative_days=m.negative_days,
                best_day=f"{m.best_day_pct:.2f}%",
                worst_day=f"{m.worst_day_pct:.2f}%",
            )

        # Per-strategy breakdown
        if hasattr(self, "strategy_metrics") and self.strategy_metrics:
            logger.info("PER-STRATEGY BREAKDOWN")
            for strat, sm in self.strategy_metrics.items():
                logger.info(
                    f"Strategy: {strat}",
                    trades=sm["n_trades"],
                    winrate=f"{sm['winrate']:.1%}",
                    pf=sm["profit_factor"],
                    net_pnl=f"${sm['net_pnl']:.2f}",
                    avg_hold=f"{sm['avg_hold_min']:.0f}m",
                )

        logger.info("=" * 60)

    def to_dict(self) -> dict[str, Any]:
        """Return all metrics as a flat dictionary for JSON serialization."""
        m = self.metrics
        return {
            "n_trades": m.n_trades,
            "n_wins": m.n_wins,
            "n_losses": m.n_losses,
            "winrate": round(m.winrate, 4),
            "gross_profit": round(m.gross_profit, 2),
            "gross_loss": round(m.gross_loss, 2),
            "net_pnl": round(m.net_pnl, 2),
            "avg_pnl": round(m.avg_pnl, 2),
            "avg_win": round(m.avg_win, 2),
            "avg_loss": round(m.avg_loss, 2),
            "profit_factor": round(m.profit_factor, 2),
            "avg_win_loss_ratio": round(m.avg_win_loss_ratio, 2),
            "ev_per_trade": round(m.ev_per_trade, 2),
            "max_consecutive_losers": m.max_consecutive_losers,
            "max_consecutive_winners": m.max_consecutive_winners,
            "avg_hold_minutes": round(m.avg_hold_seconds / 60, 1),
            "median_hold_minutes": round(m.median_hold_seconds / 60, 1),
            "total_commission": round(m.total_commission, 2),
            "total_slippage": round(m.total_slippage, 2),
            "avg_r": round(m.avg_r, 4),
            "median_r": round(m.median_r, 4),
            "std_r": round(m.std_r, 4),
            "pnl_r_total": round(m.pnl_r_total, 4),
            "z_score": round(m.z_score, 3),
            "max_drawdown_r": round(m.max_drawdown_r, 4),
            "final_equity": round(m.final_equity, 2),
            "total_return_pct": round(m.total_return_pct, 2),
            "cagr": round(m.cagr, 2),
            "sharpe_ratio": round(m.sharpe_ratio, 3),
            "sortino_ratio": round(m.sortino_ratio, 3),
            "trade_sharpe_ratio": round(m.trade_sharpe_ratio, 3),
            "trade_sortino_ratio": round(m.trade_sortino_ratio, 3),
            "calmar_ratio": round(m.calmar_ratio, 3),
            "max_drawdown_pct": round(m.max_drawdown_pct, 2),
            "max_drawdown_duration_days": m.max_drawdown_duration_days,
            "best_day_pct": round(m.best_day_pct, 2),
            "worst_day_pct": round(m.worst_day_pct, 2),
            "positive_days": m.positive_days,
            "negative_days": m.negative_days,
            "trading_days": m.trading_days,
            "per_strategy": self.strategy_metrics if hasattr(self, "strategy_metrics") else {},
        }

    def write_markdown(self, path: str | Path) -> Path:
        """Write a human-readable markdown report to disk."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        m = self.metrics
        lines: list[str] = []

        lines.append("# Backtest Report Card\n\n")
        if self.start_date and self.end_date:
            lines.append(f"**Period:** {self.start_date} → {self.end_date}\n")
        lines.append(f"**Initial Capital:** ${self.initial_capital:,.2f}\n\n")

        _TBL_HDR = "| Metric | Value |\n|--------|-------|\n"  # noqa: N806

        # Portfolio Performance
        lines.append("## Portfolio Performance\n\n")
        lines.append(_TBL_HDR)
        lines.append(f"| Final Equity | ${m.final_equity:,.2f} |\n")
        lines.append(f"| Net PnL | ${m.net_pnl:,.2f} |\n")
        lines.append(f"| Total Return | {m.total_return_pct:.2f}% |\n")
        lines.append(f"| CAGR | {m.cagr:.2f}% |\n")
        lines.append(f"| Sharpe Ratio (daily) | {m.sharpe_ratio:.3f} |\n")
        lines.append(f"| Sharpe Ratio (trade) | {m.trade_sharpe_ratio:.3f} |\n")
        lines.append(f"| Sortino Ratio (daily) | {m.sortino_ratio:.3f} |\n")
        lines.append(f"| Sortino Ratio (trade) | {m.trade_sortino_ratio:.3f} |\n")
        lines.append(f"| Calmar Ratio | {m.calmar_ratio:.3f} |\n")
        lines.append(f"| Max Drawdown | {m.max_drawdown_pct:.2f}% |\n")
        lines.append(f"| Max DD Duration | {m.max_drawdown_duration_days} days |\n\n")

        # Trade Statistics
        lines.append("## Trade Statistics\n\n")
        lines.append(_TBL_HDR)
        lines.append(f"| Total Trades | {m.n_trades} |\n")
        lines.append(f"| Winners | {m.n_wins} |\n")
        lines.append(f"| Losers | {m.n_losses} |\n")
        lines.append(f"| Win Rate | {m.winrate:.1%} |\n")
        lines.append(f"| Profit Factor | {m.profit_factor:.2f} |\n")
        lines.append(f"| Avg Win/Loss Ratio | {m.avg_win_loss_ratio:.2f} |\n")
        lines.append(f"| EV Per Trade | ${m.ev_per_trade:.2f} |\n")
        lines.append(f"| Gross Profit | ${m.gross_profit:,.2f} |\n")
        lines.append(f"| Gross Loss | ${m.gross_loss:,.2f} |\n")
        lines.append(f"| Avg Win | ${m.avg_win:.2f} |\n")
        lines.append(f"| Avg Loss | ${m.avg_loss:.2f} |\n")
        lines.append(f"| Max Consecutive Losers | {m.max_consecutive_losers} |\n")
        lines.append(f"| Max Consecutive Winners | {m.max_consecutive_winners} |\n\n")

        # Hold Times
        lines.append("## Hold Times\n\n")
        lines.append(_TBL_HDR)
        lines.append(f"| Avg Hold | {m.avg_hold_seconds / 60:.1f} min |\n")
        lines.append(f"| Median Hold | {m.median_hold_seconds / 60:.1f} min |\n\n")

        # Costs
        lines.append("## Costs\n\n")
        lines.append(_TBL_HDR)
        lines.append(f"| Total Commission | ${m.total_commission:.2f} |\n")
        lines.append(f"| Total Slippage | ${m.total_slippage:.2f} |\n\n")

        # Daily Stats
        if m.trading_days > 0:
            lines.append("## Daily Stats\n\n")
            lines.append(_TBL_HDR)
            lines.append(f"| Trading Days | {m.trading_days} |\n")
            lines.append(f"| Positive Days | {m.positive_days} |\n")
            lines.append(f"| Negative Days | {m.negative_days} |\n")
            lines.append(f"| Best Day | {m.best_day_pct:.2f}% |\n")
            lines.append(f"| Worst Day | {m.worst_day_pct:.2f}% |\n\n")

        # Per-strategy breakdown
        if hasattr(self, "strategy_metrics") and self.strategy_metrics:
            lines.append("## Per-Strategy Breakdown\n\n")
            lines.append("| Strategy | Trades | Win% | PF | Net PnL | Avg PnL | Avg Hold |\n")
            lines.append("|----------|--------|------|-----|---------|---------|----------|\n")
            for strat, sm in self.strategy_metrics.items():
                lines.append(
                    f"| {strat} | {sm['n_trades']} | {sm['winrate']:.1%} "
                    f"| {sm['profit_factor']:.2f} | ${sm['net_pnl']:.2f} "
                    f"| ${sm['avg_pnl']:.2f} | {sm['avg_hold_min']:.0f}m |\n"
                )
            lines.append("\n")

        # Trade Log
        if self.trades:
            lines.append("## Trade Log\n\n")
            lines.append("| # | Strategy | Symbol | Side | Qty | Entry | Exit | PnL | Hold |\n")
            lines.append("|---|----------|--------|------|-----|-------|------|-----|------|\n")
            for i, t in enumerate(self.trades, 1):
                hold_min = (t.exit_time - t.entry_time).total_seconds() / 60 if t.exit_time and t.entry_time else 0
                lines.append(
                    f"| {i} | {t.strategy or 'unknown'} | {t.symbol} | {t.side} | {t.qty:.0f} "
                    f"| ${t.entry_price:.2f} | ${t.exit_price:.2f} "
                    f"| ${t.pnl:.2f} | {hold_min:.0f}m |\n"
                )
            lines.append("\n")

        out.write_text("".join(lines))
        return out
