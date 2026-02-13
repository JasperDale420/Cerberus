from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agent.llm import LLMClient
from src.agent.models import ActionType, AgentAction, StrategyDailyStats
from src.agent.stage2 import Stage2Tuner
from src.agent.stage3 import Stage3Proposer
from src.analysis.db import DatabaseDatabase
from src.analysis.schema import AgentAction as DbAgentAction
from src.analysis.schema import StrategyStatsDaily as DbStrategyStatsDaily
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


class Agent:
    """
    Offline agent that analyzes strategy performance and adjusts configuration.
    Stage 1: Health & Risk Adjustments (Deterministic)
    Stage 2: Parameter Tuning (Deterministic Offline)
    Stage 3: Code Proposals (LLM Generative)
    """

    def __init__(
        self,
        logger: StructuredLogger,
        config_loader: ConfigLoader,
        config_path: str = "config/strategies.auto.yaml",
        config_path_or_dir: str = "config",
        stage2_evaluator=None,
        stage3_evaluator=None,
        llm_client: LLMClient | None = None,
    ):
        self.logger = logger
        self.config_loader = config_loader
        self.config_path_or_dir = config_path_or_dir
        # PRD 2.1/10.5: write overrides next to the active config suite.
        # If the caller doesn't explicitly set `config_path`, derive it from `config_path_or_dir`,
        # which can be either a directory path or a config file path.
        try:
            default = Path("config/strategies.auto.yaml")
            provided = Path(str(config_path))
            if provided == default:
                base = Path(str(config_path_or_dir))
                base_dir = base.parent if base.suffix else base
                self.config_path = str(base_dir / "strategies.auto.yaml")
            else:
                self.config_path = str(provided)
        except Exception:
            self.config_path = config_path

        # Lazily initialize LLM client; Stage 1 and deterministic Stage 2 should not depend on it.
        self.llm_client: LLMClient | None = llm_client

        # Helpers
        self.stage2_tuner = Stage2Tuner(logger, config_loader, config_path_or_dir, evaluator=stage2_evaluator)
        self.stage3_proposer = Stage3Proposer(
            logger,
            config_loader,
            config_path_or_dir,
            llm_client=llm_client,
            evaluator=stage3_evaluator,
        )

    def _load_stage1_params(self) -> Tuple[int, int, float, float]:
        cfg = self.config_loader.load_config(self.config_path_or_dir)
        agent_cfg = (cfg.get("agent") or {}) if isinstance(cfg, dict) else {}
        stage1 = (agent_cfg.get("stage1") or {}) if isinstance(agent_cfg, dict) else {}
        window_days = int(stage1.get("window_days", 30))
        min_trades = int(stage1.get("min_trades", 20))
        z_high = float(stage1.get("z_high", 1.645))
        max_drawdown_r = float(stage1.get("max_drawdown_r", 10.0))
        return window_days, min_trades, z_high, max_drawdown_r

    def analyze_performance(
        self, stats_list: List[StrategyDailyStats], as_of: Optional[datetime] = None
    ) -> List[AgentAction]:
        """
        Analyzes daily stats and generates actions.
        """
        actions: List[AgentAction] = []
        _window_days, min_trades, z_high, max_dd_r = self._load_stage1_params()

        now = as_of or datetime.now(timezone.utc)
        for stats in stats_list:
            # PRD 9.1 Stage 1 Rules
            # 1. Filter insufficient data
            if stats.n_trades < min_trades:
                self.logger.info(
                    "Agent Stage 1: insufficient data",
                    strategy=stats.strategy,
                    regime=stats.regime,
                    n_trades=stats.n_trades,
                    min_trades=min_trades,
                )
                continue

            # 2. Compute Z-Score
            # se = std_r / sqrt(n_trades)
            # z = expectancy / se
            if stats.std_r > 0:
                se = stats.std_r / (stats.n_trades**0.5)
                z = stats.expectancy / se
                # P2.2 fix: Cap extreme z-scores for numerical stability
                z = max(-100.0, min(100.0, z))
            else:
                z = 0.0

            # PRD 9.1: if z < -z_high and drawdown large -> tighten risk / possibly disable.
            if z < -z_high and stats.max_drawdown_r >= max_dd_r:
                actions.append(
                    AgentAction(
                        timestamp=now,
                        action_type=ActionType.DISABLE_STRATEGY,
                        strategy=stats.strategy,
                        regime=stats.regime,
                        details={
                            "expectancy": stats.expectancy,
                            "z_score": z,
                            "n_trades": stats.n_trades,
                            "max_drawdown_r": stats.max_drawdown_r,
                            "max_drawdown_r_threshold": max_dd_r,
                        },
                        reason=f"Negative expectancy with significance and drawdown (z={z:.2f} < -{z_high}, dd={stats.max_drawdown_r:.2f}R >= {max_dd_r:.2f}R)",
                    )
                )
                continue

            # If significance is negative but drawdown isn't large yet, tighten risk only.
            if z < -z_high:
                actions.append(
                    AgentAction(
                        timestamp=now,
                        action_type=ActionType.REDUCE_RISK,
                        strategy=stats.strategy,
                        regime=stats.regime,
                        details={
                            "expectancy": stats.expectancy,
                            "z_score": z,
                            "n_trades": stats.n_trades,
                            "max_drawdown_r": stats.max_drawdown_r,
                            "max_drawdown_r_threshold": max_dd_r,
                        },
                        reason=f"Negative expectancy with significance (z={z:.2f} < -{z_high}); reducing risk",
                    )
                )
                continue

            # Excessive drawdown alone -> reduce risk.
            if stats.max_drawdown_r >= max_dd_r:
                actions.append(
                    AgentAction(
                        timestamp=now,
                        action_type=ActionType.REDUCE_RISK,
                        strategy=stats.strategy,
                        regime=stats.regime,
                        details={
                            "max_drawdown_r": stats.max_drawdown_r,
                            "max_drawdown_r_threshold": max_dd_r,
                        },
                        reason=f"Drawdown exceeded threshold ({stats.max_drawdown_r:.2f}R >= {max_dd_r:.2f}R)",
                    )
                )

        return actions

    def _load_recent_stats(self, db: DatabaseDatabase, as_of: Optional[datetime] = None) -> List[StrategyDailyStats]:
        window_days, _min_trades, _z_high, _max_dd_r = self._load_stage1_params()
        now = as_of or datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=window_days)).date()

        # NOTE: SQLAlchemy expires ORM instances on commit by default, so we must extract
        # scalar values inside the session context to avoid DetachedInstanceError.
        rows_data: List[dict[str, Any]] = []
        with db.get_session() as session:
            rows: List[DbStrategyStatsDaily] = (
                session.query(DbStrategyStatsDaily)
                .filter(DbStrategyStatsDaily.date >= cutoff)
                .order_by(DbStrategyStatsDaily.date.asc())
                .all()
            )
            for r in rows:
                raw_regime = str(getattr(r, "regime", "") or "")
                rows_data.append(
                    {
                        "strategy": str(getattr(r, "strategy", "") or ""),
                        "regime": self._normalize_regime_name(raw_regime),
                        "n_trades": int(getattr(r, "n_trades", 0) or 0),
                        "winrate": float(getattr(r, "winrate", 0.0) or 0.0),
                        "avg_r": float(getattr(r, "avg_r", 0.0) or 0.0),
                        "std_r": float(getattr(r, "std_r", 0.0) or 0.0),
                        "pnl_r_total": float(getattr(r, "pnl_r_total", 0.0) or 0.0),
                    }
                )

        # Aggregate across days into per-(strategy, regime) stats (PRD 9.1 best-effort).
        # We treat each day's StrategyStatsDaily as a summary of that day's trades.
        bucket: Dict[tuple[str, str], Dict[str, Any]] = {}
        for row in rows_data:
            key = (str(row["strategy"]), str(row["regime"]))
            b = bucket.setdefault(
                key,
                {
                    "n_trades": 0,
                    "win_w": 0.0,
                    "mean_r_w": 0.0,
                    "m2_r": 0.0,  # pooled sum of squares for variance
                    "pnl_r_total": 0.0,
                    "equity_curve_r": [],
                },
            )
            n = int(row["n_trades"])
            if n <= 0:
                continue
            mean_r = float(row["avg_r"])
            std_r = float(row["std_r"])
            pnl_r_total = float(row["pnl_r_total"])
            b["n_trades"] += n
            b["win_w"] += float(row["winrate"]) * n
            b["mean_r_w"] += mean_r * n
            # Pooled variance components: (n-1)*std^2 + n*(mean - overall_mean)^2 computed later.
            b["m2_r"] += max(0, n - 1) * (std_r**2)
            b["pnl_r_total"] += pnl_r_total
            b["equity_curve_r"].append(pnl_r_total)

        def _max_drawdown_r(daily_pnl_r: List[float]) -> float:
            equity = 0.0
            peak = 0.0
            max_dd = 0.0
            for x in daily_pnl_r:
                equity += float(x)
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)
            return float(max_dd)

        out: List[StrategyDailyStats] = []
        for (strategy, regime), agg in bucket.items():
            n_trades = int(agg["n_trades"])
            mean_r = (float(agg["mean_r_w"]) / n_trades) if n_trades > 0 else 0.0
            # Add between-day mean drift component into pooled variance.
            # This is an approximation because we don't have per-trade R distribution here.
            # m2_total = within + between.
            # between = sum(n_i * (mean_i - mean_r)^2). We don't have all mean_i after loop,
            # so we rely on within-only pooled variance (conservative) for determinism.
            var_r = (float(agg["m2_r"]) / max(1, n_trades - 1)) if n_trades > 1 else 0.0
            std_r = float(var_r**0.5)
            max_dd = _max_drawdown_r(list(agg["equity_curve_r"]))
            out.append(
                StrategyDailyStats(
                    date=now.date(),
                    strategy=strategy,
                    regime=regime,
                    n_trades=n_trades,
                    winrate=(float(agg["win_w"]) / n_trades) if n_trades > 0 else 0.0,
                    avg_r=float(mean_r),
                    std_r=float(std_r),
                    max_drawdown_r=float(max_dd),
                    expectancy=float(mean_r),
                    total_pnl_r=float(agg["pnl_r_total"]),
                )
            )
        return out

    def _normalize_regime_name(self, regime: str) -> str:
        """Map legacy/serialized regime placeholders to a usable regime key."""
        val = str(regime or "").strip().lower()
        if val in ("", "{}", "none", "null"):
            return "chop"
        return val

    def _persist_actions(self, db: DatabaseDatabase, actions: List[AgentAction]) -> None:
        for action in actions:

            def _write_action(session: Any, a: AgentAction = action) -> None:
                session.add(
                    DbAgentAction(
                        timestamp=a.timestamp,
                        action_type=a.action_type.value,
                        strategy=a.strategy,
                        regime=a.regime,
                        details_json=a.details,
                        human_reviewed=False,
                        approved=False,
                    )
                )

            db.write(
                "agent_action",
                _write_action,
            )

    def tune_parameters(
        self,
        stats: StrategyDailyStats,
        current_config: Dict[str, Any],
        *,
        as_of: Optional[datetime] = None,
    ) -> List[AgentAction]:
        """
        Stage 2: deterministic parameter tuning via offline evaluation.
        """
        # Load necessary params to pass to tuner
        _window_days, min_trades, _z_high, max_dd_r = self._load_stage1_params()
        return self.stage2_tuner.tune_parameters(
            stats,
            current_config,
            min_trades=min_trades,
            max_dd_r=max_dd_r,
            as_of=as_of,
        )

    def propose_code_changes(self, stats: StrategyDailyStats, strategy_file_path: str) -> List[AgentAction]:
        """
        Stage 3: Generates a new strategy variant based on performance and source code.
        """
        return self.stage3_proposer.propose_code_changes(stats, strategy_file_path)

    def run_cycle_with_db(self, db: Any, as_of: Optional[datetime] = None) -> List[AgentAction]:
        """
        Run full agent cycle: load stats, analyze, persist actions, apply config changes.

        Stage 1: Health & Risk Adjustments (always runs)
        Stage 2: Parameter Tuning (if enabled and strategy has sufficient data)
        """
        now = as_of or datetime.now(timezone.utc)
        stats_list = self._load_recent_stats(db, as_of=now)

        # Stage 1: Health & Risk Adjustments
        actions = self.analyze_performance(stats_list, as_of=now)

        # Stage 2: Parameter Tuning (PRD 9.2)
        # Only tune strategies that weren't disabled by Stage 1
        disabled_strategies = {a.strategy for a in actions if a.action_type == ActionType.DISABLE_STRATEGY}

        cfg = self.config_loader.load_config(self.config_path_or_dir)
        strategies_cfg = cfg.get("strategies", {})
        if isinstance(strategies_cfg.get("strategies"), dict):
            strategies_cfg = strategies_cfg.get("strategies", {})

        for stats in stats_list:
            if stats.strategy in disabled_strategies:
                continue

            # Get current strategy config for tuning
            current_config = strategies_cfg.get(stats.strategy, {})
            if isinstance(current_config, dict):
                current_config = {k: v for k, v in current_config.items() if k not in ("enabled", "activation")}
            else:
                current_config = {}

            tune_actions = self.tune_parameters(stats, current_config, as_of=now)
            actions.extend(tune_actions)

        if actions:
            self._persist_actions(db, actions)
            self.apply_actions(actions)
        return actions

    def run_weekly_analysis(self, db: Any, as_of: Optional[datetime] = None) -> List[AgentAction]:
        """
        Run weekly analysis cycle (Stage 3).

        Generates a weekly report with:
        - Performance summary by strategy/regime
        - Feature importance analysis
        - Model/feature recommendations

        Should be scheduled for Friday EOD.
        """
        now = as_of or datetime.now(timezone.utc)

        # Load week's stats (use 7-day window instead of default)
        week_start = now - timedelta(days=7)

        # Get all trades from the past week for analysis
        from src.analysis.schema import Trade as DbTrade

        weekly_trades: List[Dict[str, Any]] = []
        with db.get_session() as session:
            trades = (
                session.query(DbTrade)
                .filter(DbTrade.entry_time >= week_start)
                .filter(DbTrade.exit_time.isnot(None))
                .all()
            )
            for t in trades:
                weekly_trades.append(
                    {
                        "symbol": str(getattr(t, "symbol", "")),
                        "strategy": str(getattr(t, "strategy", "")),
                        "regime_at_entry": str(getattr(t, "regime_at_entry", "")),
                        "pnl_r": float(getattr(t, "pnl_r", 0.0) or 0.0),
                        "pnl_net": float(getattr(t, "pnl_net", 0.0) or 0.0),
                        "entry_time": getattr(t, "entry_time", None),
                        "exit_time": getattr(t, "exit_time", None),
                        "features_json": getattr(t, "features_json", {}),
                        "regime_tags_entry_json": getattr(t, "regime_tags_entry_json", {}),
                    }
                )

        if not weekly_trades:
            self.logger.info(
                "No trades found for weekly analysis",
                week_start=week_start.isoformat(),
            )
            return []

        # Generate weekly report
        actions = self._generate_weekly_report(weekly_trades, now)

        if actions:
            self._persist_actions(db, actions)

        return actions

    def _generate_weekly_report(self, trades: List[Dict[str, Any]], as_of: datetime) -> List[AgentAction]:
        """
        Generate weekly analysis report using LLM.
        """
        import json
        from pathlib import Path

        from src.agent.approval import get_stage3_approval

        cfg = self.config_loader.load_config(self.config_path_or_dir)
        agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
        stage3 = agent_cfg.get("stage3", {}) if isinstance(agent_cfg, dict) else {}

        enabled = bool(stage3.get("enabled", False))
        approved, env_key = get_stage3_approval(stage3)

        if not enabled:
            self.logger.info("Agent Stage 3 (weekly analysis) disabled")
            return []

        if not approved:
            self.logger.warning(
                "Agent Stage 3 not approved; skipping weekly report",
                approval_env_var=env_key,
            )
            return []

        # Compute summary statistics
        stats_by_strategy: Dict[str, Dict[str, Any]] = {}
        for t in trades:
            strat = t["strategy"]
            if strat not in stats_by_strategy:
                stats_by_strategy[strat] = {
                    "n_trades": 0,
                    "wins": 0,
                    "total_r": 0.0,
                    "total_pnl": 0.0,
                }
            stats_by_strategy[strat]["n_trades"] += 1
            stats_by_strategy[strat]["total_r"] += t["pnl_r"]
            stats_by_strategy[strat]["total_pnl"] += t["pnl_net"]
            if t["pnl_r"] > 0:
                stats_by_strategy[strat]["wins"] += 1

        for s in stats_by_strategy.values():
            s["winrate"] = s["wins"] / s["n_trades"] if s["n_trades"] > 0 else 0.0
            s["avg_r"] = s["total_r"] / s["n_trades"] if s["n_trades"] > 0 else 0.0

        # Initialize LLM client if needed
        if self.llm_client is None:
            self.llm_client = LLMClient(self.config_loader, self.logger)

        # Generate report via LLM
        system_prompt = """You are a Senior Quantitative Researcher analyzing weekly trading performance.
Your goal is to provide actionable insights for improving strategy performance.

Output a JSON object with:
1. "executive_summary": 2-3 sentence overview of the week's performance
2. "strategy_analysis": List of per-strategy insights with recommendations
3. "feature_recommendations": Suggested new features to add for ML/signal improvement
4. "model_recommendations": Suggested model architecture or threshold changes
5. "risk_recommendations": Any risk management adjustments needed

Be specific and data-driven. Reference actual numbers from the data provided."""

        prompt = f"""Analyze the following weekly trading data:

**Performance Summary by Strategy:**
{json.dumps(stats_by_strategy, indent=2)}

**Total Trades:** {len(trades)}
**Date Range:** Week ending {as_of.strftime("%Y-%m-%d")}

**Sample Trade Features (first 5):**
{json.dumps([t.get("features_json", {}) for t in trades[:5]], indent=2)}

Provide your analysis as a JSON object."""

        try:
            response = self.llm_client.complete(prompt, system_prompt=system_prompt)
            # Parse LLM response
            payload = response.replace("```json", "").replace("```", "").strip()
            report_data = json.loads(payload)
        except Exception as e:
            self.logger.error(
                "Failed to generate weekly report via LLM",
                error=str(e),
                exc_info=True,
            )
            # Fall back to basic report
            report_data = {
                "executive_summary": f"Analyzed {len(trades)} trades this week.",
                "strategy_analysis": [{"strategy": k, **v} for k, v in stats_by_strategy.items()],
                "feature_recommendations": [],
                "model_recommendations": [],
                "risk_recommendations": [],
                "error": str(e),
            }

        # Write report to artifacts
        week_num = as_of.isocalendar()[1]
        year = as_of.year
        report_dir = Path("artifacts/weekly_reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / f"week_{year}_{week_num:02d}.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "generated_at": as_of.isoformat(),
                    "week": week_num,
                    "year": year,
                    "n_trades": len(trades),
                    "stats_by_strategy": stats_by_strategy,
                    "analysis": report_data,
                },
                f,
                indent=2,
                default=str,
            )

        # Also write markdown summary
        md_path = report_dir / f"week_{year}_{week_num:02d}.md"
        md_content = f"""# Weekly Trading Report - Week {week_num}, {year}

**Generated:** {as_of.strftime("%Y-%m-%d %H:%M UTC")}
**Total Trades:** {len(trades)}

## Executive Summary

{report_data.get("executive_summary", "No summary available.")}

## Strategy Performance

| Strategy | Trades | Win Rate | Avg R | Total PnL |
|----------|--------|----------|-------|-----------|
"""
        for strat, s in stats_by_strategy.items():
            md_content += (
                f"| {strat} | {s['n_trades']} | {s['winrate']:.1%} | {s['avg_r']:.2f}R | ${s['total_pnl']:.2f} |\n"
            )

        md_content += f"""
## Feature Recommendations

{chr(10).join(["- " + str(r) for r in report_data.get("feature_recommendations", ["None"])])}

## Model Recommendations

{chr(10).join(["- " + str(r) for r in report_data.get("model_recommendations", ["None"])])}

## Risk Recommendations

{chr(10).join(["- " + str(r) for r in report_data.get("risk_recommendations", ["None"])])}
"""
        with open(md_path, "w") as f:
            f.write(md_content)

        self.logger.info(
            "Generated weekly analysis report",
            report_path=str(report_path),
            md_path=str(md_path),
            n_trades=len(trades),
        )

        return [
            AgentAction(
                timestamp=as_of,
                action_type=ActionType.CODE_PROPOSAL,  # Reuse for reports
                strategy="weekly_analysis",
                regime=None,
                details={
                    "report_file": str(report_path),
                    "markdown_file": str(md_path),
                    "n_trades": len(trades),
                    "week": week_num,
                    "year": year,
                },
                reason="Weekly Stage 3 analysis report",
            )
        ]

    def apply_actions(self, actions: List[AgentAction]) -> None:
        """
        Apply agent actions to configuration files.
        Backward compatibility wrapper for the refactored API.
        """
        from pathlib import Path

        import yaml

        for action in actions:
            if action.action_type == ActionType.TUNE_PARAM:
                # Extract parameters from action details
                new_params = action.details.get("new_params", {})
                window_days = action.details.get("window_days", 30)
                if new_params:
                    path = Path(self.config_path)
                    if path.exists():
                        with open(path) as f:
                            data = yaml.safe_load(f) or {}
                    else:
                        data = {}

                    if action.strategy not in data:
                        data[action.strategy] = {}
                    if "params" not in data[action.strategy]:
                        data[action.strategy]["params"] = {}
                    if "metadata" not in data[action.strategy]:
                        data[action.strategy]["metadata"] = {}

                    data[action.strategy]["params"].update(new_params)
                    data[action.strategy]["metadata"]["last_optimized"] = action.timestamp.strftime("%Y-%m-%d")
                    data[action.strategy]["metadata"]["window_days"] = window_days

                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w") as f:
                        yaml.safe_dump(data, f, default_flow_style=False)
            elif action.action_type in (
                ActionType.DISABLE_STRATEGY,
                ActionType.REDUCE_RISK,
            ):
                # PRD 9.1: Write risk adjustments to strategies.auto.yaml
                path = Path(self.config_path)
                if path.exists():
                    with open(path) as f:
                        data = yaml.safe_load(f) or {}
                else:
                    data = {}

                if action.strategy not in data:
                    data[action.strategy] = {}

                # Determine target config location based on regime
                if action.regime:
                    # Regime-specific override
                    if "regimes" not in data[action.strategy]:
                        data[action.strategy]["regimes"] = {}
                    if action.regime not in data[action.strategy]["regimes"]:
                        data[action.strategy]["regimes"][action.regime] = {}
                    target_cfg = data[action.strategy]["regimes"][action.regime]
                else:
                    # Strategy-level override (no regime specified)
                    target_cfg = data[action.strategy]

                if action.action_type == ActionType.DISABLE_STRATEGY:
                    target_cfg["enabled"] = False
                    target_cfg["disabled_at"] = action.timestamp.strftime("%Y-%m-%d")
                    target_cfg["disabled_reason"] = action.reason
                else:  # REDUCE_RISK
                    # Halve the risk per trade (PRD 9.1 default action)
                    current_risk = float(target_cfg.get("max_risk_per_trade", 50.0))
                    new_risk = current_risk / 2.0
                    # Floor at 0 when below epsilon
                    target_cfg["max_risk_per_trade"] = 0.0 if new_risk < 0.1 else new_risk
                    target_cfg["risk_reduced_at"] = action.timestamp.strftime("%Y-%m-%d")
                    target_cfg["risk_reduced_reason"] = action.reason

                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    yaml.safe_dump(data, f, default_flow_style=False)

                self.logger.info(
                    "Agent action applied to config",
                    action_type=action.action_type.value,
                    strategy=action.strategy,
                    regime=action.regime,
                    config_path=str(path),
                )
