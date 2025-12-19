import difflib
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.agent.bars_provider import JsonlBarsProvider
from src.agent.llm import LLMClient
from src.agent.models import ActionType, AgentAction, StrategyDailyStats
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
        # Optional injectable evaluator for deterministic Stage 2 (unit tests / offline runners).
        self.stage2_evaluator = stage2_evaluator
        # Optional injectable evaluator for Stage 3 gating (unit tests / offline runners).
        self.stage3_evaluator = stage3_evaluator

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

    def _load_recent_stats(
        self, db: DatabaseDatabase, as_of: Optional[datetime] = None
    ) -> List[StrategyDailyStats]:
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
                rows_data.append(
                    {
                        "strategy": str(getattr(r, "strategy", "") or ""),
                        "regime": str(getattr(r, "regime", "") or ""),
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

    def _persist_actions(
        self, db: DatabaseDatabase, actions: List[AgentAction]
    ) -> None:
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

        Behavior:
        - If `agent.stage2.enabled` is false: returns [].
        - Else: enumerates candidates from `agent.stage2.search_space[strategy]` and evaluates them
          using `self.stage2_evaluator` (must be deterministic).
        """
        cfg = self.config_loader.load_config(self.config_path_or_dir)
        agent_cfg = (cfg.get("agent") or {}) if isinstance(cfg, dict) else {}
        stage2 = (agent_cfg.get("stage2") or {}) if isinstance(agent_cfg, dict) else {}
        enabled = bool(stage2.get("enabled", False))
        if not enabled:
            return []

        # PRD 9.2: parameter tuning must be offline/deterministic.
        # Enforce an offline bars source when using the built-in evaluator; custom evaluators
        # may provide deterministic behavior without requiring bars access.
        offline_dir = str(stage2.get("offline_bars_dir", "")).strip()
        if self.stage2_evaluator is None and not offline_dir:
            self.logger.error(
                "Agent Stage 2 enabled but no offline bars source configured",
                required_key="agent.stage2.offline_bars_dir",
            )
            raise ValueError(
                "Agent Stage 2 requires agent.stage2.offline_bars_dir for offline determinism"
            )

        # Deterministic default: thread `as_of` from the caller (PRD 11.1).
        now = as_of or datetime.now(timezone.utc)

        raw_search_space = (
            stage2.get("search_space") if isinstance(stage2, dict) else None
        )
        search_space: Dict[str, Any] = (
            dict(raw_search_space) if isinstance(raw_search_space, dict) else {}
        )
        raw_strat_space = search_space.get(stats.strategy)
        strat_space = raw_strat_space if isinstance(raw_strat_space, dict) else {}
        if not strat_space:
            self.logger.info(
                "Agent Stage 2: no search space configured",
                strategy=stats.strategy,
                regime=stats.regime,
            )
            return []

        evaluator = self.stage2_evaluator
        if evaluator is None:
            from src.agent.stage2 import DeterministicStage2Evaluator

            def _clock(now: datetime = now) -> datetime:
                return now

            # Build evaluator from merged config using offline bars (PRD 9.2).
            evaluator = DeterministicStage2Evaluator(
                cfg if isinstance(cfg, dict) else {},
                self.config_loader,
                self.logger,
                clock=_clock,
                bars_provider=JsonlBarsProvider(Path(offline_dir)),
            )

        def _regime() -> Any:
            # stats.regime can be stored as string in DB-derived stats.
            r = str(stats.regime or "").strip().lower()
            from src.core.domain import Regime as RegimeEnum

            if r == "bull":
                return RegimeEnum.BULL
            if r == "bear":
                return RegimeEnum.BEAR
            return RegimeEnum.CHOP

        # Evaluate baseline metrics for the current config.
        if callable(evaluator) and not hasattr(evaluator, "evaluate"):
            baseline_metrics = evaluator(stats, dict(current_config))
        else:
            m0 = evaluator.evaluate(
                stats.strategy, _regime(), dict(current_config), as_of=now
            )
            baseline_metrics = {
                "expectancy": float(m0.expectancy),
                "max_drawdown_r": float(m0.max_drawdown_r),
                "n_trades": int(m0.n_trades),
            }

        baseline_expectancy = float(baseline_metrics.get("expectancy", 0.0))
        baseline_dd = float(baseline_metrics.get("max_drawdown_r", 0.0))
        baseline_n = int(baseline_metrics.get("n_trades", 0))

        # PRD 9.2: require sufficient sample size.
        _window_days, min_trades, _z_high, max_dd_r = self._load_stage1_params()
        if baseline_n < int(min_trades):
            self.logger.info(
                "Agent Stage 2: insufficient baseline sample size",
                strategy=stats.strategy,
                regime=stats.regime,
                n_trades=baseline_n,
                min_trades=min_trades,
            )
            return []

        # Build candidate list (cartesian product) deterministically.
        keys = sorted(strat_space.keys())
        values_list = []
        for k in keys:
            v = strat_space.get(k)
            if isinstance(v, list):
                values_list.append([x for x in v])
            else:
                values_list.append([v])

        import itertools

        best_params: Dict[str, Any] = {}
        best_score: Optional[float] = None
        best_metrics: Dict[str, Any] = {}

        for combo in itertools.product(*values_list):
            cand = dict(zip(keys, combo, strict=True))
            merged = {**current_config, **cand}
            if callable(evaluator) and not hasattr(evaluator, "evaluate"):
                metrics = evaluator(stats, merged)
            else:
                m = evaluator.evaluate(stats.strategy, _regime(), merged, as_of=now)
                metrics = {
                    "expectancy": float(m.expectancy),
                    "max_drawdown_r": float(m.max_drawdown_r),
                    "n_trades": int(m.n_trades),
                }
            # PRD 9.2: choose candidate that improves expectancy and controls drawdown with enough trades.
            expectancy = float(metrics.get("expectancy", 0.0))
            dd = float(metrics.get("max_drawdown_r", 0.0))
            n = int(metrics.get("n_trades", 0))

            if n < int(min_trades):
                continue
            if dd > float(max_dd_r):
                continue
            if expectancy <= baseline_expectancy:
                continue
            if dd > baseline_dd:
                continue

            # Deterministic objective: maximize expectancy, then minimize drawdown, then maximize n_trades.
            score = expectancy * 1_000_000.0 - dd * 1_000.0 + n

            if best_score is None or score > best_score:
                best_score = score
                best_params = cand
                best_metrics = dict(metrics)

        if not best_params:
            return []

        return [
            AgentAction(
                timestamp=now,
                action_type=ActionType.TUNE_PARAM,
                strategy=stats.strategy,
                regime=stats.regime,
                details={
                    "new_params": best_params,
                    "metrics": best_metrics,
                    "baseline_metrics": baseline_metrics,
                    "window_days": int(
                        ((stage2 or {}).get("window_days", 30))
                        if isinstance(stage2, dict)
                        else 30
                    ),
                },
                reason="Deterministic Stage 2 parameter tuning",
            )
        ]

    def propose_code_changes(
        self, stats: StrategyDailyStats, strategy_file_path: str
    ) -> List[AgentAction]:
        """
        Stage 3: Generates a new strategy variant based on performance and source code.
        """
        try:
            cfg = self.config_loader.load_config(self.config_path_or_dir)
            agent_cfg = (cfg.get("agent") or {}) if isinstance(cfg, dict) else {}
            stage3 = (
                (agent_cfg.get("stage3") or {}) if isinstance(agent_cfg, dict) else {}
            )
            enabled = bool(stage3.get("enabled", False))
            env_key = str(stage3.get("approval_env_var", "CERBERUS_STAGE3_APPROVED"))
            approved = str(os.getenv(env_key, "")).strip().lower() in (
                "1",
                "true",
                "yes",
            )
            write_to_src = bool(stage3.get("write_to_src", False))
            propose_scanner_profiles = bool(
                stage3.get("propose_scanner_profiles", False)
            )
            backtest = (
                (stage3.get("backtest") or {}) if isinstance(stage3, dict) else {}
            )
            backtest_enabled = bool(backtest.get("enabled", True))
            min_trades = int(backtest.get("min_trades", 0))
            max_dd_r = float(backtest.get("max_drawdown_r", float("inf")))
            min_expectancy_delta = float(backtest.get("min_expectancy_delta", 0.0))

            # PRD 9.3: Stage 3 must not change live trading behavior without explicit human approval.
            if not enabled:
                self.logger.info(
                    "Agent Stage 3 disabled; skipping code proposal",
                    strategy=stats.strategy,
                    regime=stats.regime,
                )
                return []
            if not approved:
                self.logger.warning(
                    "Agent Stage 3 not approved; skipping code proposal",
                    strategy=stats.strategy,
                    regime=stats.regime,
                    approval_env_var=env_key,
                )
                return []

            offline_dir = str(stage3.get("offline_bars_dir", "")).strip()
            if backtest_enabled and self.stage3_evaluator is None and not offline_dir:
                self.logger.error(
                    "Agent Stage 3 backtest enabled but no offline bars source configured",
                    required_key="agent.stage3.offline_bars_dir",
                )
                raise ValueError(
                    "Agent Stage 3 backtest requires agent.stage3.offline_bars_dir for offline determinism"
                )

            if self.llm_client is None:
                self.llm_client = LLMClient(self.config_loader, self.logger)
            with open(strategy_file_path, "r") as f:
                source_code = f.read()

            scanner_profiles_source = ""
            if propose_scanner_profiles:
                try:
                    with open("src/scanner/profiles.py", "r") as f:
                        scanner_profiles_source = f.read()
                except Exception as e:
                    self.logger.warning(
                        "Failed to load scanner profiles for Stage 3 proposal",
                        error=str(e),
                    )

            system_prompt = """You are a Senior Quantitative Architect for the Cerberus Trading System.
Your goal is to iterate on underperforming trading strategies to improve their "expectancy" (average R per trade) and reduce "drawdown".
You adhere to the following principles:
1. Scientific Method: Changes must be motivated by data (the provided stats).
2. Incremental Refinement (Annealing): Prefer small, targeted adjustments (adding a filter, tightening a stop, adjusting a threshold) over rewriting the entire strategy. Radical changes increase risk.
3. Risk First: Your primary constraint is safety. Never remove risk controls.
4. Vertical Slice Architecture: The strategy is a self-contained unit. Do not invent external dependencies.
5. Determinism: Ensure the new logic remains deterministic (no random numbers).

Output valid Python code only."""

            prompt = f"""
            The strategy '{stats.strategy}' is underperforming based on recent trading data.

            Current Performance Stats:
            - Win Rate: {stats.winrate:.2f}
            - Avg R (Expectancy): {stats.avg_r:.2f}
            - Max Drawdown R: {stats.max_drawdown_r:.2f}
            - Number of Trades: {stats.n_trades}

            Source Code:
            ```python
            {source_code}
            ```

            Scanner Profiles (context for symbol selection):
            ```python
            {scanner_profiles_source}
            ```

            **Task:**
            Propose a **V2** version of this strategy class to improve its Expectancy (Avg R) and/or reduce its Max Drawdown.
            The system is in an "annealing" phase—we want to converge on a stable, profitable configuration.

            **Guidelines:**
            1. Analyze the logic: identify potential weak points (e.g., entering too early in chop, stops too loose, taking trades against the trend).
            2. Propose a **targeted** fix. Examples:
               - Add a regime filter (e.g., only trade if `market_state.regime == Regime.BULL`).
               - Add a technical filter (e.g., `adx > 25`).
               - Tighten exit logic (e.g., reduce `max_hold_minutes`).
               - Adjust entry triggers.
            3. If the Scanner Profile is too loose, you may propose an updated `src/scanner/profiles.py` as well.
            4. **Do not** rewrite the entire class structure. Keep specific logic that works.
            5. Ensure the new class name is `{stats.strategy}_v2` (or increment version).

            **Output format:**
            Return a JSON object with strictly these keys:
            - "analysis": "Brief reasoning for the change"
            - "strategy_code": "Full Python code for the new strategy class"
            - "scanner_profiles_code": "Full Python code for src/scanner/profiles.py (or empty string if no change)"
            """

            response = self.llm_client.complete(prompt, system_prompt=system_prompt)
            payload = response.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(payload)
                if not isinstance(parsed, dict):
                    raise ValueError("stage3_response_not_dict")
                new_code = str(parsed.get("strategy_code", "") or "").strip()
                new_profiles_code = str(
                    parsed.get("scanner_profiles_code", "") or ""
                ).strip()
            except Exception:
                new_code = response.replace("```python", "").replace("```", "").strip()
                new_profiles_code = ""

            # PRD 9.3: Backtest candidate versions and gate on objective thresholds.
            baseline_metrics = None
            candidate_metrics = None
            gate_passed = None
            if backtest_enabled:
                # Stage 3 uses the strategy name from stats to fetch baseline params for evaluation.
                strategies_cfg = (
                    cfg.get("strategies") if isinstance(cfg, dict) else None
                )
                if isinstance(strategies_cfg, dict) and isinstance(
                    strategies_cfg.get("strategies"), dict
                ):
                    strategies_cfg = strategies_cfg.get("strategies")
                baseline_params = (
                    (strategies_cfg or {}).get(stats.strategy, {})
                    if isinstance(strategies_cfg, dict)
                    else {}
                )
                if isinstance(baseline_params, dict):
                    baseline_params = {
                        k: v for k, v in baseline_params.items() if k != "enabled"
                    }
                else:
                    baseline_params = {}

                from src.core.domain import Regime as RegimeEnum

                r = str(getattr(stats, "regime", "") or "").strip().lower()
                regime = RegimeEnum.CHOP
                if r in ("bull", "bear", "chop"):
                    regime = RegimeEnum(r)

                # Use a single timestamp for the whole Stage 3 evaluation/artifact set.
                now = datetime.now(timezone.utc)
                if self.stage3_evaluator is not None:
                    out = self.stage3_evaluator(stats, source_code, new_code, cfg)
                    baseline_metrics = out.get("baseline_metrics")
                    candidate_metrics = out.get("candidate_metrics")
                else:
                    from src.agent.stage2 import DeterministicStage2Evaluator
                    from src.agent.stage3 import DeterministicStage3Evaluator

                    provider = (
                        JsonlBarsProvider(Path(offline_dir)) if offline_dir else None
                    )
                    stage2 = DeterministicStage2Evaluator(
                        cfg, self.config_loader, self.logger, bars_provider=provider
                    )
                    stage3_eval = DeterministicStage3Evaluator(
                        cfg, self.config_loader, self.logger, bars_provider=provider
                    )
                    baseline_metrics = stage2.evaluate(
                        stats.strategy, regime, dict(baseline_params), as_of=now
                    )
                    candidate_metrics = stage3_eval.evaluate_code(
                        new_code,
                        strategy_params=dict(baseline_params),
                        regime=regime,
                        as_of=now,
                    )

                def _m(m: Any) -> dict:
                    if m is None:
                        return {}
                    if isinstance(m, dict):
                        return m
                    return {
                        "expectancy": float(getattr(m, "expectancy", 0.0) or 0.0),
                        "max_drawdown_r": float(
                            getattr(m, "max_drawdown_r", 0.0) or 0.0
                        ),
                        "n_trades": int(getattr(m, "n_trades", 0) or 0),
                    }

                b = _m(baseline_metrics)
                c = _m(candidate_metrics)
                baseline_metrics = b
                candidate_metrics = c
                gate_passed = (
                    int(c.get("n_trades", 0)) >= int(min_trades)
                    and float(c.get("max_drawdown_r", 0.0)) <= float(max_dd_r)
                    and float(c.get("expectancy", 0.0))
                    >= float(b.get("expectancy", 0.0)) + float(min_expectancy_delta)
                )
                if not gate_passed:
                    self.logger.warning(
                        "Agent Stage 3 gate failed; not emitting proposal",
                        strategy=stats.strategy,
                        regime=stats.regime,
                        baseline_metrics=b,
                        candidate_metrics=c,
                        min_trades=min_trades,
                        max_drawdown_r=max_dd_r,
                        min_expectancy_delta=min_expectancy_delta,
                    )
                    return []

            # Save proposal (prefer artifacts by default to avoid unintended repo mutations).
            proposal_dir = (
                "src/strategies/proposals" if write_to_src else "artifacts/proposals"
            )
            os.makedirs(proposal_dir, exist_ok=True)
            proposal_filename = f"{stats.strategy}_v2.py"
            proposal_path = os.path.join(proposal_dir, proposal_filename)

            with open(proposal_path, "w") as f:
                f.write(new_code)

            profiles_proposal_path = ""
            profiles_diff_path = ""
            if propose_scanner_profiles and new_profiles_code:
                profiles_proposal_path = os.path.join(
                    proposal_dir, "scanner_profiles_v2.py"
                )
                with open(profiles_proposal_path, "w") as f:
                    f.write(new_profiles_code)

                profiles_diff_path = profiles_proposal_path + ".diff"
                try:
                    diff_text_profiles = "".join(
                        difflib.unified_diff(
                            scanner_profiles_source.splitlines(keepends=True),
                            new_profiles_code.splitlines(keepends=True),
                            fromfile="src/scanner/profiles.py",
                            tofile=profiles_proposal_path,
                            lineterm="",
                        )
                    )
                    with open(profiles_diff_path, "w") as f:
                        f.write(diff_text_profiles)
                except Exception as e:
                    self.logger.warning(
                        "Failed to write scanner profiles diff",
                        error=str(e),
                    )

            # PRD 9.3: output a patch/diff and summary for human review.
            diff_path = proposal_path + ".diff"
            diff_text = "".join(
                difflib.unified_diff(
                    source_code.splitlines(keepends=True),
                    new_code.splitlines(keepends=True),
                    fromfile=strategy_file_path,
                    tofile=proposal_path,
                    lineterm="",
                )
            )
            with open(diff_path, "w") as f:
                f.write(diff_text)

            summary_path = proposal_path + ".summary.json"
            summary = {
                "strategy": stats.strategy,
                "regime": stats.regime,
                "source_file": strategy_file_path,
                "proposal_file": proposal_path,
                "diff_file": diff_path,
                "stats": {
                    "winrate": float(stats.winrate),
                    "avg_r": float(stats.avg_r),
                    "max_drawdown_r": float(stats.max_drawdown_r),
                    "n_trades": int(stats.n_trades),
                },
                "stage3_gate": {
                    "enabled": bool(backtest_enabled),
                    "passed": bool(gate_passed) if gate_passed is not None else None,
                    "thresholds": {
                        "min_trades": int(min_trades),
                        "max_drawdown_r": float(max_dd_r),
                        "min_expectancy_delta": float(min_expectancy_delta),
                    },
                    "baseline_metrics": baseline_metrics,
                    "candidate_metrics": candidate_metrics,
                },
                "scanner_profile_proposal": {
                    "enabled": bool(propose_scanner_profiles),
                    "proposal_file": profiles_proposal_path or None,
                    "diff_file": profiles_diff_path or None,
                },
                "sha256": {
                    "source": hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
                    "proposal": hashlib.sha256(new_code.encode("utf-8")).hexdigest(),
                },
                "created_at": now.isoformat(),
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            return [
                AgentAction(
                    timestamp=now,
                    action_type=ActionType.CODE_PROPOSAL,
                    strategy=stats.strategy,
                    regime=stats.regime,
                    details={
                        "proposal_path": proposal_path,
                        "diff_path": diff_path,
                        "summary_path": summary_path,
                        "scanner_profiles_proposal_path": profiles_proposal_path
                        or None,
                        "scanner_profiles_diff_path": profiles_diff_path or None,
                    },
                    reason="LLM generated code proposal for underperformance",
                )
            ]

        except Exception as e:
            self.logger.error(
                "Failed to propose code changes", strategy=stats.strategy, error=str(e)
            )

        return []

    def apply_actions(self, actions: List[AgentAction]):
        """
        Applies actions by writing to strategies.auto.yaml.
        """
        if not actions:
            return

        # Load existing auto config or empty
        current_config: Dict[str, Any] = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    current_config = yaml.safe_load(f) or {}
            except Exception as e:
                self.logger.error("Failed to load existing auto config", error=str(e))

        # Apply changes
        for action in actions:
            self.logger.info("Applying agent action", action=action)

            raw_strat_config = current_config.get(action.strategy)
            strat_config: Dict[str, Any] = (
                dict(raw_strat_config) if isinstance(raw_strat_config, dict) else {}
            )

            def _reduce_risk(value: Any) -> float:
                """
                PRD 9.1: decrease risk downward, down to zero if needed.

                Determinism: round to cents and clamp below 1 cent to 0.0.
                """
                try:
                    prior_f = float(value)
                except Exception:
                    prior_f = 0.0
                if prior_f <= 0.0:
                    return 0.0
                new = max(0.0, prior_f * 0.5)
                new = round(new, 2)
                if new <= 0.01:
                    return 0.0
                return float(new)

            if action.action_type == ActionType.DISABLE_STRATEGY:
                # PRD 9.1: disable per (strategy, regime) when regime provided.
                if action.regime:
                    raw_regimes = strat_config.get("regimes")
                    regimes = dict(raw_regimes) if isinstance(raw_regimes, dict) else {}
                    raw_r_cfg = regimes.get(action.regime)
                    r_cfg = dict(raw_r_cfg) if isinstance(raw_r_cfg, dict) else {}
                    r_cfg["enabled"] = False
                    regimes[action.regime] = r_cfg
                    strat_config["regimes"] = regimes
                else:
                    strat_config["enabled"] = False
            elif action.action_type == ActionType.ENABLE_STRATEGY:
                if action.regime:
                    raw_regimes = strat_config.get("regimes")
                    regimes = dict(raw_regimes) if isinstance(raw_regimes, dict) else {}
                    raw_r_cfg = regimes.get(action.regime)
                    r_cfg = dict(raw_r_cfg) if isinstance(raw_r_cfg, dict) else {}
                    r_cfg["enabled"] = True
                    regimes[action.regime] = r_cfg
                    strat_config["regimes"] = regimes
                else:
                    strat_config["enabled"] = True
            elif action.action_type == ActionType.REDUCE_RISK:
                # PRD 9.1: tighten risk downward via max_risk_per_trade override (per regime if provided).
                base_cfg = self.config_loader.load_config(self.config_path_or_dir)
                base_risk = (
                    ((base_cfg.get("risk") or {}).get("max_risk_per_trade"))
                    if isinstance(base_cfg, dict)
                    else None
                )
                if base_risk is None:
                    base_risk = 50.0
                if action.regime:
                    raw_regimes = strat_config.get("regimes")
                    regimes = dict(raw_regimes) if isinstance(raw_regimes, dict) else {}
                    raw_r_cfg = regimes.get(action.regime)
                    r_cfg = dict(raw_r_cfg) if isinstance(raw_r_cfg, dict) else {}
                    prior = r_cfg.get(
                        "max_risk_per_trade",
                        strat_config.get("max_risk_per_trade", base_risk),
                    )
                    r_cfg["max_risk_per_trade"] = _reduce_risk(prior)
                    regimes[action.regime] = r_cfg
                    strat_config["regimes"] = regimes
                else:
                    prior = strat_config.get("max_risk_per_trade", base_risk)
                    strat_config["max_risk_per_trade"] = _reduce_risk(prior)

            elif action.action_type == ActionType.TUNE_PARAM:
                # PRD 9.2: write tuned params + metadata into strategies.auto.yaml.
                params = {}
                if isinstance(strat_config, dict) and isinstance(
                    strat_config.get("params"), dict
                ):
                    params = dict(strat_config.get("params") or {})
                if isinstance(action.details, dict) and isinstance(
                    action.details.get("new_params"), dict
                ):
                    params.update(action.details.get("new_params") or {})
                strat_config["params"] = params

                meta = {}
                if isinstance(strat_config, dict) and isinstance(
                    strat_config.get("metadata"), dict
                ):
                    meta = dict(strat_config.get("metadata") or {})
                ts = action.timestamp
                try:
                    last_opt = ts.date().isoformat()
                except Exception:
                    last_opt = ""
                meta["last_optimized"] = last_opt
                if isinstance(action.details, dict) and "window_days" in action.details:
                    try:
                        meta["window_days"] = int(
                            action.details.get("window_days") or 0
                        )
                    except Exception:
                        meta["window_days"] = 0
                if isinstance(action.details, dict) and isinstance(
                    action.details.get("metrics"), dict
                ):
                    meta["metrics"] = action.details.get("metrics")
                strat_config["metadata"] = meta

            current_config[action.strategy] = strat_config
            action.applied = True

        # Write back
        try:
            with open(self.config_path, "w") as f:
                yaml.safe_dump(current_config, f, sort_keys=True)
            self.logger.info("Updated auto configuration", path=self.config_path)
        except Exception as e:
            self.logger.error("Failed to write auto config", error=str(e))

    def run_cycle(self):
        """
        Orchestrates the full agent analysis cycle.
        """
        self.logger.info("Agent cycle started (use run_cycle_with_db for Stage 1)")

    def run_cycle_with_db(
        self, db: Optional[DatabaseDatabase], as_of: Optional[datetime] = None
    ) -> None:
        if db is None:
            self.logger.warning("Agent Stage 1 skipped: no DB provided")
            return

        window_days, min_trades, z_high, max_dd_r = self._load_stage1_params()
        self.logger.info(
            "Agent Stage 1 starting",
            window_days=window_days,
            min_trades=min_trades,
            z_high=z_high,
            max_drawdown_r=max_dd_r,
        )

        stats = self._load_recent_stats(db, as_of=as_of)

        stage1_actions = self.analyze_performance(stats, as_of=as_of)
        if not stage1_actions:
            self.logger.info("Agent Stage 1: no actions produced")

        # PRD 9.2: Stage 2 parameter tuning (config-gated inside tune_parameters).
        # Determinism: pass `as_of` through.
        stage2_actions: List[AgentAction] = []
        cfg = self.config_loader.load_config(self.config_path_or_dir)
        strategies_cfg = (cfg.get("strategies") or {}) if isinstance(cfg, dict) else {}
        for s in stats:
            raw_curr = strategies_cfg.get(s.strategy)
            curr: Dict[str, Any] = dict(raw_curr) if isinstance(raw_curr, dict) else {}
            # If overrides use `params`, flattening in ConfigLoader means `curr` already contains tuned keys.
            stage2_actions.extend(self.tune_parameters(s, curr, as_of=as_of))

        stage3_actions: List[AgentAction] = []
        for s in stats:
            # PRD 9.3: Trigger Stage 3 if Stage 1 suggests failure (Disable or Reduce Risk).
            related = [
                a
                for a in stage1_actions
                if a.strategy == s.strategy
                and a.action_type
                in (ActionType.DISABLE_STRATEGY, ActionType.REDUCE_RISK)
            ]
            if related:
                # Attempt to resolve source file path (convention: src/strategies/{name}.py)
                strategy_file = f"src/strategies/{s.strategy}.py"
                if os.path.exists(strategy_file):
                    self.logger.info(
                        "Triggering Agent Stage 3 (Code Proposal)", strategy=s.strategy
                    )
                    stage3_actions.extend(self.propose_code_changes(s, strategy_file))
                else:
                    self.logger.warning(
                        "Agent Stage 3 skipped: strategy source not found",
                        strategy=s.strategy,
                        path=strategy_file,
                    )

        actions = [*stage1_actions, *stage2_actions, *stage3_actions]
        if not actions:
            self.logger.info("Agent: no actions produced across stages")
            return

        self.apply_actions(actions)
        self._persist_actions(db, actions)
        self.logger.info(
            "Agent cycle complete",
            stage1_actions=len(stage1_actions),
            stage2_actions=len(stage2_actions),
            actions=len(actions),
        )
