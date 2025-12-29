from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Type, TypedDict, cast

from src.agent.bars_provider import BarsProvider, JsonlBarsProvider
from src.agent.llm import LLMClient
from src.agent.models import ActionType, AgentAction, StrategyDailyStats
from src.core.config import ConfigLoader
from src.core.domain import Bar, MarketState, Regime, RiskMode, SymbolState
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.strategies.base import BaseStrategy


@dataclass(frozen=True)
class Stage3Metrics:
    expectancy: float
    max_drawdown_r: float
    n_trades: int


class _PendingEntry(TypedDict):
    side: str
    stop: float
    target: float
    risk_per_share: float


class _OpenTrade(_PendingEntry):
    entry: float
    entry_time: datetime


class DeterministicStage3Evaluator:
    """
    PRD 9.3: deterministic offline evaluation for Stage 3 candidate code.

    Uses the same deterministic bar-replay rules as Stage 2:
    - Entry at next bar open after a signal.
    - Exit at stop/target if crossed intrabar (stop prioritized if both hit).
    - One position per symbol at a time.
    - Computes PnL in R using signal stop distance as initial risk.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        config_loader: Any,
        logger: StructuredLogger,
        clock: Optional[Callable[[], datetime]] = None,
        bars_provider: Optional[BarsProvider] = None,
    ) -> None:
        self.config = config
        self.config_loader = config_loader
        self.logger = logger
        self.clock: Callable[[], datetime] = clock or (
            lambda: datetime.now(timezone.utc)
        )
        self.bars_provider = bars_provider
        self.alpaca = (
            AlpacaClient(config_loader, logger) if bars_provider is None else None
        )

    def _bars_window(self, as_of: datetime) -> tuple[datetime, datetime]:
        agent_cfg = (
            (self.config.get("agent") or {}) if isinstance(self.config, dict) else {}
        )
        stage3 = (agent_cfg.get("stage3") or {}) if isinstance(agent_cfg, dict) else {}
        backtest = (stage3.get("backtest") or {}) if isinstance(stage3, dict) else {}
        window_days = int(backtest.get("window_days", 30))
        end = as_of
        start = end - timedelta(days=max(1, window_days))
        return start, end

    def _symbols(self) -> List[str]:
        agent_cfg = (
            (self.config.get("agent") or {}) if isinstance(self.config, dict) else {}
        )
        stage3 = (agent_cfg.get("stage3") or {}) if isinstance(agent_cfg, dict) else {}
        backtest = (stage3.get("backtest") or {}) if isinstance(stage3, dict) else {}
        symbols = backtest.get("symbols") if isinstance(backtest, dict) else None
        if isinstance(symbols, list) and symbols:
            return sorted({str(s).upper() for s in symbols if s})
        uni = self.config.get("universe") if isinstance(self.config, dict) else None
        uni_symbols = uni.get("symbols") if isinstance(uni, dict) else None
        if isinstance(uni_symbols, list):
            return sorted({str(s).upper() for s in uni_symbols if s})
        return []

    def _fetch_bars(self, symbol: str, start: datetime, end: datetime) -> List[Bar]:
        if self.bars_provider is not None:
            return list(
                self.bars_provider.get_bars(symbol, start, end, timeframe="1Min")
            )
        if self.alpaca is None:
            return []

        data = self.alpaca.get_historical_bars(symbol, start, end, timeframe="1Min")
        rows = data.get("bars") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return []

        out: List[Bar] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            t = r.get("t") or r.get("timestamp")
            if isinstance(t, str):
                t = datetime.fromisoformat(t.replace("Z", "+00:00"))
            if not isinstance(t, datetime):
                continue
            out.append(
                Bar(
                    symbol=str(symbol).upper(),
                    time=t,
                    open=float(r.get("o") or r.get("open") or 0.0),
                    high=float(r.get("h") or r.get("high") or 0.0),
                    low=float(r.get("l") or r.get("low") or 0.0),
                    close=float(r.get("c") or r.get("close") or 0.0),
                    volume=float(r.get("v") or r.get("volume") or 0.0),
                )
            )
        return out

    def _extract_strategy_class(self, module: ModuleType) -> Type[BaseStrategy]:
        candidates: List[Type[BaseStrategy]] = []
        for obj in module.__dict__.values():
            if not isinstance(obj, type):
                continue
            try:
                if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                    candidates.append(obj)
            except Exception:
                continue
        if not candidates:
            raise ValueError("No BaseStrategy subclass found in candidate code")
        return sorted(candidates, key=lambda c: c.__name__)[0]

    def _load_candidate_module(self, code: str) -> ModuleType:
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]
        module_name = f"cerberus_stage3_candidate_{digest}"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"{module_name}.py"
            path.write_text(code, encoding="utf-8")

            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ValueError("Failed to create module spec for Stage 3 candidate")

            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.modules.pop(module_name, None)

            return mod

    def evaluate_code(
        self,
        code: str,
        *,
        strategy_params: Dict[str, Any],
        regime: Regime,
        as_of: Optional[datetime] = None,
    ) -> Stage3Metrics:
        now = as_of or self.clock()
        start, end = self._bars_window(now)
        symbols = self._symbols()
        if not symbols:
            self.logger.warning(
                "Stage 3 evaluator has no symbols; returning empty metrics"
            )
            return Stage3Metrics(expectancy=0.0, max_drawdown_r=0.0, n_trades=0)

        mod = self._load_candidate_module(code)
        cls = self._extract_strategy_class(mod)
        strat = cls(dict(strategy_params), self.logger)

        pnl_r_series: List[float] = []

        # Pre-fetch bars for all symbols in parallel
        from concurrent.futures import ThreadPoolExecutor

        bars_by_symbol: Dict[str, List[Bar]] = {}
        max_workers = min(10, len(symbols))
        if max_workers > 0:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._fetch_bars, sym, start, end): sym
                    for sym in symbols
                }
                for future in futures:
                    sym = futures[future]
                    try:
                        bars_by_symbol[sym] = future.result()
                    except Exception:
                        bars_by_symbol[sym] = []

        for sym in symbols:
            bars = bars_by_symbol.get(sym, [])
            if len(bars) < 3:
                continue

            state = SymbolState(
                symbol=sym,
                bars=__import__("collections").deque(maxlen=500),
                indicators={},
                position=None,
                open_orders={},
                allowed_strategies=[getattr(strat, "name", "candidate")],
                meta={},
            )
            market = MarketState(
                time=bars[0].time,
                regime=regime,
                index_symbol=str(self.config.get("index_symbol", "SPY")),
                index_price=0.0,
                index_return=0.0,
                realized_vol=0.0,
                daily_pnl=0.0,
                risk_mode=RiskMode.NORMAL,
                meta={},
            )

            pending_entry: _PendingEntry | None = None
            open_trade: _OpenTrade | None = None
            for i in range(len(bars)):
                bar = bars[i]
                market.time = bar.time

                if pending_entry is not None and open_trade is None:
                    entry = float(bar.open)
                    open_trade = cast(
                        _OpenTrade,
                        {**pending_entry, "entry": entry, "entry_time": bar.time},
                    )
                    pending_entry = None

                if open_trade is not None:
                    side = open_trade["side"]
                    stop = float(open_trade["stop"])
                    target = float(open_trade["target"])
                    entry = float(open_trade["entry"])
                    rps = float(open_trade["risk_per_share"])
                    if rps <= 0:
                        open_trade = None
                    else:
                        stop_hit = (
                            (bar.low <= stop) if side == "buy" else (bar.high >= stop)
                        )
                        target_hit = (
                            (bar.high >= target)
                            if side == "buy"
                            else (bar.low <= target)
                        )
                        exit_px = None
                        if stop_hit:
                            exit_px = stop
                        elif target_hit:
                            exit_px = target
                        if exit_px is not None:
                            pnl_r = (
                                ((exit_px - entry) / rps)
                                if side == "buy"
                                else ((entry - exit_px) / rps)
                            )
                            pnl_r_series.append(float(pnl_r))
                            open_trade = None

                state.bars.append(bar)
                if open_trade is None and pending_entry is None:
                    sig = strat.on_bar(sym, bar, state, market)
                    if sig is None:
                        continue
                    risk_per_share = abs(sig.entry_price - sig.stop_price)
                    if risk_per_share <= 0:
                        continue
                    pending_entry = {
                        "side": sig.side.value,
                        "stop": float(sig.stop_price),
                        "target": float(sig.target_price),
                        "risk_per_share": float(risk_per_share),
                    }

        n = len(pnl_r_series)
        expectancy = float(sum(pnl_r_series) / n) if n > 0 else 0.0
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in pnl_r_series:
            equity += float(r)
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        return Stage3Metrics(
            expectancy=expectancy, max_drawdown_r=float(max_dd), n_trades=int(n)
        )


class Stage3Proposer:
    """
    Orchestrates Stage 3 LLM-based strategy evolution.
    """

    def __init__(
        self,
        logger: StructuredLogger,
        config_loader: ConfigLoader,
        config_path_or_dir: str,
        llm_client: Optional[LLMClient] = None,
        evaluator: Optional[Any] = None,
    ):
        self.logger = logger
        self.config_loader = config_loader
        self.config_path_or_dir = config_path_or_dir
        self.llm_client = llm_client
        self.evaluator = evaluator

    def propose_code_changes(
        self,
        stats: StrategyDailyStats,
        strategy_file_path: str,
    ) -> List[AgentAction]:
        cfg = self.config_loader.load_config(self.config_path_or_dir)
        agent_cfg = (cfg.get("agent") or {}) if isinstance(cfg, dict) else {}
        stage3 = (agent_cfg.get("stage3") or {}) if isinstance(agent_cfg, dict) else {}
        enabled = bool(stage3.get("enabled", False))
        env_key = str(stage3.get("approval_env_var", "CERBERUS_STAGE3_APPROVED"))
        approved = str(os.getenv(env_key, "")).strip().lower() in (
            "1",
            "true",
            "yes",
        )
        write_to_src = bool(stage3.get("write_to_src", False))
        propose_scanner_profiles = bool(stage3.get("propose_scanner_profiles", False))
        backtest = (stage3.get("backtest") or {}) if isinstance(stage3, dict) else {}
        backtest_enabled = bool(backtest.get("enabled", True))
        min_trades = int(backtest.get("min_trades", 0))
        max_dd_r = float(backtest.get("max_drawdown_r", float("inf")))
        min_expectancy_delta = float(backtest.get("min_expectancy_delta", 0.0))

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
        if backtest_enabled and self.evaluator is None and not offline_dir:
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

        # Validation Logic (Backtest & Gate)
        baseline_metrics = None
        candidate_metrics = None
        gate_passed = None

        if backtest_enabled:
            # Baseline params load
            strategies_cfg = cfg.get("strategies") if isinstance(cfg, dict) else None
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

            now = datetime.now(timezone.utc)

            if self.evaluator is not None and not hasattr(
                self.evaluator, "evaluate_code"
            ):
                # Custom evaluator injected
                out = self.evaluator(stats, source_code, new_code, cfg)
                baseline_metrics = out.get("baseline_metrics")
                candidate_metrics = out.get("candidate_metrics")
            else:
                # Use default deterministic evaluators
                from src.agent.stage2 import DeterministicStage2Evaluator

                provider = JsonlBarsProvider(Path(offline_dir)) if offline_dir else None
                evaluator_s2 = DeterministicStage2Evaluator(
                    cfg, self.config_loader, self.logger, bars_provider=provider
                )
                evaluator_s3 = (
                    self.evaluator
                    if self.evaluator
                    else DeterministicStage3Evaluator(
                        cfg, self.config_loader, self.logger, bars_provider=provider
                    )
                )

                baseline_metrics_obj = evaluator_s2.evaluate(
                    stats.strategy, regime, dict(baseline_params), as_of=now
                )
                baseline_metrics = {
                    "expectancy": baseline_metrics_obj.expectancy,
                    "max_drawdown_r": baseline_metrics_obj.max_drawdown_r,
                    "n_trades": baseline_metrics_obj.n_trades,
                }

                candidate_metrics_obj = evaluator_s3.evaluate_code(
                    new_code,
                    strategy_params=dict(baseline_params),
                    regime=regime,
                    as_of=now,
                )
                candidate_metrics = {
                    "expectancy": candidate_metrics_obj.expectancy,
                    "max_drawdown_r": candidate_metrics_obj.max_drawdown_r,
                    "n_trades": candidate_metrics_obj.n_trades,
                }

            def _m(m: Any) -> dict:
                if m is None:
                    return {}
                if isinstance(m, dict):
                    return m
                return {
                    "expectancy": float(getattr(m, "expectancy", 0.0) or 0.0),
                    "max_drawdown_r": float(getattr(m, "max_drawdown_r", 0.0) or 0.0),
                    "n_trades": int(getattr(m, "n_trades", 0) or 0),
                }

            baseline_metrics = _m(baseline_metrics)
            candidate_metrics = _m(candidate_metrics)

            gate_passed = (
                int(candidate_metrics.get("n_trades", 0)) >= int(min_trades)
                and float(candidate_metrics.get("max_drawdown_r", 0.0))
                <= float(max_dd_r)
                and float(candidate_metrics.get("expectancy", 0.0))
                >= float(baseline_metrics.get("expectancy", 0.0))
                + float(min_expectancy_delta)
            )

            if not gate_passed:
                self.logger.warning(
                    "Agent Stage 3 gate failed; not emitting proposal",
                    strategy=stats.strategy,
                    regime=stats.regime,
                    baseline_metrics=baseline_metrics,
                    candidate_metrics=candidate_metrics,
                    min_trades=min_trades,
                    max_drawdown_r=max_dd_r,
                    min_expectancy_delta=min_expectancy_delta,
                )
                return []

        # Artifact generation
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
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        return [
            AgentAction(
                timestamp=datetime.now(timezone.utc),
                action_type=ActionType.CODE_PROPOSAL,
                strategy=stats.strategy,
                regime=stats.regime,
                details={
                    "proposal_file": proposal_path,
                    "diff_file": diff_path,
                    "summary_file": summary_path,
                    "scanner_profiles_proposal_file": (
                        profiles_proposal_path if profiles_proposal_path else None
                    ),
                    "scanner_profiles_diff_file": (
                        profiles_diff_path if profiles_diff_path else None
                    ),
                },
                reason="LLM Stage 3 code proposal",
            )
        ]
