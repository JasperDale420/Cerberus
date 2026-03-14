from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, cast

from src.agent.bars_provider import BarsProvider, JsonlBarsProvider
from src.agent.models import ActionType, AgentAction, StrategyDailyStats
from src.analytics.optimizer import GridSearchOptimizer
from src.analytics.walk_forward import WalkForwardManager
from src.core.config import ConfigLoader
from src.core.domain import Bar, MarketState, Regime, RiskMode, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


@dataclass(frozen=True)
class Stage2Metrics:
    expectancy: float
    max_drawdown_r: float
    n_trades: int
    profit_factor: float = 0.0
    win_rate: float = 0.0


class _PendingEntry(TypedDict):
    side: str
    stop: float
    target: float
    risk_per_share: float
    max_hold_seconds: int | None


class _OpenTrade(_PendingEntry):
    entry: float
    entry_time: datetime


class DeterministicStage2Evaluator:
    """
    PRD 9.2: deterministic parameter tuning via offline backtests/replay.

    This evaluator performs a minimal deterministic bar-replay simulation:
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
    ):
        self.config = config
        self.config_loader = config_loader
        self.logger = logger
        self.clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))

        self.bars_provider = bars_provider
        self.alpaca: Any = None  # Legacy: only used when no bars_provider is set

    def _fetch_bars(self, symbol: str, start: datetime, end: datetime) -> List[Bar]:
        if self.bars_provider is not None:
            return list(self.bars_provider.get_bars(symbol, start, end, timeframe="1Min"))
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

    def _bars_window(self, as_of: datetime) -> tuple[datetime, datetime]:
        agent_cfg = (self.config.get("agent") or {}) if isinstance(self.config, dict) else {}
        stage2 = (agent_cfg.get("stage2") or {}) if isinstance(agent_cfg, dict) else {}
        window_days = int(stage2.get("window_days", 30))
        end = as_of
        start = end - timedelta(days=max(1, window_days))
        return start, end

    def _symbols(self) -> List[str]:
        stage2 = (
            ((self.config.get("agent") or {}).get("stage2") or {}) if isinstance(self.config.get("agent"), dict) else {}
        )
        symbols = stage2.get("symbols") if isinstance(stage2, dict) else None
        if isinstance(symbols, list) and symbols:
            return sorted({str(s).upper() for s in symbols if s})
        # Fall back to configured universe symbols for determinism.
        uni = self.config.get("universe") if isinstance(self.config, dict) else None
        uni_symbols = uni.get("symbols") if isinstance(uni, dict) else None
        if isinstance(uni_symbols, list):
            return sorted({str(s).upper() for s in uni_symbols if s})
        return []

    def _strategy_instance(self, strategy_name: str, params: Dict[str, Any]) -> BaseStrategy:
        # Minimal deterministic mapping for existing strategies.
        from src.strategies.failed_breakout import FailedBreakoutStrategy
        from src.strategies.flow_momentum import FlowMomentumStrategy
        from src.strategies.gap_fill import GapFillStrategy
        from src.strategies.index_mean_reversion import IndexMeanReversionStrategy
        from src.strategies.momentum_continuation import MomentumContinuationStrategy
        from src.strategies.orb import ORBStrategy
        from src.strategies.trend_pullback import TrendPullbackStrategy
        from src.strategies.vix_spike_fade import VixSpikeFadeStrategy
        from src.strategies.vwap_reversion import VWAPReversionStrategy
        from src.strategies.vwap_trend_rider import VWAPTrendRiderStrategy

        mapping = {
            "vwap_reversion": VWAPReversionStrategy,
            "orb": ORBStrategy,
            "trend_pullback": TrendPullbackStrategy,
            "failed_breakout": FailedBreakoutStrategy,
            "vwap_trend_rider": VWAPTrendRiderStrategy,
            "index_mean_reversion": IndexMeanReversionStrategy,
            "flow_momentum": FlowMomentumStrategy,
            "gap_fill": GapFillStrategy,
            "vix_spike_fade": VixSpikeFadeStrategy,
            "momentum_continuation": MomentumContinuationStrategy,
        }
        cls = mapping.get(strategy_name)
        if cls is None:
            raise ValueError(f"Unknown strategy for Stage 2: {strategy_name}")
        cls_any = cast(Any, cls)
        inst = cls_any(params, self.logger)
        return cast(BaseStrategy, inst)

    def evaluate(
        self,
        strategy_name: str,
        regime: Regime,
        params: Dict[str, Any],
        as_of: Optional[datetime] = None,
        start_override: Optional[datetime] = None,
        end_override: Optional[datetime] = None,
    ) -> Stage2Metrics:
        now = as_of or self.clock()
        if start_override and end_override:
            start, end = start_override, end_override
        else:
            start, end = self._bars_window(now)
        symbols = self._symbols()
        if not symbols:
            self.logger.warning("Stage 2 evaluator has no symbols; returning empty metrics")
            return Stage2Metrics(expectancy=0.0, max_drawdown_r=0.0, n_trades=0)

        strat = self._strategy_instance(strategy_name, params)

        pnl_r_series: List[float] = []

        # Pre-fetch bars for all symbols in parallel
        from concurrent.futures import ThreadPoolExecutor

        bars_by_symbol: Dict[str, List[Bar]] = {}
        max_workers = min(10, len(symbols))
        if max_workers > 0:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._fetch_bars, sym, start, end): sym for sym in symbols}
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
                bars=__import__("collections").deque(maxlen=500),  # Safe dynamic import for deque
                indicators={},
                position=None,
                open_orders={},
                allowed_strategies=[strategy_name],
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

                # Fill entry at next bar open deterministically.
                if pending_entry is not None and open_trade is None:
                    entry = float(bar.open)
                    open_trade = cast(
                        _OpenTrade,
                        {**pending_entry, "entry": entry, "entry_time": bar.time},
                    )
                    pending_entry = None

                # Manage open trade exits using intrabar extremes (stop prioritized).
                if open_trade is not None:
                    side = open_trade["side"]  # "buy" or "sell"
                    stop = float(open_trade["stop"])
                    target = float(open_trade["target"])
                    entry = float(open_trade["entry"])
                    rps = float(open_trade["risk_per_share"])

                    if rps <= 0:
                        open_trade = None
                    else:
                        stop_hit = (bar.low <= stop) if side == "buy" else (bar.high >= stop)
                        target_hit = (bar.high >= target) if side == "buy" else (bar.low <= target)

                        exit_px = None
                        if stop_hit:
                            exit_px = stop
                        elif target_hit:
                            exit_px = target

                        # Deterministic max-hold exit if configured.
                        max_hold_sec = open_trade.get("max_hold_seconds")
                        if exit_px is None and isinstance(max_hold_sec, int) and max_hold_sec > 0:
                            dt = (bar.time - open_trade["entry_time"]).total_seconds()
                            if dt >= max_hold_sec:
                                exit_px = float(bar.close)

                        if exit_px is not None:
                            pnl_r = ((exit_px - entry) / rps) if side == "buy" else ((entry - exit_px) / rps)
                            pnl_r_series.append(float(pnl_r))
                            open_trade = None

                # Append bar and compute signal at close.
                state.bars.append(bar)
                if open_trade is None and pending_entry is None:
                    sig = strat.on_bar(sym, bar, state, market)
                    if sig is None:
                        continue

                    risk_per_share = abs(sig.entry_price - sig.stop_price)
                    if risk_per_share <= 0:
                        continue

                    # Defer entry fill to next bar open.
                    pending_entry = {
                        "side": sig.side.value,
                        "stop": float(sig.stop_price),
                        "target": float(sig.target_price),
                        "risk_per_share": float(risk_per_share),
                        "max_hold_seconds": (
                            int(float(params.get("max_hold_minutes", 0)) * 60)
                            if params.get("max_hold_minutes") is not None
                            else None
                        ),
                    }

        n = len(pnl_r_series)
        expectancy = float(sum(pnl_r_series) / n) if n > 0 else 0.0

        # Profit factor: sum of wins / abs(sum of losses)
        wins = [r for r in pnl_r_series if r > 0]
        losses = [r for r in pnl_r_series if r < 0]
        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = min(gross_win / gross_loss, 999.0) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

        # Win rate
        win_rate = float(len(wins) / n) if n > 0 else 0.0

        # Max drawdown on cumulative R.
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in pnl_r_series:
            equity += float(r)
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)

        return Stage2Metrics(
            expectancy=expectancy,
            max_drawdown_r=float(max_dd),
            n_trades=int(n),
            profit_factor=float(profit_factor),
            win_rate=float(win_rate),
        )


class Stage2Tuner:
    """
    Orchestrates deterministic parameter tuning.
    """

    def __init__(
        self,
        logger: StructuredLogger,
        config_loader: ConfigLoader,
        config_path_or_dir: Optional[str] = None,
        evaluator: Optional[DeterministicStage2Evaluator] = None,
    ):
        self.logger = logger
        self.config_loader = config_loader
        self.config_path_or_dir = config_path_or_dir
        self.evaluator = evaluator
        self.optimizer = GridSearchOptimizer(logger)
        self.wf_manager = WalkForwardManager(logger)

    def tune_parameters(
        self,
        stats: StrategyDailyStats,
        current_config: Dict[str, Any],
        min_trades: int,
        max_dd_r: float,
        *,
        as_of: Optional[datetime] = None,
    ) -> List[AgentAction]:
        """
        Stage 2: deterministic parameter tuning via offline evaluation.
        """
        cfg = self.config_loader.load_config(self.config_path_or_dir)
        agent_cfg = (cfg.get("agent") or {}) if isinstance(cfg, dict) else {}
        stage2 = (agent_cfg.get("stage2") or {}) if isinstance(agent_cfg, dict) else {}
        enabled = bool(stage2.get("enabled", False))
        if not enabled:
            return []

        # PRD 9.2: parameter tuning must be offline/deterministic.
        offline_dir = str(stage2.get("offline_bars_dir", "")).strip()
        if self.evaluator is None and not offline_dir:
            self.logger.error(
                "Agent Stage 2 enabled but no offline bars source configured",
                required_key="agent.stage2.offline_bars_dir",
            )
            raise ValueError("Agent Stage 2 requires agent.stage2.offline_bars_dir for offline determinism")

        now = as_of or datetime.now(timezone.utc)

        raw_search_space = stage2.get("search_space") if isinstance(stage2, dict) else None
        search_space: Dict[str, Any] = dict(raw_search_space) if isinstance(raw_search_space, dict) else {}
        strat_space = search_space.get(stats.strategy, {})
        if not strat_space:
            return []

        evaluator = self.evaluator
        if evaluator is None:
            evaluator = DeterministicStage2Evaluator(
                cfg if isinstance(cfg, dict) else {},
                self.config_loader,
                self.logger,
                clock=lambda: now,
                bars_provider=JsonlBarsProvider(Path(offline_dir)),
            )

        def _get_regime_enum() -> Regime:
            r = str(stats.regime or "").strip().lower()
            if r == "bull":
                return Regime.BULL
            if r == "bear":
                return Regime.BEAR
            return Regime.CHOP

        regime_enum = _get_regime_enum()

        def _evaluate_params(params: Dict[str, Any]) -> Dict[str, Any]:
            merged = {**current_config, **params}
            if callable(evaluator) and not hasattr(evaluator, "evaluate"):
                m_raw = evaluator(stats, merged)
                m_dict = cast(Dict[str, Any], m_raw)
                return {
                    "expectancy": float(m_dict.get("expectancy", 0.0)),
                    "max_drawdown_r": float(m_dict.get("max_drawdown_r", 0.0)),
                    "n_trades": int(m_dict.get("n_trades", 0)),
                }

            m_metrics = evaluator.evaluate(stats.strategy, regime_enum, merged, as_of=now)
            return {
                "expectancy": float(m_metrics.expectancy),
                "max_drawdown_r": float(m_metrics.max_drawdown_r),
                "n_trades": int(m_metrics.n_trades),
                "profit_factor": float(m_metrics.profit_factor),
                "win_rate": float(m_metrics.win_rate),
            }

        def _score_fn(metrics: Dict[str, Any]) -> float:
            pf = float(metrics.get("profit_factor", 0.0))
            n = int(metrics.get("n_trades", 0))
            dd = float(metrics.get("max_drawdown_r", 0.0))
            expectancy = float(metrics.get("expectancy", 0.0))
            # Composite: PF × √trades × (1 - drawdown_penalty) + expectancy tiebreaker
            # DD penalty caps at 1.0 (100% penalty) when DD reaches 10R
            dd_penalty = 1.0 - min(dd / 10.0, 1.0)
            return pf * (n**0.5) * dd_penalty + expectancy

        # Baseline Evaluation
        baseline_metrics = _evaluate_params({})
        baseline_expectancy = float(baseline_metrics["expectancy"])
        baseline_dd = float(baseline_metrics["max_drawdown_r"])

        if int(baseline_metrics["n_trades"]) < int(min_trades):
            self.logger.info(
                "Insufficient baseline trades",
                strategy=stats.strategy,
                n=baseline_metrics["n_trades"],
            )
            return []

        # Parameter Search
        best_params, best_metrics = self.optimizer.optimize(
            evaluate_fn=_evaluate_params,
            search_space=strat_space,
            min_trades=min_trades,
            score_fn=_score_fn,
        )

        if not best_params:
            return []

        # Baseline checks: skip if no improvement
        if float(best_metrics["expectancy"]) <= baseline_expectancy:
            return []
        if float(best_metrics["max_drawdown_r"]) > baseline_dd and float(best_metrics["max_drawdown_r"]) > float(
            max_dd_r
        ):
            # Only allow DD increase if it's within global limits
            return []

        # optional Walk-Forward Validation
        wf_cfg = stage2.get("walk_forward", {})
        if wf_cfg.get("enabled", False):
            step = int(wf_cfg.get("step_days", 7))
            windows = self.wf_manager.get_windows(now, int(stage2.get("window_days", 30)), step)

            wf_results = []
            for start, end in windows:

                def _wf_eval(params_inner: Dict[str, Any], s=start, e=end) -> Dict[str, Any]:
                    merged = {**current_config, **params_inner}
                    m = evaluator.evaluate(
                        stats.strategy,
                        regime_enum,
                        merged,
                        start_override=s,
                        end_override=e,
                    )
                    return {"expectancy": m.expectancy, "n_trades": m.n_trades}

                wf_results.append(_wf_eval(best_params))

            if not self.wf_manager.evaluate_stability(wf_results):
                self.logger.warning(
                    "Optimization rejected due to walk-forward instability",
                    strategy=stats.strategy,
                    best_params=best_params,
                )
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
                    "walk_forward": wf_cfg.get("enabled", False),
                },
                reason=f"Deterministic Stage 2 tuning (Improved expectancy from {baseline_expectancy:.3f} to {best_metrics['expectancy']:.3f})",
            )
        ]
