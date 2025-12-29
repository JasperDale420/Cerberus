from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, cast

from src.agent.bars_provider import BarsProvider, JsonlBarsProvider
from src.agent.models import ActionType, AgentAction, StrategyDailyStats
from src.core.config import ConfigLoader
from src.core.domain import Bar, MarketState, Regime, RiskMode, SymbolState
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.strategies.base import BaseStrategy


@dataclass(frozen=True)
class Stage2Metrics:
    expectancy: float
    max_drawdown_r: float
    n_trades: int


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
        self.clock: Callable[[], datetime] = clock or (
            lambda: datetime.now(timezone.utc)
        )

        self.bars_provider = bars_provider
        self.alpaca = (
            AlpacaClient(config_loader, logger) if bars_provider is None else None
        )

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

    def _bars_window(self, as_of: datetime) -> tuple[datetime, datetime]:
        agent_cfg = (
            (self.config.get("agent") or {}) if isinstance(self.config, dict) else {}
        )
        stage2 = (agent_cfg.get("stage2") or {}) if isinstance(agent_cfg, dict) else {}
        window_days = int(stage2.get("window_days", 30))
        end = as_of
        start = end - timedelta(days=max(1, window_days))
        return start, end

    def _symbols(self) -> List[str]:
        stage2 = (
            ((self.config.get("agent") or {}).get("stage2") or {})
            if isinstance(self.config.get("agent"), dict)
            else {}
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

    def _strategy_instance(
        self, strategy_name: str, params: Dict[str, Any]
    ) -> BaseStrategy:
        # Minimal deterministic mapping for existing strategies.
        from src.strategies.failed_breakout import FailedBreakoutStrategy
        from src.strategies.flow_momentum import FlowMomentumStrategy
        from src.strategies.gap_fill import GapFillStrategy
        from src.strategies.index_mean_reversion import IndexMeanReversionStrategy
        from src.strategies.orb import ORBStrategy
        from src.strategies.trend_pullback import TrendPullbackStrategy
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
    ) -> Stage2Metrics:
        now = as_of or self.clock()
        start, end = self._bars_window(now)
        symbols = self._symbols()
        if not symbols:
            self.logger.warning(
                "Stage 2 evaluator has no symbols; returning empty metrics"
            )
            return Stage2Metrics(expectancy=0.0, max_drawdown_r=0.0, n_trades=0)

        strat = self._strategy_instance(strategy_name, params)

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
                bars=__import__("collections").deque(
                    maxlen=500
                ),  # Safe dynamic import for deque
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

                        # Deterministic max-hold exit if configured.
                        max_hold_sec = open_trade.get("max_hold_seconds")
                        if (
                            exit_px is None
                            and isinstance(max_hold_sec, int)
                            and max_hold_sec > 0
                        ):
                            dt = (bar.time - open_trade["entry_time"]).total_seconds()
                            if dt >= max_hold_sec:
                                exit_px = float(bar.close)

                        if exit_px is not None:
                            pnl_r = (
                                ((exit_px - entry) / rps)
                                if side == "buy"
                                else ((entry - exit_px) / rps)
                            )
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

        # Max drawdown on cumulative R.
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in pnl_r_series:
            equity += float(r)
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)

        return Stage2Metrics(
            expectancy=expectancy, max_drawdown_r=float(max_dd), n_trades=int(n)
        )


class Stage2Tuner:
    """
    Orchestrates deterministic parameter tuning.
    """

    def __init__(
        self,
        logger: StructuredLogger,
        config_loader: ConfigLoader,
        config_path_or_dir: str,
        evaluator: Optional[Any] = None,
    ):
        self.logger = logger
        self.config_loader = config_loader
        self.config_path_or_dir = config_path_or_dir
        self.evaluator = evaluator

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
            raise ValueError(
                "Agent Stage 2 requires agent.stage2.offline_bars_dir for offline determinism"
            )

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

        evaluator = self.evaluator
        if evaluator is None:

            def _clock(now: datetime = now) -> datetime:
                return now

            evaluator = DeterministicStage2Evaluator(
                cfg if isinstance(cfg, dict) else {},
                self.config_loader,
                self.logger,
                clock=_clock,
                bars_provider=JsonlBarsProvider(Path(offline_dir)),
            )

        def _regime() -> Any:
            r = str(stats.regime or "").strip().lower()
            from src.core.domain import Regime as RegimeEnum

            if r == "bull":
                return RegimeEnum.BULL
            if r == "bear":
                return RegimeEnum.BEAR
            return RegimeEnum.CHOP

        # Evaluate baseline
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

        if baseline_n < int(min_trades):
            self.logger.info(
                "Agent Stage 2: insufficient baseline sample size",
                strategy=stats.strategy,
                regime=stats.regime,
                n_trades=baseline_n,
                min_trades=min_trades,
            )
            return []

        # Build candidates
        keys = sorted(strat_space.keys())
        values_list = []
        for k in keys:
            v = strat_space.get(k)
            if isinstance(v, list):
                values_list.append([x for x in v])
            else:
                values_list.append([v])

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
