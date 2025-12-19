from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Type, TypedDict, cast

from src.agent.bars_provider import BarsProvider
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
        for sym in symbols:
            bars = self._fetch_bars(sym, start, end)
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
