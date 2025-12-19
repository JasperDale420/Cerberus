from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from src.agent.core import Agent
from src.analysis.analytics import AnalyticsEngine
from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Order as DbOrder
from src.analysis.schema import Signal as DbSignal
from src.analysis.schema import StrategyStatsDaily
from src.analysis.schema import Trade as DbTrade
from src.core.config import ConfigLoader
from src.core.domain import (
    Bar,
    OrderSide,
    Regime,
    Signal,
    SymbolFeatures,
    SymbolState,
    WatchlistSymbol,
)
from src.core.logger import StructuredLogger
from src.engine.execution import ExecutionEngine
from src.engine.orders import NoopOrderExecutor
from src.strategies.base import BaseStrategy


class _TwoTradeStrategy(BaseStrategy):
    name = "e2e_strategy"

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state,
    ) -> Signal | None:
        fired = int(symbol_state.meta.get("fired", 0) or 0)
        if fired >= 2:
            return None
        if symbol_state.position is not None:
            return None
        if len(symbol_state.bars) < 1:
            return None

        symbol_state.meta["fired"] = fired + 1
        entry = float(bar.close)
        stop = entry - 1.0
        target = entry + 2.0
        return Signal(
            symbol=symbol,
            side=OrderSide.BUY,
            size_hint=0.0,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            strategy=self.name,
            regime=market_state.regime,
            generated_at=bar.time,
            meta={"source": "prd-e2e"},
        )


def _write_yaml(path: Path, obj: dict) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=True))


@pytest.mark.e2e
def test_prd_success_metric_vertical_slice_offline(tmp_path: Path) -> None:
    """
    PRD 1.2: Validate the success-metric vertical slice end-to-end, offline + deterministic:
    scanner → execution engine → risk → orders → logs → analytics → agent config adjustment.
    """

    db_path = tmp_path / "slice.db"

    # Config suite (ConfigLoader loads config.yaml, strategies.yaml, risk.yaml, scanner.yaml,
    # universe.yaml, logging.yaml, plus agent override strategies.auto.yaml).
    _write_yaml(
        tmp_path / "config.yaml",
        {
            "database_url": f"sqlite:///{db_path}",
            "timezone": "US/Eastern",
            "index_symbol": "SPY",
            "log_level": "INFO",
            "agent": {
                "stage1": {
                    "window_days": 30,
                    "min_trades": 2,
                    "z_high": 1.0,
                    "max_drawdown_r": 10.0,
                }
            },
        },
    )
    _write_yaml(
        tmp_path / "risk.yaml",
        {
            "risk": {
                "max_daily_loss": 1_000_000.0,
                "max_risk_per_trade": 50.0,
                "max_open_risk": 1_000_000.0,
                "max_trades_per_day": 1000,
                "max_notional_per_order": 1_000_000.0,
                "max_notional_per_symbol": 1_000_000.0,
                "risk_mode": "normal",
            }
        },
    )
    _write_yaml(tmp_path / "scanner.yaml", {"scanner": {"enabled": False}})
    _write_yaml(tmp_path / "universe.yaml", {"universe": {"symbols": ["AAPL"]}})
    _write_yaml(tmp_path / "logging.yaml", {"logging": {"console_level": "INFO"}})
    _write_yaml(
        tmp_path / "strategies.yaml",
        {"strategies": {"e2e_strategy": {"enabled": True}}},
    )

    logger = StructuredLogger("PRD-SLICE", level="INFO")
    loader = ConfigLoader(config_dir=str(tmp_path), logger=logger)
    cfg = loader.load_config(str(tmp_path / "config.yaml"))

    db = DatabaseDatabase(
        loader,
        logger,
        config=cfg,
        config_path_or_dir=str(tmp_path / "config.yaml"),
    )
    db.init_db()

    fixed_now = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)

    def clock():
        return fixed_now

    engine = ExecutionEngine(
        config=cfg,
        logger=logger,
        db=db,
        alpaca_client=None,
        clock=clock,
    )
    engine.order_executor = NoopOrderExecutor(logger, db=db, clock=clock)  # type: ignore
    engine.market_state.regime = Regime.CHOP
    engine.register_strategy(_TwoTradeStrategy({}, logger))

    features = SymbolFeatures(
        symbol="AAPL",
        price=100.0,
        atr_pct=0.01,
        avg_volume=1_000_000.0,
        intraday_range_pct=0.01,
        gap_pct=0.0,
        ema20_slope=0.0,
        ema_trend_strength=0.0,
        distance_from_vwap=0.0,
        premarket_volume=0.0,
        adx=25.0,
        distance_from_ema20=0.0,
        prior_day_high=101.0,
        prior_day_low=99.0,
        bb_upper=102.0,
        bb_lower=98.0,
        price_zscore=0.0,
        flow_zscore=0.0,
        call_put_ratio=1.0,
        large_sweeps_count=0,
        aggressive_flow_share=0.0,
        last_updated=fixed_now,
        extra={},
    )
    wl = [
        WatchlistSymbol(
            symbol="AAPL",
            score=1.0,
            strategies=["e2e_strategy"],
            features=features,
        )
    ]
    engine.apply_scan_result(wl)

    # Generate signals and orders, then inject fills to close two trades.
    t0 = fixed_now
    for i, exit_price in enumerate([99.0, 99.5]):
        bar = Bar(
            symbol="AAPL",
            time=t0 + timedelta(minutes=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000.0,
        )
        engine.on_bar("AAPL", bar)

        state = engine.symbol_states["AAPL"]
        pending = state.meta.get("pending_entries")
        assert isinstance(pending, dict) and pending

        corr_id = sorted(pending.keys())[0]
        qty = float(pending[corr_id]["qty"])

        engine.on_fill(
            {
                "symbol": "AAPL",
                "side": "buy",
                "qty": qty,
                "price": 100.0,
                "timestamp": bar.time + timedelta(seconds=1),
                "strategy": "e2e_strategy",
                "correlation_id": corr_id,
            }
        )
        engine.on_fill(
            {
                "symbol": "AAPL",
                "side": "sell",
                "qty": qty,
                "price": float(exit_price),
                "timestamp": bar.time + timedelta(seconds=2),
                "strategy": "e2e_strategy",
                "correlation_id": corr_id,
            }
        )

    # Verify DB artifacts: signals/orders/trades exist.
    with db.get_session() as session:
        assert session.query(DbSignal).count() >= 2
        assert session.query(DbOrder).count() >= 2
        assert session.query(DbTrade).count() == 2

    analytics = AnalyticsEngine(db, logger)
    analytics.run_daily_aggregation(target_date=fixed_now.date())
    with db.get_session() as session:
        assert session.query(StrategyStatsDaily).count() >= 1

    agent = Agent(logger, loader, config_path_or_dir=str(tmp_path / "config.yaml"))
    agent.run_cycle_with_db(db, as_of=fixed_now)

    auto_path = tmp_path / "strategies.auto.yaml"
    assert auto_path.exists()
    auto_cfg = yaml.safe_load(auto_path.read_text())
    assert isinstance(auto_cfg, dict)
    assert "e2e_strategy" in auto_cfg
    assert auto_cfg["e2e_strategy"]["regimes"]["chop"]["max_risk_per_trade"] == 25.0

    merged = loader.load_config(str(tmp_path / "config.yaml"))

    from src.engine.risk import RiskManager

    rm = RiskManager(merged, logger)
    ss = SymbolState("AAPL", deque(maxlen=10), {}, None, {}, [], {})
    sig = Signal(
        symbol="AAPL",
        side=OrderSide.BUY,
        size_hint=0.0,
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        strategy="e2e_strategy",
        regime=Regime.CHOP,
        generated_at=fixed_now,
        meta={},
    )
    intents = rm.apply(sig, ss, engine.market_state, current_positions=[])
    assert intents
    assert intents[0].qty == 25
