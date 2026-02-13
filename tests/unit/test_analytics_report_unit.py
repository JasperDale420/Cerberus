from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.analysis.analytics import AnalyticsEngine
from src.analysis.schema import Trade as DbTrade
from src.core.logger import StructuredLogger


@dataclass
class _DbStub:
    config: Dict[str, Any]

    def get_session(self):  # pragma: no cover
        raise NotImplementedError


@pytest.mark.unit
def test_analytics_write_daily_report_includes_breakdowns(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    logger = StructuredLogger("test_daily_report", level="INFO")
    db = _DbStub(config={"timezone": "UTC"})
    engine = AnalyticsEngine(db, logger)  # type: ignore

    rows = []
    base = datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc)
    for i in range(10):
        rows.append(
            {
                "strategy": "S1" if i < 7 else "S2",
                "regime": "bull" if i % 2 == 0 else "bear",
                "pnl": float(i - 3),
                "pnl_r": float((i - 5) / 2),
                "mae_r": float(i % 3),
                "mfe_r": float((i % 4) + 0.5),
                "commission": 0.1,
                "slippage": 0.05,
                "exit_time": base,
                "scanner_score": float(i + 1),
                "flow_zscore": float((i + 1) / 2),
                "win": 1 if (i - 3) > 0 else 0,
            }
        )
        base = base.replace(hour=(14 + i) % 24)

    df = pd.DataFrame(rows)
    out_path = engine._write_daily_report(date(2025, 1, 1), df)
    assert out_path.exists()
    text = out_path.read_text()
    assert "# Cerberus Daily Report – 2025-01-01" in text
    assert "## Strategy/Regime" in text
    assert "## PnL By Exit Hour" in text
    assert "## Scanner Score Deciles" in text
    assert "## Flow Strength Deciles" in text


@pytest.mark.unit
def test_daily_aggregation_computes_drawdown_losers_and_z_score(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    db = MagicMock()
    db.config = {"timezone": "UTC"}
    logger = StructuredLogger("test_daily_agg_metrics", level="INFO")
    mock_session = MagicMock()

    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = None
    db.get_session.return_value = mock_cm

    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    trades = [
        DbTrade(
            strategy="StratA",
            regime_at_entry="bull",
            pnl_net=10.0,
            pnl_r=1.0,
            exit_time=now,
        ),
        DbTrade(
            strategy="StratA",
            regime_at_entry="bull",
            pnl_net=10.0,
            pnl_r=1.0,
            exit_time=now,
        ),
        DbTrade(
            strategy="StratA",
            regime_at_entry="bull",
            pnl_net=-10.0,
            pnl_r=-1.0,
            exit_time=now,
        ),
    ]
    mock_session.execute.return_value.scalars.return_value.all.return_value = trades

    engine = AnalyticsEngine(db, logger)  # type: ignore
    engine.run_daily_aggregation(date(2025, 1, 1))

    added_objs = [call[0][0] for call in mock_session.add.call_args_list]
    assert len(added_objs) == 1
    stats = added_objs[0]
    assert stats.strategy == "StratA"
    assert stats.max_drawdown_r == pytest.approx(1.0)
    assert stats.max_consecutive_losers == 1
    assert stats.z_score > 0
