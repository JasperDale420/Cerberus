from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.analytics.optuna_harness import build_wfo_run_context


@pytest.mark.unit
def test_build_wfo_run_context_changes_when_signature_changes() -> None:
    generated_at = datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc)
    base_config = {"universe": {"symbols": ["SPY", "QQQ", "AAPL"]}}

    first = build_wfo_run_context(
        strategy_name="trend_rider_pro",
        full_start="2024-01-02",
        full_end="2024-12-31",
        min_train_months=3,
        test_months=1,
        holdout_months=2,
        n_trials=50,
        mode="rolling",
        base_config=base_config,
        data_dir="data/bars_2024",
        generated_at=generated_at,
    )
    second = build_wfo_run_context(
        strategy_name="trend_rider_pro",
        full_start="2020-01-01",
        full_end="2024-12-31",
        min_train_months=3,
        test_months=1,
        holdout_months=2,
        n_trials=50,
        mode="rolling",
        base_config=base_config,
        data_dir="data/bars_5yr",
        generated_at=generated_at,
    )

    assert first.run_tag != second.run_tag
    assert first.storage_uri != second.storage_uri
    assert first.study_name_for_window(0) != second.study_name_for_window(0)


@pytest.mark.unit
def test_build_wfo_run_context_honors_explicit_run_tag() -> None:
    ctx = build_wfo_run_context(
        strategy_name="trend_rider_pro",
        full_start="2020-01-01",
        full_end="2024-12-31",
        min_train_months=3,
        test_months=1,
        holdout_months=2,
        n_trials=50,
        mode="rolling",
        base_config={"universe": {"symbols": ["SPY", "QQQ"]}},
        data_dir="data/bars_5yr",
        run_tag="large-clean-run",
    )

    assert ctx.run_tag == "large-clean-run"
    assert "large-clean-run" in ctx.storage_uri
    assert ctx.study_name_for_window(4) == "trend_rider_pro_large-clean-run_wfo_4"
