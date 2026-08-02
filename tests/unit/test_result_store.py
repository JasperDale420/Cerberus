from datetime import datetime, timezone

import pytest

from src.backtest.result_store import list_backtest_runs, load_backtest_result, save_backtest_result


@pytest.mark.unit
def test_saved_timestamp_is_timezone_aware_utc(tmp_path):
    run_id = save_backtest_result({}, {"s": "a"}, "2024-01-01", "2024-06-01", results_dir=tmp_path)
    loaded = load_backtest_result(run_id, results_dir=tmp_path)
    assert loaded is not None
    ts = datetime.fromisoformat(loaded["timestamp"])
    assert ts.tzinfo is not None, "timestamp must be timezone-aware"
    assert ts.utcoffset() == timezone.utc.utcoffset(None)


@pytest.mark.unit
def test_save_and_load(tmp_path):
    report = {"metrics": {"sharpe": 1.5}, "equity_curve": [100000, 101000]}
    run_id = save_backtest_result(report, {"strategy": "orb"}, "2024-01-01", "2024-06-01", results_dir=tmp_path)
    loaded = load_backtest_result(run_id, results_dir=tmp_path)
    assert loaded is not None
    assert loaded["metrics"]["sharpe"] == 1.5
    assert loaded["run_id"] == run_id


@pytest.mark.unit
def test_load_nonexistent(tmp_path):
    assert load_backtest_result("nonexistent", results_dir=tmp_path) is None


@pytest.mark.unit
def test_list_runs(tmp_path):
    save_backtest_result({"metrics": {}}, {"s": "a"}, "2024-01-01", "2024-03-01", results_dir=tmp_path)
    save_backtest_result({"metrics": {}}, {"s": "b"}, "2024-04-01", "2024-06-01", results_dir=tmp_path)
    runs = list_backtest_runs(results_dir=tmp_path)
    assert len(runs) == 2


@pytest.mark.unit
def test_deterministic_run_id(tmp_path):
    id1 = save_backtest_result({}, {"x": 1}, "2024-01-01", "2024-06-01", results_dir=tmp_path)
    id2 = save_backtest_result({}, {"x": 1}, "2024-01-01", "2024-06-01", results_dir=tmp_path)
    assert id1 == id2
