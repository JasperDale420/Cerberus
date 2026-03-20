import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.backtest_api import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.unit
def test_list_runs_empty(client, tmp_path):
    with patch("src.backtest.result_store.RESULTS_DIR", tmp_path):
        resp = client.get("/api/backtest/runs")
        assert resp.status_code == 200
        assert resp.json() == []


@pytest.mark.unit
def test_get_equity_not_found(client):
    with patch("src.api.backtest_api.load_backtest_result", return_value=None):
        resp = client.get("/api/backtest/runs/nonexistent/equity")
        assert resp.status_code == 404


@pytest.mark.unit
def test_get_equity_found(client):
    mock_result = {"equity_curve": [100000, 101000], "benchmark": {"total_return_pct": 5.0}}
    with patch("src.api.backtest_api.load_backtest_result", return_value=mock_result):
        resp = client.get("/api/backtest/runs/abc123/equity")
        assert resp.status_code == 200
        data = resp.json()
        assert "equity_curve" in data
        assert len(data["equity_curve"]) == 2


@pytest.mark.unit
def test_get_trades_not_found(client):
    with patch("src.api.backtest_api.load_backtest_result", return_value=None):
        resp = client.get("/api/backtest/runs/nonexistent/trades")
        assert resp.status_code == 404


@pytest.mark.unit
def test_get_trades_found(client):
    mock_result = {"trades": [{"symbol": "AAPL", "side": "buy", "qty": 100}]}
    with patch("src.api.backtest_api.load_backtest_result", return_value=mock_result):
        resp = client.get("/api/backtest/runs/abc123/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["trades"]) == 1


@pytest.mark.unit
def test_get_monte_carlo(client):
    mock_result = {"monte_carlo": {"prob_loss_pct": 15.0, "n_simulations": 10000}}
    with patch("src.api.backtest_api.load_backtest_result", return_value=mock_result):
        resp = client.get("/api/backtest/runs/abc123/monte-carlo")
        assert resp.status_code == 200
        assert resp.json()["prob_loss_pct"] == 15.0


@pytest.mark.unit
def test_get_monte_carlo_not_found(client):
    with patch("src.api.backtest_api.load_backtest_result", return_value=None):
        resp = client.get("/api/backtest/runs/nonexistent/monte-carlo")
        assert resp.status_code == 404


@pytest.mark.unit
def test_get_regime_splits(client):
    mock_result = {"diagnostics": {"regime_mismatches": [{"regime": "bull", "sharpe": 1.2}]}}
    with patch("src.api.backtest_api.load_backtest_result", return_value=mock_result):
        resp = client.get("/api/backtest/runs/abc123/regime-splits")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["regime"] == "bull"


@pytest.mark.unit
def test_get_sensitivity(client):
    mock_result = {"param_sensitivity": {"stop_loss": {"values": [0.01, 0.02], "sharpe": [1.1, 1.3]}}}
    with patch("src.api.backtest_api.load_backtest_result", return_value=mock_result):
        resp = client.get("/api/wfo/runs/abc123/sensitivity")
        assert resp.status_code == 200
        data = resp.json()
        assert "stop_loss" in data


@pytest.mark.unit
def test_get_sensitivity_not_found(client):
    with patch("src.api.backtest_api.load_backtest_result", return_value=None):
        resp = client.get("/api/wfo/runs/nonexistent/sensitivity")
        assert resp.status_code == 404


@pytest.mark.unit
def test_list_runs_with_data(client, tmp_path):
    # Write a mock result file
    result = {
        "run_id": "test123",
        "timestamp": "2026-03-20T12:00:00",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
    }
    (tmp_path / "test123.json").write_text(json.dumps(result))
    with patch("src.backtest.result_store.RESULTS_DIR", tmp_path):
        resp = client.get("/api/backtest/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "test123"
