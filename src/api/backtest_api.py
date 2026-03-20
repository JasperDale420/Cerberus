from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.backtest.result_store import list_backtest_runs, load_backtest_result

app = FastAPI(title="Cerberus Backtest API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/backtest/runs")
def get_runs():
    return list_backtest_runs()


@app.get("/api/backtest/runs/{run_id}/equity")
def get_equity(run_id: str):
    result = load_backtest_result(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    return {"equity_curve": result.get("equity_curve", []), "benchmark": result.get("benchmark")}


@app.get("/api/backtest/runs/{run_id}/trades")
def get_trades(run_id: str):
    result = load_backtest_result(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    return {"trades": result.get("trades", [])}


@app.get("/api/backtest/runs/{run_id}/monte-carlo")
def get_monte_carlo(run_id: str):
    result = load_backtest_result(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    return result.get("monte_carlo", {})


@app.get("/api/backtest/runs/{run_id}/regime-splits")
def get_regime_splits(run_id: str):
    result = load_backtest_result(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    return result.get("diagnostics", {}).get("regime_mismatches", [])


@app.get("/api/wfo/runs/{run_id}/sensitivity")
def get_sensitivity(run_id: str):
    result = load_backtest_result(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    return result.get("param_sensitivity", {})
