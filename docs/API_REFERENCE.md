# API Reference

Cerberus exposes a read-only FastAPI service for backtest and walk-forward analysis outputs.

## Service Basics

- App: `src.api.backtest_api:app`
- Default local URL: `http://localhost:8002`
- CORS allowlist: `http://localhost:5173`
- Authentication: none (local trusted environment)

Start server:
```bash
uvicorn src.api.backtest_api:app --port 8002
```

## Endpoints

### `GET /api/backtest/runs`
Returns all stored backtest runs.

### `GET /api/backtest/runs/{run_id}/equity`
Returns equity curve and benchmark payload.

Response shape:
```json
{
  "equity_curve": [],
  "benchmark": {}
}
```

### `GET /api/backtest/runs/{run_id}/trades`
Returns normalized trade list for a run.

Response shape:
```json
{
  "trades": []
}
```

### `GET /api/backtest/runs/{run_id}/monte-carlo`
Returns Monte Carlo summary for a run.

### `GET /api/backtest/runs/{run_id}/regime-splits`
Returns regime mismatch diagnostics extracted from report diagnostics.

### `GET /api/backtest/runs/{run_id}/autocorrelation`
Returns return autocorrelation diagnostics.

### `GET /api/backtest/runs/{run_id}/factor-attribution`
Returns factor attribution payload.

### `GET /api/backtest/runs/{run_id}/correlation`
Returns strategy correlation and diversification metrics.

### `GET /api/backtest/runs/{run_id}/statistical-tests`
Returns statistical test outputs.

Response shape:
```json
{
  "psr": {},
  "min_backtest_length": {}
}
```

### `GET /api/wfo/runs/{run_id}/sensitivity`
Returns walk-forward parameter sensitivity output.

## Error Behavior

For endpoints with `{run_id}`:
- Returns HTTP `404` with `"Run not found"` if the run is missing.

## Data Source Contract

All endpoints read from persisted run artifacts through:
- `list_backtest_runs()`
- `load_backtest_result(run_id)`

These are implemented in `src/backtest/result_store.py`.
