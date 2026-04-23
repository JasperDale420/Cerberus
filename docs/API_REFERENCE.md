# API Reference

Cerberus exposes a FastAPI service for backtest and walk-forward analytics artifacts.

## Service

- App: `src.api.backtest_api:app`
- Default local URL: `http://localhost:8002`
- CORS allowlist: `http://localhost:5173`

Start locally:

```bash
uv run uvicorn src.api.backtest_api:app --port 8002
```

## Endpoints

### `GET /api/backtest/runs`

Returns available backtest run metadata from result storage.

Response shape:

```json
[
  {
    "run_id": "string",
    "created_at": "string",
    "strategy": "string"
  }
]
```

### `GET /api/backtest/runs/{run_id}/equity`

Returns equity curve and benchmark data.

Response shape:

```json
{
  "equity_curve": [],
  "benchmark": {}
}
```

### `GET /api/backtest/runs/{run_id}/trades`

Returns trade list for a run.

Response shape:

```json
{
  "trades": []
}
```

### `GET /api/backtest/runs/{run_id}/monte-carlo`

Returns Monte Carlo output (if available).

### `GET /api/backtest/runs/{run_id}/regime-splits`

Returns regime mismatch diagnostics.

### `GET /api/backtest/runs/{run_id}/autocorrelation`

Returns return autocorrelation analysis.

### `GET /api/backtest/runs/{run_id}/factor-attribution`

Returns factor attribution analysis.

### `GET /api/backtest/runs/{run_id}/correlation`

Returns strategy correlation analysis.

### `GET /api/backtest/runs/{run_id}/statistical-tests`

Returns statistical test metrics:

- `psr`
- `min_backtest_length`

### `GET /api/wfo/runs/{run_id}/sensitivity`

Returns walk-forward parameter sensitivity output.

## Error Behavior

All run-specific endpoints return HTTP 404 (`Run not found`) when `run_id` does not exist.

## Storage Contract

This API reads from the backtest result store via:

- `src.backtest.result_store.list_backtest_runs`
- `src.backtest.result_store.load_backtest_result`

The API does not execute backtests; it serves already materialized result artifacts.
