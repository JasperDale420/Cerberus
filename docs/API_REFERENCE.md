# API Reference

Cerberus exposes a single read-only FastAPI service that serves backtest and walk-forward optimization (WFO) artifacts to the EmpireUI dashboard. This is the only HTTP surface — the trading loop itself does not listen on any port.

## Service

- App: `src.api.backtest_api:app`
- Default local URL: `http://localhost:8002`
- CORS allowlist: `http://localhost:5173`
- Data source: `src.backtest.result_store.list_backtest_runs`, `src.backtest.result_store.load_backtest_result`

Start locally:

```bash
uv run uvicorn src.api.backtest_api:app --port 8002
```

The service is **read-only** — it does not execute backtests, only serves materialized JSON artifacts written under `artifacts/` (gitignored).

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

Returns Monte Carlo output (if `analytics.monte_carlo.enabled: true` was set for the run).

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

- `psr` — Probabilistic Sharpe Ratio
- `min_backtest_length` — minimum bars needed for significance

### `GET /api/wfo/runs/{run_id}/sensitivity`

Returns walk-forward parameter sensitivity output (Spearman rank correlation of trial params vs objective).

## Error Behavior

All run-specific endpoints return HTTP 404 (`Run not found`) when `run_id` does not exist.

## Storage Contract

| Concept | Where |
|---|---|
| Artifact directory | `artifacts/` (gitignored) |
| Result loader | `src.backtest.result_store.load_backtest_result` |
| Run lister | `src.backtest.result_store.list_backtest_runs` |
| JSON schema | Stable across writers — both `scripts/run_backtest.py` and `scripts/run_wfo.py` emit the same envelope |

The API does not execute backtests; it serves already materialized result artifacts.

## Consumers

- **EmpireUI** — primary consumer; renders equity curves, trade tables, regime-split charts.
- **Athena** — post-trade LLM analyst; reads the same JSON to ground its narrative reports.

## Notes

The API is intended for local/LAN use — there is no auth layer. Do not expose it to the public internet without a reverse proxy and auth in front of it.

## Related Docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — service topology
- [`TESTING.md`](../TESTING.md) — how to run backtests and WFO
- [`CODEBASE_SUMMARY.md`](CODEBASE_SUMMARY.md) — `src/api/` and `src/backtest/` index
