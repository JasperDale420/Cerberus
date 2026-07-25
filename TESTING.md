# Testing Guide

Cerberus uses `pytest` with strict markers and a 68% line-coverage gate (`--cov-fail-under=68`). The suite is organized by test type and by component. Backtesting and walk-forward optimization (WFO) are first-class verification flows in addition to unit/integration testing.

## Quick Commands

```bash
# Full default suite
uv run pytest

# By marker
uv run pytest -m unit          # fast, isolated
uv run pytest -m integration   # real DB / file I/O / cross-component
uv run pytest -m contract      # boundary tests for external APIs
uv run pytest -m e2e           # full system flow
uv run pytest -m slow          # opt-in; tests >1s

# Single file / single test
uv run pytest tests/test_risk.py
uv run pytest -k "test_signal_rejected_on_daily_loss"

# With coverage gate (CI equivalent)
uv run pytest --cov=src --cov-fail-under=68

# Make targets
make test / test-ci / test-unit / test-integration / test-contract / test-e2e
make test-hmm        # HMM sidecar tests only
```

## Marker Definitions

Defined in `pyproject.toml` (`[tool.pytest.ini_options].markers`):

| Marker | Meaning |
|---|---|
| `unit` | Fast, isolated. No I/O, no network. |
| `integration` | Real DB (SQLite tmp), file I/O, or component interactions. |
| `contract` | Boundary tests against external API shapes (Alpaca, UW, Gateway). |
| `e2e` | Full system flow — `--run-once`, scheduler, backtest. |
| `slow` | Anything >1s — opt-in, excluded from default runs. |

`--strict-markers` is on, so a typo in a marker name fails collection.

## Test Layout

```
tests/
├── conftest.py                # Safe defaults: ALPACA_PAPER=True, fake creds, fake gateway URL
├── unit/                      # pure unit tests
├── integration/
├── e2e/
├── smoke/                     # lightweight startup checks
├── strategies/                # per-strategy unit tests
├── data/                      # data-pipeline fixtures
├── test_*.py                  # per-module suites at top level
├── benchmark_performance.py
└── repro.py                   # bug-repro harness
```

### Conftest Defaults

`tests/conftest.py` sets these env vars before any module is imported, so production credentials are never required to run tests:

| Var | Test value |
|---|---|
| `ALPACA_API_KEY` | `test` |
| `ALPACA_SECRET_KEY` | `test` |
| `ALPACA_PAPER` | `True` |
| `DATA_INGESTION_URL` | `http://central.test` |

## Backtest & Analytics Test Suites

| Test File | Covers |
|---|---|
| `tests/unit/test_fill_models.py` | FillModel protocol, fixed/volume-aware models, factory |
| `tests/unit/test_overnight_handling.py` | Overnight config defaults, per-strategy flatten logic |
| `tests/unit/test_data_quality.py` | Gap detection, zero volume, outliers, coverage, staleness |
| `tests/unit/test_benchmark.py` | Alpha, beta, capture ratios, return percentages |
| `tests/unit/test_monte_carlo.py` | Bootstrap simulation, confidence intervals, edge cases |
| `tests/unit/test_param_sensitivity.py` | Spearman ranking, few trials, constant params |
| `tests/unit/test_diagnostics.py` | Strategy ranking, regime mismatch, time edge, hold analysis |
| `tests/unit/test_wfo_holdout.py` | HoldoutResult structure, pass/fail logic |
| `tests/unit/test_result_store.py` | Save/load round-trip, listing, deterministic IDs |
| `tests/unit/test_backtest_api.py` | Backtest API endpoints (happy path + 404) |

## Backtesting as Verification

Backtests are run from the CLI and produce JSON artifacts in `artifacts/`. They are **not** part of `pytest` — they're validation runs you launch manually before promoting a config change.

```bash
# Quick smoke (defined in config/backtest_smoke.yaml)
uv run python scripts/run_backtest.py --config config/backtest_smoke.yaml \
    --start-date 2024-01-03 --end-date 2024-01-10

# Deterministic offline replay (uses local JSONL bars)
uv run python scripts/run_backtest.py --config config/config.yaml \
    --start-date 2024-01-03 --end-date 2024-01-10 \
    --offline-bars-dir data/replay_bars
```

Results land under `artifacts/` and are served by the FastAPI service in `src/api/backtest_api.py`. See [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md).

## Walk-Forward Optimization

Optuna-driven WFO is the standard pre-deployment check for any strategy parameter change.

```bash
uv run python scripts/run_wfo.py                       # default WFO config
uv run python scripts/run_wfo_robust.py                 # robust scoring
uv run python scripts/run_holdout.py                    # holdout-only
uv run python scripts/run_oos_validation.py             # OOS validation
uv run python scripts/run_param_sweep.py                # raw parameter sweep
```

Post-process with `scripts/analyze_wfo_results.py`, `scripts/extract_wfo_insights.py`, `scripts/wfo_dashboard.py` (interactive dashboard).

## Pre-Commit & Quality Gates

`.pre-commit-config.yaml` runs on every commit: `ruff check` + `ruff format`, `detect-secrets`.

CI equivalent:

```bash
make lint
make type-check
make security
uv run pytest --cov=src --cov-fail-under=68
```

## Coverage

```bash
uv run pytest --cov=src --cov-report=term-missing
uv run pytest --cov=src --cov-report=html   # opens htmlcov/index.html
```

Coverage config (`pyproject.toml` `[tool.coverage.*]`):
- `branch = true`, `source = ["src"]`
- `omit = ["tests/*", "scripts/*", "*/__init__.py"]`
- Excluded patterns: `pragma: no cover`, `def __repr__`, `if __name__ == "__main__":`, `raise NotImplementedError`, `if TYPE_CHECKING:`

`coverage.xml` is also generated for CI/SonarQube.

## Async Tests

`asyncio_mode = "auto"` — any `async def test_...` is auto-detected. No `@pytest.mark.asyncio` decorator needed.

## Filter Warnings

`filterwarnings = ["ignore::DeprecationWarning"]` is set globally. Strategy code that legitimately needs to surface a deprecation should use `warnings.warn(..., FutureWarning)` instead.

## Test Rules

- Deterministic tests only — no `time.sleep()`, no real network calls in unit tests
- Prefer fixtures and dependency injection
- Clock/time should be injectable or fixed in tests

## Backtest & Analytics Test Suites

| Test File | Tests | Covers |
|---|---|---|
| `tests/unit/test_fill_models.py` | 13 | FillModel protocol, fixed/volume-aware models, factory |
| `tests/unit/test_overnight_handling.py` | 6 | Overnight config defaults, per-strategy flatten logic |
| `tests/unit/test_data_quality.py` | 5 | Gap detection, zero volume, outliers, coverage, staleness |
| `tests/unit/test_benchmark.py` | 4 | Alpha, beta, capture ratios, return percentages |
| `tests/unit/test_monte_carlo.py` | 5 | Bootstrap simulation, confidence intervals, edge cases |
| `tests/unit/test_param_sensitivity.py` | 4 | Spearman ranking, few trials, constant params |
| `tests/unit/test_diagnostics.py` | 5 | Strategy ranking, regime mismatch, time edge, hold analysis |
| `tests/unit/test_wfo_holdout.py` | 2 | HoldoutResult structure, pass/fail logic |
| `tests/unit/test_result_store.py` | 4 | Save/load round-trip, listing, deterministic IDs |
| `tests/unit/test_backtest_api.py` | 11 | All 6 API endpoints (happy path + 404) |

Run all backtest tests:
```bash
uv run pytest tests/unit/test_fill_models.py tests/unit/test_overnight_handling.py tests/unit/test_data_quality.py tests/unit/test_benchmark.py tests/unit/test_monte_carlo.py tests/unit/test_param_sensitivity.py tests/unit/test_diagnostics.py tests/unit/test_wfo_holdout.py tests/unit/test_result_store.py tests/unit/test_backtest_api.py -v
```
