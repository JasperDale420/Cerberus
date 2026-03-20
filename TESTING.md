# Testing Guide

Cerberus uses `pytest` with unit, integration, contract, e2e, and smoke coverage.

## Quick Commands

```bash
make test
make test-ci
make test-unit
make test-integration
make test-contract
make test-e2e
```

## Test Layout

- `tests/unit/` isolated logic tests
- `tests/integration/` module interaction tests
- `tests/contract/` external boundary contracts
- `tests/e2e/` end-to-end workflows
- `tests/smoke/` lightweight runtime smoke checks
- `tests/backtest/` backtest-specific behavior
- `tests/strategies/` strategy-focused behavior

## Pytest Configuration

See `pyproject.toml`:

- markers: `unit`, `integration`, `contract`, `e2e`
- strict markers enabled
- async mode enabled

## Coverage

- enforced by Makefile targets via pytest-cov
- `coverage.xml` generated for CI/Sonar

## Test Rules

- deterministic tests only
- no real network calls in unit tests
- prefer fixtures and dependency injection
- clock/time should be injectable or fixed in tests

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
