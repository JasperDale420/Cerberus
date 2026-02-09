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
