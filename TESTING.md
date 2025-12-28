# Testing

This repository uses `pytest` with `pytest-asyncio` and `pytest-cov`.

## Quickstart

```bash
python -m pip install -r requirements.txt
make test
```

CI runs `make test-ci`.

## Test Pyramid

- **Unit** (`-m unit`): pure logic with fakes/mocks; no network.
- **Contract** (`-m contract`): boundary tests that validate request shape/paths against a fake transport (no real HTTP).
- **Integration** (`-m integration`): real local dependencies (SQLite, filesystem) with temp directories.
- **E2E/Smoke** (`-m e2e`): small offline workflow across multiple modules (strategy → engine → risk → order → DB).

## Commands

```bash
make test
make test-ci
make test-unit
make test-contract
make test-integration
make test-e2e
```

Notes:
- `make test` / `make test-ci` enforce the coverage threshold (`--cov-fail-under=70`).
- `pytest` alone runs without the coverage gate so you can run focused subsets locally.

## Determinism / No Live Calls

Tests are written to avoid live networks and real broker actions:
- HTTP calls are mocked at the transport layer (e.g., `httpx.MockTransport`).
- Alpaca interactions are replaced with `MagicMock` stubs.

`tests/conftest.py` also sets safe default environment variables early, so a local `.env` does not accidentally introduce real credentials into test runs.

## Coverage Notes

Coverage is enforced at 70% for CI viability. `src/main.py` is omitted from coverage because it is an orchestration entrypoint that performs long-running loops and real external integrations; covering it deterministically would require significant refactoring or heavy fakes.

## Troubleshooting

- If you see `pytest-asyncio` warnings about loop scope, check `pyproject.toml` (`asyncio_default_fixture_loop_scope = "function"`).
- If a test unexpectedly attempts network access, ensure the code path is using fakes/mocks and not `CentralApiClient`’s default `httpx.Client`.
