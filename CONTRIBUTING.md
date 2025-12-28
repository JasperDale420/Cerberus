# Contributing / Development Workflow

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install pre-commit
pre-commit install
```

## Quality (Local)

- Run the full hygiene stack: `pre-commit run --all-files`
- Lint only: `ruff check .`
- Format check: `black --check .`
- Type-check: `mypy .`
- Security: `make security`

## Tests + Coverage

- All tests + coverage gate: `make test`
- CI-equivalent (JUnit + coverage.xml): `make test-ci`
- Marked subsets: `make test-unit`, `make test-contract`, `make test-integration`, `make test-e2e`

`coverage.xml` is written at repo root for Sonar.

## CI Notes

- GitHub Actions runs `pre-commit run --all-files` and `make test-ci`.
- Sonar analysis uses `sonar-project.properties` and consumes `coverage.xml` (and JUnit if configured).

## Secrets Hygiene

- `detect-secrets` is enforced in pre-commit using `.secrets.baseline`.
- If you add a new *non-secret* string that trips the scanner, update the baseline intentionally:
  - `detect-secrets scan --update .secrets.baseline`
