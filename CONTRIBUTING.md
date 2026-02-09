# Contributing

## Local Setup

```bash
python -m pip install -r requirements.txt
python -m pip install pre-commit
pre-commit install
```

## Development Workflow

1. Create a branch from `main`.
2. Implement changes with tests.
3. Run quality gates.
4. Update `CHANGELOG.md`.
5. Open PR.

## Quality Gates

```bash
make lint
make type-check
make test
make security
pre-commit run --all-files
```

## Testing Targets

```bash
make test-unit
make test-integration
make test-contract
make test-e2e
```

## Documentation Requirements

- Keep `README.md` and affected docs in sync with code changes.
- Keep `.env.example` aligned with runtime env variables.
- Add notable changes to `CHANGELOG.md`.

## Commit Guidance

- Use clear commit messages with scope prefix, e.g. `docs: ...`, `fix: ...`, `feat: ...`.
- Avoid committing secrets, credentials, or `.env` files.
