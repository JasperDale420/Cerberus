# Security Policy

## Reporting

Report vulnerabilities privately to: `security@empire-trading.com`.
Do not file public issues for security vulnerabilities.

Include:

- affected files and commit/branch
- reproduction steps
- impact assessment
- proof-of-concept if available

## Scope Priorities

Cerberus is safety-critical trading software. Highest priority issues:

1. unintended order execution
2. risk guard bypasses
3. credential leakage
4. data integrity corruption (positions/fills/trades)
5. API abuse/rate-limit cascades

## Security Practices

- credentials must come from environment variables
- do not commit `.env` or secrets
- run local security checks:

```bash
make security
```

- use paper mode/noop paths for validation before live deployment

## Dependency and Supply Chain

- dependencies are pinned in `requirements.txt`
- CI runs pre-commit and test gates
- keep dependency updates reviewed and tested before merge

## Operational Hardening

- run healthcheck before market session:
```bash
python -m src.main --healthcheck
```
- monitor logs for repeated API/auth failures
- verify kill/flatten behaviors in paper mode before production cutover
