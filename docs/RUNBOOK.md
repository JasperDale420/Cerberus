# Operational Runbook

This runbook covers common failure scenarios for Cerberus operations. Cerberus normally runs via Docker Compose (`cerberus-trader` + `cerberus-snapshot`), not as a bare process — check Docker first, not just local processes.

## Check Current Status

Don't trust any doc's claim about whether the system is running — check directly:

```bash
docker ps --filter "name=cerberus"          # is cerberus-trader / cerberus-snapshot up?
docker logs --tail 50 cerberus_trader       # what is it actually doing right now?
docker inspect cerberus_trader --format '{{.State.Health.Status}}'
launchctl list | grep cerberus              # anything loaded outside Docker? (expected: none)
```

## Quick Health Check

```bash
uv run python -m src.main --healthcheck          # local
docker exec cerberus_trader python -m src.main --healthcheck   # inside the running container
```

## Startup Checklist

1. Confirm credentials are present in `.env`.
2. Run healthcheck.
3. Verify config path (`config/config.yaml` unless overridden).
4. Start in paper mode first.

## Common Incidents

### 1) DB locked or unavailable

Symptoms:
- `sqlite3.OperationalError: database is locked`
- Missing persisted fills/trades

Diagnostics:
```bash
lsof cerberus.db
sqlite3 cerberus.db "PRAGMA integrity_check;"
```

Note: in the Docker setup, `cerberus.db`/`ledger.db` live on the named volume `cerberus_state`, and the repo-root `cerberus.db`/`ledger.db` are symlinks into `state_export/` (populated by the `cerberus-snapshot` sidecar, refreshed every 15 min) — do not bind-mount these directly from macOS, it's what caused corruption before (see CHANGELOG).

Recovery:
1. Stop all Cerberus processes (`docker stop cerberus_trader` if running in Docker).
2. Backup the DB file.
3. Ensure a single writer process.
4. Restart and re-run healthcheck.

### 2) API failures/rate pressure

Symptoms:
- frequent 429s or fetch failures

Diagnostics:
```bash
grep "429" logs/cerberus.log | tail -20
```

Recovery:
1. Increase scanner interval in config.
2. Use offline bars for backtests.
3. Confirm API status pages and credentials.

### 3) End-of-day flatten concern

Symptoms:
- open positions near/after close

Diagnostics:
```bash
python -m src.main --healthcheck
grep -i "flatten" logs/cerberus.log | tail -50
```

Recovery:
1. Manually close in Alpaca console if needed.
2. Review error logs around close window.
3. Verify `force_flat_before_close_mins` and related risk settings.

Note: `src/main.py` does not expose a `--flatten-all` CLI flag.

### 4) Scheduler process issues

Diagnostics:
```bash
python -m src.main --scheduler
```

Recovery:
1. Validate config load path.
2. Run scheduler standalone to surface stack traces.
3. If using Docker, use scheduler profile:
```bash
docker compose --profile scheduler up -d cerberus-scheduler
```

## Emergency Stop

```bash
# Stop the Docker trader (the primary way this system runs)
docker stop cerberus_trader

# Stop any local (non-Docker) processes
pkill -f "python.*src.main"
pkill -f "python.*scheduler"
```

Then verify account/positions in broker UI and run a healthcheck.

## Key Logs and Files

- Main logs: `logs/`
- Runtime config: `config/`
- DB: `cerberus.db` (symlink to `state_export/cerberus.db` under Docker)
- Snapshots: `data/screener_snapshots/`

## Useful Commands

```bash
uv run python -m src.main --mode paper --run-once
uv run python -m src.main --eod
uv run python scripts/run_backtest.py --config config/config.yaml --start-date 2024-01-03 --end-date 2024-01-10
```

## Related Docs

- [`DEPLOYMENT.md`](DEPLOYMENT.md) — full Docker/launchd deploy details
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system topology
- [`CONFIGURATION_GUIDE.md`](CONFIGURATION_GUIDE.md) — config/env reference
