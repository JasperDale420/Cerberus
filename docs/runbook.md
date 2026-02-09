# Operational Runbook

This runbook covers common failure scenarios for Cerberus operations.

## Quick Health Check

```bash
python -m src.main --healthcheck
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

Recovery:
1. Stop all Cerberus processes.
2. Backup DB file.
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
pkill -f "python.*src.main"
pkill -f "python.*scheduler"
```

Then verify account/positions in broker UI and run local healthcheck.

## Key Logs and Files

- Main logs: `logs/`
- Runtime config: `config/`
- DB: `cerberus.db`
- Snapshots: `data/screener_snapshots/`

## Useful Commands

```bash
python -m src.main --mode paper --run-once
python -m src.main --eod
python scripts/run_backtest.py --config config/config.yaml --start-date 2024-01-03 --end-date 2024-01-10
```
