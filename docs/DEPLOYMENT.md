# Deployment Guide

How to package, deploy, and operate Cerberus. Focused on local + Docker + macOS launchd, which is how the system runs today.

> **Status (verified 2026-07-24).** The Docker Compose `cerberus-trader` service is up and running in **paper mode** (`ALPACA_PAPER=true`, `--mode paper --order-executor gateway`) — it is not stopped. Its healthcheck flapped between `healthy`/`unhealthy` during this check; that's worth watching but is not itself evidence of live/real-money trading, since the broker URL is the Alpaca paper endpoint. Separately, the macOS launchd agent `com.empire.cerberus.live` has been disabled since 2026-06-05 (`~/Library/LaunchAgents/com.empire.cerberus.live.plist.disabled.20260605-user-request`) and stays off unless the user re-enables it — do not re-enable without explicit instruction. Always confirm current state yourself with the commands in this doc rather than trusting any static claim (including this one) to stay current.

## Deploy Targets

| Target | Process | Used for |
|---|---|---|
| Local Python | `uv run python -m src.main ...` | Dev, ad-hoc backtests, debugging |
| Docker `cerberus-trader` | `docker compose up -d cerberus-trader` | Paper (or live, if ever authorized) trading loop |
| Docker `cerberus-snapshot` | starts alongside `cerberus-trader` | Exports the SQLite state (`ledger.db`, `cerberus.db`) from the Docker named volume to `./state_export` on the host every 15 min, since the DBs live on a Linux-VM-native volume and can't be bind-mounted reliably from macOS |
| Docker `cerberus-scheduler` (profile) | `docker compose --profile scheduler up -d` | APScheduler EOD/cron process — off by default, only starts with `--profile scheduler` |
| macOS launchd | `~/Library/LaunchAgents/com.empire.cerberus.live.plist` | Auto-start on user login — currently disabled |
| FastAPI sidecar | `uvicorn src.api.backtest_api:app --port 8002` | Serves backtest artifacts to EmpireUI |

## Docker

### Build

```bash
docker build -t empire/cerberus:latest .
```

Image is based on Python 3.12 (see `Dockerfile`). It installs deps via `uv`, copies `src/`, and exposes no ports (the trader pushes outbound only).

### Compose services

`docker-compose.yml` defines `cerberus-trader`, `cerberus-snapshot`, and the profile-gated `cerberus-scheduler`.

#### `cerberus-trader`

```yaml
command: python -m src.main --mode paper --order-executor gateway
restart: always
env_file: .env
volumes:
  - ./logs:/app/logs
  - ./config:/app/config:ro
  - cerberus_state:/app/state       # named volume — see note below
  - ./data:/app/data
  - /Volumes/heber/data:/Volumes/heber/data:ro
environment:
  - TZ=America/New_York
  - CERBERUS_GATEWAY_URL=http://host.docker.internal:8080
  - CERBERUS_HEBER_CATALOG_URL=http://host.docker.internal:8085/api/v1
  - CERBERUS_HEBER_DATA_ROOT=/Volumes/heber/data
healthcheck:
  test: ["CMD", "python", "-m", "src.main", "--healthcheck"]
  interval: 5m
  timeout: 60s
  retries: 3
  start_period: 60s
```

Notes:
- `host.docker.internal` connects back to host-side Data-Gateway (port 8080) and Heber catalog (port 8085).
- `config/` is mounted **read-only** so a running container cannot mutate config — the EOD agent's Stage 2 tuning writes to `config/strategies.auto.yaml`, so if that file isn't updating, check whether the mount is actually writable in your setup.
- `cerberus.db` and `ledger.db` live on the **named Docker volume** `cerberus_state`, not a bind mount — macOS's gRPC-FUSE bind-mount locking is unreliable for SQLite and corrupts the file. The `cerberus-snapshot` sidecar exports a periodic snapshot to `./state_export` on the host, which is what `cerberus.db`/`ledger.db` symlink to at the repo root.
- `restart: always` will keep the trader bouncing on crash or host reboot.

#### `cerberus-scheduler` (profile-gated)

```yaml
profiles: [scheduler]
command: python -m src.main --scheduler
restart: unless-stopped
```

Only starts when explicitly enabled:

```bash
docker compose --profile scheduler up -d cerberus-scheduler
```

### Day-to-day

```bash
# Bring trader (+ snapshot sidecar) up
docker compose up -d cerberus-trader

# Check what's actually running right now
docker ps --filter "name=cerberus"

# Tail logs
docker logs -f cerberus_trader

# Healthcheck inside container
docker exec cerberus_trader python -m src.main --healthcheck

# Stop the trader (does not remove it)
docker stop cerberus_trader

# Stop and remove
docker stop cerberus_trader && docker rm cerberus_trader
```

## macOS launchd

Templates live in `scripts/`:

- `scripts/com.cerberus.main.paper.plist` — paper-mode auto-start
- `scripts/com.cerberus.paper.plist` — alternate paper launcher

Install pattern (only when explicitly re-enabling):

```bash
cp scripts/com.cerberus.main.paper.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.cerberus.main.paper.plist
launchctl list | grep cerberus
```

Check current launchd state:

```bash
ls ~/Library/LaunchAgents/com.empire.cerberus*
# a filename containing "disabled" means it is NOT loaded
launchctl list | grep cerberus
# no output means nothing is currently loaded under launchd
```

## Local Process

For ad-hoc runs:

```bash
uv sync

# Validate environment
uv run python -m src.main --healthcheck

# Noop simulation (safest — no orders)
uv run python -m src.main --mode paper --order-executor noop --run-once

# Backtest API for EmpireUI
uv run uvicorn src.api.backtest_api:app --port 8002

# Backtest
uv run python scripts/run_backtest.py --config config/backtest_smoke.yaml \
    --start-date 2024-01-03 --end-date 2024-01-10

# End-of-day analytics + agent
uv run python -m src.main --eod
uv run python -m src.main --eod --eod-date 2026-03-19
```

## Pre-Flight Before Enabling/Re-Enabling Live

If you (the user) authorize switching to real-money `--mode live`, the operator checklist is:

1. Confirm Data-Gateway is up: `curl -s http://localhost:8080/health`.
2. Confirm Heber catalog is up (if `CERBERUS_HEBER_CATALOG_URL` is set): `curl -s http://localhost:8085/health`.
3. Confirm `.env` is current, and double-check `ALPACA_PAPER` is set the way you intend — it defaults to `True`.
4. Run `uv run python -m src.main --healthcheck`.
5. Run a noop pass: `uv run python -m src.main --mode paper --order-executor noop --run-once`.
6. Inspect `cerberus.db` and `ledger.db` for stale open positions: `sqlite3 cerberus.db "SELECT * FROM positions WHERE qty != 0;"`.
7. Decide whether to reset `daily_*` counters in `RiskManager` (handled automatically on session-date rollover).
8. Bring up the trader: `docker compose up -d cerberus-trader`.
9. Tail logs: `docker logs -f cerberus_trader` for the first several minutes.
10. Verify positions in the Alpaca UI match `cerberus.db`.

## Logs & Files on Disk

| Path | Owner |
|---|---|
| `logs/cerberus_YYYY-MM-DD.log` | Daily all-levels JSON log |
| `logs/cerberus_errors_YYYY-MM-DD.log` | WARNING+ only |
| `cerberus.db` | Symlink into `state_export/` — primary SQLite analytics DB |
| `ledger.db` | Symlink into `state_export/` — trade audit ledger (large) |
| `cerberus_backup_*.db`, `ledger_backup_*.db` | Manual point-in-time backups |
| `artifacts/` | WFO + backtest outputs (JSON) |
| `data/` | Snapshots, replay bars |
| `debug/`, `fix/`, `predict/`, `reason/` | Autoresearch / experiment scratch |

## Emergency Stop

```bash
# Stop the Docker trader (the primary way this system runs)
docker stop cerberus_trader

# Stop any local (non-Docker) processes
pkill -f "python.*src.main"
pkill -f "python.*scheduler"

# Then verify in broker UI and run a local healthcheck
uv run python -m src.main --healthcheck
```

See [`RUNBOOK.md`](RUNBOOK.md) for per-incident recovery steps.

## CI

`.github/workflows/` defines GitHub Actions for `ruff`, `mypy`, `pytest`, `bandit`, `detect-secrets`. The 68% coverage gate runs on every PR.

## Related Docs

- [`RUNBOOK.md`](RUNBOOK.md) — incident playbook
- [`CONFIGURATION_GUIDE.md`](CONFIGURATION_GUIDE.md) — env + YAML matrix
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — topology diagram
