#!/usr/bin/env bash
# Migrate Cerberus SQLite DBs off the macOS bind mount onto the cerberus_state
# named volume, and publish host-readable snapshots via symlink.
#
# Run once, from a machine with Docker Desktop, when no trading session is active
# (i.e. after the market close). Idempotent-ish: it backs up rather than destroys.
#
#   ./scripts/migrate_ledger_to_volume.sh
#
# What it does:
#   1. Builds the image (so the snapshot sidecar + DB pragmas are baked in).
#   2. Stops the trader/scheduler/snapshot containers.
#   3. Seeds the named volume from the current host DBs (WAL checkpointed), chowns it.
#   4. Seeds ./state_export and repoints Cerberus/{ledger,cerberus}.db as symlinks
#      into it, so host readers (dashboard, Athena, Heber, ledger_audit) keep working.
#   5. Brings the stack back up and shows the first snapshot.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
DBS=(ledger.db cerberus.db)

echo "==> Repo: $REPO"
command -v docker >/dev/null || { echo "docker not found"; exit 1; }

echo "==> 1/5 Building image..."
docker compose build cerberus-snapshot

echo "==> 2/5 Stopping writers (ignore 'no such service')..."
docker compose stop cerberus-trader cerberus-scheduler cerberus-snapshot 2>/dev/null || true

echo "==> 3/5 Seeding named volume from host DBs (WAL checkpointed) + chown..."
mkdir -p "$REPO/state_export"
# One throwaway root container: copy DBs into the volume, fold any WAL into the main
# file, drop the sidecar -wal/-shm, then hand ownership to the runtime user.
docker compose run --rm --user root -v "$REPO:/host:ro" cerberus-snapshot sh -c '
  set -e
  for db in ledger.db cerberus.db; do
    if [ -f "/host/$db" ]; then
      echo "   seeding $db"
      cp "/host/$db" "/app/state/$db"
      [ -f "/host/$db-wal" ] && cp "/host/$db-wal" "/app/state/$db-wal" || true
      [ -f "/host/$db-shm" ] && cp "/host/$db-shm" "/app/state/$db-shm" || true
      # Set WAL once here with exclusive access (persists in the DB header), then fold
      # any copied -wal frames into the main file. At runtime the app only needs
      # busy_timeout, which switching WAL at connect-time cannot guarantee under pooling.
      python -c "import sqlite3; c=sqlite3.connect(\"/app/state/$db\"); c.execute(\"PRAGMA journal_mode=WAL\"); c.execute(\"PRAGMA wal_checkpoint(TRUNCATE)\"); c.close()"
    else
      echo "   $db not on host — skipping (fresh DB will be created on start)"
    fi
  done
  chown -R appuser:appgroup /app/state
'

echo "==> 4/5 Seeding ./state_export and repointing host paths as symlinks..."
for db in "${DBS[@]}"; do
  if [ -f "$REPO/$db" ] && [ ! -L "$REPO/$db" ]; then
    cp "$REPO/$db" "$REPO/state_export/$db"
    mv "$REPO/$db" "$REPO/${db}.premigration.$(date +%Y%m%d%H%M%S)"
    rm -f "$REPO/${db}-wal" "$REPO/${db}-shm"
    echo "   backed up host $db"
  fi
  # (Re)create the symlink readers follow.
  ln -sfn "state_export/$db" "$REPO/$db"
  echo "   $db -> state_export/$db"
done

echo "==> 5/5 Starting stack..."
docker compose up -d cerberus-trader cerberus-snapshot

echo "==> Waiting for first snapshot..."
sleep 5
docker compose logs --tail 20 cerberus-snapshot || true

echo
echo "Done. Verify with:  docker compose logs -f cerberus-snapshot"
echo "Host readers now follow: $REPO/ledger.db -> state_export/ledger.db"
