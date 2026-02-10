# Cerberus + Data-Gateway + Heber Implementation Checklist

## How To Use This
- Execute in order.
- Do not skip phase gates.
- Keep changes small and commit per checklist section.

## Phase 0: Contract and Environment Readiness

### Cerberus
- [x] Add integration env vars to Cerberus runtime settings (`src/core/settings.py` or equivalent env access layer).
- [x] Add startup validation for required variables in gateway/heber modes.
- [x] Add health probes for Gateway and Heber to `src/core/health.py`.

### Data-Gateway
- [ ] Confirm Cerberus client record in `Data-Gateway/config/clients.yaml` includes required providers/feeds.
- [ ] Confirm `gateway/config.py` values for sink and cache are explicit per environment.
- [ ] Confirm `/health` and `/health/ready` are wired in deployment checks.

### Heber
- [ ] Confirm `heber/config.py` stream settings match Gateway stream topic (`heber:events`).
- [ ] Confirm Redis, Postgres, and data root are valid for the target environment.

### Gate
- [ ] One command smoke test script succeeds for:
  - Cerberus -> Gateway authenticated call
  - Gateway -> Redis stream publish
  - Heber consumer -> Bronze/Silver write

## Phase 1: Cerberus Data Adapter Cut-In (No Behavior Change)

### Files To Edit (Cerberus)
- `src/data/api_client.py`
  - [x] Replace non-versioned paths with Data-Gateway routes (`/api/v1/alpaca/...`, `/api/v1/uw/...`).
  - [x] Add `X-Gateway-Key` header support.
  - [ ] Add robust timeout/retry classification for 401/403/429/5xx.
- `src/data/fetcher.py`
  - [x] Introduce backend interface (`legacy` vs `gateway`) with same return shape.
- `src/data/pipeline.py`
  - [x] Route fetch calls through backend interface; preserve feature semantics.
- `src/scanner/universe.py`
  - [x] Route screener calls through adapter instead of direct Alpaca client when enabled.
- `src/main.py`
  - [x] Wire backend selection flag from env/config at composition root.

### Tests (Cerberus)
- [x] Extend `tests/contract/test_central_api_client_contract.py` for actual Data-Gateway route contracts.
- [x] Add parity tests for bars/trades/flow between legacy and gateway modes.

### Gate
- [ ] `CERBERUS_DATA_BACKEND=legacy` behaves unchanged.
- [ ] `CERBERUS_DATA_BACKEND=gateway` passes scanner + feature pipeline integration tests.
- [ ] `CERBERUS_DATA_BACKEND=dual` emits comparable outputs with acceptable delta.

## Phase 2: Data-Gateway Stream Sink Activation

### Files To Edit (Data-Gateway)
- `gateway/main.py`
  - [ ] Ensure sink initialization and dispatch limits are environment-tunable.
  - [ ] Ensure startup logs include active sink mode and stream target.
- `gateway/config.py`
  - [ ] Verify sink config defaults are safe; no accidental enablement in wrong env.
- `gateway/core/data_sink.py`
  - [ ] Confirm backpressure handling/metrics and dedupe behavior are visible in logs/metrics.
- `gateway/core/redis_sink.py`
  - [ ] Confirm publish payload and serialization shape stays stable.

### Tests (Data-Gateway)
- [ ] Add/extend tests for sink publish success/failure and backpressure drops.
- [ ] Validate auth and permission path under load.

### Gate
- [ ] Ingest stream volume stable.
- [ ] Backpressure drops near zero under normal profile.
- [ ] No critical sink publish failure bursts.

## Phase 3: Heber Consumer and Silver Validation

### Files To Edit (Heber)
- `heber/models/envelope.py`
  - [ ] Keep strict compatibility with Data-Gateway envelope fields.
- `heber/writer/consumer.py`
  - [ ] Ensure retry/DLQ behavior is deterministic and logged clearly.
  - [ ] Ensure payload validators for required feeds match current feed schemas.
- `heber/writer/silver.py`
  - [ ] Ensure field mappings for critical feeds (bars, trades, quotes, flow_alerts, market_tide) are current.
- `heber/schemas/silver.py`
  - [ ] Confirm schema versions and required columns for Cerberus consumption.

### Tests (Heber)
- [ ] Extend consumer reliability tests for malformed/partial envelopes.
- [ ] Validate Silver schema alignment for required feeds.

### Gate
- [ ] DLQ remains within threshold.
- [ ] Silver write errors are not sustained.
- [ ] Required datasets are queryable with expected columns and types.

## Phase 4: Cerberus Heber Read Integration (Shadow)

### Files To Edit (Cerberus)
- `src/data/pipeline.py`
  - [ ] Add Heber-backed read mode for historical/replay paths.
- `src/analysis/*` and replay paths
  - [ ] Use point-in-time reads (`ts_available`-aware) for offline evaluation.
- `src/core/health.py`
  - [ ] Add Heber read freshness checks.

### Supporting Files (Heber)
- `heber/sdk/client.py`
  - [ ] Confirm usage pattern for `read_asof` and `asof_join` in Cerberus adapters.

### Tests
- [ ] Add shadow parity suite comparing current vs Heber-backed feature inputs.
- [ ] Add anti-leakage tests that fail on lookahead.

### Gate
- [ ] Shadow outputs stable for multiple sessions.
- [ ] No anti-leakage violations.

## Phase 5: Live Cutover and Cleanup

### Cutover Steps
- [ ] Set `CERBERUS_DATA_BACKEND=gateway`.
- [ ] Set `CERBERUS_STORAGE_BACKEND=heber` (or `dual` during confidence window).
- [ ] Keep `CERBERUS_FAILOVER_TO_LEGACY=true` initially.

### Post-Cutover Cleanup
- [ ] Remove legacy direct data fetch paths not needed for runtime.
- [ ] Keep order execution path untouched unless explicitly migrating it.
- [ ] Update docs:
  - `docs/architecture.md`
  - `docs/runbook.md`
  - `docs/environment-variables.md`

### Gate
- [ ] Stable live operation through agreed soak period.
- [ ] Rollback path tested and documented.

## Suggested Ownership Split
- Cerberus team:
  - adapter integration, feature pipeline switching, cutover toggles, runtime health
- Data-Gateway team:
  - route contract stability, auth/permissions, stream sink reliability
- Heber team:
  - stream consumption robustness, schema quality, as-of read correctness

## Rollback Quick Actions
- [ ] Set Cerberus back to legacy:
  - `CERBERUS_DATA_BACKEND=legacy`
  - `CERBERUS_STORAGE_BACKEND=sqlite`
- [ ] If needed, disable Gateway sink:
  - `GATEWAY_DATA_SINK_ENABLED=false`
- [ ] Keep Heber services up for post-incident replay and diagnostics.
