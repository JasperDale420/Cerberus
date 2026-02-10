# Cerberus + Data-Gateway + Heber Migration Roadmap

## Objective
Migrate Cerberus from direct provider + local-only data assumptions to:
1) Data-Gateway for all market/flow ingest paths.
2) Heber for canonical, point-in-time-safe storage reads.

This roadmap is phased with explicit entry/exit criteria and rollback at each phase.

## Proposed Feature Flags and Env Toggles
Add these in Cerberus config/runtime settings before code migration:
- `CERBERUS_DATA_BACKEND=legacy|gateway|dual`
- `CERBERUS_STORAGE_BACKEND=sqlite|heber|dual`
- `CERBERUS_GATEWAY_URL=http://...`
- `CERBERUS_GATEWAY_KEY=...`
- `CERBERUS_HEBER_CATALOG_URL=http://...`
- `CERBERUS_HEBER_DATA_ROOT=/path/...` (if local filesystem reads)
- `CERBERUS_DUAL_READ_COMPARE=true|false`
- `CERBERUS_FAILOVER_TO_LEGACY=true|false`

Data-Gateway/Heber side toggles already present:
- `GATEWAY_DATA_SINK_ENABLED=true|false`
- `GATEWAY_DATA_SINK_REDIS_URL=...`
- `HEBER_REDIS_STREAM_NAME=heber:events`
- `HEBER_REDIS_DLQ_STREAM_NAME=heber:events:dlq`

## Phase 0: Pre-Migration Hardening

### Scope
- Freeze contract for minimum required feeds/routes.
- Confirm Redis stream wiring and environment parity.

### Actions
- Verify Data-Gateway routes and auth with Cerberus key from `Data-Gateway/config/clients.yaml`.
- Verify Heber consumer can parse current Data-Gateway envelope (`heber/models/envelope.py`).
- Define feed list required by Cerberus runtime:
  - bars, trades, quotes
  - flow_alerts, gex
- Capture baseline Cerberus performance/reliability metrics on legacy path.

### Exit Criteria
- Signed interface mapping doc (route + payload + auth headers).
- Smoke test passes for all required routes from Cerberus host.
- Heber consumer healthy with no unexpected DLQ growth for test traffic.

### Rollback
- None needed (no production path changes yet).

## Phase 1: Gateway Client Integration in Cerberus (Read Path Only)

### Scope
- Introduce Data-Gateway-backed adapters while keeping current direct providers available.

### Actions
- Refactor Cerberus data access seams:
  - `src/data/api_client.py` -> align to `/api/v1/alpaca/...` and `/api/v1/uw/...` routes.
  - Add auth header support (`X-Gateway-Key`).
- Wire `FeaturePipeline`/`DataFetcher` to use adapter based on `CERBERUS_DATA_BACKEND`:
  - `legacy`: existing `AlpacaClient` + `UnusualWhalesClient`
  - `gateway`: Data-Gateway client only
  - `dual`: both, with diff logging
- Keep execution/order submission unchanged.

### Exit Criteria
- Unit and integration tests pass in `gateway` mode.
- In `dual` mode, data parity error rate under agreed threshold (for example <1%).
- No increase in scanner failure rate or runtime exceptions.

### Rollback
- Set `CERBERUS_DATA_BACKEND=legacy`.
- Disable dual compare if noisy.

## Phase 2: Stream-to-Heber Activation (Write Path)

### Scope
- Turn on canonical stream publication from Data-Gateway to Heber.

### Actions
- Enable Data-Gateway sink:
  - `GATEWAY_DATA_SINK_ENABLED=true`
  - valid `GATEWAY_DATA_SINK_REDIS_URL`
- Confirm stream topic alignment (`heber:events`).
- Start/verify Heber consumer group and write loop:
  - Bronze and Silver writes healthy.
- Set alerts for:
  - sink publish failures/backpressure drops
  - Heber write errors and DLQ growth

### Exit Criteria
- Steady-state ingest observed for target feeds.
- Silver files are queryable and schema-valid for required feeds.
- No sustained DLQ accumulation under normal load.

### Rollback
- Disable sink (`GATEWAY_DATA_SINK_ENABLED=false`) to stop fanout.
- Continue Cerberus reads from `legacy` or `gateway` REST without Heber dependency.

## Phase 3: Heber Read Introduction (Shadow)

### Scope
- Start reading selected datasets from Heber in shadow mode while still trading on primary path.

### Actions
- Add Heber read adapter in Cerberus for replay/analytics paths first.
- Use `read_asof` for all historical/decision-time reads in shadow jobs.
- Compare:
  - feature outputs
  - signal counts
  - decision deltas
- Start with non-order-impact paths (offline analysis, reports, replay).

### Exit Criteria
- Shadow output stable and reproducible for multiple trading days.
- No anti-leakage violations detected.
- Acceptable latency for required query windows.

### Rollback
- Set `CERBERUS_STORAGE_BACKEND=sqlite`.
- Keep Heber pipeline active but detached from decision path.

## Phase 4: Cutover for Data Reads

### Scope
- Move live Cerberus read path fully to Data-Gateway + Heber where applicable.

### Actions
- Set:
  - `CERBERUS_DATA_BACKEND=gateway`
  - `CERBERUS_STORAGE_BACKEND=heber|dual`
- Keep `CERBERUS_FAILOVER_TO_LEGACY=true` for first cutover window.
- Monitor live deltas and error budgets tightly.

### Exit Criteria
- Full trading session(s) without critical incidents.
- Data freshness and fill/signal quality within baseline tolerance.
- Ops runbook validated by on-call/operator.

### Rollback
- Immediate switch:
  - `CERBERUS_DATA_BACKEND=legacy`
  - `CERBERUS_STORAGE_BACKEND=sqlite`
- Keep ingestion pipelines running for postmortem replay.

## Phase 5: Decommission Legacy Read Paths

### Scope
- Remove direct provider data dependencies from Cerberus runtime path.

### Actions
- Remove/restrict usage of direct client fetches in:
  - `src/data/alpaca.py`
  - `src/data/unusual_whales.py`
  - `src/data/fetcher.py`
- Keep only trading/order APIs that still belong in Cerberus.
- Update docs and runbooks.

### Exit Criteria
- No runtime codepaths depend on direct provider market-data fetches.
- Operational docs updated and validated.

### Rollback
- Short-term rollback remains env flag-based only if legacy code retained during grace period.

## Test Strategy by Phase

### Cerberus
- Existing tests to extend:
  - `tests/contract/test_central_api_client_contract.py`
  - `tests/integration/test_vertical_slice_3_scanner_exec.py`
  - `tests/integration/test_engine_persists_regime_and_scanner_snapshots.py`
  - `tests/unit/test_execution_engine_trade_persistence_unit.py`
- Add gateway-mode contract tests for exact route/payload mapping.
- Add dual-read diff tests (legacy vs gateway) for bars/trades/flow/gex.

### Data-Gateway
- Validate health/auth/metrics:
  - `tests/test_auth.py`
  - `tests/test_metrics.py`
  - `tests/test_live_provider_smoke.py`
- Add sink integration tests around backpressure and dispatch failures.

### Heber
- Validate consumer/schema/reliability:
  - `tests/test_writer_consumer_reliability.py`
  - `tests/test_silver_model_schema_alignment.py`
  - `tests/test_stream_naming_conventions.py`
  - `tests/test_silver_flush_config.py`

### End-to-End
- Synthetic replay day with known expected outputs:
  - Gateway ingest -> stream -> Heber Silver -> Cerberus read -> signal generation.
- Pass criteria:
  - no critical errors
  - bounded parity deltas
  - no anti-leakage violations

## Operational Guardrails
- Define hard stop criteria for cutover day:
  - sustained 5xx from Data-Gateway
  - Heber DLQ surge above threshold
  - stale data age above threshold
- Keep rollback commands in one runbook with single env toggle changes.
- Keep legacy mode deployable until at least one week of stable cutover operation.
