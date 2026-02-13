# Gateway + Heber Soak Report

- Run date (UTC): `2026-02-12T03:46:23.858861+00:00`
- Status: **PASS**
- Duration seconds: `90`
- Poll interval seconds: `15`
- Gateway URL: `http://localhost:8080`
- Redis URL: `redis://localhost:6379/0`
- Stream: `heber:events`
- DLQ stream: `heber:events:dlq`

## Summary Metrics

- Poll count: `7`
- Gateway readiness failures: `0`
- Sink publish delta: `0.0`
- Stream length delta: `6`
- DLQ growth: `0`
- Bronze fresh seen: `True`
- Silver fresh seen: `True`
- Poll error snapshots: `0`

## Verdict

- No blocking issues observed in this soak window.

## Poll Timeline

| UTC Timestamp | Gateway Ready | Sink Ready | Sink Counter | Stream Len | DLQ Len | Bronze Fresh | Silver Fresh | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-02-12T03:44:51.659475+00:00 | yes | yes | 1.0 | 50735 | 1784 | yes | yes |  |
| 2026-02-12T03:45:07.076564+00:00 | yes | yes | 1.0 | 50736 | 1784 | yes | yes |  |
| 2026-02-12T03:45:22.429968+00:00 | yes | yes | 1.0 | 50737 | 1784 | yes | yes |  |
| 2026-02-12T03:45:37.752461+00:00 | yes | yes | 1.0 | 50738 | 1784 | yes | yes |  |
| 2026-02-12T03:45:53.080542+00:00 | yes | yes | 1.0 | 50739 | 1784 | yes | yes |  |
| 2026-02-12T03:46:08.495702+00:00 | yes | yes | 1.0 | 50740 | 1784 | yes | yes |  |
| 2026-02-12T03:46:23.858580+00:00 | yes | yes | 1.0 | 50741 | 1784 | yes | yes |  |
