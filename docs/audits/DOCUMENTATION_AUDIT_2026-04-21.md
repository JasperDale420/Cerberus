# Documentation Audit — 2026-04-21

## Scope

- Repository: Cerberus
- Path: `/Users/jacobmcmillan/.codex/worktrees/6c7d/Cerberus`
- Skill: `updating-documentation`

## Inventory Check

Tier 1 required:

- `README.md` — present
- `CHANGELOG.md` — present
- `AGENTS.md` — added in this run
- `.env.example` — present
- `docs/ARCHITECTURE.md` — canonicalized in this run

Tier 2 required for production service:

- `PRD.md` — present
- `docs/RUNBOOK.md` — canonicalized in this run
- `docs/API_REFERENCE.md` — added in this run

Tier 3 recommended:

- `CONTRIBUTING.md` — present
- `TESTING.md` — present
- `SECURITY.md` — present
- `DEVELOPER_NOTES.md` — present

## Remediation Performed

1. Renamed:
   - `docs/architecture.md` -> `docs/ARCHITECTURE.md`
   - `docs/runbook.md` -> `docs/RUNBOOK.md`
2. Created `AGENTS.md` with project-specific coding guidance.
3. Created `docs/API_REFERENCE.md` for the FastAPI backtest endpoints.
4. Updated internal links in `README.md` to canonical doc names.
5. Updated `CHANGELOG.md` under `[Unreleased]`.

## Notes

- `CLAUDE.md` remains in the repo as a legacy guidance artifact.
- Historical changelog entries reference prior lowercase doc names; those entries were not rewritten.
