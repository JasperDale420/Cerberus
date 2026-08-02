from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results/backtest")


def _make_run_id(config: dict, start_date: str, end_date: str) -> str:
    key = f"{json.dumps(config, sort_keys=True)}:{start_date}:{end_date}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def save_backtest_result(
    report_dict: dict[str, Any],
    config: dict,
    start_date: str,
    end_date: str,
    results_dir: Path | None = None,
) -> str:
    """Save backtest result as JSON. Returns run_id."""
    base = results_dir or RESULTS_DIR
    base.mkdir(parents=True, exist_ok=True)
    run_id = _make_run_id(config, start_date, end_date)

    result = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "start_date": start_date,
        "end_date": end_date,
        **report_dict,
    }

    path = base / f"{run_id}.json"
    path.write_text(json.dumps(result, default=str, indent=2))
    return run_id


def load_backtest_result(run_id: str, results_dir: Path | None = None) -> dict | None:
    """Load a backtest result by run_id. Returns None if not found."""
    base = results_dir or RESULTS_DIR
    path = base / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_backtest_runs(results_dir: Path | None = None) -> list[dict]:
    """List all backtest runs with summary metadata."""
    base = results_dir or RESULTS_DIR
    if not base.exists():
        return []
    runs = []
    for f in sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            runs.append(
                {
                    "run_id": data.get("run_id", f.stem),
                    "timestamp": data.get("timestamp"),
                    "start_date": data.get("start_date"),
                    "end_date": data.get("end_date"),
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return runs
