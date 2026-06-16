from __future__ import annotations

from pathlib import Path

from scripts import wfo_dashboard


def test_discover_wfo_runs_returns_deterministic_sorted_runs(tmp_path: Path) -> None:
    legacy_json = tmp_path / "orb_v2_wfo_results.json"
    legacy_json.write_text("{}")
    (tmp_path / "orb_v2_study.db").write_text("")

    trp_dir = tmp_path / "runs" / "trend_rider_pro" / "20260306T120000Z-3sym-25tr-abcd1234"
    trp_dir.mkdir(parents=True)
    (trp_dir / "wfo_results.json").write_text("{}")
    (trp_dir / "study.db").write_text("")

    mrp_dir = tmp_path / "runs" / "mean_reversion_pro" / "20260306T130000Z-3sym-25tr-zzzz9999"
    mrp_dir.mkdir(parents=True)
    (mrp_dir / "wfo_results.json").write_text("{}")
    (mrp_dir / "study.db").write_text("")

    runs = wfo_dashboard.discover_wfo_runs(tmp_path)

    assert [(run["strategy_name"], run["run_tag"]) for run in runs] == [
        ("mean_reversion_pro", "20260306T130000Z-3sym-25tr-zzzz9999"),
        ("orb_v2", "legacy"),
        ("trend_rider_pro", "20260306T120000Z-3sym-25tr-abcd1234"),
    ]


def test_build_summary_payload_keeps_run_identity_and_metrics() -> None:
    payload = wfo_dashboard.build_summary_payload(
        {
            "trend_rider_pro::20260306T120000Z-3sym-25tr-abcd1234": {
                "strategy_name": "trend_rider_pro",
                "run_tag": "20260306T120000Z-3sym-25tr-abcd1234",
                "display_label": "Trend Rider Pro [20260306T120000Z-3sym-25tr-abcd1234]",
                "windows": [{"window_idx": 0}, {"window_idx": 1}],
                "wfo_result": {
                    "windows": [{}, {}],
                    "wfo_efficiency_ratio": 1.21,
                    "deflated_sharpe_ratio": 0.97,
                    "oos_metrics": [{"net_pnl": 100.0}, {"net_pnl": -50.0}],
                    "param_stability": {
                        "stop_atr_mult": {"stable": True},
                        "max_hold_minutes": {"stable": False},
                    },
                },
                "artifact_dir": "tmp/wfo/trend_rider_pro/run-alpha",
                "result_path": "tmp/wfo/trend_rider_pro/run-alpha/wfo_results.json",
            }
        }
    )

    assert payload["runs"][0]["run_tag"] == "20260306T120000Z-3sym-25tr-abcd1234"
    assert payload["runs"][0]["status"] == "complete"
    assert payload["runs"][0]["window_count"] == 2
    assert payload["runs"][0]["profitable_oos_windows"] == 1
    assert payload["runs"][0]["stable_params"] == ["stop_atr_mult"]
    assert payload["runs"][0]["unstable_params"] == ["max_hold_minutes"]


def test_ensure_labeled_result_copy_backfills_finished_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "trend_rider_pro" / "20260306T120000Z-3sym-25tr-abcd1234"
    run_dir.mkdir(parents=True)
    run = {
        "strategy_name": "trend_rider_pro",
        "run_tag": "20260306T120000Z-3sym-25tr-abcd1234",
        "json_path": run_dir / "wfo_results.json",
    }

    labeled_path = wfo_dashboard.ensure_labeled_result_copy(run, {"strategy": "trend_rider_pro"})

    assert labeled_path == run_dir / "trend_rider_pro_20260306T120000Z-3sym-25tr-abcd1234_wfo_results.json"
    assert labeled_path.read_text() == '{\n  "strategy": "trend_rider_pro"\n}'
