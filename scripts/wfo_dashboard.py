"""WFO Performance Dashboard — Interactive HTML report.

Reads from Optuna study databases and WFO result JSONs to produce
a self-contained Plotly HTML dashboard comparing all strategies.

Usage::

    # Generate dashboard (reads from artifacts/optimization/)
    PYTHONPATH=. python scripts/wfo_dashboard.py

    # Auto-open in browser
    PYTHONPATH=. python scripts/wfo_dashboard.py --open
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    go = None
    make_subplots = None
    PLOTLY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data extraction from Optuna study DBs
# ---------------------------------------------------------------------------

STRATEGIES = ["trend_rider_pro", "mean_reversion_pro", "orb_v2"]
STRATEGY_LABELS = {
    "trend_rider_pro": "Trend Rider Pro",
    "mean_reversion_pro": "Mean Reversion Pro",
    "orb_v2": "ORB V2",
    "flow_alpha": "Flow Alpha",
    "pair_trading_v2": "Pair Trading V2",
}
STRATEGY_COLORS = {
    "trend_rider_pro": "#2196F3",
    "mean_reversion_pro": "#FF9800",
    "orb_v2": "#4CAF50",
    "flow_alpha": "#9C27B0",
    "pair_trading_v2": "#F44336",
}


def load_study_db(db_path: Path) -> dict[str, Any]:
    """Extract WFO window data from an Optuna study database."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    # Get all WFO studies
    c.execute("SELECT study_id, study_name FROM studies")
    studies = c.fetchall()
    wfo_studies = sorted(
        [(sid, sname) for sid, sname in studies if "wfo" in sname],
        key=lambda x: x[1],
    )

    windows: list[dict] = []
    for study_id, study_name in wfo_studies:
        window_idx = int(study_name.split("_")[-1])

        # Trial counts
        c.execute(
            "SELECT COUNT(*) FROM trials WHERE study_id=? AND state='COMPLETE'",
            (study_id,),
        )
        n_complete = c.fetchone()[0]
        c.execute(
            "SELECT COUNT(*) FROM trials WHERE study_id=? AND state='PRUNED'",
            (study_id,),
        )
        n_pruned = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM trials WHERE study_id=?", (study_id,))
        n_total = c.fetchone()[0]

        # Best trial
        c.execute(
            """
            SELECT t.trial_id
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id=? AND t.state='COMPLETE'
            ORDER BY tv.value DESC LIMIT 1
            """,
            (study_id,),
        )
        best_row = c.fetchone()
        if not best_row:
            continue
        best_trial_id = best_row[0]

        # Best IS score
        c.execute(
            "SELECT value FROM trial_values WHERE trial_id=?",
            (best_trial_id,),
        )
        is_score = c.fetchone()[0]

        # Best params
        c.execute(
            "SELECT param_name, param_value FROM trial_params WHERE trial_id=?",
            (best_trial_id,),
        )
        params = {row[0]: row[1] for row in c.fetchall()}

        # User attributes (metrics)
        c.execute(
            "SELECT key, value_json FROM trial_user_attributes WHERE trial_id=?",
            (best_trial_id,),
        )
        attrs = {row[0]: json.loads(row[1]) for row in c.fetchall()}

        # All completed trial scores for this window (for distribution plot)
        c.execute(
            """
            SELECT tv.value
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id=? AND t.state='COMPLETE'
            ORDER BY tv.value DESC
            """,
            (study_id,),
        )
        all_scores = [row[0] for row in c.fetchall()]

        windows.append(
            {
                "window_idx": window_idx,
                "study_name": study_name,
                "n_complete": n_complete,
                "n_pruned": n_pruned,
                "n_total": n_total,
                "is_score": is_score,
                "params": params,
                "attrs": attrs,
                "all_scores": all_scores,
            }
        )

    conn.close()
    return {"windows": windows}


def load_wfo_results(json_path: Path) -> dict[str, Any] | None:
    """Load completed WFO results JSON if available."""
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


def discover_wfo_runs(artifact_dir: Path) -> list[dict[str, Any]]:
    """Discover completed WFO runs from both isolated and legacy artifact layouts."""
    runs: list[dict[str, Any]] = []

    for json_path in sorted((artifact_dir / "runs").glob("*/*/wfo_results.json")):
        strategy_name = json_path.parent.parent.name
        run_tag = json_path.parent.name
        runs.append(
            {
                "strategy_name": strategy_name,
                "run_tag": run_tag,
                "db_path": json_path.parent / "study.db",
                "json_path": json_path,
            }
        )

    for json_path in sorted(artifact_dir.glob("*_wfo_results.json")):
        strategy_name = json_path.name.removesuffix("_wfo_results.json")
        runs.append(
            {
                "strategy_name": strategy_name,
                "run_tag": "legacy",
                "db_path": artifact_dir / f"{strategy_name}_study.db",
                "json_path": json_path,
            }
        )

    return sorted(runs, key=lambda run: (run["strategy_name"], run["run_tag"], str(run["json_path"])))


def build_summary_payload(all_data: dict[str, dict]) -> dict[str, Any]:
    """Build a deterministic machine-readable summary for all discovered WFO runs."""
    runs: list[dict[str, Any]] = []

    for data in sorted(all_data.values(), key=lambda item: (item["strategy_name"], item["run_tag"])):
        wfo_result = data.get("wfo_result") or {}
        oos_metrics = wfo_result.get("oos_metrics", [])
        param_stability = wfo_result.get("param_stability", {})
        stable_params = sorted(name for name, stats in param_stability.items() if stats.get("stable"))
        unstable_params = sorted(name for name, stats in param_stability.items() if not stats.get("stable"))

        runs.append(
            {
                "strategy_name": data["strategy_name"],
                "run_tag": data["run_tag"],
                "label": data["display_label"],
                "window_count": len(wfo_result.get("windows", [])) or len(data.get("windows", [])),
                "status": "complete" if wfo_result else "running",
                "wfo_efficiency_ratio": wfo_result.get("wfo_efficiency_ratio"),
                "deflated_sharpe_ratio": wfo_result.get("deflated_sharpe_ratio"),
                "profitable_oos_windows": sum(1 for metric in oos_metrics if metric.get("net_pnl", 0) > 0),
                "oos_window_count": len(oos_metrics),
                "total_oos_net_pnl": round(sum(metric.get("net_pnl", 0.0) for metric in oos_metrics), 2),
                "stable_params": stable_params,
                "unstable_params": unstable_params,
                "artifact_dir": data.get("artifact_dir"),
                "result_path": data.get("result_path"),
            }
        )

    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "runs": runs}


def ensure_labeled_result_copy(run: dict[str, Any], wfo_result: dict[str, Any] | None) -> Path | None:
    """Backfill a labeled per-run result file for completed isolated runs."""
    if not wfo_result or run["run_tag"] == "legacy":
        return None

    labeled_path = run["json_path"].parent / f"{run['strategy_name']}_{run['run_tag']}_wfo_results.json"
    if not labeled_path.exists():
        labeled_path.write_text(json.dumps(wfo_result, indent=2))
    return labeled_path


def _display_label(data: dict[str, Any]) -> str:
    base_label = STRATEGY_LABELS.get(data["strategy_name"], data["strategy_name"])
    return f"{base_label} [{data['run_tag']}]"


def _strategy_color(data: dict[str, Any]) -> str:
    return STRATEGY_COLORS.get(data["strategy_name"], "#666")


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def build_is_scores_chart(all_data: dict[str, dict]) -> go.Figure:
    """Bar chart of IS scores per window per strategy."""
    fig = go.Figure()

    for strategy_name, data in all_data.items():
        windows = data.get("windows", [])
        if not windows:
            continue
        label = data["display_label"]
        color = _strategy_color(data)

        x = [f"W{w['window_idx']+1}" for w in windows]
        y = [w["is_score"] for w in windows]

        fig.add_trace(
            go.Bar(
                name=label,
                x=x,
                y=y,
                marker_color=color,
                text=[f"{v:.2f}" for v in y],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="In-Sample Best Scores per WFO Window",
        xaxis_title="WFO Window",
        yaxis_title="Composite IS Score",
        barmode="group",
        template="plotly_white",
        height=400,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def build_oos_comparison_chart(all_data: dict[str, dict]) -> go.Figure | None:
    """IS vs OOS scores from completed WFO results."""
    has_oos = False
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=["IS vs OOS Score (closer to diagonal = less overfit)"],
    )

    for strategy_name, data in all_data.items():
        wfo_result = data.get("wfo_result")
        if not wfo_result:
            continue
        has_oos = True
        label = data["display_label"]
        color = _strategy_color(data)

        is_scores = wfo_result.get("is_scores", [])
        oos_scores = wfo_result.get("oos_scores", [])

        fig.add_trace(
            go.Scatter(
                x=is_scores,
                y=oos_scores,
                mode="markers+text",
                name=label,
                marker={"color": color, "size": 10},
                text=[f"W{i+1}" for i in range(len(is_scores))],
                textposition="top center",
            )
        )

    if not has_oos:
        return None

    # Add diagonal line (no-overfit reference)
    fig.add_shape(
        type="line",
        x0=-5,
        y0=-5,
        x1=10,
        y1=10,
        line={"color": "gray", "width": 1, "dash": "dash"},
    )
    fig.update_layout(
        xaxis_title="IS Score",
        yaxis_title="OOS Score",
        template="plotly_white",
        height=450,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def build_metrics_table(all_data: dict[str, dict]) -> go.Figure:
    """Table showing key metrics per strategy per window."""
    headers = [
        "Strategy",
        "Window",
        "IS Score",
        "Trades",
        "Sharpe",
        "Trade Sharpe",
        "PF",
        "Win %",
        "Max DD %",
        "Avg Hold (min)",
    ]
    rows: list[list[str]] = []

    for strategy_name, data in all_data.items():
        label = data["display_label"]
        for w in data.get("windows", []):
            a = w.get("attrs", {})
            rows.append(
                [
                    label,
                    f"W{w['window_idx']+1}",
                    f"{w['is_score']:.2f}",
                    str(int(a.get("n_trades", 0))),
                    f"{a.get('sharpe', 0):.3f}",
                    f"{a.get('trade_sharpe', 0):.3f}",
                    f"{a.get('profit_factor', 0):.2f}",
                    f"{a.get('winrate', 0)*100:.1f}",
                    f"{a.get('max_dd_pct', 0):.1f}",
                    f"{a.get('avg_hold_min', 0):.0f}",
                ]
            )

    if not rows:
        return go.Figure()

    # Transpose for table
    cols = list(zip(*rows, strict=False))

    # Color-code IS Score column
    is_scores = [float(r[2]) for r in rows]
    is_colors = []
    for s in is_scores:
        if s >= 2.0:
            is_colors.append("rgba(76, 175, 80, 0.3)")
        elif s >= 0.5:
            is_colors.append("rgba(255, 235, 59, 0.3)")
        elif s > -100:
            is_colors.append("rgba(255, 152, 0, 0.3)")
        else:
            is_colors.append("rgba(244, 67, 54, 0.3)")

    cell_colors = [["white"] * len(rows)] * 2 + [is_colors] + [["white"] * len(rows)] * 7

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": headers,
                    "fill_color": "#37474F",
                    "font": {"color": "white", "size": 12},
                    "align": "center",
                },
                cells={
                    "values": list(cols),
                    "fill_color": cell_colors,
                    "align": "center",
                    "font": {"size": 11},
                    "height": 28,
                },
            )
        ]
    )
    fig.update_layout(
        title="Detailed Metrics per WFO Window (In-Sample Best Trial)",
        height=max(300, 50 + 30 * len(rows)),
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


def build_param_stability_chart(all_data: dict[str, dict]) -> go.Figure:
    """Heatmap of parameter CV (coefficient of variation) per strategy."""
    strategies_with_params: list[str] = []
    all_params: dict[str, dict[str, float]] = {}

    for strategy_name, data in all_data.items():
        windows = data.get("windows", [])
        if len(windows) < 2:
            continue

        # Compute CV for each param
        param_keys: set[str] = set()
        for w in windows:
            param_keys.update(w["params"].keys())

        stab: dict[str, float] = {}
        for key in sorted(param_keys):
            vals = [w["params"].get(key) for w in windows if key in w["params"]]
            numeric = [v for v in vals if isinstance(v, (int, float))]
            if len(numeric) < 2:
                continue
            mean_val = sum(numeric) / len(numeric)
            variance = sum((v - mean_val) ** 2 for v in numeric) / len(numeric)
            std_val = math.sqrt(variance)
            cv = std_val / abs(mean_val) if mean_val != 0 else 0
            stab[key] = round(cv, 3)

        if stab:
            label = data["display_label"]
            strategies_with_params.append(label)
            all_params[label] = stab

    if not strategies_with_params:
        return go.Figure()

    # Build heatmap data
    all_param_names = sorted(set().union(*(s.keys() for s in all_params.values())))
    z: list[list[float]] = []
    text: list[list[str]] = []

    for strat_label in strategies_with_params:
        row: list[float] = []
        text_row: list[str] = []
        for p in all_param_names:
            cv = all_params[strat_label].get(p, float("nan"))
            row.append(cv if not math.isnan(cv) else 0)
            if math.isnan(cv):
                text_row.append("N/A")
            elif cv < 0.15:
                text_row.append(f"{cv:.3f} ✅")
            elif cv < 0.3:
                text_row.append(f"{cv:.3f} ⚠️")
            else:
                text_row.append(f"{cv:.3f} 🔴")
        z.append(row)
        text.append(text_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=all_param_names,
            y=strategies_with_params,
            text=text,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#4CAF50"],
                [0.15, "#8BC34A"],
                [0.3, "#FFEB3B"],
                [0.5, "#FF9800"],
                [1.0, "#F44336"],
            ],
            zmin=0,
            zmax=0.6,
            colorbar={"title": "CV", "tickvals": [0, 0.15, 0.3, 0.5]},
        )
    )
    fig.update_layout(
        title="Parameter Stability (CV < 0.15 = stable, > 0.30 = unstable/overfitting)",
        xaxis_title="Parameter",
        height=250 + 50 * len(strategies_with_params),
        template="plotly_white",
    )
    return fig


def build_efficiency_chart(all_data: dict[str, dict]) -> go.Figure | None:
    """Bar chart of WFO efficiency ratios from completed results."""
    names: list[str] = []
    ratios: list[float] = []
    colors: list[str] = []

    for strategy_name, data in all_data.items():
        wfo = data.get("wfo_result")
        if not wfo:
            continue
        label = data["display_label"]
        eff = wfo.get("wfo_efficiency_ratio", 0)
        names.append(label)
        ratios.append(eff)
        colors.append(_strategy_color(data))

    if not names:
        return None

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=ratios,
                marker_color=colors,
                text=[f"{r:.3f}" for r in ratios],
                textposition="outside",
            )
        ]
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="Target (0.5)")
    fig.update_layout(
        title="WFO Efficiency Ratio (OOS / IS) — Higher = Less Overfit",
        yaxis_title="Efficiency Ratio",
        template="plotly_white",
        height=350,
    )
    return fig


def build_score_distribution_chart(all_data: dict[str, dict]) -> go.Figure:
    """Box plots of all trial scores per strategy per window."""
    fig = go.Figure()

    for strategy_name, data in all_data.items():
        label = data["display_label"]
        color = _strategy_color(data)
        for w in data.get("windows", []):
            scores = w.get("all_scores", [])
            # Filter out extreme floor values for readability
            scores = [s for s in scores if s > -100]
            if not scores:
                continue
            fig.add_trace(
                go.Box(
                    y=scores,
                    name=f"{label} W{w['window_idx']+1}",
                    marker_color=color,
                    boxmean=True,
                )
            )

    fig.update_layout(
        title="Trial Score Distributions per Window (floor values excluded)",
        yaxis_title="Composite Score",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return fig


def build_summary_card(all_data: dict[str, dict]) -> str:
    """Build an HTML summary card with key takeaways."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        '<div style="background:#f5f5f5; padding:20px; border-radius:8px; margin-bottom:20px;">',
        '<h2 style="margin-top:0;">Cerberus WFO Dashboard</h2>',
        f'<p style="color:#666;">Generated: {generated_at}</p>',
        "<table style='width:100%; border-collapse:collapse;'>",
        "<tr style='border-bottom:2px solid #ddd;'>",
        "<th style='text-align:left; padding:8px;'>Strategy</th>",
        "<th>Windows</th><th>Status</th>",
        "<th>Avg IS</th><th>WFO Eff.</th><th>Assessment</th>",
        "</tr>",
    ]

    for strategy_name, data in all_data.items():
        label = data["display_label"]
        color = _strategy_color(data)
        windows = data.get("windows", [])
        n_windows = len(windows)
        wfo = data.get("wfo_result")
        expected_windows = len(wfo.get("windows", [])) if wfo else n_windows

        # Status
        status = f"✅ Complete ({n_windows}/{expected_windows})" if wfo else f"🔄 Running ({n_windows}/{expected_windows})"

        # Avg IS
        is_scores = [w["is_score"] for w in windows if w["is_score"] > -200]
        avg_is = sum(is_scores) / len(is_scores) if is_scores else 0
        avg_is_str = f"{avg_is:.2f}"

        # Efficiency
        eff = wfo.get("wfo_efficiency_ratio", 0) if wfo else "—"
        eff_str = f"{eff:.3f}" if isinstance(eff, float) else eff

        # Assessment
        n_positive = sum(1 for s in is_scores if s > 0)
        if not is_scores:
            assessment = "No data"
        elif avg_is < -100:
            assessment = "❌ Not viable"
        elif n_positive < len(is_scores) * 0.5:
            assessment = "⚠️ Inconsistent"
        elif avg_is > 1.5:
            assessment = "✅ Promising"
        else:
            assessment = "🟡 Marginal"

        lines.append(
            f"<tr style='border-bottom:1px solid #eee;'>"
            f"<td style='padding:8px;'><span style='color:{color};'>●</span> {label}</td>"
            f"<td style='text-align:center;'>{n_windows}</td>"
            f"<td style='text-align:center;'>{status}</td>"
            f"<td style='text-align:center;'>{avg_is_str}</td>"
            f"<td style='text-align:center;'>{eff_str}</td>"
            f"<td style='text-align:center;'>{assessment}</td>"
            f"</tr>"
        )

    lines.append("</table>")

    # Regime gating note
    lines.append(
        '<p style="margin-top:15px; color:#555; font-size:0.9em;">'
        "<strong>Regime Gating:</strong> TRP & ORB V2 trade in BULL/BEAR only. "
        "MRP trades CHOP only. ORB V2 has additional 5-axis activation policy.<br>"
        "<strong>Not Optimizable:</strong> Flow Alpha (needs live options flow data), "
        "Pair Trading V2 (needs pair-specific cointegrated data).</p>"
    )
    lines.append("</div>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="WFO Performance Dashboard")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open dashboard in browser after generating",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/optimization",
        help="Directory containing study DBs and result JSONs",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)

    # Load data for each completed run
    all_data: dict[str, dict] = {}
    for run in discover_wfo_runs(artifact_dir):
        db_data = load_study_db(run["db_path"])
        wfo_result = load_wfo_results(run["json_path"])
        labeled_result_path = ensure_labeled_result_copy(run, wfo_result)
        run_key = f"{run['strategy_name']}::{run['run_tag']}"

        if db_data.get("windows") or wfo_result:
            all_data[run_key] = {
                "strategy_name": run["strategy_name"],
                "run_tag": run["run_tag"],
                "display_label": f"{STRATEGY_LABELS.get(run['strategy_name'], run['strategy_name'])} [{run['run_tag']}]",
                "windows": db_data.get("windows", []),
                "wfo_result": wfo_result,
                "artifact_dir": str(run["json_path"].parent),
                "result_path": str(labeled_result_path or run["json_path"]),
            }

    if not all_data:
        print("No WFO data found in", artifact_dir)
        return

    # Build charts
    figures: list[tuple[str, go.Figure | None]] = []
    if PLOTLY_AVAILABLE:
        figures = [
            ("is_scores", build_is_scores_chart(all_data)),
            ("oos_comparison", build_oos_comparison_chart(all_data)),
            ("efficiency", build_efficiency_chart(all_data)),
            ("metrics_table", build_metrics_table(all_data)),
            ("param_stability", build_param_stability_chart(all_data)),
            ("score_distribution", build_score_distribution_chart(all_data)),
        ]

    # Build HTML
    summary_html = build_summary_card(all_data)

    chart_htmls: list[str] = []
    for _name, fig in figures:
        if fig is not None:
            chart_htmls.append(fig.to_html(full_html=False, include_plotlyjs=False))

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Cerberus WFO Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }}
        .chart-container {{ background: white; border-radius: 8px; padding: 15px;
                           margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
    </style>
</head>
<body>
{summary_html}
{''.join(f'<div class="chart-container">{ch}</div>' for ch in chart_htmls)}
<p style="color:#666; text-align:center;">
    {"Interactive charts skipped because plotly is not installed in this environment." if not PLOTLY_AVAILABLE else ""}
</p>
<p style="text-align:center; color:#999; margin-top:30px;">
    Cerberus WFO Dashboard — re-run <code>PYTHONPATH=. python scripts/wfo_dashboard.py --open</code> to refresh
</p>
</body>
</html>"""

    out_path = artifact_dir / "wfo_dashboard.html"
    out_path.write_text(html)
    print(f"Dashboard written to {out_path}")

    summary_path = artifact_dir / "wfo_dashboard_summary.json"
    summary_path.write_text(json.dumps(build_summary_payload(all_data), indent=2))
    print(f"Summary written to {summary_path}")

    if args.open:
        webbrowser.open(f"file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
