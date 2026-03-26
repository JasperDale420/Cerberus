"""Autoresearch scoring wrapper for Pair Trading V2 optimization.

Runs pair_trading_v2 in isolation against 5-year data (2020-2025) and
outputs a single AUTORESEARCH_SCORE line for the autoresearch loop.

Disables all other strategies in-memory to ensure only pair trades
contribute to the composite score.
"""

import asyncio
import copy
import json
import sys

import yaml

from src.backtest.runner import run_backtest

START = "2020-01-02"
END = "2024-12-31"
CONFIG_PATH = "config/backtest_v2.yaml"
DATA_DIR = "data/bars_5yr"


def compute_composite_score(metrics: dict) -> float:
    pnl = metrics.get("net_pnl", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    pf = metrics.get("profit_factor", 0)
    winrate = metrics.get("winrate", 0)
    n_trades = metrics.get("n_trades", 0)

    pnl_score = pnl / 10000.0
    sharpe_score = sharpe / 2.0
    pf_score = max(0, pf - 1.0)
    wr_score = (winrate - 0.50) / 0.15

    if n_trades < 50:
        trade_score = -1.0
    elif n_trades < 200:
        trade_score = (n_trades - 50) / 150.0 * 0.5
    elif n_trades <= 800:
        trade_score = 0.5 + 0.5 * (1.0 - abs(n_trades - 400) / 400.0)
    elif n_trades <= 2000:
        trade_score = max(0, 0.5 - (n_trades - 800) / 2400.0)
    else:
        trade_score = -0.5

    composite = 0.30 * pnl_score + 0.25 * sharpe_score + 0.20 * pf_score + 0.15 * wr_score + 0.10 * trade_score
    return round(composite, 4)


async def main():
    # Suppress verbose INFO logging to keep output clean
    import logging

    logging.getLogger().setLevel(logging.WARNING)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.WARNING)

    # Load config and disable all strategies except pair_trading_v2
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    modified_config = copy.deepcopy(config)
    for strat_name, strat_cfg in modified_config.get("strategies", {}).items():
        if strat_name != "pair_trading_v2":
            strat_cfg["enabled"] = False

    # Ensure pair_trading_v2 is in all routing lists (so it activates regardless of regime)
    for regime in modified_config.get("strategy_routing", {}):
        routing = modified_config["strategy_routing"][regime]
        if "pair_trading_v2" not in routing:
            routing.append("pair_trading_v2")

    # Patch ConfigLoader to use our in-memory config
    from unittest.mock import patch as _mock_patch

    from src.core.config import ConfigLoader

    def _patched_load(self, config_path_or_dir=None):
        return copy.deepcopy(modified_config)

    with _mock_patch.object(ConfigLoader, "load_config", _patched_load):
        report = await run_backtest(START, END, CONFIG_PATH, data_dir=DATA_DIR)

    if report is None:
        print(json.dumps({"autoresearch_score": -999, "error": "backtest_failed"}), file=sys.stderr)
        print("AUTORESEARCH_SCORE=-999")
        sys.exit(1)

    metrics = report.to_dict()
    score = compute_composite_score(metrics)

    result = {
        "autoresearch_score": score,
        "net_pnl": round(metrics.get("net_pnl", 0), 2),
        "sharpe": round(metrics.get("sharpe_ratio", 0), 3),
        "profit_factor": round(metrics.get("profit_factor", 0), 2),
        "winrate": round(metrics.get("winrate", 0) * 100, 1),
        "n_trades": metrics.get("n_trades", 0),
        "max_drawdown_pct": round(metrics.get("max_drawdown_pct", 0), 2),
        "total_return_pct": round(metrics.get("total_return_pct", 0), 2),
        "calmar": round(metrics.get("calmar_ratio", 0), 3),
        "avg_pnl": round(metrics.get("avg_pnl", 0), 2),
        "max_consec_losers": metrics.get("max_consecutive_losers", 0),
    }

    print(json.dumps(result), file=sys.stderr)
    print(f"AUTORESEARCH_SCORE={score}")


if __name__ == "__main__":
    asyncio.run(main())
