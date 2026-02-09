#!/usr/bin/env python3
"""
Generate 5-year isolation backtest configs for each strategy.
Creates individual config files from the ORB template.
"""

from pathlib import Path

import yaml

CONFIG_DIR = Path("config/backtest_5yr")
TEMPLATE_PATH = CONFIG_DIR / "config.yaml"

# Strategy configurations with OPEN activation filters for unbiased testing
# All regimes/trends/sessions allowed to measure true signal quality
OPEN_ACTIVATION = {
    "session": ["opening", "midday", "power_hour"],
    "trend": ["up", "down", "flat"],
    "vol": ["low", "normal", "high"],
    "liquidity": ["good"],
    "risk": ["risk_on", "neutral", "risk_off"],
    "min_confidence": 0.0,
}

STRATEGIES = {
    "orb": {
        "enabled": True,
        "orb_minutes": 15,
        "risk_reward": 2.0,
        "stop_loss_pct": 0.05,
        "min_gap_pct": 0.0,
        "min_flow_zscore": 0.0,
        "activation": OPEN_ACTIVATION,
    },
    "vwap_reversion": {
        "enabled": True,
        "band_sigma": 2.0,
        "confirmation": "none",
        "activation": OPEN_ACTIVATION,
    },
    # ARCHIVED: trend_pullback - No profitable edge found in 5yr backtest
    "vwap_trend_rider": {
        "enabled": True,
        "vwap_slope_threshold": 0.001,
        "ema_confirmation_period": 9,
        "activation": OPEN_ACTIVATION,
    },
    "momentum_continuation": {
        "enabled": True,
        "params": {
            "breakout_lookback": 5,
            "vol_mult": 2.0,
            "close_position": 0.75,
            "risk_reward": 1.5,
        },
    },
    "fusion_v1": {
        "enabled": True,
        "params": {
            "orb_minutes": 15,
            "min_dof_score": 0.4,
            "min_flow_bias": 0.2,
            "min_relative_strength": 0.002,
            "require_flow": False,
            "stop_atr_mult": 2.0,
            "target_atr_mult": 4.0,
        },
        "scanner_override": {
            "enabled": True,
            "interval_minutes": 15,
        },
    },
    # ARCHIVED: failed_breakout - No profitable edge found in 5yr backtest
    "index_mean_reversion": {
        "enabled": True,
        "bb_period": 20,
        "bb_std": 2.5,
        "rsi_period": 2,
        "rsi_oversold": 20,
        "rsi_overbought": 80,
        "stop_pct": 0.05,
        "target_pct": 0.03,
        "activation": OPEN_ACTIVATION,
    },
    "gap_fill": {
        "enabled": True,
        "min_gap_pct": 0.02,
        "max_gap_pct": 0.10,
        "activation": OPEN_ACTIVATION,
    },
}


def generate_config(strategy_name: str, strategy_config: dict) -> dict:
    """Generate a config for a single strategy isolation test."""
    # Load template
    with open(TEMPLATE_PATH) as f:
        config = yaml.safe_load(f)

    # Update database URL
    config["database_url"] = f"sqlite:///backtest_5yr_{strategy_name}.db"

    # Update strategy routing - only this strategy
    config["strategy_routing"] = {
        "bull": [strategy_name],
        "bear": [strategy_name],
        "chop": [strategy_name],
    }

    # Disable all strategies
    for strat in config["strategies"]:
        config["strategies"][strat] = {"enabled": False}

    # Enable and configure target strategy
    config["strategies"][strategy_name] = strategy_config

    # Apply scanner override if present
    if "scanner_override" in strategy_config:
        config["scanner"].update(strategy_config["scanner_override"])
        # Remove it from the strategy params so pydantic doesn't complain
        if "scanner_override" in config["strategies"][strategy_name]:
            del config["strategies"][strategy_name]["scanner_override"]

    return config


def main():
    print(f"Generating strategy isolation configs in {CONFIG_DIR}")

    for strategy_name, strategy_config in STRATEGIES.items():
        output_path = CONFIG_DIR / f"config_{strategy_name}.yaml"

        config = generate_config(strategy_name, strategy_config)

        with open(output_path, "w") as f:
            f.write(
                f"# 5-Year Backtest Config - {strategy_name.upper()} Strategy Isolation\n"
            )
            f.write(
                f"# Purpose: Test {strategy_name} strategy alone on 5 years of data (2020-2024)\n"
            )
            f.write("#\n")
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"  Created: {output_path}")

    print(f"\nGenerated {len(STRATEGIES)} strategy configs.")
    print("\nTo run a backtest:")
    print(
        "  python scripts/run_backtest.py --config config/backtest_5yr/config_<strategy>.yaml \\"
    )
    print("      --start-date 2020-01-02 --end-date 2024-12-31 \\")
    print("      --offline-bars data/offline_bars_5yr/ \\")
    print("      --output results/backtest_5yr_<strategy>.json")


if __name__ == "__main__":
    main()
