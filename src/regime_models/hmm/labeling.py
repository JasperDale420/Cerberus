from __future__ import annotations

from typing import SupportsInt, cast

import numpy as np
import pandas as pd


def label_hidden_states(features: pd.DataFrame, states: np.ndarray) -> dict[int, str]:
    state_frame = features.copy()
    state_frame["hidden_state"] = states
    grouped = (
        state_frame.groupby("hidden_state", as_index=True)[["log_return_1", "realized_vol_20"]]
        .mean()
        .rename(columns={"log_return_1": "avg_return", "realized_vol_20": "avg_vol"})
    )
    if grouped.empty:
        return {}

    max_vol = float(grouped["avg_vol"].max())
    labels: dict[int, str] = {}
    for state_id, row in grouped.iterrows():
        avg_return = float(row["avg_return"])
        avg_vol = float(row["avg_vol"])
        if avg_vol >= max_vol and avg_return < 0.0:
            label = "stress_down"
        elif avg_vol >= max_vol and avg_return > 0.0:
            label = "stress_up"
        elif avg_return > 0.0:
            label = "trend_up"
        elif avg_return < 0.0:
            label = "trend_down"
        else:
            label = "neutral"
        raw_state = state_id.item() if hasattr(state_id, "item") else state_id
        state_key = int(cast(SupportsInt, raw_state))
        labels[state_key] = label
    return labels
