from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SensitivityResult:
    param_name: str
    values: list[float]
    scores: list[float]
    correlation: float
    sensitivity_rank: int


def analyze_param_sensitivity(
    trials_data: dict[str, list[float]],
) -> list[SensitivityResult]:
    scores = np.array(trials_data["score"])
    param_names = [k for k in trials_data if k != "score"]

    correlations: list[tuple[str, float, list[float]]] = []
    for name in param_names:
        values = np.array(trials_data[name])
        if np.std(values) < 1e-10:
            correlations.append((name, 0.0, trials_data[name]))
            continue
        corr, _ = stats.spearmanr(values, scores)
        correlations.append((name, abs(float(corr)), trials_data[name]))

    correlations.sort(key=lambda x: x[1], reverse=True)

    results = []
    for rank, (name, _abs_corr, values) in enumerate(correlations, start=1):
        if np.std(np.array(values)) < 1e-10:
            signed_corr = 0.0
        else:
            signed_corr, _ = stats.spearmanr(np.array(values), scores)
        results.append(
            SensitivityResult(
                param_name=name,
                values=values,
                scores=trials_data["score"],
                correlation=float(signed_corr),
                sensitivity_rank=rank,
            )
        )
    return results
