from __future__ import annotations

from unittest.mock import MagicMock

import src.analytics.optimizer as optimizer_module
from src.analytics.optimizer import GridSearchOptimizer


def test_grid_search_iterates_combinations_lazily(monkeypatch) -> None:
    logger = MagicMock()
    opt = GridSearchOptimizer(logger)
    events: list[str] = []

    def fake_product(*_args, **_kwargs):
        for idx in range(2):
            events.append(f"combo:{idx}")
            yield (idx,)

    monkeypatch.setattr(optimizer_module.itertools, "product", fake_product)

    def evaluate_fn(params):
        events.append(f"eval:{params['x']}")
        return {"n_trades": 10, "expectancy": float(params["x"])}

    best_params, best_metrics = opt.optimize(
        evaluate_fn=evaluate_fn,
        search_space={"x": [0, 1]},
        min_trades=1,
    )

    assert best_params == {"x": 1}
    assert best_metrics["expectancy"] == 1.0
    # Laziness check: each combination is evaluated as soon as it is produced.
    assert events == ["combo:0", "eval:0", "combo:1", "eval:1"]
