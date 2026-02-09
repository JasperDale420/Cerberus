from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

from src.core.logger import StructuredLogger


class WalkForwardManager:
    """
    Manages rolling parameter optimization windows (Walk-Forward).

    PRD 7.2: Refine strategy parameters dynamically based on regime transitions
    and periodic re-training on rolling windows.
    """

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    def get_windows(
        self, end_date: datetime, n_days: int, step_days: int
    ) -> List[tuple[datetime, datetime]]:
        """
        Produce a list of (start, end) windows for rolling optimization.
        """
        windows = []
        current_end = end_date
        while len(windows) < 5:  # Limit to 5 windows for now
            current_start = current_end - timedelta(days=n_days)
            windows.append((current_start, current_end))
            current_end = current_end - timedelta(days=step_days)
        return list(reversed(windows))

    def evaluate_stability(self, window_results: List[Dict[str, Any]]) -> bool:
        """
        Check if parameters are stable across walk-forward windows.
        If results across windows are highly volatile, the parameters might be overfit.
        """
        if len(window_results) < 2:
            return True

        expectancies = [r.get("expectancy", 0.0) for r in window_results]
        # Simple heuristic: at least 70% of windows must be positive
        pos_windows = len([e for e in expectancies if e > 0])
        stability_score = pos_windows / len(expectancies)

        self.logger.info(
            "Walk-forward stability check",
            expectancies=expectancies,
            score=stability_score,
        )

        return stability_score >= 0.7
