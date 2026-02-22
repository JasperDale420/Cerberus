from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from src.core.logger import StructuredLogger


class HealthMonitor:
    """
    Tracks and logs health metrics for the execution engine.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: StructuredLogger,
        run_id: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.config = config
        self.logger = logger
        self.run_id = run_id
        self.clock = clock or (lambda: datetime.now(timezone.utc))

        self.bars_processed = 0
        self.signals_generated = 0
        self.orders_submitted = 0
        self.error_counts: Dict[str, int] = {
            "execution": 0,
            "scanner": 0,
            "regime": 0,
            "strategy": 0,
            "risk": 0,
            "orders": 0,
            "db": 0,
        }
        self._last_health_log_time: Optional[datetime] = None
        self._health_log_interval_sec = int(config.get("health_log_interval_sec", 300))

    def record_error(self, module: str) -> None:
        """Increments error count for a module."""
        try:
            self.error_counts[module] = int(self.error_counts.get(module, 0)) + 1
        except Exception as e:
            self.logger.warning(
                "Failed to record error count",
                module=module,
                error=str(e),
            )

    def check_and_log(self, now: Optional[datetime] = None) -> None:
        """Logs health metrics if the interval has passed."""
        if now is None:
            now = self.clock()

        if self._last_health_log_time is None:
            self._last_health_log_time = now
            return

        delta = (now - self._last_health_log_time).total_seconds()
        if delta < self._health_log_interval_sec:
            return

        self._last_health_log_time = now
        self.logger.info(
            "Health",
            bars_processed=self.bars_processed,
            signals_generated=self.signals_generated,
            orders_submitted=self.orders_submitted,
            error_counts=self.error_counts,
            run_id=self.run_id,
        )

    def update_config(self, config: Dict[str, Any]) -> None:
        """Updates configuration for health monitoring."""
        self.config = config
        self._health_log_interval_sec = int(config.get("health_log_interval_sec", self._health_log_interval_sec))
