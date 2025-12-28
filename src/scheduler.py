import logging
import subprocess
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone

logger = logging.getLogger("CerberusScheduler")


class CerberusScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.scheduler = BlockingScheduler()
        self.tz = timezone(config.get("timezone", "America/New_York"))
        self._setup_logging()

    def _setup_logging(self):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "service": "Scheduler", "message": "%(message)s"}'
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def start(self):
        """
        Starts the blocking scheduler.
        """
        # Parse schedule time from config or default to 09:25 ET
        schedule_time = self.config.get("schedule_time", "09:25")
        hour, minute = map(int, schedule_time.split(":"))

        # Add job to run Mon-Fri
        trigger = CronTrigger(
            day_of_week="mon-fri", hour=hour, minute=minute, timezone=self.tz
        )

        self.scheduler.add_job(
            self._run_daily_session,
            trigger=trigger,
            id="cerberus_daily_session",
            name="Cerberus Daily Trading Session",
            replace_existing=True,
        )

        logger.info(
            f"Scheduler started. Next run scheduled for: {trigger.get_next_fire_time(None, datetime.now(self.tz))}"
        )

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped by user.")

    def _run_daily_session(self):
        """
        Spawns a subprocess to run the main trading logic.
        This ensures a clean memory slate for each daily run.
        """
        logger.info("Starting daily trading session subprocess...")

        cmd = [sys.executable, "-m", "src.main", "--mode", "live"]

        # Pass through config path if specified in args (not easily available here unless we store args)
        # For now, we assume default config or rely on the fact that src.main defaults to config/config.yaml
        # If we wanted to support custom config paths in the scheduled job, we'd need to pass that into CerberusScheduler.
        if self.config.get("config_path"):
            cmd.extend(["--config", self.config["config_path"]])

        try:
            # Run completely isolated subprocess
            # capturing output is optional, but we probably want it streamed to stdout
            result = subprocess.run(cmd, capture_output=False, check=False)

            if result.returncode == 0:
                logger.info("Daily session completed successfully.")
            else:
                logger.error(
                    f"Daily session failed with exit code {result.returncode}."
                )
        except Exception as e:
            logger.error(f"Failed to launch subprocess: {e}")
