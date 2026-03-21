import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.core.logger import get_logger

logger = get_logger("cerberus.scheduler")


class CerberusScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.scheduler = BlockingScheduler()
        self.tz = ZoneInfo(config.get("timezone", "America/New_York"))

    def start(self):
        """
        Starts the blocking scheduler.
        """
        # Parse schedule time from config or default to 09:25 ET
        schedule_time = self.config.get("schedule_time", "09:25")
        hour, minute = map(int, schedule_time.split(":"))

        # Add job to run Mon-Fri
        trigger = CronTrigger(day_of_week="mon-fri", hour=hour, minute=minute, timezone=self.tz)

        self.scheduler.add_job(
            self._run_daily_session,
            trigger=trigger,
            id="cerberus_daily_session",
            name="Cerberus Daily Trading Session",
            replace_existing=True,
        )

        # Keep daily-only scheduling as default for backward compatibility.
        if bool(self.config.get("enable_weekly_analysis", False)):
            self.add_weekly_analysis_job()

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

        mode = self.config.get("mode", "paper")
        cmd = [sys.executable, "-m", "src.main", "--mode", mode]

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
                logger.error(f"Daily session failed with exit code {result.returncode}.")
        except Exception as e:
            logger.error(f"Failed to launch subprocess: {e}")

    def _run_weekly_analysis(self):
        """
        Run weekly Stage 3 analysis after Friday market close.
        Generates performance report with feature/model recommendations.
        """
        logger.info("Starting weekly analysis (Stage 3)...")

        cmd = [sys.executable, "-m", "src.main", "--mode", "weekly-analysis"]

        if self.config.get("config_path"):
            cmd.extend(["--config", self.config["config_path"]])

        try:
            result = subprocess.run(cmd, capture_output=False, check=False)

            if result.returncode == 0:
                logger.info("Weekly analysis completed successfully.")
            else:
                logger.error(f"Weekly analysis failed with exit code {result.returncode}.")
        except Exception as e:
            logger.error(f"Failed to launch weekly analysis subprocess: {e}")

    def add_weekly_analysis_job(self):
        """
        Add Friday EOD job for Stage 3 weekly analysis.
        Runs at 16:30 ET (after market close).
        """
        trigger = CronTrigger(day_of_week="fri", hour=16, minute=30, timezone=self.tz)

        self.scheduler.add_job(
            self._run_weekly_analysis,
            trigger=trigger,
            id="cerberus_weekly_analysis",
            name="Cerberus Weekly Analysis (Stage 3)",
            replace_existing=True,
        )

        logger.info(
            f"Weekly analysis job scheduled. Next run: {trigger.get_next_fire_time(None, datetime.now(self.tz))}"
        )
