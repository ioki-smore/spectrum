from datetime import datetime
from typing import Set, Dict

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import AppConfig
from core.discovery import IntervalDiscovery
from core.pipeline import Pipeline
from core.reporting import ReportHandler
from utils.errors import ErrorCode
from utils.logger import get_logger

logger = get_logger("core.service")


class AnomalyDetectionService:
    """
    Manages the lifecycle of the anomaly detection service.
    
    Responsibilities:
    1. Interval Discovery: Periodically scans for new data folders.
    2. Job Scheduling: Dynamically schedules training and detection jobs for each discovered interval.
    3. Error Handling: Acts as a supervisor, catching and logging exceptions from jobs to prevent crashes.
    4. Feedback Loop: processes user feedback to trigger incremental model updates.
    
    Uses `APScheduler` (BlockingScheduler) to manage periodic tasks.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.discovery = IntervalDiscovery(config)
        self.reporting = ReportHandler(config)

        # Registry of active pipelines: {interval_name: Pipeline}
        self.pipelines: Dict[str, Pipeline] = {}

        # Track which intervals have jobs scheduled to avoid duplicates
        self.scheduled_intervals: Set[str] = set()

        # Track last training time per interval for periodic forced retraining
        self._last_train_time: Dict[str, datetime] = {}

        # Initial discovery
        self._refresh_intervals()

        # Initialize scheduler with thread pool
        # We use a ThreadPoolExecutor to allow multiple jobs (e.g., detect for different intervals)
        # to run concurrently.
        executors = {'default': ThreadPoolExecutor(16)}
        self.scheduler = BlockingScheduler(executors=executors)
        self._setup_scheduler()

    def _setup_scheduler(self):
        """Configure scheduler listeners and global maintenance jobs."""
        # Add listener to log crashes
        self.scheduler.add_listener(self._job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

        # --- Global Jobs ---

        # 1. Discovery Job (Every 1 hour)
        # Scans for new data directories and initializes pipelines for them.
        self.scheduler.add_job(self._job_discovery, trigger=IntervalTrigger(hours=1), id="discovery_job",
            name="discovery_job")

        # 2. Feedback Processing Job
        # Periodically checks for new user feedback (False Alarms) and retrains models.
        feedback_interval = self.config.training.feedback_interval
        self.scheduler.add_job(self.process_feedback, trigger=IntervalTrigger(minutes=feedback_interval),
            id="feedback_processing", name="feedback_processing", max_instances=1)

    def run(self):
        """
        Start the service and block until stopped.
        This is the main entry point for the application.
        """
        logger.info("Starting RueAI Anomaly Detection Service...")

        if not self.pipelines:
            logger.warning(f"No data intervals discovered in {self.config.data.source_path} (waiting for data...)")

        # Schedule jobs for initial pipelines
        for interval in self.pipelines:
            self._schedule_interval(interval)

        logger.info(f"Scheduler started. Monitoring {len(self.scheduled_intervals)} intervals.")

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass

    def stop(self):
        """Stop the service gracefully, waiting for running jobs to finish if possible."""
        logger.info("Stopping services...")
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        logger.info("Service stopped gracefully")

    def process_feedback(self) -> int:
        """
        Periodic job to process user feedback.
        
        Workflow:
        1. Read pending feedback items (False Alarms) from the reporting system.
        2. For each feedback item, trigger `incremental_train` on the corresponding pipeline.
        3. Mark items as processed upon success.
        
        Returns:
            int: Number of successfully processed feedback items.
        """
        try:
            df = self.reporting.read_pending_feedback()
            if df is None:
                return 0

            logger.info(f"Processing {len(df)} feedback items")

            processed_indices = []
            for row in df.iter_rows(named=True):
                interval = row['interval']
                if interval in self.pipelines:
                    try:
                        # Trigger fine-tuning for the specific time range
                        res = self.pipelines[interval].incremental_train(row['start_time'], row['end_time'])
                        if res.is_ok():
                            processed_indices.append(row['_idx'])
                    except Exception as e:
                        logger.error(f"Error processing feedback for interval {interval}: {e}")
                else:
                    logger.warning(f"Unknown interval in feedback: {interval}")

            if processed_indices:
                self.reporting.mark_processed(processed_indices)
                logger.info(f"Successfully processed {len(processed_indices)} items")

            return len(processed_indices)
        except Exception as e:
            logger.error(f"Error in process_feedback job: {e}")
            return 0

    # --- Internal Helpers ---

    def _refresh_intervals(self):
        """
        Discover and initialize pipelines for new intervals.
        Idempotent: skips intervals that are already initialized.
        """
        discovered = self.discovery.discover()
        for interval in discovered:
            if interval not in self.pipelines:
                logger.info(f"Initializing new pipeline for interval: {interval}")
                self.pipelines[interval] = Pipeline(interval, self.config)
        return discovered

    def _job_listener(self, event: JobExecutionEvent):
        """APScheduler listener to log job successes and failures."""
        if event.exception:
            import traceback
            tb = "".join(traceback.format_tb(event.traceback)) if event.traceback else ""
            logger.error(f"Job {event.job_id} CRASHED: {event.exception}\n{tb}")
        else:
            logger.debug(f"Job {event.job_id} executed successfully.")

    def _job_discovery(self):
        """Wrapper for periodic discovery task."""
        try:
            self._refresh_intervals()
            for interval in self.pipelines:
                self._schedule_interval(interval)
        except Exception as e:
            logger.error(f"Error in discovery job: {e}")

    def _schedule_interval(self, interval: str):
        """
        Schedule train and detect jobs for a specific interval.
        Ensures we don't schedule duplicate jobs for the same interval.
        """
        if interval in self.scheduled_intervals:
            return

        # 1. Training Job
        # Runs less frequently (e.g., every 6 hours or 1 day)
        training_interval = self.config.training.interval
        self.scheduler.add_job(self._job_train, trigger=IntervalTrigger(minutes=training_interval),
            id=f"train_{interval}", name=f"train_{interval}", args=[interval], replace_existing=True, max_instances=1,
            next_run_time=datetime.now()  # Trigger immediately on startup
        )

        # 2. Detection Job
        # Runs frequently (e.g., every 1 min or 5 mins)
        detection_interval = self.config.detection.interval
        self.scheduler.add_job(self._job_detect, trigger=IntervalTrigger(minutes=detection_interval),
            id=f"detect_{interval}", name=f"detect_{interval}", args=[interval], replace_existing=True, max_instances=1)

        self.scheduled_intervals.add(interval)
        logger.info(f"Scheduled jobs for interval: {interval}")

    # --- Job Wrappers ---
    # These wrappers ensure that exceptions in pipeline logic don't crash the scheduler thread.

    def _job_train(self, interval_id: str):
        """
        Job wrapper: Train model for a specific interval.
        Handles initial training and periodic forced retraining for concept drift.
        """
        try:
            pipeline = self.pipelines.get(interval_id)
            if not pipeline:
                return

            if not pipeline.is_trained:
                # Initial training
                logger.info(f"[{interval_id}] Triggering initial training")
                res = pipeline.train()
                if res.is_err() and res.err_value != ErrorCode.DATA_INSUFFICIENT:
                    logger.error(f"[{interval_id}] Training failed: {res.err_value}")
                elif res.is_ok():
                    self._last_train_time[interval_id] = datetime.now()
            else:
                # Periodic forced retraining to combat concept drift
                retrain_days = int(self.config.models.get("retrain_interval_days", 7))
                last_train = self._last_train_time.get(interval_id)
                if last_train is None or (datetime.now() - last_train).days >= retrain_days:
                    logger.info(f"[{interval_id}] Triggering periodic retraining (drift prevention)")
                    res = pipeline.train(force=True)
                    if res.is_err():
                        logger.error(f"[{interval_id}] Periodic retraining failed: {res.err_value}")
                    else:
                        self._last_train_time[interval_id] = datetime.now()
        except Exception as e:
            logger.error(f"[{interval_id}] Unexpected error in training job: {e}")

    def _job_detect(self, interval_id: str):
        """
        Job wrapper: Run detection for a specific interval.
        """
        try:
            pipeline = self.pipelines.get(interval_id)
            if not pipeline:
                return

            if not pipeline.is_trained:
                # Can't detect without a model. Wait for training job.
                return

            res = pipeline.detect()
            if res.is_ok():
                events = res.unwrap()
                if events:
                    self.reporting.append(events)
            else:
                logger.error(f"[{interval_id}] Detection failed: {res.err_value}")
        except Exception as e:
            logger.error(f"[{interval_id}] Unexpected error in detection job: {e}")
