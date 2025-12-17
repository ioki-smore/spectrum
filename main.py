import argparse
import sys
import signal
import time
import json
import logging
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from config import config
from core.manager import manager
from utils.logger import get_logger

logger = get_logger("main")

# Reduce apscheduler logging verbosity
logging.getLogger('apscheduler').setLevel(logging.WARNING)

class SpectrumService:
    """
    Main service class for the Spectrum Anomaly Detection System.
    Handles service lifecycle, signal processing, and CLI command dispatch.
    """
    
    def __init__(self):
        self.running = False
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_listener(self._job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    def _job_listener(self, event):
        if event.exception:
            logger.error(f"Job {event.job_id} failed: {event.exception}")
        else:
            logger.debug(f"Job {event.job_id} executed successfully.")

    def signal_handler(self, sig, frame):
        """Handle system signals for graceful shutdown."""
        logger.info("Received shutdown signal. Stopping services...")
        self.running = False
        try:
            self.scheduler.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
        sys.exit(0)

    def start(self):
        """
        Start the long-running detection service.
        Schedules periodic tasks and keeps the main thread alive.
        """
        logger.info("Starting Spectrum Anomaly Detection Service...")
        
        # 1. Validate Configuration
        if not config.validate():
            logger.error("Configuration validation failed. Exiting.")
            # TODO：内层不要直接退出
            sys.exit(1)

        # 2. Register Signal Handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # 3. Schedule Tasks
        try:
            # Unified Pipeline (Training + Detection)
            # Runs every interval (default 5 mins). Logic inside run_pipeline handles conditional training.
            pipeline_interval = config.detection.interval_minutes * 60
            self.scheduler.add_job(
                manager.run_pipeline,
                trigger=IntervalTrigger(seconds=pipeline_interval),
                id="anomaly_pipeline",
                name="anomaly_pipeline",
                replace_existing=True,
                coalesce=True,
                max_instances=1
            )
            logger.info(f"Scheduled anomaly pipeline every {pipeline_interval // 60} minutes")

            # Feedback Processing
            feedback_interval = 5 * 60 # 5 minutes
            self.scheduler.add_job(
                manager.process_feedback,
                trigger=IntervalTrigger(seconds=feedback_interval),
                id="feedback_processing",
                name="feedback_processing",
                replace_existing=True,
                coalesce=True,
                max_instances=1
            )
            logger.info(f"Scheduled feedback processing every {feedback_interval // 60} minutes")

            # Start Scheduler
            self.scheduler.start()
            self.running = True
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            self.signal_handler(None, None)
        except Exception as e:
            logger.error(f"Service runtime error: {e}")
            self.signal_handler(None, None)

    def train(self, interval: str):
        """
        Trigger manual training.
        """
        if not config.validate():
            logger.error("Configuration invalid.")
            sys.exit(1)
            
        logger.info(f"Triggering manual training for interval: {interval}")
        try:
            if interval == 'all':
                manager.train_all()
            elif interval in manager.managers:
                manager.managers[interval].train()
            else:
                logger.error(f"Interval '{interval}' not found. Available: {list(manager.managers.keys())}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            sys.exit(1)

    def detect(self, interval: str):
        """
        Trigger manual detection.
        """
        if not config.validate():
            logger.error("Configuration invalid.")
            sys.exit(1)
            
        logger.info(f"Triggering manual detection for interval: {interval}")
        try:
            if interval == 'all':
                manager.detect_all()
            elif interval in manager.managers:
                result = manager.managers[interval].detect()
                # Pretty print result for CLI user
                print(json.dumps(result, indent=2, default=str))
            else:
                logger.error(f"Interval '{interval}' not found. Available: {list(manager.managers.keys())}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Spectrum: Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py start             # Start the background service
  python main.py train --interval 5min   # Train model for 5min interval
  python main.py detect --interval all   # Run detection on all intervals
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Command: start
    _start = subparsers.add_parser('start', help='Start the long-running service')
    
    # Command: train
    train = subparsers.add_parser('train', help='Trigger manual training')
    train.add_argument('--interval', type=str, default='all', help='Interval to train (default: all)')
    
    # Command: detect
    detect = subparsers.add_parser('detect', help='Trigger manual detection')
    detect.add_argument('--interval', type=str, default='all', help='Interval to detect (default: all)')

    args = parser.parse_args()
    service = SpectrumService()

    if args.command == 'start':
        # TODO：在这里退出
        service.start()
    elif args.command == 'train':
        service.train(args.interval)
    elif args.command == 'detect':
        service.detect(args.interval)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    # try catch
    main()
