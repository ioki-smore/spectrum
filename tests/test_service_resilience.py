import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.service import AnomalyDetectionService
from config import AppConfig, DataConfig, ModelsConfig
from utils.errors import Ok


class TestServiceResilience(unittest.TestCase):
    def setUp(self):
        # Setup basic config
        self.config = AppConfig(data=DataConfig(source_path="/tmp/data"), models=ModelsConfig(save_path="/tmp/models"))

        # Patch dependencies
        self.patchers = []

        # Patch IntervalDiscovery
        self.mock_discovery_cls = patch('core.service.IntervalDiscovery').start()
        self.mock_discovery = self.mock_discovery_cls.return_value
        self.mock_discovery.discover.return_value = ["1min"]
        self.patchers.append(self.mock_discovery_cls)

        # Patch ReportHandler
        self.mock_reporting_cls = patch('core.service.ReportHandler').start()
        self.mock_reporting = self.mock_reporting_cls.return_value
        self.patchers.append(self.mock_reporting_cls)

        # Patch Pipeline
        self.mock_pipeline_cls = patch('core.service.Pipeline').start()
        self.mock_pipeline = self.mock_pipeline_cls.return_value
        self.patchers.append(self.mock_pipeline_cls)

        # Patch Scheduler
        self.mock_scheduler_cls = patch('core.service.BlockingScheduler').start()
        self.mock_scheduler = self.mock_scheduler_cls.return_value
        self.patchers.append(self.mock_scheduler_cls)

        # Capture logs
        self.logger_patcher = patch('core.service.logger')
        self.mock_logger = self.logger_patcher.start()
        self.patchers.append(self.logger_patcher)

        self.service = AnomalyDetectionService(self.config)
        # Manually set pipeline for "1min" since constructor calls _refresh_intervals
        self.service.pipelines["1min"] = self.mock_pipeline

    def tearDown(self):
        for p in self.patchers:
            p.stop()

    def test_job_train_handles_exception(self):
        """Test that _job_train catches unexpected exceptions."""
        # Setup pipeline to raise exception on train
        self.mock_pipeline.is_trained = False
        self.mock_pipeline.train.side_effect = Exception("Boom! Training crashed")

        # Execute job manually
        try:
            self.service._job_train("1min")
        except Exception:
            self.fail("_job_train raised an exception instead of catching it")

        # Verify error logged
        self.mock_logger.error.assert_called()
        args, _ = self.mock_logger.error.call_args
        self.assertIn("Unexpected error in training job", args[0])
        self.assertIn("Boom! Training crashed", str(args[0]))

    def test_job_detect_handles_exception(self):
        """Test that _job_detect catches unexpected exceptions."""
        self.mock_pipeline.is_trained = True
        self.mock_pipeline.detect.side_effect = Exception("Boom! Detection crashed")

        try:
            self.service._job_detect("1min")
        except Exception:
            self.fail("_job_detect raised an exception instead of catching it")

        self.mock_logger.error.assert_called()
        args, _ = self.mock_logger.error.call_args
        self.assertIn("Unexpected error in detection job", args[0])
        self.assertIn("Boom! Detection crashed", str(args[0]))

    def test_job_discovery_handles_exception(self):
        """Test that _job_discovery catches unexpected exceptions."""
        self.service.discovery.discover.side_effect = Exception("Boom! Discovery crashed")

        try:
            self.service._job_discovery()
        except Exception:
            self.fail("_job_discovery raised an exception instead of catching it")

        self.mock_logger.error.assert_called()
        args, _ = self.mock_logger.error.call_args
        self.assertIn("Error in discovery job", args[0])
        self.assertIn("Boom! Discovery crashed", str(args[0]))

    def test_process_feedback_handles_exception(self):
        """Test that process_feedback catches unexpected exceptions."""
        self.service.reporting.read_pending_feedback.side_effect = Exception("Boom! Feedback crashed")

        try:
            res = self.service.process_feedback()
        except Exception:
            self.fail("process_feedback raised an exception instead of catching it")

        self.assertEqual(res, 0)
        self.mock_logger.error.assert_called()
        args, _ = self.mock_logger.error.call_args
        self.assertIn("Error in process_feedback job", args[0])
        self.assertIn("Boom! Feedback crashed", str(args[0]))

    def test_process_feedback_individual_interval_exception(self):
        """Test that one interval failing in feedback doesn't stop others."""
        # Setup feedback data with 2 rows
        mock_df = MagicMock()
        mock_df.__len__.return_value = 2

        row1 = {'interval': '1min', 'start_time': 100, 'end_time': 200, '_idx': 1}
        row2 = {'interval': '2min', 'start_time': 300, 'end_time': 400, '_idx': 2}

        mock_df.iter_rows.return_value = [row1, row2]
        self.service.reporting.read_pending_feedback.return_value = mock_df

        # Setup pipelines
        pipeline1 = MagicMock()
        pipeline1.incremental_train.side_effect = Exception("Interval 1 crash")

        pipeline2 = MagicMock()
        pipeline2.incremental_train.return_value = Ok(None)  # Ok

        self.service.pipelines = {'1min': pipeline1, '2min': pipeline2}

        res = self.service.process_feedback()

        # Should have processed 1 item (the second one)
        self.assertEqual(res, 1)

        # Check that error was logged for interval 1
        self.mock_logger.error.assert_any_call("Error processing feedback for interval 1min: Interval 1 crash")

        # Check that success was logged
        self.mock_logger.info.assert_any_call("Successfully processed 1 items")


if __name__ == '__main__':
    unittest.main()
