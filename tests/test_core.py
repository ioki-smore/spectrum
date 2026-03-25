import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import AppConfig, DataConfig, ModelsConfig, TrainingConfig, DetectionConfig, LoggingConfig
from core.pipeline import Pipeline
from core.discovery import IntervalDiscovery
from core.reporting import ReportHandler
from data.loader import DataLoader
from core.state import StateManager
from utils.errors import ErrorCode, Ok, Err


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch('data.loader.logger')
        self.mock_logger = self.logger_patcher.start()

        self.test_dir = tempfile.mkdtemp()
        self.source_path = Path(self.test_dir) / "data" / "source"
        self.interval = "1min"
        self.interval_path = self.source_path / self.interval
        self.interval_path.mkdir(parents=True, exist_ok=True)

        self.models_path = Path(self.test_dir) / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Create config objects
        self.data_config = DataConfig(source_path=str(self.source_path))
        self.models_config = ModelsConfig(save_path=str(self.models_path))

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        self.logger_patcher.stop()

    def create_dummy_csv(self, filename, timestamps):
        df = pl.DataFrame(
            {"timestamp": timestamps, "value": np.random.rand(len(timestamps)), "label": [0] * len(timestamps)})
        df.write_csv(self.interval_path / filename)

    def test_load_training_data(self):
        # Create 7 compliant files + 1 non-compliant file
        interval_ms = 60 * 1000  # 1min
        points_per_day = 1440

        # Create 7 valid files
        start_base = 1000000000000
        ms_per_day = 24 * 3600 * 1000

        for i in range(7):
            day_start = start_base + i * ms_per_day
            # Full day data (perfect)
            timestamps = [day_start + j * interval_ms for j in range(points_per_day)]
            self.create_dummy_csv(f"day_{i}.csv", timestamps)

        # Create 1 non-compliant file (only 50 points)
        bad_start = start_base + 8 * ms_per_day
        bad_timestamps = [bad_start + j * interval_ms for j in range(50)]
        self.create_dummy_csv("bad_day.csv", bad_timestamps)

        state_manager = StateManager(self.interval, self.models_path)
        loader = DataLoader(self.interval, self.data_config, state_manager)
        df = loader.load_training_data(7)  # Duration 7 days

        self.assertIsNotNone(df)
        self.assertTrue(df.is_ok())
        df_val = df.unwrap()
        # Expected length: 7 * 1440 = 10080. Bad file (50) should be discarded.
        self.assertEqual(len(df_val), 10080)

        # Verify bad data is not in df
        timestamps = df_val['timestamp'].to_list()
        self.assertNotIn(bad_start, timestamps)

    def test_load_new_data(self):
        # Use test directory for state persistence
        state_path = Path(self.test_dir) / "state"
        state_manager = StateManager(self.interval, state_path)
        loader = DataLoader(self.interval, self.data_config, state_manager)
        interval_ms = 60 * 1000

        # 1. No data initially
        res = loader.load_new_data()
        self.assertTrue(res.is_err())
        self.assertEqual(res.err_value, ErrorCode.DATA_NOT_FOUND)

        # 2. Add file with valid dense data
        # 10 points spaced by interval
        timestamps1 = [1000 + i * interval_ms for i in range(10)]
        self.create_dummy_csv("file1.csv", timestamps1)

        res = loader.load_new_data()
        self.assertTrue(res.is_ok())
        df = res.unwrap()
        self.assertEqual(len(df), 10)
        # Verify state is NOT updated until commit
        self.assertEqual(loader.last_timestamp, 0)
        res = loader.commit(timestamps1[-1])
        self.assertTrue(res.is_ok())
        self.assertEqual(loader.last_timestamp, timestamps1[-1])

        # 3. Add newer file
        timestamps2 = [timestamps1[-1] + (i + 1) * interval_ms for i in range(5)]
        self.create_dummy_csv("file2.csv", timestamps2)

        res = loader.load_new_data()
        self.assertTrue(res.is_ok())
        df = res.unwrap()
        self.assertEqual(len(df), 5)
        res = loader.commit(timestamps2[-1])
        self.assertTrue(res.is_ok())
        self.assertEqual(loader.last_timestamp, timestamps2[-1])


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch('core.pipeline.logger')
        self.mock_logger = self.logger_patcher.start()

        self.test_dir = tempfile.mkdtemp()

        # Create AppConfig
        self.config = AppConfig(data=DataConfig(source_path=str(Path(self.test_dir) / "data")),
            models=ModelsConfig(save_path=str(Path(self.test_dir) / "models")), training=TrainingConfig(data_window=7),
            detection=DetectionConfig(summary_file=str(Path(self.test_dir) / "results" / "summary.csv")),
            logging=LoggingConfig())

        self.interval = "1min"
        self.pipeline = Pipeline(self.interval, self.config)

        # Mock Loader and Processor
        self.pipeline.loader = MagicMock()
        self.pipeline.processor = MagicMock()
        self.pipeline.model = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        self.logger_patcher.stop()

    @patch('core.pipeline.DataProcessor')
    @patch('core.pipeline.AnomalyDetector')
    def test_train_success(self, MockAnomalyDetector, MockDataProcessor):
        # Mock data spanning 7 days with sufficient density
        points = 11000
        timestamps = np.arange(points) * 60 * 1000  # 1 min steps
        df = pl.DataFrame({"timestamp": timestamps, "val": np.random.rand(points)})

        self.pipeline.loader.load_training_data.return_value = Ok(df)
        self.pipeline.loader.interval_ms = 60 * 1000  # 1 min

        # Setup Mock DataProcessor instance (Shadow Copy)
        mock_processor_instance = MockDataProcessor.return_value
        mock_processor_instance.transform.return_value = Ok(df)
        mock_processor_instance.fit.return_value = Ok(None)
        mock_processor_instance.save.return_value = Ok(None)
        mock_processor_instance.columns = ["val"]  # Needed for feature count

        # Mock the instance returned by AnomalyDetector constructor
        mock_model_instance = MockAnomalyDetector.return_value
        mock_model_instance.predict.return_value = Ok(np.random.rand(points))
        mock_model_instance.fit.return_value = Ok(None)
        mock_model_instance.save.return_value = Ok(None)

        self.pipeline.train()

        self.pipeline.loader.load_training_data.assert_called_once()

        # Verify calls on the NEW processor instance
        mock_processor_instance.fit.assert_called_once()
        mock_processor_instance.save.assert_called_once()

        # Verify AnomalyDetector was initialized
        MockAnomalyDetector.assert_called()

        # Verify fit and save were called on the mock instance
        mock_model_instance.fit.assert_called_once()
        mock_model_instance.save.assert_called_once()

        # Verify Hot Swap
        self.assertEqual(self.pipeline.processor, mock_processor_instance)
        self.assertEqual(self.pipeline.model, mock_model_instance)

    def _create_dummy_model_files(self):
        """Helper to create dummy model files for is_trained check."""
        # Main ensemble file
        self.pipeline.ensemble_path.parent.mkdir(parents=True, exist_ok=True)
        self.pipeline.ensemble_path.touch()
        self.pipeline.processor_path.touch()

        # Sub-models
        # Default enabled: usad, lstm, sr
        for name in ["USAD", "LSTM", "SR"]:
            fname = f"{self.interval}_ensemble_{name}_{self.interval}.pth"
            (self.pipeline.model_dir / fname).touch()

    def test_train_skip_existing_model(self):
        # Create files
        self._create_dummy_model_files()

        # Set threshold in state to simulate trained state
        self.pipeline.state_manager.set_threshold(0.5)

        self.pipeline.train()

        self.pipeline.loader.load_training_data.assert_not_called()

    def test_train_insufficient_data(self):
        # Mock load_training_data to return Error
        self.pipeline.loader.load_training_data.return_value = Err(ErrorCode.DATA_INSUFFICIENT)

        # Ensure model does NOT exist
        if self.pipeline.ensemble_path.exists():
            self.pipeline.ensemble_path.unlink()
        self.pipeline.state_manager.clear()

        res = self.pipeline.train()
        self.assertTrue(res.is_err())
        self.assertEqual(res.err_value, ErrorCode.DATA_INSUFFICIENT)

        self.pipeline.processor.fit.assert_not_called()

    def test_train_no_data(self):
        self.pipeline.loader.load_training_data.return_value = Err(ErrorCode.DATA_NOT_FOUND)
        res = self.pipeline.train()
        self.assertTrue(res.is_err())
        self.assertEqual(res.err_value, ErrorCode.DATA_NOT_FOUND)
        self.pipeline.processor.fit.assert_not_called()

    def test_detect_success(self):
        # Setup paths
        self._create_dummy_model_files()

        # Mock threshold state
        self.pipeline.state_manager.set_threshold(0.5)

        self.pipeline.processor.fitted = True
        self.pipeline.processor.columns = ["val"]

        # Mock data
        df = pl.DataFrame({"timestamp": [100], "val": [0.1]})
        self.pipeline.loader.load_new_data.return_value = Ok(df)
        self.pipeline.processor.transform.return_value = Ok(df)
        self.pipeline.model.predict_with_contributions.return_value = Ok(
            {"consensus": np.array([0.1]), "details": {"model1": np.array([0.1])}, "contributions": None})

        res = self.pipeline.detect()

        # Expect Result.ok([])
        if res.is_err():
            print(f"DEBUG: detect failed with {res}")

        self.assertTrue(res.is_ok())
        self.assertEqual(res.unwrap(), [])
        self.pipeline.model.predict_with_contributions.assert_called_once()

    def test_detect_anomaly(self):
        # Setup as above
        self._create_dummy_model_files()

        # Mock threshold state
        self.pipeline.state_manager.set_threshold(5.0)

        self.pipeline.processor.fitted = True
        self.pipeline.processor.columns = ["val"]

        df = pl.DataFrame({"timestamp": [100], "val": [10.0]})
        self.pipeline.loader.load_new_data.return_value = Ok(df)
        self.pipeline.processor.transform.return_value = Ok(df)

        self.pipeline.model.predict_with_contributions.return_value = Ok(
            {"consensus": np.array([10.0]), "details": {"model1": np.array([10.0])},
             "contributions": np.array([[10.0]])})

        # Mock post_processor to pass through the raw anomaly mask
        self.pipeline.post_processor = MagicMock()
        self.pipeline.post_processor.process.side_effect = lambda raw, data, cols: raw

        res = self.pipeline.detect()

        # Expect Result.ok with one event
        self.assertTrue(res.is_ok())
        events = res.unwrap()
        self.assertIsInstance(events, list)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event['interval'], self.interval)
        self.assertTrue(event['top_k_metrics'] is not None)

    def test_incremental_train(self):
        # Setup paths
        self._create_dummy_model_files()

        # Create dummy threshold
        self.pipeline.state_manager.set_threshold(0.5)

        # Mock pl.scan_csv to return a dataframe
        df = pl.DataFrame({"timestamp": [100], "val": [10.0], "label": [0], "time": [0]})

        # Force processor to be ready
        self.pipeline.processor.fitted = True

        with patch('polars.scan_csv') as mock_scan:
            mock_lazy = MagicMock()
            mock_scan.return_value = mock_lazy
            mock_lazy.filter.return_value = mock_lazy
            mock_lazy.collect.return_value = df

            self.pipeline.processor.transform.return_value = Ok(df)
            self.pipeline.processor.columns = ["val"]  # Match feature columns (excluding metadata)

            # Setup Mocks for _create_model
            # self.pipeline.model is already set in setUp, so _ensure_model_loaded won't call _create_model.
            # _create_model will be called ONCE for shadow model creation.
            mock_shadow_model = MagicMock()
            mock_shadow_model.load.return_value = Ok(None)
            mock_shadow_model.fit.return_value = Ok(None)
            mock_shadow_model.save.return_value = Ok(None)
            mock_shadow_model.predict.return_value = Ok(np.array([10.0]))

            with patch('core.pipeline.Pipeline._create_model', return_value=mock_shadow_model) as mock_create_model:
                # Ensure source path is mocked correctly
                self.pipeline.loader.subdir_path = Path(self.test_dir) / "data" / "1min"
                self.pipeline.loader.use_subdir = True

                res = self.pipeline.incremental_train(100, 200)
                self.assertTrue(res.is_ok())

                # Verify _create_model was called once
                self.assertEqual(mock_create_model.call_count, 1)

                # Verify Shadow Model was used for training
                mock_shadow_model.load.assert_called()
                mock_shadow_model.fit.assert_called_with(unittest.mock.ANY, update_normalization=False)

                # Verify threshold updated
                self.assertAlmostEqual(self.pipeline.state_manager.threshold, 10.5)

                # Verify Hot Swap: pipeline.model should now be the shadow model
                self.assertEqual(self.pipeline.model, mock_shadow_model)


class TestDiscovery(unittest.TestCase):
    def test_discover(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "data" / "source"
            source_path.mkdir(parents=True)
            (source_path / "15min").mkdir()
            (source_path / "1h").mkdir()

            config = AppConfig(data=DataConfig(source_path=str(source_path)))

            discovery = IntervalDiscovery(config)
            intervals = discovery.discover()
            self.assertEqual(intervals, ['15min', '1h'])

    def test_discover_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "data" / "source"
            source_path.mkdir(parents=True)
            # Create files matching pattern
            (source_path / "data_1min.csv").touch()
            (source_path / "data_5min.csv").touch()

            config = AppConfig(data=DataConfig(source_path=str(source_path)))

            discovery = IntervalDiscovery(config)
            intervals = discovery.discover()
            self.assertEqual(sorted(intervals), ['1min', '5min'])


class TestReporting(unittest.TestCase):
    def test_append_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_file = Path(tmpdir) / "summary.csv"
            config = AppConfig(detection=DetectionConfig(summary_file=str(summary_file)))

            reporting = ReportHandler(config)

            events = [
                {"interval": "1min", "start_time": 100, "end_time": 200, "top_k_metrics": "cpu", "is_false_alarm": True,
                    "processed": False}]

            reporting.append(events)

            self.assertTrue(summary_file.exists())

            # Test read pending
            pending_df = reporting.read_pending_feedback()
            self.assertIsNotNone(pending_df)
            self.assertEqual(len(pending_df), 1)
            self.assertEqual(pending_df['interval'][0], "1min")

            # Test mark processed
            idx = pending_df['_idx'][0]
            reporting.mark_processed([idx])

            # Verify file updated
            df = pl.read_csv(summary_file)
            self.assertTrue(df['processed'][0])


class TestConfig(unittest.TestCase):
    def test_validation(self):
        # 1. Valid Config
        config = AppConfig()
        self.assertTrue(config.validate())

        # 2. Invalid Models Config
        config = AppConfig()
        config.models.window_size = 0
        self.assertFalse(config.validate())

        config = AppConfig()
        config.models.epochs = -1
        self.assertFalse(config.validate())

        config = AppConfig()
        config.models.pot_risk = 1.5
        self.assertFalse(config.validate())

        # 3. Invalid Training Config
        config = AppConfig()
        config.training.interval = 0
        self.assertFalse(config.validate())

        config = AppConfig()
        config.training.start_time = "25:00"  # Invalid time
        self.assertFalse(config.validate())

        # 4. Invalid Detection Config
        config = AppConfig()
        config.detection.top_k = 0
        self.assertFalse(config.validate())
        
        config = AppConfig()
        config.detection.timeout = 0
        self.assertFalse(config.validate())

        # 5. Invalid Logging Config
        config = AppConfig()
        config.logging.retention = -1
        self.assertFalse(config.validate())


if __name__ == '__main__':
    unittest.main()
