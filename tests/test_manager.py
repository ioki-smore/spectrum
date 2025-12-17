import unittest
from unittest.mock import MagicMock, patch
import tempfile
import shutil
import polars as pl
import numpy as np
from pathlib import Path

# Assume we can import these after patching config if needed, 
# or patch the config object directly since it's already imported in modules.
from config import config, DataConfig, ModelsConfig, TrainingConfig, DetectionConfig
from core.manager import IntervalManager, Manager
from data.loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch('data.loader.logger')
        self.mock_logger = self.logger_patcher.start()
        
        self.test_dir = tempfile.mkdtemp()
        self.source_path = Path(self.test_dir) / "data" / "source"
        self.interval = "1min"
        self.interval_path = self.source_path / self.interval
        self.interval_path.mkdir(parents=True, exist_ok=True)
        
        # Patch config to point to test dir
        self.original_data_config = config.data
        config.data = DataConfig(source_path=str(self.source_path))

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        config.data = self.original_data_config
        self.logger_patcher.stop()

    def create_dummy_csv(self, filename, timestamps):

        df = pl.DataFrame({
            "timestamp": timestamps,
            "value": np.random.rand(len(timestamps)),
            "label": [0] * len(timestamps)
        })
        df.write_csv(self.interval_path / filename)

    def test_load_training_data(self):
        # Create 7 compliant files + 1 non-compliant file
        interval_ms = 60 * 1000 # 1min
        points_per_day = 1440
        
        # Create 7 valid files
        # We need specific filenames? Loader validates by content length vs interval.
        # But filename is used for logging.
        # Timestamps should be distinct days.
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
        
        loader = DataLoader(self.interval)
        df = loader.load_training_data("7d")
        
        self.assertIsNotNone(df)
        # Expected length: 7 * 1440 = 10080. Bad file (50) should be discarded.
        self.assertEqual(len(df), 10080)
        
        # Verify bad data is not in df
        timestamps = df['timestamp'].to_list()
        self.assertNotIn(bad_start, timestamps)

    def test_load_new_data(self):
        # Use test directory for state persistence
        state_dir = Path(self.test_dir) / "state"
        loader = DataLoader(self.interval, state_dir=state_dir)
        interval_ms = 60 * 1000
        
        # 1. No data initially
        df = loader.load_new_data()
        self.assertIsNone(df) 
        
        # 2. Add file with valid dense data
        # 10 points spaced by interval
        timestamps1 = [1000 + i * interval_ms for i in range(10)]
        self.create_dummy_csv("file1.csv", timestamps1)
        
        df = loader.load_new_data()
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 10)
        self.assertEqual(loader.last_timestamp, timestamps1[-1])
        
        # 3. Add newer file
        timestamps2 = [timestamps1[-1] + (i+1) * interval_ms for i in range(5)]
        self.create_dummy_csv("file2.csv", timestamps2)
        
        df = loader.load_new_data()
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 5)
        self.assertEqual(loader.last_timestamp, timestamps2[-1])


class TestIntervalManager(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch('core.manager.logger')
        self.mock_logger = self.logger_patcher.start()
        
        self.test_dir = tempfile.mkdtemp()
        
        # Mock Config
        self.original_data = config.data
        self.original_models = config.models
        self.original_training = config.training
        self.original_detection = config.detection

        config.data = DataConfig(source_path=str(Path(self.test_dir) / "data"))
        config.models = ModelsConfig(save_path=str(Path(self.test_dir) / "models"))
        config.training = TrainingConfig(data_window='7d')
        config.detection = DetectionConfig(summary_file=str(Path(self.test_dir) / "results" / "summary.csv"))
        
        self.interval = "1min"
        self.manager = IntervalManager(self.interval)
        
        # Mock Loader and Processor
        self.manager.loader = MagicMock()
        self.manager.processor = MagicMock()
        self.manager.model = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        config.data = self.original_data
        config.models = self.original_models
        config.training = self.original_training
        config.detection = self.original_detection
        self.logger_patcher.stop()

    def test_train_success(self):
        # Mock data spanning 7 days with sufficient density
        # 1 min interval -> ~1440 points/day -> ~10080 points/week
        # We need > 90% of this
        points = 11000
        timestamps = np.arange(points) * 60 * 1000 # 1 min steps
        df = pl.DataFrame({
            "timestamp": timestamps, 
            "val": np.random.rand(points)
        })
        
        self.manager.loader.load_training_data.return_value = df
        self.manager.loader.interval_ms = 60 * 1000 # 1 min
        
        self.manager.processor.transform.return_value = df
        
        # Ensure model does NOT exist
        self.manager.model_path = MagicMock()
        self.manager.model_path.exists.return_value = False
        self.manager.processor_path = MagicMock()
        self.manager.threshold_path = MagicMock()
        self.manager.threshold_path.exists.return_value = False
        
        # Mock predict for POT calculation
        self.manager.model.predict.return_value = np.random.rand(points)

        self.manager.train()
        
        self.manager.loader.load_training_data.assert_called_once()
        self.manager.processor.fit.assert_called_once()
        self.manager.processor.save.assert_called_once()
        self.manager.model.fit.assert_called_once()
        self.manager.model.save.assert_called_once()

    def test_train_skip_existing_model(self):
        self.manager.model_path = MagicMock()
        self.manager.model_path.exists.return_value = True
        self.manager.threshold_path = MagicMock()
        self.manager.threshold_path.exists.return_value = True
        
        self.manager.train()
        
        self.manager.loader.load_training_data.assert_not_called()

    def test_train_insufficient_data(self):
        # Mock load_training_data to return None (Loader handles check)
        self.manager.loader.load_training_data.return_value = None
        
        # Ensure model does NOT exist
        self.manager.model_path = MagicMock()
        self.manager.model_path.exists.return_value = False
        self.manager.threshold_path = MagicMock()
        self.manager.threshold_path.exists.return_value = False
        
        self.manager.train()
        
        self.manager.processor.fit.assert_not_called()

    def test_train_no_data(self):
        self.manager.model_path = MagicMock()
        self.manager.model_path.exists.return_value = False
        self.manager.threshold_path = MagicMock()
        self.manager.threshold_path.exists.return_value = False
        self.manager.loader.load_training_data.return_value = None
        self.manager.train()
        self.manager.processor.fit.assert_not_called()

    def test_detect_success(self):
        # Setup paths
        self.manager.model_path = Path(self.test_dir) / "models" / "model.pth"
        self.manager.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.manager.model_path.touch()
        
        # Mock threshold file
        self.manager.threshold_path = Path(self.test_dir) / "models" / f"{self.interval}_threshold.json"
        with open(self.manager.threshold_path, 'w') as f:
            import json
            json.dump({"threshold": 0.5}, f)
        
        self.manager.processor_path = Path(self.test_dir) / "models" / "proc.joblib"
        self.manager.processor_path.touch()
        
        self.manager.processor.fitted = True
        
        # Mock data
        df = pl.DataFrame({"timestamp": [100], "val": [0.1]})
        self.manager.loader.load_new_data.return_value = df
        self.manager.processor.transform.return_value = df
        self.manager.model.predict.return_value = np.array([0.1])
        
        res = self.manager.detect()
        
        # Expect empty list for no anomalies
        self.assertEqual(res, [])
        self.manager.model.predict.assert_called_once()
        
    def test_detect_anomaly(self):
        # Setup as above
        self.manager.model_path = Path(self.test_dir) / "models" / "model.pth"
        self.manager.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.manager.model_path.touch()
        
        # Mock threshold file
        self.manager.threshold_path = Path(self.test_dir) / "models" / f"{self.interval}_threshold.json"
        with open(self.manager.threshold_path, 'w') as f:
            import json
            json.dump({"threshold": 5.0}, f)
        
        self.manager.processor.fitted = True
        
        df = pl.DataFrame({"timestamp": [100], "val": [10.0]})
        self.manager.loader.load_new_data.return_value = df
        self.manager.processor.transform.return_value = df
        
        self.manager.model.predict.return_value = np.array([10.0])
        # Mock contribution
        self.manager.model.get_contribution.return_value = np.array([[10.0]])

        res = self.manager.detect()
        
        # Expect list with one event
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 1)
        event = res[0]
        self.assertEqual(event['interval'], self.interval)
        self.assertTrue(event['top_k_metrics'] is not None)


    def test_incremental_train(self):
        # Setup paths
        self.manager.model_path = Path(self.test_dir) / "models" / "model.pth"
        self.manager.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.manager.threshold_path = Path(self.test_dir) / "models" / f"{self.interval}_threshold.json"
        
        # Create dummy threshold
        with open(self.manager.threshold_path, 'w') as f:
            import json
            json.dump({"threshold": 0.5}, f)
            
        # Mock pl.scan_csv to return a dataframe
        df = pl.DataFrame({"timestamp": [100], "val": [10.0], "label": [0], "time": [0]})
        
        with patch('polars.scan_csv') as mock_scan:
            mock_lazy = MagicMock()
            mock_scan.return_value = mock_lazy
            mock_lazy.filter.return_value = mock_lazy
            mock_lazy.collect.return_value = df
            
            self.manager.processor.transform.return_value = df
            self.manager.model.predict.return_value = np.array([10.0])
            self.manager.loader.source_path = Path(self.test_dir) / "data"
            
            self.manager.incremental_train(100, 200)
            
            # Verify fit called with update_stats=False
            self.manager.model.fit.assert_called_with(df, update_stats=False)
            
            # Verify threshold updated (10.0 > 0.5, new should be 10.0 * 1.05 = 10.5)
            with open(self.manager.threshold_path, 'r') as f:
                data = json.load(f)
                self.assertAlmostEqual(data['threshold'], 10.5)


class TestManager(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch('core.manager.logger')
        self.mock_logger = self.logger_patcher.start()
        
        self.original_data = config.data
        self.original_detection = config.detection

    def tearDown(self):
        config.data = self.original_data
        config.detection = self.original_detection
        self.logger_patcher.stop()

    @patch('core.manager.IntervalManager')
    def test_auto_infer_intervals(self, MockIntervalManager):
        # 1. Test auto-inference success
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "data" / "source"
            source_path.mkdir(parents=True)
            (source_path / "15min").mkdir()
            (source_path / "1h").mkdir()
            
            # Patch config to point to this source path
            config.data = DataConfig(source_path=str(source_path))
            
            mgr = Manager()
            self.assertEqual(mgr.intervals, ['15min', '1h'])
            self.assertIn('15min', mgr.managers)
            self.assertIn('1h', mgr.managers)

        # 2. Test fallback when no directories found
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "data" / "source"
            source_path.mkdir(parents=True)
            # No subdirs
            
            config.data = DataConfig(source_path=str(source_path))
            
            mgr = Manager()
            self.assertEqual(mgr.intervals, [])

    @patch('core.manager.IntervalManager')
    def test_train_all(self, MockIntervalManager):
        # Create temp dir for source to ensure no files are found
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            
            # Setup config with empty source path
            config.data = DataConfig(source_path=str(source_path))
            
            # Create dummy interval directories so inference works
            (source_path / "1min").mkdir()
            (source_path / "5min").mkdir()
            
            # Ensure distinct mocks are returned
            MockIntervalManager.side_effect = [MagicMock(), MagicMock()]
            
            mgr = Manager()
            mgr.train_all()
            
            self.assertEqual(len(mgr.managers), 2)
            for i_mgr in mgr.managers.values():
                i_mgr.train.assert_called_once()

    @patch('core.manager.IntervalManager')
    def test_detect_all(self, MockIntervalManager):
        # Create temp dir for source to ensure no files are found
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            
            # Create dummy interval directory
            (source_path / "1min").mkdir()
            
            config.data = DataConfig(source_path=str(source_path))
            mgr = Manager()
            
            # Mock return from detect (List[Dict])
            mock_instance = MockIntervalManager.return_value
            mock_instance.detect.return_value = [{
                "interval": "1min", "start_time": 1, "end_time": 2, 
                "top_k_metrics": "val", "is_false_alarm": False, "processed": False
            }]
            
            summary_file = Path(tmpdir) / "summary.csv"
            mgr.summary_file = summary_file
            
            mgr.detect_all()
            
            self.assertTrue(summary_file.exists())
            df = pl.read_csv(summary_file)
            self.assertEqual(len(df), 1)
            # Schema check: top_k_metrics should exist
            self.assertIn("top_k_metrics", df.columns)

    def test_process_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_file = Path(tmpdir) / "summary.csv"
            
            # Create summary with pending feedback (using new schema)
            df = pl.DataFrame({
                "interval": ["1min"],
                "start_time": [100],
                "end_time": [200],
                "top_k_metrics": ["val"],
                "is_false_alarm": [True],
                "processed": [False]
            })
            df.write_csv(summary_file)
            
            # Setup manager with mock interval manager
            config.detection = DetectionConfig(summary_file=str(summary_file))
            
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            (source_path / "1min").mkdir()
            config.data = DataConfig(source_path=str(source_path))

            with patch('core.manager.IntervalManager') as MockIntervalManager:
                mock_im = MockIntervalManager.return_value
                
                mgr = Manager()
                mgr.process_feedback()
                
                # Check if incremental_train called
                mock_im.incremental_train.assert_called_with(100, 200)
                
                # Check if summary file updated
                df_new = pl.read_csv(summary_file)
                self.assertTrue(df_new['processed'][0])


    def test_run_pipeline(self):
        with patch('core.manager.IntervalManager') as MockIntervalManager:
            mock_im = MockIntervalManager.return_value
            mgr = Manager()
            
            # Case 1: Train returns True (Ready) -> Detect called
            mock_im.train.return_value = True
            mock_im.detect.return_value = [{"res": 1}]
            
            mgr.run_pipeline()
            
            mock_im.train.assert_called()
            mock_im.detect.assert_called()
            
            # Case 2: Train returns False (Not Ready) -> Detect NOT called
            mock_im.reset_mock()
            mock_im.train.return_value = False
            
            mgr.run_pipeline()
            
            mock_im.train.assert_called()
            mock_im.detect.assert_not_called()

if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()
