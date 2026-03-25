import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.pipeline import Pipeline
from config import AppConfig, DataConfig, ModelsConfig, TrainingConfig, DetectionConfig
from utils.errors import ErrorCode, Ok


class TestSchemaChange(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = AppConfig(data=DataConfig(source_path=str(Path(self.test_dir) / "data")),
            models=ModelsConfig(save_path=str(Path(self.test_dir) / "models")), training=TrainingConfig(),
            detection=DetectionConfig())
        self.interval = "1min"
        self.pipeline = Pipeline(self.interval, self.config)

        # Create dummy model files to simulate "trained" state
        self.pipeline.model_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline.ensemble_path.touch()
        self.pipeline.processor_path.touch()
        self.pipeline.state_manager.set_threshold(0.5)

        # Create dummy sub-model files
        # Defaults: USAD, LSTM, SR
        for name in ["USAD", "LSTM", "SR"]:
            sub_path = self.pipeline.model_dir / f"{self.interval}_ensemble_{name}_{self.interval}.pth"
            sub_path.touch()

        # Mock processor
        self.pipeline.processor = MagicMock()
        self.pipeline.processor.fitted = True
        self.pipeline.processor.columns = ["feature_a"]

        # Mock loader
        self.pipeline.loader = MagicMock()

        # Mock model
        self.pipeline.model = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_detect_schema_change_triggers_reset(self):
        # 1. Setup incoming data with DIFFERENT schema (feature_b instead of feature_a)
        df_new = pl.DataFrame({"timestamp": [1000], "feature_b": [1.0]})
        self.pipeline.loader.load_new_data.return_value = Ok(df_new)

        # 2. Mock _prepare_data to return DATA_INVALID_SCHEMA logic check
        # Actually Pipeline._prepare_data logic is:
        # if current != trained: return Err(DATA_INVALID_SCHEMA)
        # We rely on the real _prepare_data logic, but we need to mock extract_feature_columns if needed, 
        # or just let it run since it uses the dataframe.
        # But wait, Pipeline._prepare_data uses self.processor.columns which we mocked.

        # We need to make sure processor.load works or is skipped.
        # _prepare_data calls self.processor.load if not fitted.
        # We set fitted=True, so it skips load.

        # 3. Call detect()
        res = self.pipeline.detect()

        # 4. Verify result is Err(DATA_INVALID_SCHEMA)
        self.assertTrue(res.is_err())
        self.assertEqual(res.err_value, ErrorCode.DATA_INVALID_SCHEMA)

        # 5. Verify artifacts are deleted (reset)
        self.assertFalse(self.pipeline.ensemble_path.exists(), "Ensemble file should be deleted")
        self.assertFalse(self.pipeline.processor_path.exists(), "Processor file should be deleted")
        self.assertIsNone(self.pipeline.model, "Model should be None")
        self.assertIsNone(self.pipeline.state_manager.threshold, "Threshold should be None")

        # 6. Verify backup was created
        versions_dir = Path(self.test_dir) / "models" / "versions" / self.interval
        self.assertTrue(versions_dir.exists())
        # Should have 1 version directory
        versions = [x for x in versions_dir.iterdir() if x.is_dir()]
        self.assertEqual(len(versions), 1)


if __name__ == '__main__':
    unittest.main()
