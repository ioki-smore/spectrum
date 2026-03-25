import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.usad import USAD
from models.anomaly_detection import AnomalyDetector
from models.base import BaseModel
from utils.errors import Result, Ok, Err, ErrorCode


class TestUSAD(unittest.TestCase):
    def setUp(self):
        # Patch logger
        self.logger_patcher = patch('models.usad.logger')
        self.mock_logger = self.logger_patcher.start()

        self.config = {'window_size': 10, 'latent_size': 5, 'epochs': 1, 'batch_size': 2}
        self.input_dim = 2
        self.usad = USAD("test_usad", self.config, self.input_dim)

    def tearDown(self):
        self.logger_patcher.stop()

    def test_init(self):
        self.assertEqual(self.usad.window_size, 10)
        self.assertEqual(self.usad.feature_dim, 2)
        self.assertIsNotNone(self.usad.model)

    def test_fit_success(self):
        # Create dummy data
        df = pl.DataFrame(
            {"f1": np.random.rand(20), "f2": np.random.rand(20), "timestamp": range(20), "label": [0] * 20})

        res = self.usad.fit(df)
        self.assertTrue(res.is_ok())

        # We check state by ensuring model is in training mode or parameters exist  # Just passing fit without error is good for unit test here

    def test_predict_success(self):
        df = pl.DataFrame({"f1": np.random.rand(20), "f2": np.random.rand(20)})

        res = self.usad.predict(df)
        self.assertTrue(res.is_ok())
        scores = res.unwrap()
        # Expected length: 20 samples, window 10, step 1 -> 20 - 10 + 1 = 11
        self.assertEqual(len(scores), 11)
        self.assertIsInstance(scores, np.ndarray)

    def test_fit_empty_data(self):
        df = pl.DataFrame({"f1": [], "f2": []})
        res = self.usad.fit(df)
        self.assertTrue(res.is_ok())
        # Should log error about dataset creation or empty
        self.assertTrue(self.mock_logger.error.called or self.mock_logger.warning.called)


class MockModel(BaseModel):
    def fit(self, data) -> Result[None]:
        return Ok(None)

    def predict(self, data) -> Result[np.ndarray]:
        # Return dummy scores of length 1
        return Ok(np.array([0.5]))

    def get_contribution(self, data, top_k=None) -> Result[np.ndarray]:
        # Return dummy contribution
        return Ok(np.array([[0.5]]))


class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch('models.anomaly_detection.logger')
        self.mock_logger = self.logger_patcher.start()

        self.models = [MockModel("m1", {}), MockModel("m2", {})]
        self.detector = AnomalyDetector("ens", self.models)

    def tearDown(self):
        self.logger_patcher.stop()

    def test_fit(self):
        res = self.detector.fit(None)
        self.assertTrue(
            res.is_ok())  # Should call fit on all models.   # Since MockModel does nothing, we just ensure no error.

    def test_predict(self):
        # We need to fit first to populate stats
        self.detector.fit(None)

        res = self.detector.predict(None)
        self.assertTrue(res.is_ok())
        scores = res.unwrap()
        # MockModel returns [0.5]. 
        # Mean=0.5, Std=0 (-> 1.0).
        # Z-score = (0.5 - 0.5) / 1.0 = 0.0
        # Ensemble averages them -> 0.0
        self.assertEqual(scores[0], 0.0)

    def test_partial_failure(self):
        # Make one model fail
        failing_model = MagicMock(spec=BaseModel)
        failing_model.name = "fail"
        # Simulate predictable failure (returning Err)
        failing_model.predict.return_value = Err(ErrorCode.MODEL_PREDICT_FAILED)

        # We need to set up failing model behavior for fit as well
        failing_model.fit.return_value = Ok(None)

        working_model = MockModel("work", {})

        det = AnomalyDetector("ens_fail", [failing_model, working_model])

        # Fit to setup stats
        det.fit(None)

        res = det.predict(None)
        self.assertTrue(res.is_ok())
        scores = res.unwrap()

        # Should rely on working model
        # Working model predicts 0.5 -> Z-score 0.0
        self.assertEqual(scores[0], 0.0)


if __name__ == '__main__':
    unittest.main()
