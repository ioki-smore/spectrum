import unittest
from unittest.mock import MagicMock, patch
import torch
import polars as pl
import numpy as np
import shutil
import tempfile
from pathlib import Path

from models.usad import USAD
from models.ensemble import Ensemble
from models.base import BaseModel

class TestUSAD(unittest.TestCase):
    def setUp(self):
        # Patch logger
        self.logger_patcher = patch('models.usad.logger')
        self.mock_logger = self.logger_patcher.start()
        
        self.config = {
            'window_size': 10,
            'latent_size': 5,
            'epochs': 1,
            'batch_size': 2
        }
        self.input_dim = 2
        self.usad = USAD("test_usad", self.config, self.input_dim)
        
        # Force CPU
        self.usad.device = torch.device('cpu')
        self.usad.model.to('cpu')

    def tearDown(self):
        self.logger_patcher.stop()

    def test_init(self):
        self.assertEqual(self.usad.window_size, 10)
        self.assertEqual(self.usad.feature_dim, 2)
        self.assertIsNotNone(self.usad.model)

    def test_fit_success(self):
        # Create dummy data
        df = pl.DataFrame({
            "f1": np.random.rand(20),
            "f2": np.random.rand(20),
            "timestamp": range(20),
            "label": [0]*20
        })
        
        self.usad.fit(df)
            
        # We check state by ensuring model is in training mode or parameters exist
        # Just passing fit without error is good for unit test here

    def test_predict_success(self):
        df = pl.DataFrame({
            "f1": np.random.rand(20),
            "f2": np.random.rand(20)
        })
        
        scores = self.usad.predict(df)
        # Expected length: 20 samples, window 10, step 1 -> 20 - 10 + 1 = 11
        self.assertEqual(len(scores), 11)
        self.assertIsInstance(scores, np.ndarray)

    def test_fit_empty_data(self):
        df = pl.DataFrame({"f1": [], "f2": []})
        self.usad.fit(df)
        # Should log error about dataset creation or empty
        self.assertTrue(self.mock_logger.error.called or self.mock_logger.warning.called)

class MockModel(BaseModel):
    def fit(self, data):
        pass
    def predict(self, data):
        # Return dummy scores of length 1
        return np.array([0.5])
    def get_contribution(self, data, top_k=None):
        # Return dummy contribution
        return np.array([[0.5]])

class TestEnsemble(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch('models.ensemble.logger')
        self.mock_logger = self.logger_patcher.start()
        
        self.models = [MockModel("m1", {}), MockModel("m2", {})]
        self.ensemble = Ensemble("ens", self.models)

    def tearDown(self):
        self.logger_patcher.stop()

    def test_fit(self):
        self.ensemble.fit(None)
        # Should call fit on all models. 
        # Since MockModel does nothing, we just ensure no error.

    def test_predict(self):
        # We need to fit first to populate stats
        self.ensemble.fit(None)
        
        scores = self.ensemble.predict(None)
        # MockModel returns [0.5]. 
        # Mean=0.5, Std=0 (-> 1.0).
        # Z-score = (0.5 - 0.5) / 1.0 = 0.0
        # Ensemble averages them -> 0.0
        self.assertEqual(scores[0], 0.0)

    def test_partial_failure(self):
        # Make one model fail
        failing_model = MagicMock(spec=BaseModel)
        failing_model.name = "fail"
        failing_model.predict.side_effect = Exception("Boom")
        
        # We need to set up failing model behavior for fit as well, or catch it
        failing_model.fit.side_effect = None # Fit succeeds?
        # If fit fails, stats are default (0, 1)
        
        working_model = MockModel("work", {})
        
        ens = Ensemble("ens_fail", [failing_model, working_model])
        
        # Fit to setup stats
        ens.fit(None)
        
        scores = ens.predict(None)
        
        # Should rely on working model
        # Working model predicts 0.5 -> Z-score 0.0
        self.assertEqual(scores[0], 0.0)

if __name__ == '__main__':
    unittest.main()
