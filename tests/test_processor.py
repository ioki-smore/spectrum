import unittest
import tempfile
import shutil
import polars as pl
import numpy as np
from pathlib import Path
from data.processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.processor_path = Path(self.test_dir) / "processor.joblib"
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_init(self):
        p1 = DataProcessor(method='standard')
        self.assertEqual(p1.method, 'standard')
        self.assertFalse(p1.fitted)
        
        p2 = DataProcessor(method='minmax')
        self.assertEqual(p2.method, 'minmax')
        
        with self.assertRaises(ValueError):
            DataProcessor(method='invalid')

    def test_fit_transform_standard(self):
        processor = DataProcessor(method='standard')
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "timestamp": [1, 2, 3, 4, 5] # Should be ignored
        })
        
        processor.fit(df)
        self.assertTrue(processor.fitted)
        self.assertEqual(processor.columns, ["a", "b"])
        
        # Check mean and std (approximately)
        # a: mean=3, std=sqrt(2) ~= 1.414
        # b: mean=30, std=sqrt(200) ~= 14.14
        
        # Transform
        df_trans = processor.transform(df)
        
        # Check if timestamp is preserved and untouched
        self.assertEqual(df_trans['timestamp'].to_list(), [1, 2, 3, 4, 5])
        
        # Check Z-score of middle element (should be 0)
        self.assertAlmostEqual(df_trans['a'][2], 0.0)
        self.assertAlmostEqual(df_trans['b'][2], 0.0)

    def test_fit_transform_minmax(self):
        processor = DataProcessor(method='minmax')
        df = pl.DataFrame({
            "a": [0.0, 10.0],
            "b": [-5.0, 5.0]
        })
        
        processor.fit(df)
        df_trans = processor.transform(df)
        
        # a: 0->0, 10->1
        self.assertAlmostEqual(df_trans['a'][0], 0.0)
        self.assertAlmostEqual(df_trans['a'][1], 1.0)
        
        # b: -5->0, 5->1
        self.assertAlmostEqual(df_trans['b'][0], 0.0)
        self.assertAlmostEqual(df_trans['b'][1], 1.0)

    def test_save_load(self):
        processor = DataProcessor(method='standard')
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        processor.fit(df)
        
        processor.save(str(self.processor_path))
        self.assertTrue(self.processor_path.exists())
        
        # Load into new instance
        p2 = DataProcessor(method='standard')
        p2.load(str(self.processor_path))
        
        self.assertTrue(p2.fitted)
        self.assertEqual(p2.columns, ["a", "b"])
        
        # Verify transform works on loaded processor
        df_trans = p2.transform(df)
        self.assertIsNotNone(df_trans)

    def test_transform_not_fitted(self):
        processor = DataProcessor()
        with self.assertRaises(ValueError):
            processor.transform(pl.DataFrame({"a": [1]}))

    def test_transform_missing_cols(self):
        processor = DataProcessor()
        df_train = pl.DataFrame({"a": [1.0], "b": [2.0]})
        processor.fit(df_train)
        
        df_test = pl.DataFrame({"a": [1.0]}) # Missing 'b'
        with self.assertRaises(RuntimeError): # transform wraps exceptions in RuntimeError
            processor.transform(df_test)

if __name__ == '__main__':
    unittest.main()
