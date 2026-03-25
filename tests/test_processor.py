import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.processor import DataProcessor
from utils.errors import ErrorCode


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
        df = pl.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0], "timestamp": [1, 2, 3, 4, 5]
                # Should be ignored
            })

        res = processor.fit(df)
        self.assertTrue(res.is_ok())
        self.assertTrue(processor.fitted)
        self.assertEqual(processor.columns, ["a", "b"])

        # Transform
        res = processor.transform(df)
        self.assertTrue(res.is_ok())
        df_trans = res.unwrap()

        # Check if timestamp is preserved and untouched
        self.assertEqual(df_trans['timestamp'].to_list(), [1, 2, 3, 4, 5])

        # Check Z-score of middle element (should be 0)
        self.assertAlmostEqual(df_trans['a'][2], 0.0)
        self.assertAlmostEqual(df_trans['b'][2], 0.0)

    def test_fit_transform_minmax(self):
        processor = DataProcessor(method='minmax')
        df = pl.DataFrame({"a": [0.0, 10.0], "b": [-5.0, 5.0]})

        res = processor.fit(df)
        self.assertTrue(res.is_ok())
        res = processor.transform(df)
        self.assertTrue(res.is_ok())
        df_trans = res.unwrap()

        # a: 0->0, 10->1
        self.assertAlmostEqual(df_trans['a'][0], 0.0)
        self.assertAlmostEqual(df_trans['a'][1], 1.0)

        # b: -5->0, 5->1
        self.assertAlmostEqual(df_trans['b'][0], 0.0)
        self.assertAlmostEqual(df_trans['b'][1], 1.0)

    def test_save_load(self):
        processor = DataProcessor(method='standard')
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        res = processor.fit(df)
        self.assertTrue(res.is_ok())

        res = processor.save(str(self.processor_path))
        self.assertTrue(res.is_ok())
        self.assertTrue(self.processor_path.exists())

        # Load into new instance
        p2 = DataProcessor(method='standard')
        res = p2.load(str(self.processor_path))
        self.assertTrue(res.is_ok())

        self.assertTrue(p2.fitted)
        self.assertEqual(p2.columns, ["a", "b"])

        # Verify transform works on loaded processor
        res = p2.transform(df)
        self.assertTrue(res.is_ok())
        df_trans = res.unwrap()
        self.assertIsNotNone(df_trans)

    def test_transform_not_fitted(self):
        processor = DataProcessor()
        res = processor.transform(pl.DataFrame({"a": [1]}))
        self.assertTrue(res.is_err())
        self.assertEqual(res.err_value, ErrorCode.PROCESSOR_NOT_FITTED)

    def test_transform_missing_cols(self):
        processor = DataProcessor()
        df_train = pl.DataFrame({"a": [1.0], "b": [2.0]})
        res = processor.fit(df_train)
        self.assertTrue(res.is_ok())

        df_test = pl.DataFrame({"a": [1.0]})  # Missing 'b'
        res = processor.transform(df_test)
        self.assertTrue(res.is_err())
        self.assertEqual(res.err_value, ErrorCode.PROCESSOR_MISSING_COLUMNS)


if __name__ == '__main__':
    unittest.main()
