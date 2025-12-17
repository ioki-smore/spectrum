import numpy as np
import polars as pl
from typing import Optional, Union, Dict, List
import joblib
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# TODO：优化下
from utils.logger import get_logger

logger = get_logger("data.processor")

class DataProcessor:
    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: 'standard' (Z-score) or 'minmax'
        """
        self.method = method
        self.fitted = False
        self.columns: List[str] = []
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method {method}")

    def fit(self, df: pl.DataFrame, columns: Optional[list] = None):
        """
        Compute statistics for normalization using sklearn.
        """
        try:
            if columns is None:
                # Assume all numeric columns except metadata columns if present
                meta_cols = ['timestamp', 'label', 'time', 'datetime', 'Label']
                columns = [
                    c for c in df.columns 
                    if c not in meta_cols and df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                ]

            if not columns:
                raise ValueError("No numeric columns to fit. Check your data schema or specify columns explicitly.")

            self.columns = columns
            
            # Convert to numpy for sklearn
            data = df.select(self.columns).to_numpy()
            self.scaler.fit(data)
            self.fitted = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to fit data processor: {e}")

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise ValueError("Processor not fitted")

        try:
            # Verify columns exist
            missing = [c for c in self.columns if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns for transform: {missing}")
            
            data = df.select(self.columns).to_numpy()
            transformed = self.scaler.transform(data)
            
            # Map back to Polars
            # Create list of Series to update
            new_series = [
                pl.Series(self.columns[i], transformed[:, i]) 
                for i in range(len(self.columns))
            ]
            
            return df.with_columns(new_series)
            
        except Exception as e:
            raise RuntimeError(f"Failed to transform data: {e}")

    def save(self, path: str):
        try:
            path_obj = os.path.dirname(path)
            if path_obj and not os.path.exists(path_obj):
                os.makedirs(path_obj, exist_ok=True)
            
            state = {
                'scaler': self.scaler,
                'columns': self.columns,
                'method': self.method,
                'fitted': self.fitted
            }
            joblib.dump(state, path)
        except Exception as e:
            raise IOError(f"Failed to save processor state to {path}: {e}")

    def load(self, path: str):
        if os.path.exists(path):
            try:
                state = joblib.load(path)
                self.scaler = state['scaler']
                self.columns = state['columns']
                self.method = state['method']
                self.fitted = state['fitted']
            except Exception as e:
                raise IOError(f"Failed to load processor state from {path}: {e}")
        else:
            raise FileNotFoundError(f"Processor state not found at {path}")
