"""
Data preprocessing and normalization module.
Handles feature scaling (Z-score, MinMax) and persistence of preprocessing state.
"""

from pathlib import Path
from typing import List, Optional, Union

import joblib
import polars as pl
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.errors import ErrorCode, Result, Ok, Err
from utils.logger import get_logger

logger = get_logger("data.processor")

# Default columns to exclude from feature scaling (metadata)
DEFAULT_EXCLUDED_COLS = frozenset({'timestamp', 'label', 'time', 'datetime', 'Label'})


class DataProcessor:
    """
    Handles data normalization and feature scaling.
    Wraps sklearn scalers to work seamlessly with Polars DataFrames.
    """

    def __init__(self, method: str = 'standard'):
        """
        Initialize the DataProcessor.

        Args:
            method: Scaling method. Options:
                - 'standard': Z-score normalization (mean=0, std=1)
                - 'minmax': Min-max scaling (0-1 range)
        
        Raises:
            ValueError: If the specified method is not supported.
        """
        self.method = method
        self.is_fitted = False
        self.feature_columns: List[str] = []

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

    @property
    def fitted(self) -> bool:
        """Alias for is_fitted compatibility."""
        return self.is_fitted

    @property
    def columns(self) -> List[str]:
        """Alias for feature_columns compatibility."""
        return self.feature_columns

    def fit(self, df: pl.DataFrame, columns: Optional[List[str]] = None) -> Result[None]:
        """
        Compute scaling statistics from the training data.

        Args:
            df: Training data DataFrame.
            columns: List of columns to scale. If None, automatically selects 
                     all numeric columns excluding metadata.

        Returns:
            Result[None]: Ok on success, Err on validation failure.
        """
        target_columns = columns

        # Auto-select columns if not provided
        if target_columns is None:
            target_columns = [col for col in df.columns if
                col not in DEFAULT_EXCLUDED_COLS and df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]

        if not target_columns:
            logger.warning("No valid numeric columns found for scaling.")
            return Err(ErrorCode.DATA_INVALID_SCHEMA)

        self.feature_columns = target_columns

        # Select data and fit scaler
        data_matrix = df.select(self.feature_columns).to_numpy()
        self.scaler.fit(data_matrix)
        self.is_fitted = True
        logger.debug(f"Processor fitted on {len(self.feature_columns)} columns.")
        return Ok(None)

    def transform(self, df: pl.DataFrame) -> Result[pl.DataFrame]:
        """
        Apply the learned scaling to the data.

        Args:
            df: Data DataFrame to transform.

        Returns:
            Result[pl.DataFrame]: New DataFrame with scaled columns, or error.
        """
        if not self.is_fitted:
            return Err(ErrorCode.PROCESSOR_NOT_FITTED)

        # Validate schema
        missing_columns = [col for col in self.feature_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Schema mismatch. Missing columns: {missing_columns}")
            return Err(ErrorCode.PROCESSOR_MISSING_COLUMNS)

        data_matrix = df.select(self.feature_columns).to_numpy()
        scaled_matrix = self.scaler.transform(data_matrix)

        # Map transformed data back to Polars Series
        scaled_series = [pl.Series(name, scaled_matrix[:, idx]) for idx, name in enumerate(self.feature_columns)]

        # Return new DataFrame with replaced columns
        return Ok(df.with_columns(scaled_series))

    def save(self, path: Union[str, Path]) -> Result[None]:
        """
        Persist the processor state to disk.

        Args:
            path: Destination file path.
        """
        save_path = Path(path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {'scaler': self.scaler, 'columns': self.feature_columns, 'method': self.method,
            'fitted': self.is_fitted}
        joblib.dump(state, save_path)
        return Ok(None)

    def load(self, path: Union[str, Path]) -> Result[None]:
        """
        Restore the processor state from disk.

        Args:
            path: Source file path.
        """
        load_path = Path(path)

        if not load_path.exists():
            return Err(ErrorCode.PROCESSOR_LOAD_FAILED)

        state = joblib.load(load_path)
        self.scaler = state['scaler']
        self.feature_columns = state.get('columns', [])  # Handle backward compat if needed
        self.method = state.get('method', 'standard')
        self.is_fitted = state.get('fitted', False)
        return Ok(None)
