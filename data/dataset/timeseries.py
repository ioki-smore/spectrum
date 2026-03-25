import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from utils.logger import get_logger

logger = get_logger("data.dataset.timeseries")


class TimeSeriesDataset(Dataset):
    def __init__(self, data: pl.DataFrame, window_size: int, step: int = 1):
        """
        Args:
            data: Polars DataFrame containing the time series data.
                  Assumes all columns except 'timestamp', 'label', 'time' are features.
            window_size: Length of the sliding window.
            step: Stride for the sliding window.
        """
        self.window_size = window_size
        self.step = step

        # Select feature columns
        feature_cols = [c for c in data.columns if c not in ['timestamp', 'label', 'time']]
        if not feature_cols:
            raise ValueError("No feature columns found in dataframe.")

        # FIXME: Exception will bubble up
        self.values = data.select(feature_cols).to_numpy()

        # Check for NaNs or Infs
        if np.isnan(self.values).any() or np.isinf(self.values).any():
            # Option: Fill with 0 or raise.
            # Since processor should handle this, raising suggests a pipeline failure.
            raise ValueError("Dataset contains NaNs or Infs after processing.")

        # Keep labels if present, otherwise None
        if 'label' in data.columns:
            # FIXME: Exception will bubble up
            self.labels = data['label'].to_numpy()
        else:
            self.labels = None

        self.n_samples = len(self.values)

        if self.n_samples < window_size:
            self.indices = []
        else:
            self.indices = list(range(0, self.n_samples - window_size + 1, step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size

        window_data = self.values[start:end]

        # Convert to float32 tensor
        # Ensure array is writable to avoid PyTorch warning
        if not window_data.flags.writeable:
            window_data = window_data.copy()

        window_tensor = torch.from_numpy(window_data).float()

        if self.labels is not None:
            # Return last label in window or majority? usually for anomaly detection 
            # we might want the label of the last point or the whole window.
            # Let's return the window labels for now.
            window_labels = self.labels[start:end]
            if not window_labels.flags.writeable:
                window_labels = window_labels.copy()
            return window_tensor, torch.from_numpy(window_labels)

        # If no labels, just return data (for unsupervised training/inference)
        return [window_tensor]
