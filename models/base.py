from abc import ABC, abstractmethod
import polars as pl
import numpy as np
import torch
from pathlib import Path
from typing import Any, Optional, Union

class BaseModel(ABC):
    def __init__(self, name: str, config: Any):
        self.name = name
        self.config = config
        self.model = None

    def get_param(self, key: str, default: Any = None) -> Any:
        """Helper to get config parameter from dict or object."""
        # if config is dict
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        # if config is object with .get() method (like ModelsConfig)
        if hasattr(self.config, 'get'):
            return self.config.get(key, default)
        # fallback to attribute access
        return getattr(self.config, key, default)

    @abstractmethod
    def fit(self, train_data: Union[pl.DataFrame, Any]) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, data: Union[pl.DataFrame, Any]) -> np.ndarray:
        """Predict anomalies.
        
        Returns:
            np.ndarray: Anomaly scores for each sample.
        """
        pass

    @abstractmethod
    def get_contribution(self, data: Union[pl.DataFrame, Any]) -> np.ndarray:
        """Get anomaly contribution per feature.
        
        Returns:
            np.ndarray: Shape (n_samples, n_features), Z-score normalized contributions.
        """
        pass

    def save(self, path: str) -> None:
        """Save model state dictionary to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
             torch.save(self.model.state_dict(), path)
        else:
            raise ValueError("Model is not initialized or trained.")

    def load(self, path: str) -> None:
        """Load model state dictionary from path."""
        path = Path(path)
        if path.exists():
            if self.model is None:
                raise ValueError("Model must be initialized before loading weights.")
            
            # Determine device from model parameters
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = 'cpu'
                
            try:
                self.model.load_state_dict(torch.load(path, map_location=device))
            except Exception as e:
                raise RuntimeError(f"Failed to load model weights from {path}: {e}")
        else:
            raise FileNotFoundError(f"Model file not found at {path}")
