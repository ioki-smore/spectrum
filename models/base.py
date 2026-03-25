"""
Base abstract class for all anomaly detection models.
Defines the standard interface for training, inference, and persistence.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import numpy as np
import polars as pl
import torch

from utils.errors import Result, Ok, Err, ErrorCode


class BaseModel(ABC):
    """
    Abstract base class for anomaly detection models.
    
    Attributes:
        name (str): Unique name for the model instance.
        config (Any): Configuration object or dictionary containing model parameters.
        model (Any): The underlying model object (e.g., PyTorch module), if applicable.
    """

    def __init__(self, name: str, config: Any):
        self.name = name
        self.config = config
        self.model = None

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration parameter.
        Supports both dictionary access and object attribute access.
        
        Args:
            key: Parameter name.
            default: Default value if not found.
            
        Returns:
            The parameter value.
        """
        # Case 1: Config is a dictionary
        if isinstance(self.config, dict):
            return self.config.get(key, default)

        # Case 2: Config is an object with a .get() method (e.g., ModelsConfig)
        if hasattr(self.config, 'get'):
            return self.config.get(key, default)

        # Case 3: Config is a standard object (attribute access)
        return getattr(self.config, key, default)

    @abstractmethod
    def fit(self, train_data: Union[pl.DataFrame, Any]) -> Result[None]:
        """
        Train the model on the provided data.
        
        Args:
            train_data: Training data, typically a Polars DataFrame or numpy array.
            
        Returns:
            Result[None]: Ok(None) on success, Err on failure.
        """
        pass

    @abstractmethod
    def predict(self, data: Union[pl.DataFrame, Any]) -> Result[np.ndarray]:
        """
        Generate anomaly scores for the provided data.
        
        Args:
            data: Input data for inference.
            
        Returns:
            Result[np.ndarray]: Array of anomaly scores (higher usually means more anomalous).
        """
        pass

    @abstractmethod
    def get_contribution(self, data: Union[pl.DataFrame, Any]) -> Result[np.ndarray]:
        """
        Calculate feature-wise anomaly contributions.
        
        Args:
            data: Input data.
            
        Returns:
            Result[np.ndarray]: Matrix of shape (n_samples, n_features) representing 
                                the contribution of each feature to the anomaly score.
        """
        pass

    def predict_and_contribute(self, data: Union[pl.DataFrame, Any]) -> Result[tuple]:
        """
        Single-pass inference returning both anomaly scores and feature contributions.
        
        Default implementation calls predict() and get_contribution() separately.
        Sub-models should override this to share a single forward pass for efficiency.
        
        Returns:
            Result[tuple]: (scores: np.ndarray, contributions: np.ndarray)
        """
        res = self.predict(data)
        if res.is_err():
            return res
        scores = res.unwrap()

        cres = self.get_contribution(data)
        if cres.is_err():
            return cres
        contributions = cres.unwrap()

        return Ok((scores, contributions))

    def save(self, path: Union[str, Path]) -> Result[None]:
        """
        Persist the model state to disk.
        Default implementation saves `self.model.state_dict()` using torch.
        
        Args:
            path: Destination file path.
        """
        save_path = Path(path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            torch.save(self.model.state_dict(), save_path)
            return Ok(None)
        else:
            return Err(ErrorCode.MODEL_NOT_TRAINED)

    def load(self, path: Union[str, Path]) -> Result[None]:
        """
        Restore the model state from disk.
        Default implementation loads state dict using torch.
        
        Args:
            path: Source file path.
        """
        load_path = Path(path)

        if not load_path.exists():
            return Err(ErrorCode.MODEL_NOT_FOUND)

        if self.model is None:
            return Err(ErrorCode.MODEL_NOT_READY)

        state_dict = torch.load(load_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        return Ok(None)
