import torch
import torch.nn as nn
import torch.fft
import polars as pl
import numpy as np
from typing import Any

from pathlib import Path

from .base import BaseModel
from utils.device import get_device, to_device
from utils.logger import get_logger
from data.dataset.timeseries import TimeSeriesDataset
from torch.utils.data import DataLoader

logger = get_logger("models.sr")

class SR(BaseModel):
    def __init__(self, name: str, config: Any, input_dim: int):
        super().__init__(name, config)
        
        self.window_size = self.get_param('window_size', 64)
        self.batch_size = self.get_param('batch_size', 128)
        self.device = get_device()
        self.q = self.get_param('sr_filter_size', 3) 
        
        # Dictionary to store mean/std for each feature index
        self.stats = {} 

    def _compute_saliency(self, data: pl.DataFrame) -> np.ndarray:
        """
        Computes the Spectral Residual Saliency Map for the given data.
        Returns:
            np.ndarray: Shape (n_samples, n_features)
        """
        try:
            dataset = TimeSeriesDataset(data, self.window_size)
        except ValueError as e:
            logger.error(f"Failed to create dataset for SR: {e}")
            return np.array([])

        if len(dataset) == 0:
            return np.array([])

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        results = []
        
        with torch.no_grad():
            for batch in loader:
                x = batch[0] # (batch, window, features)
                x = to_device(x, self.device)
                
                # FFT on time dimension (dim=1)
                transf = torch.fft.fft(x, dim=1)
                real = transf.real
                imag = transf.imag
                
                mag = torch.sqrt(real**2 + imag**2)
                phase = torch.atan2(imag, real)
                
                eps = 1e-8
                log_mag = torch.log(mag + eps)
                
                # Spectral Residual
                log_mag_perm = log_mag.permute(0, 2, 1) 
                
                pad = self.q // 2
                avg_log_mag = nn.functional.avg_pool1d(
                    log_mag_perm, 
                    kernel_size=self.q, 
                    stride=1, 
                    padding=pad, 
                    count_include_pad=False
                )
                
                if avg_log_mag.shape[-1] != log_mag_perm.shape[-1]:
                     avg_log_mag = nn.functional.interpolate(avg_log_mag, size=log_mag_perm.shape[-1])

                avg_log_mag = avg_log_mag.permute(0, 2, 1)
                
                spectral_residual = log_mag - avg_log_mag
                
                sr_exp = torch.exp(spectral_residual)
                real_new = sr_exp * torch.cos(phase)
                imag_new = sr_exp * torch.sin(phase)
                
                complex_new = torch.complex(real_new, imag_new)
                saliency = torch.fft.ifft(complex_new, dim=1)
                
                # Reconstruction error (Saliency Map)
                # (batch, window, features)
                # We want the score for the last point in the window? 
                # Or the average over the window? 
                # SR usually highlights anomalies in the window. 
                # Let's take the squared magnitude.
                rec_squared = torch.abs(saliency)**2
                
                # For per-point score, we typically take the last point or average.
                # Taking the mean over the window is robust.
                batch_scores = torch.mean(rec_squared, dim=1) # (batch, features)
                
                results.append(batch_scores.cpu().numpy())
                
        if not results:
            return np.array([])
            
        return np.concatenate(results, axis=0) # (n_samples, n_features)

    def fit(self, train_data: pl.DataFrame):
        logger.info(f"Training SR model {self.name} (learning stats)...")
        
        saliency_map = self._compute_saliency(train_data)
        if len(saliency_map) == 0:
            logger.warning("SR produced empty saliency map during fit.")
            return

        # Learn mean/std per feature
        # saliency_map shape: (n_samples, n_features)
        n_features = saliency_map.shape[1]
        
        self.stats = {}
        for i in range(n_features):
            col_scores = saliency_map[:, i]
            mean = float(np.mean(col_scores))
            std = float(np.std(col_scores))
            if std == 0: std = 1e-6
            self.stats[i] = (mean, std)
            
        logger.info(f"SR stats learned for {n_features} features.")

    def save(self, path: str):
        import torch
        state = {
            "model": "SR",
            "stats": self.stats
        }
        torch.save(state, path)

    def load(self, path: str):
        import torch
        if not Path(path).exists():
            raise FileNotFoundError(f"SR model file not found at {path}")
        # PyTorch 2.6+ requires weights_only=False for loading dicts with numpy types
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.stats = state.get("stats", {})

    def predict(self, data: pl.DataFrame) -> np.ndarray:
        saliency_map = self._compute_saliency(data)
        if len(saliency_map) == 0:
            return np.array([])
            
        # Normalize per feature
        n_features = saliency_map.shape[1]
        norm_scores_list = []
        
        for i in range(n_features):
            mean, std = self.stats.get(i, (0.0, 1.0))
            # Z-score
            z = (saliency_map[:, i] - mean) / std
            norm_scores_list.append(z)
            
        # Stack: (n_features, n_samples) -> (n_samples, n_features)
        norm_scores = np.stack(norm_scores_list, axis=1)
        
        # Aggregate across features to get single anomaly score
        # Mean Z-score across features
        return np.mean(norm_scores, axis=1)

    def get_contribution(self, data: pl.DataFrame) -> np.ndarray:
        """
        Returns normalized Z-scores per feature.
        """
        saliency_map = self._compute_saliency(data)
        if len(saliency_map) == 0:
            return np.array([])

        n_features = saliency_map.shape[1]
        norm_scores_list = []
        
        for i in range(n_features):
            mean, std = self.stats.get(i, (0.0, 1.0))
            z = (saliency_map[:, i] - mean) / std
            norm_scores_list.append(z)
            
        return np.stack(norm_scores_list, axis=1)
