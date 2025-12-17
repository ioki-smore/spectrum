import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import polars as pl
import numpy as np
from typing import Any

from .base import BaseModel
from utils.device import get_device, to_device
from utils.logger import get_logger
from data.dataset.timeseries import TimeSeriesDataset

logger = get_logger("models.lstm")

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0
        )
        self.decoder = nn.LSTM(
            hidden_dim, input_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0
        )
        # We might need a linear layer to map back to input dim exactly if decoder output is hidden dim
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # Encoder
        # _, (hidden, cell) = self.encoder(x)
        # We can use the output of encoder or just the hidden state. 
        # For reconstruction, we often use the full sequence.
        enc_out, (hidden, cell) = self.encoder(x)
        
        # Decoder
        # To reconstruct the sequence, we can feed enc_out or start with hidden.
        # Simple autoencoder: predict sequence from encoded representation.
        # But standard LSTM AE often mirrors the input.
        # Let's simple pass enc_out to decoder (which acts as a reconstructor here)
        # Note: input_dim of decoder must match hidden_dim of encoder if we feed enc_out.
        # Wait, usually Encoder maps Input -> Hidden. Decoder maps Hidden -> Input.
        
        # Simpler architecture:
        # Encoder: Input -> Hidden
        # Decoder: Hidden -> Reconstructed Input (step by step or all at once)
        
        # Let's use a symmetric architecture where input_dim -> hidden -> input_dim
        # But using LSTM for both.
        
        # Actually, for anomaly detection, a simple LSTM prediction or reconstruction is used.
        # Let's stick to reconstruction.
        
        # Encoder
        enc_out, _ = self.encoder(x)
        
        # Decoder - we want to map back to input space.
        # If we use LSTM as decoder, it expects input size matching its defined input_dim.
        # If we defined decoder input_dim as hidden_dim, we can feed enc_out.
        
        dec_out, _ = self.decoder(enc_out)
        
        # Map to original dimension
        # The decoder output is (batch, seq, input_dim) if we set it right.
        # My init: decoder(hidden_dim, input_dim) -> expects input of size hidden_dim.
        # enc_out is (batch, seq, hidden_dim). So this matches.
        
        return dec_out

class LSTM(BaseModel):
    def __init__(self, name: str, config: Any, input_dim: int):
        super().__init__(name, config)
        
        self.window_size = self.get_param('window_size', 64)
        self.hidden_dim = self.get_param('lstm_hidden_dim', 32)
        self.num_layers = self.get_param('lstm_layers', 1)
        self.epochs = self.get_param('epochs', 10)
        self.batch_size = self.get_param('batch_size', 128)
        self.device = get_device()
        
        self.model = LSTMAutoencoder(input_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def fit(self, train_data: pl.DataFrame):
        logger.info(f"Training LSTM model {self.name}...")
        try:
            dataset = TimeSeriesDataset(train_data, self.window_size)
        except ValueError as e:
            logger.error(f"Failed to create dataset for LSTM: {e}")
            return

        if len(dataset) == 0:
            logger.warning("LSTM Dataset is empty.")
            return

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            losses = []
            for batch in train_loader:
                batch_data = batch[0] # (batch, window, features)
                batch_data = to_device(batch_data, self.device)
                
                self.optimizer.zero_grad()
                output = self.model(batch_data)
                loss = self.criterion(output, batch_data)
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
            
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.epochs:
                logger.info(f"LSTM Epoch [{epoch+1}/{self.epochs}], Loss: {np.mean(losses):.4f}")

        # Compute feature stats
        self._compute_feature_stats(dataset)

    def _compute_feature_stats(self, dataset):
        logger.info(f"Computing feature error statistics for {self.name}...")
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_contribs = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                batch_data = to_device(batch_data, self.device)
                
                output = self.model(batch_data)
                # Reconstruction error per feature (averaged over window)
                error = torch.mean((batch_data - output) ** 2, dim=1)
                all_contribs.append(error.cpu().numpy())
        
        if all_contribs:
            all_contribs = np.concatenate(all_contribs, axis=0) # (n_samples, n_features)
            
            self.feature_stats = {}
            for i in range(all_contribs.shape[1]):
                vals = all_contribs[:, i]
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                if std == 0: std = 1e-6
                self.feature_stats[i] = (mean, std)
            logger.info(f"LSTM feature stats computed.")
        else:
            self.feature_stats = {}

    def save(self, path: str):
        import torch
        state = {
            "model_state": self.model.state_dict(),
            "feature_stats": getattr(self, 'feature_stats', {})
        }
        torch.save(state, path)

    def load(self, path: str):
        import torch
        from pathlib import Path
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found at {path}")
            
        try:
            # PyTorch 2.6+ requires weights_only=False for loading dicts with numpy types
            state = torch.load(path, map_location=self.device, weights_only=False)
            if "model_state" in state:
                self.model.load_state_dict(state["model_state"])
                self.feature_stats = state.get("feature_stats", {})
            else:
                self.model.load_state_dict(state)
                self.feature_stats = {}
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")

    def predict(self, data: pl.DataFrame) -> np.ndarray:
        self.model.eval()
        try:
            dataset = TimeSeriesDataset(data, self.window_size)
        except ValueError as e:
            logger.error(f"Failed to create dataset for LSTM predict: {e}")
            return np.array([])

        if len(dataset) == 0:
            return np.array([])

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        results = []
        
        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                batch_data = to_device(batch_data, self.device)
                
                output = self.model(batch_data)
                
                # Reconstruction error per sample (averaged over window)
                # output: (batch, window, feature)
                # error: (batch, window, feature)
                error = torch.mean((batch_data - output) ** 2, dim=[1, 2])
                results.append(error.cpu().numpy())
                
        if not results:
            return np.array([])
            
        return np.concatenate(results)

    def get_contribution(self, data: pl.DataFrame) -> np.ndarray:
        self.model.eval()
        try:
            dataset = TimeSeriesDataset(data, self.window_size)
        except ValueError:
            return np.array([])

        if len(dataset) == 0:
            return np.array([])

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        results = []
        
        feature_stats = getattr(self, 'feature_stats', {})
        
        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                batch_data = to_device(batch_data, self.device)
                
                output = self.model(batch_data)
                
                # Reconstruction error per feature (averaged over window)
                # (batch, window, feature) -> mean(dim=1) -> (batch, feature)
                error = torch.mean((batch_data - output) ** 2, dim=1)
                
                if feature_stats:
                    error_np = error.cpu().numpy()
                    norm_list = []
                    for i in range(error_np.shape[1]):
                        mean, std = feature_stats.get(i, (0.0, 1.0))
                        z = (error_np[:, i] - mean) / std
                        norm_list.append(z)
                    results.append(np.stack(norm_list, axis=1))
                else:
                    results.append(error.cpu().numpy())
                
        if not results:
            return np.array([])
            
        return np.concatenate(results)
