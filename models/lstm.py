from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset.timeseries import TimeSeriesDataset
from utils.errors import Result, Ok, Err, ErrorCode
from utils.logger import get_logger
from .base import BaseModel

logger = get_logger("models.lstm")


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0)
        # We might need a linear layer to map back to input dim exactly if decoder output is hidden dim
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        enc_out, _ = self.encoder(x)
        dec_out, _ = self.decoder(enc_out)
        return dec_out


class LSTM(BaseModel):
    def __init__(self, name: str, config: Any, input_dim: int):
        super().__init__(name, config)

        self.window_size = self.get_param('window_size', 64)
        self.hidden_dim = self.get_param('lstm_hidden_dim', 32)
        self.num_layers = self.get_param('lstm_layers', 1)
        self.epochs = self.get_param('epochs', 10)
        self.batch_size = self.get_param('batch_size', 128)
        self.error_check_window = int(self.get_param('lstm_error_check_window', 5))
        self.device = 'cpu'

        self.model = LSTMAutoencoder(input_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        self.criterion = nn.MSELoss()

    def fit(self, train_data: pl.DataFrame) -> Result[None]:
        logger.info(f"Training LSTM model {self.name}...")
        # FIXME: ValueError will bubble up
        dataset = TimeSeriesDataset(train_data, self.window_size)

        if len(dataset) == 0:
            logger.warning("LSTM Dataset is empty.")
            return Ok(None)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model.train()
        for epoch in range(self.epochs):
            losses = []
            for batch in train_loader:
                batch_data = batch[0]  # (batch, window, features)
                batch_data = batch_data.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(batch_data)
                loss = self.criterion(output, batch_data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                losses.append(loss.item())

            avg_loss = float(np.mean(losses))
            self.scheduler.step(avg_loss)

            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.epochs:
                logger.info(f"LSTM Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")

        # Compute feature stats
        self._compute_feature_stats(dataset)
        return Ok(None)

    def _compute_feature_stats(self, dataset):
        logger.info(f"Computing feature error statistics for {self.name}...")
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_contribs = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                batch_data = batch_data.to(self.device)

                output = self.model(batch_data)
                # Reconstruction error per feature (averaged over window)
                diff = (batch_data - output) ** 2
                if self.error_check_window > 0 and self.error_check_window < self.window_size:
                    diff = diff[:, -self.error_check_window:, :]
                
                error = torch.mean(diff, dim=1)
                all_contribs.append(error.cpu().numpy())

        if all_contribs:
            all_contribs = np.concatenate(all_contribs, axis=0)  # (n_samples, n_features)

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

    def save(self, path: str) -> Result[None]:
        # FIXME: Exception will bubble up
        state = {"model_state": self.model.state_dict(), "feature_stats": getattr(self, 'feature_stats', {})}
        torch.save(state, path)
        return Ok(None)

    def load(self, path: str) -> Result[None]:
        if not Path(path).exists():
            return Err(ErrorCode.MODEL_NOT_FOUND)

        # FIXME: Exception will bubble up
        # PyTorch 2.6+ requires weights_only=False for loading dicts with numpy types
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state" in state:
            self.model.load_state_dict(state["model_state"])
            self.feature_stats = state.get("feature_stats", {})
            self.error_check_window = state.get("error_check_window", self.error_check_window)
        else:
            self.model.load_state_dict(state)
            self.feature_stats = {}
        return Ok(None)

    def predict(self, data: pl.DataFrame) -> Result[np.ndarray]:
        self.model.eval()
        # FIXME: ValueError will bubble up
        dataset = TimeSeriesDataset(data, self.window_size)

        if len(dataset) == 0:
            return Ok(np.array([]))

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        results = []

        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                batch_data = batch_data.to(self.device)

                output = self.model(batch_data)

                # Reconstruction error per sample (averaged over window)
                # output: (batch, window, feature)
                # error: (batch, window, feature)
                diff = (batch_data - output) ** 2
                if self.error_check_window > 0 and self.error_check_window < self.window_size:
                    diff = diff[:, -self.error_check_window:, :]

                error = torch.mean(diff, dim=[1, 2])
                results.append(error.cpu().numpy())

        if not results:
            return Ok(np.array([]))

        return Ok(np.concatenate(results))

    def predict_and_contribute(self, data: pl.DataFrame) -> Result[tuple]:
        """Single forward pass returning both scores and contributions."""
        self.model.eval()
        dataset = TimeSeriesDataset(data, self.window_size)

        if len(dataset) == 0:
            return Ok((np.array([]), np.array([])))

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        score_results = []
        contrib_results = []
        feature_stats = getattr(self, 'feature_stats', {})

        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0].to(self.device)

                # Single forward pass
                output = self.model(batch_data)

                diff = (batch_data - output) ** 2
                if self.error_check_window > 0 and self.error_check_window < self.window_size:
                    diff = diff[:, -self.error_check_window:, :]

                # Scores: mean over window and features
                score_results.append(torch.mean(diff, dim=[1, 2]).cpu().numpy())

                # Contributions: mean over window only → (batch, features)
                error = torch.mean(diff, dim=1)
                if feature_stats:
                    error_np = error.cpu().numpy()
                    norm_list = []
                    for i in range(error_np.shape[1]):
                        mean, std = feature_stats.get(i, (0.0, 1.0))
                        norm_list.append((error_np[:, i] - mean) / std)
                    contrib_results.append(np.stack(norm_list, axis=1))
                else:
                    contrib_results.append(error.cpu().numpy())

        if not score_results:
            return Ok((np.array([]), np.array([])))

        return Ok((np.concatenate(score_results), np.concatenate(contrib_results)))

    def get_contribution(self, data: pl.DataFrame) -> Result[np.ndarray]:
        self.model.eval()
        # FIXME: ValueError will bubble up
        dataset = TimeSeriesDataset(data, self.window_size)

        if len(dataset) == 0:
            return Ok(np.array([]))

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        results = []

        feature_stats = getattr(self, 'feature_stats', {})

        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                batch_data = batch_data.to(self.device)

                output = self.model(batch_data)

                # Reconstruction error per feature (averaged over window)
                # (batch, window, feature) -> mean(dim=1) -> (batch, feature)
                diff = (batch_data - output) ** 2
                if self.error_check_window > 0 and self.error_check_window < self.window_size:
                    diff = diff[:, -self.error_check_window:, :]
                
                error = torch.mean(diff, dim=1)

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
            return Ok(np.array([]))

        return Ok(np.concatenate(results))
