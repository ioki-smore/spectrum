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

logger = get_logger("models.usad")


class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w


class USADModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        # Check for numerical instability
        if torch.isnan(loss1) or torch.isnan(loss2):
            raise RuntimeError("NaN loss detected during training step.")
        return loss1, loss2

    def forward(self, batch):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3


class USAD(BaseModel):
    def __init__(self, name: str, config: Any, input_dim: int):
        super().__init__(name, config)

        self.window_size = self.get_param('window_size', 64)
        self.w_size = input_dim * self.window_size

        self.feature_dim = input_dim
        self.z_size = self.get_param('latent_size', 10)
        self.epochs = self.get_param('epochs', 10)
        self.batch_size = self.get_param('batch_size', 128)
        self.error_check_window = int(self.get_param('usad_error_check_window', 0))
        self.device = 'cpu'

        # Calculate flattened input size
        self.input_size = self.window_size * self.feature_dim

        self.model = USADModel(self.input_size, self.z_size).to(self.device)
        self.optimizer1 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()))
        self.optimizer2 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder2.parameters()))

    def fit(self, train_data: pl.DataFrame) -> Result[None]:
        logger.info(f"Training USAD model {self.name}...")
        # FIXME: ValueError will bubble up
        dataset = TimeSeriesDataset(train_data, self.window_size)

        if len(dataset) == 0:
            logger.warning("Dataset is empty after windowing. Increase data size or decrease window size.")
            return Ok(None)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model.train()
        for epoch in range(self.epochs):
            losses1 = []
            losses2 = []
            for batch in train_loader:
                # batch is a list [window_tensor, (optional labels)]
                # window_tensor shape: (batch, window, features)
                batch_data = batch[0]
                batch_data = batch_data.view(batch_data.size(0), -1)  # Flatten
                batch_data = batch_data.to(self.device)

                # Train AE1
                # FIXME: RuntimeError will bubble up
                # Note: Two forward passes are intentional in USAD — Phase 2
                # needs the encoder already updated by Phase 1's gradient step.
                self.optimizer1.zero_grad()
                loss1, _ = self.model.training_step(batch_data, epoch + 1)
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer1.step()

                # Train AE2 (encoder now updated by Phase 1)
                self.optimizer2.zero_grad()
                _, loss2 = self.model.training_step(batch_data, epoch + 1)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer2.step()

                losses1.append(loss1.item())
                losses2.append(loss2.item())

            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.epochs:
                logger.info(
                    f"Epoch [{epoch + 1}/{self.epochs}], Loss1: {np.mean(losses1):.4f}, Loss2: {np.mean(losses2):.4f}")

        # Compute and store per-feature error statistics (Mean/Std) on training data
        # This is needed for Z-score normalization in get_contribution
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
                batch_data_view = batch_data.view(batch_data.size(0), -1)
                batch_data_view = batch_data_view.to(self.device)

                w1 = self.model.decoder1(self.model.encoder(batch_data_view))
                w2 = self.model.decoder2(self.model.encoder(w1))

                diff1 = (batch_data_view - w1) ** 2
                diff2 = (batch_data_view - w2) ** 2
                diff = 0.5 * diff1 + 0.5 * diff2

                diff_reshaped = diff.view(batch_data.size(0), self.window_size, self.feature_dim)
                if self.error_check_window > 0 and self.error_check_window < self.window_size:
                    diff_reshaped = diff_reshaped[:, -self.error_check_window:, :]
                contrib = torch.mean(diff_reshaped, dim=1)  # (batch, features)
                all_contribs.append(contrib.cpu().numpy())

        if all_contribs:
            all_contribs = np.concatenate(all_contribs, axis=0)  # (n_samples, n_features)

            # Learn mean/std per feature
            self.feature_stats = {}
            for i in range(self.feature_dim):
                vals = all_contribs[:, i]
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                if std == 0: std = 1e-6
                self.feature_stats[i] = (mean, std)
            logger.info(f"USAD feature stats computed for {self.feature_dim} features.")
        else:
            logger.warning("No data for computing feature stats.")
            self.feature_stats = {}

    def save(self, path: str) -> Result[None]:
        # Extend save to include feature_stats
        import torch
        # FIXME: Exception will bubble up
        state = {"model_state": self.model.state_dict(), "feature_stats": getattr(self, 'feature_stats', {}),
            "error_check_window": self.error_check_window}
        torch.save(state, path)
        return Ok(None)

    def load(self, path: str) -> Result[None]:
        import torch
        if not Path(path).exists():
            return Err(ErrorCode.MODEL_NOT_FOUND)

        # FIXME: Exception will bubble up
        # PyTorch 2.6+ requires weights_only=False for loading dicts with numpy types
        state = torch.load(path, map_location=self.device, weights_only=False)
        # Check if it's new format with feature_stats
        if "model_state" in state:
            self.model.load_state_dict(state["model_state"])
            self.feature_stats = state.get("feature_stats", {})
            self.error_check_window = state.get("error_check_window", self.error_check_window)
        else:
            # Old format (just state dict)
            self.model.load_state_dict(state)
            self.feature_stats = {}
        return Ok(None)

    def predict(self, data: pl.DataFrame, alpha: float = 0.5, beta: float = 0.5) -> Result[np.ndarray]:
        self.model.eval()
        # FIXME: ValueError will bubble up
        dataset = TimeSeriesDataset(data, self.window_size)

        if len(dataset) == 0:
            logger.warning("Data too short for prediction window.")
            return Ok(np.array([]))

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        results = []
        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                batch_data = batch_data.view(batch_data.size(0), -1)
                batch_data = batch_data.to(self.device)

                w1 = self.model.decoder1(self.model.encoder(batch_data))
                w2 = self.model.decoder2(self.model.encoder(w1))

                # Anomaly score
                diff1 = (batch_data - w1) ** 2
                diff2 = (batch_data - w2) ** 2

                if self.error_check_window > 0 and self.error_check_window < self.window_size:
                    # Only check last N steps for error (reduces trailing FPs)
                    diff1 = diff1.view(batch_data.size(0), self.window_size, self.feature_dim)
                    diff2 = diff2.view(batch_data.size(0), self.window_size, self.feature_dim)
                    diff1 = diff1[:, -self.error_check_window:, :]
                    diff2 = diff2[:, -self.error_check_window:, :]
                    diff1 = diff1.reshape(batch_data.size(0), -1)
                    diff2 = diff2.reshape(batch_data.size(0), -1)

                score = alpha * torch.mean(diff1, dim=1) + beta * torch.mean(diff2, dim=1)
                results.append(score.cpu().numpy())

        if not results:
            return Ok(np.array([]))

        return Ok(np.concatenate(results))

    def predict_and_contribute(self, data: pl.DataFrame, alpha: float = 0.5, beta: float = 0.5) -> Result[tuple]:
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
                batch_data = batch[0]
                batch_flat = batch_data.view(batch_data.size(0), -1).to(self.device)

                # Single shared forward pass
                encoded = self.model.encoder(batch_flat)
                w1 = self.model.decoder1(encoded)
                w2 = self.model.decoder2(self.model.encoder(w1))

                diff1 = (batch_flat - w1) ** 2
                diff2 = (batch_flat - w2) ** 2

                # --- Scores ---
                s_diff1, s_diff2 = diff1, diff2
                if self.error_check_window > 0 and self.error_check_window < self.window_size:
                    s_diff1 = s_diff1.view(batch_data.size(0), self.window_size, self.feature_dim)
                    s_diff2 = s_diff2.view(batch_data.size(0), self.window_size, self.feature_dim)
                    s_diff1 = s_diff1[:, -self.error_check_window:, :].reshape(batch_data.size(0), -1)
                    s_diff2 = s_diff2[:, -self.error_check_window:, :].reshape(batch_data.size(0), -1)
                score = alpha * torch.mean(s_diff1, dim=1) + beta * torch.mean(s_diff2, dim=1)
                score_results.append(score.cpu().numpy())

                # --- Contributions ---
                diff = 0.5 * diff1 + 0.5 * diff2
                diff_reshaped = diff.view(batch_data.size(0), self.window_size, self.feature_dim)
                if self.error_check_window > 0 and self.error_check_window < self.window_size:
                    diff_reshaped = diff_reshaped[:, -self.error_check_window:, :]
                contrib = torch.mean(diff_reshaped, dim=1)

                if feature_stats:
                    contrib_np = contrib.cpu().numpy()
                    norm_list = []
                    for i in range(self.feature_dim):
                        mean, std = feature_stats.get(i, (0.0, 1.0))
                        norm_list.append((contrib_np[:, i] - mean) / std)
                    contrib_results.append(np.stack(norm_list, axis=1))
                else:
                    contrib_results.append(contrib.cpu().numpy())

        if not score_results:
            return Ok((np.array([]), np.array([])))

        return Ok((np.concatenate(score_results), np.concatenate(contrib_results)))

    def get_contribution(self, data: pl.DataFrame) -> Result[np.ndarray]:
        """
        Calculates reconstruction error contribution per feature.
        Returns Z-scores based on learned stats.
        """
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
                batch_data_view = batch_data.view(batch_data.size(0), -1)
                batch_data_view = batch_data_view.to(self.device)

                w1 = self.model.decoder1(self.model.encoder(batch_data_view))
                w2 = self.model.decoder2(self.model.encoder(w1))

                diff1 = (batch_data_view - w1) ** 2
                diff2 = (batch_data_view - w2) ** 2
                diff = 0.5 * diff1 + 0.5 * diff2

                diff_reshaped = diff.view(batch_data.size(0), self.window_size, self.feature_dim)
                if self.error_check_window > 0 and self.error_check_window < self.window_size:
                    diff_reshaped = diff_reshaped[:, -self.error_check_window:, :]
                contrib = torch.mean(diff_reshaped, dim=1)  # (batch, features)

                # Normalize to Z-scores if stats are available
                if feature_stats:
                    contrib_np = contrib.cpu().numpy()
                    norm_contrib_list = []
                    for i in range(self.feature_dim):
                        mean, std = feature_stats.get(i, (0.0, 1.0))
                        # Z-score
                        z = (contrib_np[:, i] - mean) / std
                        norm_contrib_list.append(z)

                    norm_contrib = np.stack(norm_contrib_list, axis=1)
                    results.append(norm_contrib)
                else:
                    # Fallback to raw error if no stats (shouldn't happen if trained)
                    results.append(contrib.cpu().numpy())

        if not results:
            return Ok(np.array([]))

        return Ok(np.concatenate(results))
