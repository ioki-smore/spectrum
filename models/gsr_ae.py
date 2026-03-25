from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset.timeseries import TimeSeriesDataset
from utils.errors import Result, Ok, Err, ErrorCode
from utils.logger import get_logger
from .base import BaseModel

logger = get_logger("models.gsr_ae")


class _CNNAE(nn.Module):
    def __init__(self, window_size: int, latent_dim: int):
        super().__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim

        # Improved Encoder with more capacity and batch normalization
        self.enc_conv = nn.Sequential(nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(32),
            nn.ReLU(), nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(), )

        # Compute shape after convs dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, window_size)
            out = self.enc_conv(dummy)
            self.conv_out_size = out.numel()
            self.conv_out_shape = out.shape[1:]

        self.enc_linear = nn.Sequential(nn.Linear(self.conv_out_size, latent_dim * 2), nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim))
        self.dec_linear = nn.Sequential(nn.Linear(latent_dim, latent_dim * 2), nn.ReLU(),
            nn.Linear(latent_dim * 2, self.conv_out_size))

        # Improved Decoder
        self.dec_conv = nn.Sequential(nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm1d(32),
            nn.ReLU(), nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1), )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input is (Batch, Window). Needs (Batch, Channel=1, Window)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Encoder
        x = self.enc_conv(x)
        b, c, w = x.shape
        x = x.view(b, -1)
        z = self.enc_linear(x)

        # Decoder
        x = self.dec_linear(z)
        x = x.view(b, self.conv_out_shape[0], self.conv_out_shape[1])
        recon = self.dec_conv(x)

        # Handle shape mismatch due to padding/strides
        if recon.shape[2] != self.window_size:
            recon = nn.functional.interpolate(recon, size=self.window_size, mode='linear', align_corners=False)

        return recon.squeeze(1)


class GSR_AE(BaseModel):
    def __init__(self, name: str, config: Any, input_dim: int):
        super().__init__(name, config)

        self.window_size = int(self.get_param("gsr_ae_window_size", self.get_param("window_size", 64)))
        self.batch_size = int(self.get_param("batch_size", 128))
        self.epochs = int(self.get_param("gsr_ae_epochs", self.get_param("epochs", 20)))

        self.latent_dim = int(self.get_param("gsr_ae_latent_dim", 16))
        self.learning_rate = float(self.get_param("gsr_ae_lr", 1e-3))
        self.sigma = float(self.get_param("gsr_ae_sigma", 2.0))
        self.c = float(self.get_param("gsr_ae_c", 3.0))
        self.normalize = True  # Always normalize for stability
        self.use_residual_score = bool(self.get_param("gsr_ae_use_residual", True))
        self.smoothing_window = int(self.get_param("gsr_ae_smoothing_window", 3))
        self.amp_threshold = float(self.get_param("gsr_ae_amp_threshold", 5.0))
        self.th_lo_q = float(self.get_param("gsr_ae_th_lo_q", 0.15))
        self.suppression_factor = float(self.get_param("gsr_ae_suppression_factor", 0.1))
        self.score_floor = float(self.get_param("gsr_ae_score_floor", 0.0))
        self.amp_check_window = int(self.get_param("gsr_ae_amp_check_window", 5))

        # indices to check for high amplitude (bursts)
        # Defaults to all if None/Empty, but allows filtering out drifting features
        self.amp_feature_indices = self.get_param("gsr_ae_amp_feature_indices", None)

        self.feature_dim = int(input_dim)
        self.model = _CNNAE(self.window_size, self.latent_dim)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        # Paper: "minimizes the mean squared error" (Line 788)
        self.criterion = nn.MSELoss()

        self.feature_stats = {}

        self.alpha: Optional[float] = None
        self.beta: Optional[float] = None
        self.tau: Optional[float] = None
        self.th_lo: Optional[float] = None
        self.th_hi: Optional[float] = None

    def _saliency_flat(self, batch_x: torch.Tensor) -> torch.Tensor:
        saliency = self._compute_gsr_saliency(batch_x)
        b, w, f = saliency.shape
        flat = saliency.permute(0, 2, 1).contiguous().view(b * f, w)

        # Apply per-feature normalization for better sensitivity
        # Normalize each feature independently to preserve relative anomaly magnitudes
        # We need to map flat back to (b, f, w) to normalize per feature
        # flat is (b*f, w) where the order is feature-major per batch? 
        # Actually in view(b*f, w), it iterates b then f.
        # So every f rows correspond to one batch sample.

        # Reshape to (b, f, w) for normalization
        flat_reshaped = flat.view(b, f, w)

        eps = 1e-8
        # Min-Max normalization per feature per window
        feat_min = flat_reshaped.min(dim=2, keepdim=True)[0]
        feat_max = flat_reshaped.max(dim=2, keepdim=True)[0]
        flat_reshaped = (flat_reshaped - feat_min) / (feat_max - feat_min + eps)

        # Flatten back
        flat = flat_reshaped.view(b * f, w)
        return flat

    def _gaussian_kernel(self) -> torch.Tensor:
        sigma = max(self.sigma, 1e-6)
        radius = int(np.ceil(sigma * 3.0))
        xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (xs ** 2) / (sigma ** 2))
        kernel = kernel / torch.sum(kernel)
        return kernel.view(1, 1, -1)

    def _compute_gsr_saliency(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window, features) -> saliency: (batch, window, features)"""
        if x.numel() == 0:
            return x

        n = x.shape[1]
        eps = 1e-8

        # FFT along time dimension
        transf = torch.fft.fft(x, dim=1)
        mag = torch.abs(transf)
        phase = torch.angle(transf)

        # Clip magnitude to prevent log of very small numbers
        mag = torch.clamp(mag, min=eps)
        log_mag = torch.log(mag)

        kernel = self._gaussian_kernel()
        pad = kernel.shape[-1] // 2

        b, w, f = log_mag.shape
        log_mag_reshaped = log_mag.permute(0, 2, 1).contiguous().view(b * f, 1, w)
        smooth = nn.functional.conv1d(log_mag_reshaped, kernel, padding=pad)
        smooth = smooth.view(b, f, w).permute(0, 2, 1).contiguous()

        # Spectral residual
        residual = log_mag - smooth
        # Clip residual to prevent exp overflow
        residual = torch.clamp(residual, -10.0, 10.0)
        amp = torch.exp(residual)

        real_new = amp * torch.cos(phase)
        imag_new = amp * torch.sin(phase)
        complex_new = torch.complex(real_new, imag_new)

        inv = torch.fft.ifft(complex_new, dim=1)
        saliency = torch.abs(inv).pow(2)
        return saliency

    def fit(self, train_data: pl.DataFrame) -> Result[None]:
        logger.info(f"Training GSR-AE model {self.name}...")
        # FIXME: ValueError will bubble up
        dataset = TimeSeriesDataset(train_data, self.window_size)

        if len(dataset) == 0:
            logger.warning("GSR-AE Dataset is empty.")
            return Ok(None)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model.train()
        best_loss = float('inf')
        for epoch in range(self.epochs):
            losses = []
            for batch in loader:
                batch_x = batch[0]

                flat = self._saliency_flat(batch_x)

                self.optimizer.zero_grad()
                recon = self.model(flat)
                loss = self.criterion(recon, flat)

                # Check for NaN
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at epoch {epoch + 1}, skipping batch")
                    continue

                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(loss.item())

            if losses:
                avg_loss = float(np.mean(losses))
                self.scheduler.step(avg_loss)

                if avg_loss < best_loss:
                    best_loss = avg_loss

                if (epoch + 1) % 5 == 0 or (epoch + 1) == self.epochs:
                    logger.info(
                        f"GSR-AE Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")

        logger.info("Calling _compute_feature_stats...")
        self._compute_feature_stats(dataset)
        logger.info(f"After stats: th_lo={self.th_lo}, th_hi={self.th_hi}")
        return Ok(None)

    def _compute_feature_stats(self, dataset: TimeSeriesDataset):
        logger.info(f"Computing feature error statistics for {self.name}...")
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_err = []
        all_score = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0]
                flat = self._saliency_flat(batch_x)
                recon = self.model(flat)

                b = int(batch_x.shape[0])
                f = int(batch_x.shape[2])
                # Use MSE instead of MAE for more sensitivity to large errors
                err_bf = torch.mean((flat - recon) ** 2, dim=1).view(b, f)
                all_err.append(err_bf.cpu().numpy())
                # Use sum for global score (maintains proper scale)
                all_score.append(torch.sum(err_bf, dim=1).cpu().numpy())

        if not all_err:
            logger.warning("No error stats computed (empty loader?)")
            self.feature_stats = {}
            self.alpha = None
            self.beta = None
            self.tau = None
            return

        all_err_np = np.concatenate(all_err, axis=0)
        self.feature_stats = {}
        for i in range(all_err_np.shape[1]):
            vals = all_err_np[:, i]
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            if std == 0:
                std = 1e-6
            self.feature_stats[i] = (mean, std)

        score_np = np.concatenate(all_score, axis=0) if all_score else np.array([])
        logger.info(f"Score stats computed. Size: {score_np.size}")

        if score_np.size > 0:
            self.alpha = float(np.mean(score_np))
            self.beta = float(np.std(score_np))
            self.tau = self.alpha + (float(self.c) * float(self.beta))

            # Use more sensitive thresholds
            # p05 for Low Score
            # p90 for High Score (more sensitive than p99)
            self.th_lo = float(np.quantile(score_np, self.th_lo_q))
            self.th_hi = float(np.quantile(score_np, 0.90))
            logger.info(
                f"{self.name} thresholds: lo={self.th_lo:.4f} (p{int(self.th_lo_q * 100)}), hi={self.th_hi:.4f} (p90)")
        else:
            self.alpha = None
            self.beta = None
            self.tau = None
            self.th_lo = None
            self.th_hi = None

    def save(self, path: str) -> Result[None]:
        # FIXME: Exception will bubble up
        state = {"model_state": self.model.state_dict(), "feature_stats": getattr(self, "feature_stats", {}),
            "alpha": getattr(self, "alpha", None), "beta": getattr(self, "beta", None),
            "tau": getattr(self, "tau", None), "c": getattr(self, "c", 3.0), "window_size": self.window_size,
            "latent_dim": self.latent_dim, "sigma": self.sigma,
            "th_hi": getattr(self, "th_hi", None), "th_lo": getattr(self, "th_lo", None),
            "amp_feature_indices": getattr(self, "amp_feature_indices", None),
            "amp_threshold": self.amp_threshold, "amp_check_window": self.amp_check_window,
            "suppression_factor": self.suppression_factor, "score_floor": self.score_floor,
            "th_lo_q": self.th_lo_q, "smoothing_window": self.smoothing_window, }
        torch.save(state, path)
        return Ok(None)

    def load(self, path: str) -> Result[None]:
        import torch
        if not Path(path).exists():
            return Err(ErrorCode.MODEL_NOT_FOUND)

        # FIXME: Exception will bubble up
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state" in state:
            self.model.load_state_dict(state["model_state"])
            self.feature_stats = state.get("feature_stats", {})
            self.alpha = state.get("alpha")
            self.beta = state.get("beta")
            self.tau = state.get("tau")
            self.c = float(state.get("c", self.c))
            self.sigma = float(state.get("sigma", self.sigma))
            self.th_hi = state.get("th_hi")
            self.th_lo = state.get("th_lo")
            self.amp_feature_indices = state.get("amp_feature_indices")
            self.amp_threshold = float(state.get("amp_threshold", self.amp_threshold))
            self.amp_check_window = int(state.get("amp_check_window", self.amp_check_window))
            self.suppression_factor = float(state.get("suppression_factor", self.suppression_factor))
            self.score_floor = float(state.get("score_floor", self.score_floor))
            self.th_lo_q = float(state.get("th_lo_q", self.th_lo_q))
            self.smoothing_window = int(state.get("smoothing_window", self.smoothing_window))
        else:
            self.model.load_state_dict(state)
            self.feature_stats = {}
            self.alpha = None
            self.beta = None
            self.tau = None
            self.th_lo = None
            self.th_hi = None
            self.amp_feature_indices = None
        return Ok(None)

    def predict(self, data: pl.DataFrame, return_per_feature: bool = False) -> Result[np.ndarray]:
        self.model.eval()
        # FIXME: ValueError will bubble up
        dataset = TimeSeriesDataset(data, self.window_size)

        if len(dataset) == 0:
            return Ok(np.array([]))

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        results = []

        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0]
                flat = self._saliency_flat(batch_x)
                recon = self.model(flat)

                b = int(batch_x.shape[0])
                f = int(batch_x.shape[2])
                # Use MSE for consistency with training
                err_bf = torch.mean((flat - recon) ** 2, dim=1).view(b, f)

                # Default global score
                score_global = torch.sum(err_bf, dim=1)

                # --- Inverted Anomaly Logic ---
                if self.th_lo is not None and self.th_hi is not None:
                    # 1. Amplitude check
                    check_feats = batch_x
                    if self.amp_check_window > 0 and self.amp_check_window < self.window_size:
                        check_feats = batch_x[:, -self.amp_check_window:, :]

                    max_amp_per_feat, _ = torch.max(torch.abs(check_feats), dim=1)  # (b, f)

                    if self.amp_feature_indices is not None and len(self.amp_feature_indices) > 0:
                        valid_idxs = [i for i in self.amp_feature_indices if 0 <= i < f]
                        if valid_idxs:
                            max_amp_subset = max_amp_per_feat[:, valid_idxs]
                            max_amp_sample, _ = torch.max(max_amp_subset, dim=1)
                        else:
                            max_amp_sample, _ = torch.max(max_amp_per_feat, dim=1)
                    else:
                        max_amp_sample, _ = torch.max(max_amp_per_feat, dim=1)

                    is_high_amp = max_amp_sample > self.amp_threshold
                    is_low_score = score_global < self.th_lo

                    # Burst Detection (Low Score + High Amp) -> Boost
                    burst_mask = is_low_score & is_high_amp
                    if torch.any(burst_mask):
                        # Boost to th_hi * 2.0
                        score_global[burst_mask] = self.th_hi * 2.0

                    # Noise Suppression (Normal Amp + High Score) -> Suppress
                    is_normal_amp = ~is_high_amp
                    if torch.any(is_normal_amp):
                        score_global[is_normal_amp] *= self.suppression_factor
                        if self.score_floor > 0:
                            score_global[is_normal_amp] = torch.where(score_global[is_normal_amp] < self.score_floor,
                                torch.tensor(0.0, device=score_global.device), score_global[is_normal_amp])
                            # Zero out err_bf if score is floored to 0
                            floored_indices = (score_global < 1e-9) & is_normal_amp
                            if torch.any(floored_indices):
                                err_bf[floored_indices] = 0.0

                if return_per_feature:
                    results.append(err_bf.cpu().numpy())
                else:
                    results.append(score_global.cpu().numpy())

        if not results:
            return Ok(np.array([]))

        final_result = np.concatenate(results, axis=0)

        # Apply smoothing
        if self.smoothing_window > 1 and len(final_result) > self.smoothing_window:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            if final_result.ndim == 1:
                final_result = np.convolve(final_result, kernel, mode='same')
            elif final_result.ndim == 2:
                for i in range(final_result.shape[1]):
                    final_result[:, i] = np.convolve(final_result[:, i], kernel, mode='same')

        return Ok(final_result)

    def get_contribution(self, data: pl.DataFrame) -> Result[np.ndarray]:
        self.model.eval()
        # FIXME: ValueError will bubble up
        dataset = TimeSeriesDataset(data, self.window_size)

        if len(dataset) == 0:
            return Ok(np.array([]))

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        results = []

        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0]
                saliency = self._compute_gsr_saliency(batch_x)

                b, w, f = saliency.shape
                flat = saliency.permute(0, 2, 1).contiguous().view(b * f, w)
                recon = self.model(flat)

                # Use MSE for consistency
                diff = (flat - recon) ** 2
                if self.amp_check_window > 0 and self.amp_check_window < self.window_size:
                    diff = diff[:, -self.amp_check_window:]
                err = torch.mean(diff, dim=1).view(b, f)

                # --- Inverted Anomaly Logic (Contribution) ---
                if self.th_lo is not None and self.th_hi is not None:
                    # Use sum for global score
                    score = torch.sum(err, dim=1)

                    # Determine features to check (Reuse logic)
                    check_feats = batch_x
                    if self.amp_check_window > 0 and self.amp_check_window < self.window_size:
                        check_feats = batch_x[:, -self.amp_check_window:, :]

                    valid_idxs = None
                    if self.amp_feature_indices is not None and len(self.amp_feature_indices) > 0:
                        valid_idxs = [i for i in self.amp_feature_indices if 0 <= i < batch_x.shape[2]]
                        if valid_idxs:
                            check_feats = check_feats[:, :, valid_idxs]

                    max_amp_per_feat, _ = torch.max(torch.abs(check_feats), dim=1)  # (b, f_subset)
                    max_amp_sample, _ = torch.max(max_amp_per_feat, dim=1)  # (b,)

                    burst_mask = (score < self.th_lo) & (max_amp_sample > self.amp_threshold)

                    if torch.any(burst_mask):
                        target_score = self.th_hi * 2.0
                        diff = target_score - score[burst_mask]
                        diff = torch.clamp(diff, min=0.0)

                        # Distribute diff among bursting features
                        # map back to original feature indices
                        burst_feats_mask_subset = (
                                    max_amp_per_feat[burst_mask] > self.amp_threshold)  # (n_bursts, f_subset)
                        num_bursting = burst_feats_mask_subset.sum(dim=1, keepdim=True).clamp(min=1.0)
                        diff_per_feat = diff.unsqueeze(1) / num_bursting

                        # Add to error where feature is bursting
                        # need to map subset indices back to full error tensor
                        if valid_idxs:
                            # Create a zero tensor for full features
                            full_update = torch.zeros_like(err[burst_mask])
                            # Update selected indices
                            full_update[:, valid_idxs] = diff_per_feat * burst_feats_mask_subset.float()
                            err[burst_mask] += full_update
                        else:
                            err_bursts = err[burst_mask]
                            err_bursts += diff_per_feat * burst_feats_mask_subset.float()
                            err[burst_mask] = err_bursts

                err_np = err.cpu().numpy()

                norm_list = []
                for i in range(err_np.shape[1]):
                    mean, std = self.feature_stats.get(i, (0.0, 1.0))
                    z = (err_np[:, i] - mean) / std
                    norm_list.append(z)

                results.append(np.stack(norm_list, axis=1))

        if not results:
            return Ok(np.array([]))

        return Ok(np.concatenate(results, axis=0))
