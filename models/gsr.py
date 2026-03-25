from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.fft
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset.timeseries import TimeSeriesDataset
from utils.errors import Result, Ok, Err, ErrorCode
from utils.logger import get_logger
from .base import BaseModel

logger = get_logger("models.gsr")


class GSR(BaseModel):
    """Global Spectral Residual (GSR) anomaly scoring.

    Dual-score architecture:
      1. Deviation Score: Per-point absolute deviation from training baseline,
         weighted by recency within the sliding window.
      2. Spectral Score: FFT-based spectral energy shift relative to training.

    The two scores are combined with learned weights. An auto-tune grid search
    selects the best (window_size, deviation_weight, spectral_weight, threshold)
    combination by maximising the contrast ratio (p99/median) on training data.

    Designed to work with a Stage-1 Preprocessor that normalises raw data
    (e.g. ratio-to-baseline), so that normal ≈ 1.0 and anomalies >> 1.0.
    """

    def __init__(self, name: str, config: Any, input_dim: int):
        super().__init__(name, config)

        self.window_size = int(self.get_param("gsr_window_size", self.get_param("window_size", 16)))
        self.batch_size = int(self.get_param("batch_size", 256))

        # Auto tuning
        self.auto_tune = bool(self.get_param("gsr_auto_tune", True))
        self.tune_window_candidates = self.get_param("gsr_tune_window_sizes", [4, 8, 12, 16])
        self.tune_dev_weights = self.get_param("gsr_tune_dev_weights", [1.0, 2.0, 5.0])
        self.tune_spec_weights = self.get_param("gsr_tune_spec_weights", [0.0, 0.1, 0.3])

        # Score weights (deviation vs spectral)
        self.deviation_weight = float(self.get_param("gsr_deviation_weight", 1.0))
        self.spectral_weight = float(self.get_param("gsr_spectral_weight", 0.0))

        # Threshold margin: scale factor applied to auto-threshold.
        # < 1.0 lowers the threshold for higher recall (fewer missed anomalies).
        # Post-processing is expected to filter the resulting false positives.
        self.threshold_margin = float(self.get_param("gsr_threshold_margin", 0.9))

        # Numerical stability
        self.eps = float(self.get_param("gsr_eps", 1e-8))

        self.device = "cpu"

        # Learned parameters from fit()
        self.train_mean = None       # per-feature mean of last-point values
        self.train_std = None        # per-feature std
        self.train_eng_mean = None   # per-feature mean of spectral energy
        self.train_eng_std = None    # per-feature std of spectral energy
        self.train_max_dev = None    # per-feature max abs deviation in training
        self.train_max_eng = None    # per-feature max abs spectral energy deviation
        self.data_ac1 = 0.0          # lag-1 autocorrelation of preprocessed training data
        self.auto_threshold = None   # adaptive threshold from training scores

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, train_data: pl.DataFrame) -> Result[None]:
        """Learn baseline statistics from normal training data."""
        # Compute data autocorrelation before auto-tune (used for threshold)
        self._compute_data_autocorrelation(train_data)

        if self.auto_tune:
            self._auto_tune(train_data)

        dataset = TimeSeriesDataset(train_data, self.window_size)
        if len(dataset) == 0:
            return Ok(None)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_last = []
        all_eng = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)  # (b, w, f)
                all_last.append(x[:, -1, :])
                # Spectral energy
                transf = torch.fft.fft(x, dim=1)
                log_mag = torch.log(transf.abs() + self.eps)
                energy = log_mag.mean(dim=1)  # (b, f)
                all_eng.append(energy)

        if not all_last:
            return Ok(None)

        last_tensor = torch.cat(all_last, dim=0)
        eng_tensor = torch.cat(all_eng, dim=0)

        self.train_mean = last_tensor.mean(dim=0)
        self.train_std = last_tensor.std(dim=0)
        self.train_eng_mean = eng_tensor.mean(dim=0)
        self.train_eng_std = eng_tensor.std(dim=0)

        # Per-feature max absolute deviation (for multi-dim normalization)
        self.train_max_dev = torch.abs(last_tensor - self.train_mean).max(dim=0).values
        self.train_max_eng = torch.abs(eng_tensor - self.train_eng_mean).max(dim=0).values
        self.train_max_dev = torch.maximum(self.train_max_dev, torch.tensor(self.eps))
        self.train_max_eng = torch.maximum(self.train_max_eng, torch.tensor(self.eps))

        # Variance flooring
        self.train_std = torch.maximum(self.train_std, torch.tensor(self.eps))
        self.train_eng_std = torch.maximum(self.train_eng_std, torch.tensor(self.eps))

        # Compute auto threshold on training data
        self._apply_auto_threshold(train_data)

        logger.info(f"GSR fitted: mean={self.train_mean}, std={self.train_std}, "
                     f"threshold={self.auto_threshold}, ac1={self.data_ac1:.3f}")
        return Ok(None)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample anomaly score for a batch.

        Uses raw absolute deviation from the training mean.  For
        **multi-feature** data each feature's deviation is normalised by
        the training-max deviation so that features with different scales
        are comparable, then takes the max across features.

        Args:
            x: (batch, window, features)
        Returns:
            scores: (batch,)
        """
        b, w, f = x.shape
        last_val = x[:, -1, :]  # (b, f)

        # --- Deviation score ---
        raw_dev = torch.abs(last_val - self.train_mean.to(x.device))
        if f > 1:
            dev = raw_dev / (self.train_max_dev.to(x.device) + self.eps)
        else:
            dev = raw_dev

        # --- Spectral energy score ---
        transf = torch.fft.fft(x, dim=1)
        log_mag = torch.log(transf.abs() + self.eps)
        energy = log_mag.mean(dim=1)  # (b, f)
        raw_eng = torch.abs(energy - self.train_eng_mean.to(x.device))
        if f > 1:
            eng_norm = raw_eng / (self.train_max_eng.to(x.device) + self.eps)
        else:
            eng_norm = raw_eng

        # --- Combine per-feature, then take max across features ---
        combined = self.deviation_weight * dev + self.spectral_weight * eng_norm  # (b, f)
        return combined.max(dim=1).values  # (b,)

    # ------------------------------------------------------------------
    # Auto-tune
    # ------------------------------------------------------------------
    def _compute_data_autocorrelation(self, data: pl.DataFrame) -> None:
        """Compute lag-1 autocorrelation of preprocessed training data.

        High autocorrelation indicates non-stationary data (e.g. drift,
        imperfect baseline tracking) which requires a looser threshold.
        """
        cols = [c for c in data.columns if c != "timestamp"]
        ac1_vals = []
        for col in cols:
            vals = data[col].to_numpy().astype(float)
            v_centered = vals - vals.mean()
            if v_centered.var() > 1e-12 and len(v_centered) > 2:
                ac = float(np.corrcoef(v_centered[:-1], v_centered[1:])[0, 1])
                ac1_vals.append(max(ac, 0.0))
        self.data_ac1 = float(np.mean(ac1_vals)) if ac1_vals else 0.0
        logger.info(f"Data autocorrelation (lag-1): {self.data_ac1:.3f}")

    def _auto_tune(self, data: pl.DataFrame) -> None:
        """Grid search for best (window, dev_w, spec_w) on training data.

        Objective: maximise contrast = p99 / (median + eps) of scores,
        which indicates how well the scoring separates tail from bulk.
        """
        best_contrast = -1.0
        best_cfg = None

        for ws in self.tune_window_candidates:
            dataset = TimeSeriesDataset(data, int(ws))
            if len(dataset) == 0:
                continue
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            # Compute raw components once per window size
            all_last = []
            all_eng = []
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].to(self.device)
                    all_last.append(x[:, -1, :])
                    transf = torch.fft.fft(x, dim=1)
                    log_mag = torch.log(transf.abs() + self.eps)
                    all_eng.append(log_mag.mean(dim=1))

            if not all_last:
                continue

            last_t = torch.cat(all_last, dim=0)
            eng_t = torch.cat(all_eng, dim=0)

            t_mean = last_t.mean(dim=0)
            t_std = torch.maximum(last_t.std(dim=0), torch.tensor(self.eps))
            e_mean = eng_t.mean(dim=0)
            e_std = torch.maximum(eng_t.std(dim=0), torch.tensor(self.eps))

            n_features = last_t.shape[1]
            t_max_dev = torch.abs(last_t - t_mean).max(dim=0).values
            t_max_dev = torch.maximum(t_max_dev, torch.tensor(self.eps))
            e_max_dev = torch.abs(eng_t - e_mean).max(dim=0).values
            e_max_dev = torch.maximum(e_max_dev, torch.tensor(self.eps))

            # Raw deviation (max-dev normalised for multi-dim)
            raw_dev = torch.abs(last_t - t_mean)
            raw_eng = torch.abs(eng_t - e_mean)
            if n_features > 1:
                dev_all = (raw_dev / (t_max_dev + self.eps)).max(dim=1).values.numpy()
                eng_all = (raw_eng / (e_max_dev + self.eps)).max(dim=1).values.numpy()
            else:
                dev_all = raw_dev.max(dim=1).values.numpy()
                eng_all = raw_eng.max(dim=1).values.numpy()

            for dw in self.tune_dev_weights:
                for sw in self.tune_spec_weights:
                    scores = float(dw) * dev_all + float(sw) * eng_all
                    p99 = float(np.percentile(scores, 99))
                    med = float(np.median(scores))
                    contrast = p99 / (med + self.eps)
                    # Penalize spectral weight: spectral contrast on
                    # training data often doesn't generalize, so require
                    # it to be 50% better than deviation-only to win.
                    if float(sw) > 0:
                        contrast *= 0.5
                    if contrast > best_contrast:
                        best_contrast = contrast
                        best_cfg = (int(ws), float(dw), float(sw),
                                    t_mean.clone(), t_std.clone(),
                                    t_max_dev.clone(),
                                    e_mean.clone(), e_std.clone(),
                                    e_max_dev.clone())

        if best_cfg is not None:
            (self.window_size, self.deviation_weight, self.spectral_weight,
             self.train_mean, self.train_std, self.train_max_dev,
             self.train_eng_mean, self.train_eng_std, self.train_max_eng) = best_cfg
            logger.info(f"Auto-tune: ws={self.window_size}, dev_w={self.deviation_weight}, "
                         f"spec_w={self.spectral_weight}, contrast={best_contrast:.2f}")

    def _apply_auto_threshold(self, data: pl.DataFrame) -> None:
        """Set threshold adaptively from the training score distribution.

        Blends two strategies based on data autocorrelation (ac1):

          * **Stationary** (ac1 ≈ 0):  ``max * 1.5``
            A tight margin above the observed training maximum.

          * **Non-stationary** (ac1 > 0):  ``max + k * spread``
            Extrapolation using the training score spread, where
            ``k = 1 + 8 * ac1`` (capped at 6).

        The final threshold interpolates between the two:

          alpha = min(ac1 * 2, 1)
          threshold = (1 - alpha) * th_stat + alpha * th_nonstat

        This ensures stationary data gets a tight threshold (good for
        high-noise scenarios) while non-stationary data gets a loose
        threshold (good for drifting baselines).
        """
        dataset = TimeSeriesDataset(data, self.window_size)
        if len(dataset) == 0:
            return

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        scores_out = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                scores_out.append(self._score_batch(x).cpu().numpy())

        if scores_out:
            scores = np.concatenate(scores_out, axis=0)
            s_max = float(np.max(scores))
            s_med = float(np.median(scores))
            s_q75 = float(np.percentile(scores, 75))
            s_q25 = float(np.percentile(scores, 25))
            iqr = s_q75 - s_q25
            spread = max(s_max - s_med, iqr, self.eps)

            # Stationary threshold: margin above training max
            th_stat = max(s_max * 1.5, s_med + 5.0 * max(iqr, self.eps))

            # Non-stationary threshold: extrapolate using spread
            k = min(1.0 + 8.0 * self.data_ac1, 6.0)
            th_nonstat = s_max + k * spread

            # Interpolate based on autocorrelation.
            # Only blend in the non-stationary term when ac1 is clearly
            # above the baseline introduced by the rolling-median
            # preprocessor (~0.1-0.2).  Soft ramp from 0.2 to 0.5.
            alpha = max(0.0, (self.data_ac1 - 0.2) / 0.3)
            alpha = min(alpha, 1.0)
            raw_threshold = (1.0 - alpha) * th_stat + alpha * th_nonstat
            self.auto_threshold = raw_threshold * self.threshold_margin

            logger.info(f"Auto threshold: {self.auto_threshold:.4f} "
                         f"(raw={raw_threshold:.4f}, margin={self.threshold_margin}, "
                         f"max={s_max:.4f}, med={s_med:.4f}, spread={spread:.4f}, "
                         f"k={k:.2f}, alpha={alpha:.2f}, "
                         f"th_stat={th_stat:.4f}, th_nonstat={th_nonstat:.4f})")

    # ------------------------------------------------------------------
    # Predict / Contribute
    # ------------------------------------------------------------------
    def predict(self, data: pl.DataFrame) -> Result[np.ndarray]:
        dataset = TimeSeriesDataset(data, self.window_size)
        if len(dataset) == 0:
            return Ok(np.array([]))

        if self.train_mean is None:
            logger.warning("Model not fitted, fitting on test data for stats.")
            self.fit(data.head(min(len(data), 500)))

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        scores_out = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                scores_out.append(self._score_batch(x).cpu().numpy())

        if not scores_out:
            return Ok(np.array([]))

        return Ok(np.concatenate(scores_out, axis=0))

    def get_contribution(self, data: pl.DataFrame) -> Result[np.ndarray]:
        dataset = TimeSeriesDataset(data, self.window_size)
        if len(dataset) == 0:
            return Ok(np.array([]))

        if self.train_mean is None:
            self.fit(data.head(min(len(data), 500)))

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        contrib_out = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                last_val = x[:, -1, :]
                f = last_val.shape[1]
                raw_dev = torch.abs(last_val - self.train_mean.to(x.device))
                if f > 1:
                    dev = raw_dev / (self.train_max_dev.to(x.device) + self.eps)
                else:
                    dev = raw_dev
                contrib_out.append((self.deviation_weight * dev).cpu().numpy())

        if not contrib_out:
            return Ok(np.array([]))

        return Ok(np.concatenate(contrib_out, axis=0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> Result[None]:
        state = {
            "model": "GSR",
            "window_size": self.window_size,
            "batch_size": self.batch_size,
            "eps": self.eps,
            "train_mean": self.train_mean,
            "train_std": self.train_std,
            "train_eng_mean": self.train_eng_mean,
            "train_eng_std": self.train_eng_std,
            "train_max_dev": self.train_max_dev,
            "train_max_eng": self.train_max_eng,
            "data_ac1": self.data_ac1,
            "auto_tune": self.auto_tune,
            "auto_threshold": self.auto_threshold,
            "deviation_weight": self.deviation_weight,
            "spectral_weight": self.spectral_weight,
        }
        try:
            torch.save(state, path)
            return Ok(None)
        except Exception as e:
            return Err(ErrorCode.IO_WRITE_FAILED, str(e))

    def load(self, path: str) -> Result[None]:
        if not Path(path).exists():
            return Err(ErrorCode.MODEL_NOT_FOUND)

        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            self.window_size = int(state.get("window_size", self.window_size))
            self.batch_size = int(state.get("batch_size", self.batch_size))
            self.eps = float(state.get("eps", self.eps))
            self.train_mean = state.get("train_mean")
            self.train_std = state.get("train_std")
            self.train_eng_mean = state.get("train_eng_mean")
            self.train_eng_std = state.get("train_eng_std")
            self.train_max_dev = state.get("train_max_dev")
            self.train_max_eng = state.get("train_max_eng")
            self.data_ac1 = float(state.get("data_ac1", 0.0))
            self.auto_tune = bool(state.get("auto_tune", self.auto_tune))
            self.auto_threshold = state.get("auto_threshold", self.auto_threshold)
            self.deviation_weight = float(state.get("deviation_weight", self.deviation_weight))
            self.spectral_weight = float(state.get("spectral_weight", self.spectral_weight))
            return Ok(None)
        except Exception as e:
            return Err(ErrorCode.IO_READ_FAILED, str(e))
