"""
Post-processing logic for anomaly detection.
Applies business rules to filter or confirm algorithmic anomalies.
"""

import numpy as np
import polars as pl
from typing import List

from config import PostProcessingConfig
from utils.logger import get_logger

logger = get_logger("core.postprocess")

class PostProcessor:
    def __init__(self, config: PostProcessingConfig):
        self.config = config

    def process(self, anomalies: np.ndarray, data: pl.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Apply post-processing rules to filter/confirm anomalies.
        
        Args:
            anomalies: Boolean mask of anomalies detected by the algorithm.
            data: Original dataframe (should include feature columns).
            feature_cols: List of feature column names to check.
            
        Returns:
            np.ndarray: Filtered boolean mask (Confirmed Anomalies).
        """
        if not self.config.enabled:
            return anomalies

        n = len(anomalies)
        mask_amplitude = self._check_amplitude(data, feature_cols, n)
        mask_frequency = self._check_frequency(data, feature_cols, n)

        # --- Step 1: Confirm algorithm detections with active rules ---
        # Keep detections that satisfy at least one enabled rule.
        confirm_mask = np.zeros(n, dtype=bool)
        has_rules = False

        if self.config.amplitude.enabled:
            confirm_mask |= mask_amplitude
            has_rules = True

        if self.config.frequency.enabled:
            confirm_mask |= mask_frequency
            has_rules = True

        if has_rules:
            final_mask = anomalies & confirm_mask
        else:
            final_mask = anomalies.copy()

        # --- Step 2: Direction filter — suppress wrong-direction detections ---
        if self.config.direction.enabled:
            direction_mask = self._check_direction(data, feature_cols, n)
            before = int(np.sum(final_mask))
            final_mask = final_mask & direction_mask
            after = int(np.sum(final_mask))
            if before != after:
                logger.info(f"Direction filter ({self.config.direction.direction}) "
                            f"suppressed {before - after} anomalies: {before} -> {after}")

        # Log stats
        algo_count = int(np.sum(anomalies))
        final_count = int(np.sum(final_mask))
        if algo_count != final_count:
            logger.info(f"Post-processing: {algo_count} -> {final_count}")

        return final_mask

    def _check_amplitude(self, data: pl.DataFrame, feature_cols: List[str], target_len: int) -> np.ndarray:
        """
        Check if values exceed an amplitude threshold.

        Supports two modes:
          - **Absolute** (``relative_threshold == 0``): ``|value| > threshold``
          - **Relative** (``relative_threshold > 0``):
            ``value > local_baseline * relative_threshold``, where
            ``local_baseline`` is a rolling median.
        """
        if not self.config.amplitude.enabled:
            return np.zeros(target_len, dtype=bool)

        cols_to_check = self.config.amplitude.features or feature_cols
        use_relative = self.config.amplitude.relative_threshold > 0

        df_subset = data.slice(len(data) - target_len)
        mask = np.zeros(target_len, dtype=bool)

        for col in cols_to_check:
            if col in df_subset.columns:
                vals = df_subset[col].to_numpy()
                if use_relative:
                    series = df_subset[col]
                    window = self.config.amplitude.baseline_window
                    baseline = series.rolling_median(
                        window_size=window, min_periods=1, center=True
                    ).fill_null(0.0).to_numpy()
                    col_mask = vals > baseline * self.config.amplitude.relative_threshold
                else:
                    col_mask = np.abs(vals) > self.config.amplitude.threshold
                mask |= col_mask

        return mask

    def _check_direction(self, data: pl.DataFrame, feature_cols: List[str], target_len: int) -> np.ndarray:
        """
        Check deviation direction relative to a rolling-median baseline.

        Suppresses points that deviate in the wrong direction, using a
        small tolerance (1% of baseline) to avoid filtering points that
        sit marginally on the wrong side due to noise.

        Returns a boolean mask where True = keep the point.

        - direction='up':   keep points where value >= baseline - 1%
        - direction='down': keep points where value <= baseline + 1%
        - direction='both': keep all (no filtering)
        """
        direction = self.config.direction.direction.lower()
        if direction == "both":
            return np.ones(target_len, dtype=bool)

        cols_to_check = self.config.direction.features or feature_cols
        window = self.config.direction.baseline_window

        df_subset = data.slice(len(data) - target_len)
        # For each column, check if the deviation is in the right direction.
        # A point passes if ANY checked column deviates in the configured direction.
        mask = np.zeros(target_len, dtype=bool)

        for col in cols_to_check:
            if col in df_subset.columns:
                series = df_subset[col]
                baseline = series.rolling_median(window_size=window, min_periods=1).fill_null(0.0)
                vals = series.to_numpy()
                base = baseline.to_numpy()

                # Small tolerance (1% of baseline) to avoid filtering
                # anomaly points that sit marginally below the median
                # due to noise.
                tol = np.abs(base) * 0.01

                if direction == "up":
                    col_mask = vals >= base - tol
                elif direction == "down":
                    col_mask = vals <= base + tol
                else:
                    col_mask = np.ones(target_len, dtype=bool)

                mask |= col_mask

        return mask

    def _check_frequency(self, data: pl.DataFrame, feature_cols: List[str], target_len: int) -> np.ndarray:
        """
        Check for high frequency of rising points (rapid oscillation/noise).
        Metric: Ratio of positive derivatives in a sliding window.
        """
        if not self.config.frequency.enabled:
            return np.zeros(target_len, dtype=bool)

        cols_to_check = self.config.frequency.features or feature_cols
        window = self.config.frequency.window_size
        threshold = self.config.frequency.threshold

        df_subset = data.slice(len(data) - target_len)
        mask = np.zeros(target_len, dtype=bool)

        for col in cols_to_check:
            if col in df_subset.columns:
                series = df_subset[col]
                # Calculate diff
                diffs = series.diff().fill_null(0.0)
                
                # Identify rising points (diff > 0)
                rising = (diffs > 0).cast(pl.Float64)
                
                # Rolling mean of rising points = frequency/ratio
                # We use rolling_mean directly
                # Note: rolling operations in Polars need careful handling of alignment
                
                freq = rising.rolling_mean(window_size=window, min_periods=1).fill_null(0.0)
                
                col_mask = (freq > threshold).to_numpy()
                mask |= col_mask
                
        return mask
