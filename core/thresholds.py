"""
Threshold management for anomaly detection.
"""

import numpy as np

from config import AppConfig
from core.state import StateManager
from utils.errors import ErrorCode, Result, Ok, Err
from utils.logger import get_logger
from utils.thresholding import fit_pot

logger = get_logger("core.thresholds")


class ThresholdManager:
    """
    Manages reading, writing, and calculating anomaly thresholds.
    
    Strategies:
    1. POT (Peaks-Over-Threshold): Main statistical method used to set thresholds based on 
       Extreme Value Theory.
    2. Adaptive: When retraining on False Positives, automatically adjusts the threshold 
       to be slightly higher than the maximum score in the feedback window, effectively 
       suppressing the false alarm.
    3. Fallback: Uses 3-Sigma rule if no stored threshold exists.
    """

    def __init__(self, interval_id: str, state_manager: StateManager, config: AppConfig):
        self.interval_id = interval_id
        self.state_manager = state_manager
        self.config = config
        # POT baseline threshold — the "anchor" that adapted thresholds decay toward
        self._pot_baseline: float = None

    def save_threshold(self, threshold: float) -> Result[None]:
        """Persist the threshold value to the state manager."""
        res = self.state_manager.set_threshold(float(threshold))
        if res.is_err():
            logger.error(f"[{self.interval_id}] Failed to save threshold: {res.err_value}")
            return Err(res.err_value)
        return Ok(None)

    def load_threshold(self) -> Result[float]:
        """Retrieve the current persistent threshold."""
        val = self.state_manager.threshold
        if val is None:
            return Err(ErrorCode.THRESHOLD_NOT_FOUND)
        return Ok(val)

    def compute_pot_threshold(self, scores: np.ndarray) -> Result[float]:
        """
        Calculate threshold using Peak-Over-Threshold (POT) method.
        
        Args:
            scores: Anomaly scores from the training set.
            
        Returns:
            Result[float]: The calculated threshold.
        """
        res = fit_pot(scores, risk=self.config.models.get("pot_risk", 1e-4),
            level=self.config.models.get("pot_level", 0.98))
        if res.is_err():
            return Err(res.err_value)

        threshold = res.unwrap()
        self._pot_baseline = threshold
        res = self.save_threshold(threshold)
        if res.is_err():
            return Err(res.err_value)

        return Ok(threshold)

    def get_threshold_or_default(self, scores: np.ndarray) -> float:
        """
        Get the current threshold, or calculate a fallback if missing.
        
        Fallback: Mean + 3 * Std (3-Sigma rule).
        """
        res = self.load_threshold()
        if res.is_ok():
            return res.unwrap()

        logger.warning(f"[{self.interval_id}] Threshold not found. Using p99 fallback.")
        return float(np.percentile(scores, 99) * 1.1)

    def adapt_threshold(self, scores: np.ndarray) -> None:
        """
        Adaptively increase threshold based on new scores (typically from False Alarm feedback).
        
        Logic:
        - If the max score in the provided batch (the false alarm data) exceeds the current threshold,
          raise the threshold to (max_score * 1.05).
        - This ensures the specific pattern is no longer flagged as anomalous.
        """
        if len(scores) == 0:
            return

        max_score = float(scores.max())
        res = self.load_threshold()

        if res.is_ok():
            current_threshold = res.unwrap()
            if max_score > current_threshold:
                new_threshold = max_score * 1.05
                logger.info(f"[{self.interval_id}] Adapting threshold: {current_threshold:.6f} -> {new_threshold:.6f}")
                res = self.save_threshold(new_threshold)
                if res.is_err():
                    logger.warning(f"[{self.interval_id}] Failed to adapt threshold: {res.err_value}")
        else:
            logger.info(f"[{self.interval_id}] Initializing threshold from max score: {max_score:.6f}")
            res = self.save_threshold(max_score * 1.05)
            if res.is_err():
                logger.warning(f"[{self.interval_id}] Failed to initialize threshold: {res.err_value}")

    def decay_threshold(self, decay_rate: float = 0.995) -> None:
        """
        Decay the current threshold toward the POT baseline.
        
        When adapt_threshold raises the threshold to suppress false alarms,
        this method gradually pulls it back toward the statistical baseline
        to prevent long-term recall degradation.
        
        Should be called periodically (e.g., each detection cycle).
        
        Args:
            decay_rate: Exponential decay factor (0-1). Default 0.995 means
                        ~0.5% decay per call toward the baseline.
        """
        if self._pot_baseline is None:
            return

        res = self.load_threshold()
        if res.is_err():
            return

        current = res.unwrap()
        baseline = self._pot_baseline

        # Only decay if current is above baseline (was adapted upward)
        if current > baseline * 1.01:  # 1% tolerance to avoid unnecessary writes
            decayed = baseline + (current - baseline) * decay_rate
            logger.debug(f"[{self.interval_id}] Threshold decay: {current:.6f} -> {decayed:.6f} (baseline: {baseline:.6f})")
            self.save_threshold(decayed)
