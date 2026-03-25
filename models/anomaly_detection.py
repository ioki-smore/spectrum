"""
Ensemble anomaly detector implementation.
Combines multiple anomaly detection models using a soft voting strategy.
"""

from pathlib import Path
from typing import List, Any, Union, Dict

import numpy as np
import polars as pl
import torch

from utils.errors import Result, Ok, Err, ErrorCode
from utils.logger import get_logger
from .base import BaseModel

logger = get_logger("models.anomaly_detector")


class AnomalyDetector(BaseModel):
    """
    Ensemble model that aggregates scores from multiple sub-models.
    
    Strategy:
    1. Train each sub-model independently.
    2. Compute Normalization Statistics (Mean/Std) for each model's scores on training data.
    3. Compute per-model weights based on contrast ratio (p99/median).
       Models with poor contrast (< min_contrast) are auto-disabled (weight=0).
    4. During inference, normalize each model's score (Z-Score) and compute
       a weighted average (Weighted Soft Voting).
    """

    # Minimum contrast ratio (p99/median) for a model to participate in voting.
    # Models below this threshold are effectively disabled.
    MIN_CONTRAST = 1.5

    def __init__(self, name: str, models: List[BaseModel], config: Any = None):
        """
        Args:
            name: Model instance name.
            models: List of instantiated sub-models.
            config: Configuration object or dictionary.
        """
        super().__init__(name, config or {})
        self.models = models

        # normalization_stats: {model_name: (mean, std)}
        # Stores statistics for Z-score normalization of sub-model scores.
        self.normalization_stats: Dict[str, tuple] = {}

        # model_weights: {model_name: float}
        # Weights derived from training contrast ratio. 0 = disabled.
        self.model_weights: Dict[str, float] = {}

    def fit(self, train_data: Union[pl.DataFrame, Any], update_normalization: bool = True) -> Result[None]:
        """
        Train all sub-models and optionally update normalization statistics.
        
        Args:
            train_data: Training data (features).
            update_normalization: Whether to compute and store score statistics for normalization.
                                  Set to False during incremental training if stats shouldn't shift.
        """
        for model in self.models:
            # Train sub-model
            # Note: Unexpected exceptions from sub-models will propagate to the caller.
            fit_result = model.fit(train_data)

            if fit_result.is_err():
                logger.warning(f"Sub-model '{model.name}' failed to fit: {fit_result.err_value}")
                # We continue training other models to maintain partial functionality.
                continue

            if update_normalization:
                self._update_model_stats(model, train_data)

        return Ok(None)

    def predict(self, data: Union[pl.DataFrame, Any]) -> Result[np.ndarray]:
        """
        Predict anomaly scores by aggregating normalized scores from all sub-models.
        
        Returns:
            Result[np.ndarray]: Consensus anomaly scores.
        """
        res = self.predict_detailed(data)
        if res.is_err():
            return Err(res.err_value)
        return Ok(res.unwrap()["consensus"])

    def predict_detailed(self, data: Union[pl.DataFrame, Any]) -> Result[Dict[str, Any]]:
        """
        Predict and return both consensus and individual normalized scores.
        
        Returns:
            Result[Dict]: {
                "consensus": np.ndarray,
                "details": {model_name: np.ndarray}
            }
        """
        model_scores_map = {}

        # Determine if any model has positive weight; if none do, fall back to equal weights
        any_positive = any(self.model_weights.get(m.name, 1.0) > 0 for m in self.models)
        use_weights = any_positive and len(self.model_weights) > 0

        for model in self.models:
            # Skip models with zero weight, unless ALL models are disabled (fallback)
            weight = self.model_weights.get(model.name, 1.0)
            if use_weights and weight <= 0:
                continue

            # Get raw scores
            res = model.predict(data)
            if res.is_err():
                logger.warning(f"Sub-model '{model.name}' prediction failed: {res.err_value}")
                continue

            raw_scores = res.unwrap()
            if len(raw_scores) == 0:
                continue

            # Normalize scores (Z-Score)
            mean, std = self.normalization_stats.get(model.name, (0.0, 1.0))
            normalized_scores = (raw_scores - mean) / std

            model_scores_map[model.name] = normalized_scores

        if not model_scores_map:
            return Ok({"consensus": np.array([]), "details": {}})

        # Align lengths (handle potential minor mismatches due to windowing differences)
        min_length = min(len(s) for s in model_scores_map.values())
        if min_length == 0:
            return Ok({"consensus": np.array([]), "details": {}})

        # Trim all to min_length
        aligned_details = {name: scores[:min_length] for name, scores in model_scores_map.items()}

        # Weighted Soft Voting
        if use_weights:
            weights = np.array([self.model_weights.get(name, 1.0) for name in aligned_details.keys()])
        else:
            # Fallback: equal weights (all models disabled or no weights computed)
            weights = np.ones(len(aligned_details))
        total_weight = weights.sum()
        if total_weight <= 0:
            weights = np.ones(len(aligned_details))
            total_weight = weights.sum()

        stacked_scores = np.stack(list(aligned_details.values()))
        consensus_score = np.average(stacked_scores, axis=0, weights=weights)

        return Ok({"consensus": consensus_score, "details": aligned_details})

    def predict_with_contributions(self, data: Union[pl.DataFrame, Any]) -> Result[Dict[str, Any]]:
        """
        Single-pass inference: returns consensus scores, per-model details, AND contributions.
        Avoids the double forward pass of calling predict_detailed() + get_contribution() separately.
        
        Returns:
            Result[Dict]: {
                "consensus": np.ndarray,
                "details": {model_name: np.ndarray},
                "contributions": np.ndarray or None
            }
        """
        model_scores_map = {}
        model_contribs_map = {}  # per-model contributions: {name: (n, features)}
        contributions_list = []
        contrib_weights_list = []

        any_positive = any(self.model_weights.get(m.name, 1.0) > 0 for m in self.models)
        use_weights = any_positive and len(self.model_weights) > 0

        for model in self.models:
            weight = self.model_weights.get(model.name, 1.0)
            if use_weights and weight <= 0:
                continue

            # --- Single forward pass for scores + contributions ---
            res = model.predict_and_contribute(data)
            if res.is_err():
                logger.warning(f"Sub-model '{model.name}' predict_and_contribute failed: {res.err_value}")
                continue

            raw_scores, contrib = res.unwrap()
            if len(raw_scores) == 0:
                continue

            mean, std = self.normalization_stats.get(model.name, (0.0, 1.0))
            model_scores_map[model.name] = (raw_scores - mean) / std

            if contrib is not None and len(contrib) > 0:
                model_contribs_map[model.name] = contrib
                contributions_list.append(contrib)
                contrib_weights_list.append(weight)

        if not model_scores_map:
            return Ok({"consensus": np.array([]), "details": {}, "contributions": None, "model_contributions": {}})

        # Align scores
        min_length = min(len(s) for s in model_scores_map.values())
        if min_length == 0:
            return Ok({"consensus": np.array([]), "details": {}, "contributions": None, "model_contributions": {}})

        aligned_details = {name: scores[:min_length] for name, scores in model_scores_map.items()}

        # Weighted consensus
        if use_weights:
            weights = np.array([self.model_weights.get(name, 1.0) for name in aligned_details.keys()])
        else:
            weights = np.ones(len(aligned_details))
        if weights.sum() <= 0:
            weights = np.ones(len(aligned_details))

        stacked_scores = np.stack(list(aligned_details.values()))
        consensus_score = np.average(stacked_scores, axis=0, weights=weights)

        # Weighted contributions
        avg_contrib = None
        if contributions_list:
            min_clen = min(c.shape[0] for c in contributions_list)
            trimmed = [c[:min_clen] for c in contributions_list]
            cweights = np.array(contrib_weights_list)
            if cweights.sum() <= 0:
                cweights = np.ones(len(contrib_weights_list))
            stacked = np.stack(trimmed)
            avg_contrib = np.average(stacked, axis=0, weights=cweights)

        # Align per-model contributions to min_length
        aligned_model_contribs = {}
        for name, c in model_contribs_map.items():
            aligned_model_contribs[name] = c[:min_length]

        return Ok({
            "consensus": consensus_score,
            "details": aligned_details,
            "contributions": avg_contrib,
            "model_contributions": aligned_model_contribs,
        })

    def get_contribution(self, data: Union[pl.DataFrame, Any]) -> Result[np.ndarray]:
        """
        Aggregate feature contributions (attribution) from sub-models.
        Only includes models with positive weight.
        """
        contributions_list = []
        weights_list = []

        any_positive = any(self.model_weights.get(m.name, 1.0) > 0 for m in self.models)
        use_weights = any_positive and len(self.model_weights) > 0

        for model in self.models:
            weight = self.model_weights.get(model.name, 1.0)
            if use_weights and weight <= 0:
                continue

            res = model.get_contribution(data)
            if res.is_err():
                continue

            contrib = res.unwrap()
            if len(contrib) > 0:
                contributions_list.append(contrib)
                weights_list.append(weight)

        if not contributions_list:
            return Ok(np.array([]))

        # Weighted average contributions
        min_len = min(c.shape[0] for c in contributions_list)
        trimmed = [c[:min_len] for c in contributions_list]
        weights = np.array(weights_list)
        total_weight = weights.sum()
        if total_weight <= 0:
            weights = np.ones(len(weights_list))
            total_weight = weights.sum()

        stacked = np.stack(trimmed)
        avg_contribution = np.average(stacked, axis=0, weights=weights)

        return Ok(avg_contribution)

    def save(self, path: str) -> Result[None]:
        """
        Save ensemble metadata and all sub-models.
        
        Args:
            path: Path for the main ensemble metadata file.
                  Sub-models will be saved as {path_stem}_{model_name}.pth
        """
        base_path = Path(path)

        base_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Save Ensemble Metadata
        metadata = {"models": [m.name for m in self.models], "normalization_stats": self.normalization_stats,
                    "model_weights": self.model_weights}
        torch.save(metadata, base_path)

        # 2. Save Sub-models
        for model in self.models:
            # Construct unique path: e.g. "model_USAD.pth"
            sub_model_path = base_path.with_name(f"{base_path.stem}_{model.name}.pth")

            res = model.save(str(sub_model_path))
            if res.is_err():
                logger.warning(f"Failed to save sub-model '{model.name}': {res.err_value}")

        return Ok(None)

    def load(self, path: str) -> Result[None]:
        """
        Load ensemble metadata and all sub-models.
        """
        base_path = Path(path)
        if not base_path.exists():
            return Err(ErrorCode.MODEL_NOT_FOUND)

        # 1. Load Metadata
        # weights_only=False required for loading generic python dicts
        metadata = torch.load(base_path, map_location='cpu', weights_only=False)

        # Handle migration from old 'score_stats' (int keys) to 'normalization_stats' (str keys)
        if "normalization_stats" in metadata:
            self.normalization_stats = metadata["normalization_stats"]
        elif "score_stats" in metadata:
            logger.info("Migrating legacy score_stats to normalization_stats")
            raw_stats = metadata["score_stats"]
            # Attempt to map index to model name
            # This assumes self.models matches the order of saved stats
            for idx, model in enumerate(self.models):
                if idx in raw_stats:
                    self.normalization_stats[model.name] = raw_stats[idx]

        self.model_weights = metadata.get("model_weights", {})

        # 2. Load Sub-models
        for model in self.models:
            sub_model_path = base_path.with_name(f"{base_path.stem}_{model.name}.pth")

            if sub_model_path.exists():
                res = model.load(str(sub_model_path))
                if res.is_err():
                    logger.warning(f"Failed to load sub-model '{model.name}': {res.err_value}")
            else:
                logger.warning(f"Sub-model file not found: {sub_model_path}")

        return Ok(None)

    def _update_model_stats(self, model: BaseModel, data: Union[pl.DataFrame, Any]) -> None:
        """Helper to compute and store normalization stats and contrast-based weight for a model."""
        res = model.predict(data)
        if res.is_err():
            logger.warning(f"Model '{model.name}' failed to predict during stats update: {res.err_value}")
            self.normalization_stats[model.name] = (0.0, 1.0)
            self.model_weights[model.name] = 0.0
            return

        scores = res.unwrap()
        if len(scores) > 0:
            # Robust Z-Score: use median/MAD instead of mean/std
            # MAD is more resistant to heavy-tailed score distributions
            median = float(np.median(scores))
            mad = float(np.median(np.abs(scores - median)))
            # 1.4826 makes MAD consistent with std for normal distributions
            robust_scale = mad * 1.4826
            if robust_scale == 0:
                robust_scale = 1.0

            self.normalization_stats[model.name] = (median, robust_scale)

            # Compute contrast ratio: p99 / (median + eps)
            # High contrast = model distinguishes tail from bulk well
            p99 = float(np.percentile(scores, 99))
            eps = 1e-8
            contrast = p99 / (abs(median) + eps)

            if contrast < self.MIN_CONTRAST:
                logger.warning(f"Model '{model.name}' has low contrast ({contrast:.2f} < {self.MIN_CONTRAST}). "
                               f"Auto-disabling in ensemble.")
                self.model_weights[model.name] = 0.0
            else:
                # log1p to compress extreme contrast values and prevent
                # any single model from dominating the ensemble
                self.model_weights[model.name] = float(np.log1p(contrast))

            logger.info(f"Stats for {model.name} - Median: {median:.4f}, RobustScale: {robust_scale:.4f}, "
                        f"Contrast: {contrast:.2f}, Weight: {self.model_weights[model.name]:.2f}")
        else:
            logger.warning(f"Model '{model.name}' produced no scores during training.")
            self.normalization_stats[model.name] = (0.0, 1.0)
            self.model_weights[model.name] = 0.0
