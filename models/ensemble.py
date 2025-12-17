from typing import List, Any, Optional, Union, Dict
import numpy as np
import polars as pl
from .base import BaseModel
from utils.logger import get_logger

logger = get_logger("models.ensemble")


class Ensemble(BaseModel):
    """Ensemble model combining multiple anomaly detection models using soft voting."""
    
    def __init__(self, name: str, models: List[BaseModel], config: Any = None):
        super().__init__(name, config or {})
        self.models = models
        # Store training stats for score normalization: {model_index: (mean, std)}
        self.score_stats: Dict[int, tuple] = {}

    def fit(self, train_data: Union[pl.DataFrame, Any], update_stats: bool = True) -> None:
        """Train all sub-models and compute score statistics for normalization.
        
        Args:
            train_data: Training data as Polars DataFrame.
            update_stats: If True, update score normalization statistics.
        """
        for i, model in enumerate(self.models):
            try:
                model.fit(train_data)
                
                if update_stats:
                    # Calculate score stats on training data for normalization
                    scores = model.predict(train_data)
                    if len(scores) > 0:
                        mean = np.mean(scores)
                        std = np.std(scores)
                        if std == 0: std = 1.0
                        self.score_stats[i] = (mean, std)
                        logger.info(f"Model {model.name} score stats - Mean: {mean:.4f}, Std: {std:.4f}")
                    else:
                        logger.warning(f"Model {model.name} produced no scores during training.")
                        self.score_stats[i] = (0.0, 1.0)
                
            except Exception as e:
                logger.error(f"Error fitting model {model.name} in ensemble: {e}")
                if update_stats and i not in self.score_stats:
                     self.score_stats[i] = (0.0, 1.0)

    def predict(self, data: Union[pl.DataFrame, Any]) -> np.ndarray:
        normalized_scores_list = []
        
        for i, model in enumerate(self.models):
            try:
                scores = model.predict(data)
                if len(scores) == 0:
                    continue
                
                # Normalize using training stats (Z-Score)
                mean, std = self.score_stats.get(i, (0.0, 1.0))
                norm_scores = (scores - mean) / std
                
                # Clip negative values? Anomaly scores are usually positive distance. 
                # Z-score can be negative. 
                # For anomaly detection, we care about high positive values (deviations).
                # But if we average, negatives might cancel positives. 
                # Let's take absolute z-score? Or just raw z-score.
                # Usually we want "how far from normal". 
                # If normal is low score, high score is anomaly. 
                # Z-score preserves order.
                normalized_scores_list.append(norm_scores)
                
            except Exception as e:
                logger.error(f"Error predicting with model {model.name} in ensemble: {e}")
        
        if not normalized_scores_list:
            return np.array([])
            
        # Stack: (num_models, num_samples)
        # We need to handle potential length mismatches if models behave differently (unlikely with same dataset class)
        try:
            stacked = np.stack(normalized_scores_list)
            # Average Z-scores (Soft Voting)
            consensus_scores = np.mean(stacked, axis=0)
            return consensus_scores
        except ValueError as e:
            logger.error(f"Ensemble prediction shape mismatch: {e}")
            return np.array([])

    def get_contribution(self, data: Union[pl.DataFrame, Any]) -> np.ndarray:
        """
        Aggregates anomaly contributions from all models.
        Returns:
            np.ndarray: (n_samples, n_features)
        """
        contrib_list = []
        
        for i, model in enumerate(self.models):
            try:
                contrib = model.get_contribution(data)
                if len(contrib) == 0:
                    continue
                
                # All sub-models (USAD, LSTM, SR) now return Z-score normalized contributions.
                # We can safely average them.
                contrib_list.append(contrib)
                
            except Exception as e:
                logger.error(f"Error getting contribution from {model.name}: {e}")
        
        if not contrib_list:
            return np.array([])
            
        try:
            stacked = np.stack(contrib_list)
            # Average
            consensus = np.mean(stacked, axis=0)
            return consensus
        except ValueError as e:
            logger.error(f"Contribution shape mismatch: {e}")
            return np.array([])

    def save(self, path: str) -> None:
        """Save ensemble metadata and all sub-models."""
        import torch
        from pathlib import Path
        
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata to the main path
        metadata = {
            "models": [m.name for m in self.models],
            "score_stats": self.score_stats
        }
        torch.save(metadata, path)
        
        # Save sub-models
        for i, model in enumerate(self.models):
            # Construct a unique path for each model
            # e.g. "path/to/model.pth" -> "path/to/model_USAD.pth"
            model_path = base_path.with_name(f"{base_path.stem}_{model.name}.pth")
            try:
                model.save(str(model_path))
            except Exception as e:
                logger.warning(f"Failed to save sub-model {model.name}: {e}")

    def load(self, path: str) -> None:
        """Load ensemble metadata and all sub-models."""
        import torch
        from pathlib import Path
        
        base_path = Path(path)
        if not base_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found at {path}")
            
        try:
            # PyTorch 2.6+ requires weights_only=False for loading dicts with numpy types
            metadata = torch.load(path, map_location='cpu', weights_only=False)
            self.score_stats = metadata.get("score_stats", {})
        except Exception as e:
            logger.error(f"Failed to load ensemble metadata from {path}: {e}")
            return
        
        # Load sub-models
        for i, model in enumerate(self.models):
            model_path = base_path.with_name(f"{base_path.stem}_{model.name}.pth")
            if model_path.exists():
                try:
                    model.load(str(model_path))
                except Exception as e:
                    logger.warning(f"Failed to load sub-model {model.name}: {e}")
            else:
                 logger.warning(f"Sub-model file not found: {model_path}")
