"""
Pipeline orchestration for a single interval.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
from filelock import FileLock

from config import AppConfig, ModelsConfig
from core.analysis import build_events
from core.artifacts import ArtifactVersioner
from core.postprocess import PostProcessor
from core.state import StateManager
from core.thresholds import ThresholdManager
from data.preprocessor import Preprocessor
from data.loader import DataLoader
from data.processor import DataProcessor
from models import USAD, LSTM, SR, GSR, GSR_AE, AnomalyDetector
from models.historical import HistoricalThresholdModel
from utils.errors import ErrorCode, Result, Ok, Err
from utils.logger import get_logger

logger = get_logger("core.pipeline")

METADATA_COLUMNS = frozenset(['timestamp', 'label', 'time', 'datetime'])


def extract_feature_columns(df: pl.DataFrame) -> List[str]:
    """Return list of columns to be used as features (excluding metadata)."""
    return [c for c in df.columns if c not in METADATA_COLUMNS]


class Pipeline:
    """
    Manages the complete anomaly detection pipeline for a specific time interval.
    
    Responsibilities:
    1. Orchestrate data loading, preprocessing, model training, and inference.
    2. Manage model lifecycle (training, saving, loading, versioning).
    3. Ensure thread-safe updates to the model ("Hot Swap" pattern).
    4. Handle runtime schema changes and errors.
    
    Thread Safety Strategy:
    - Training (Writer): Creates a "Shadow Copy" of the model/processor. Only when training 
      is fully successful and saved to disk does it atomicly swap the reference (`self.model`).
    - Detection (Reader): Captures a local reference to the current `self.model` at the start 
      of execution to ensure consistency throughout the request, even if a hot swap occurs concurrently.
    - File Operations: Uses `FileLock` for critical sections involving file deletion or replacement.
    """

    def __init__(self, interval: str, config: AppConfig):
        self.interval = interval
        self.config = config

        # Paths
        self.model_dir = Path(config.models.save_path)
        self.ensemble_path = self.model_dir / f"{interval}_ensemble.pth"
        self.processor_path = self.model_dir / f"{interval}_processor.joblib"

        # Components
        self.state_manager = StateManager(interval, self.model_dir)
        self.loader = DataLoader(interval, config.data, self.state_manager)
        self.preprocessor_stage1 = Preprocessor(
            mode=str(config.models.get("preprocess_mode", "ratio")),
            baseline_window=int(config.models.get("preprocess_baseline_window", 61)),
            smoothing_window=int(config.models.get("preprocess_smoothing", 0)),
            fill_value=float(config.models.get("preprocess_fill_value", 0.0)),
        )
        self.processor = DataProcessor(method='standard')

        # The active model instance. 
        # READERS (detect) must capture this reference locally.
        # WRITERS (train) must update this reference atomically after training.
        self.model: Optional[AnomalyDetector] = None

        # Helpers
        self.thresholds = ThresholdManager(interval, self.state_manager, config)
        self.versioner = ArtifactVersioner(interval, self.model_dir)
        self.post_processor = PostProcessor(config.postprocessing)

        # Cache for is_trained to avoid repeated disk I/O
        self._trained_cache: Optional[bool] = None

        # Ensure we start from fresh data on startup to avoid processing stale backlogs
        self.loader.skip_to_latest()

    @property
    def is_trained(self) -> bool:
        """
        Check if the pipeline has a valid trained state on disk.
        
        Returns:
            True if ensemble, processor, and thresholds exist; False otherwise.
        """
        if self._trained_cache is not None:
            return self._trained_cache

        if not (
                self.ensemble_path.exists() and self.processor_path.exists() and self.state_manager.threshold is not None):
            self._trained_cache = False
            return False

        for name in self._get_enabled_model_names():
            model_name_prefix = name.upper()
            full_model_name = f"{model_name_prefix}_{self.interval}"
            sub_path = self.model_dir / f"{self.interval}_ensemble_{full_model_name}.pth"

            if not sub_path.exists():
                self._trained_cache = False
                return False

        self._trained_cache = True
        return True

    def _get_enabled_model_names(self) -> List[str]:
        valid_keys = {"usad", "lstm", "sr", "gsr", "gsr_ae"}
        enabled = self.config.models.enabled_models

        if not enabled:
            return list(valid_keys)

        return [m for m in enabled if m in valid_keys]

    def train(self, force: bool = False) -> Result[None]:
        """
        Execute the full training workflow.
        
        Strategy (Shadow Copy):
        1. Load data.
        2. Create a NEW `DataProcessor` instance and fit it.
        3. Create a NEW `AnomalyDetector` instance (Ensemble) and fit it.
        4. Save all artifacts to disk (atomic write).
        5. Compute thresholds using the new model.
        6. HOT SWAP: Update `self.model` and `self.processor` references atomically.
        
        Args:
            force: If True, retrain even if valid artifacts already exist.
            
        Returns:
            Result: Ok(None) on success, Err(ErrorCode) on failure.
        """
        if self.is_trained and not force:
            return Ok(None)

        logger.info(f"[{self.interval}] Starting pipeline training{'(forced)' if force else ''}...")

        # 1. Load Data
        res = self.loader.load_training_data(self.config.training.data_window)
        if res.is_err():
            logger.error(f"[{self.interval}] {res.err_value.message}")
            return Err(res.err_value)
        data = res.unwrap()
        preprocessed = self.preprocessor_stage1.transform(data)

        # 2. Train Processor (Shadow Instance)
        # We use a fresh instance to avoid modifying the active one used by 'detect'
        processor = DataProcessor(method='standard')
        res = processor.fit(preprocessed)
        if res.is_err():
            return Err(res.err_value)

        # Save Processor (Atomic)
        # DataProcessor.save internally uses atomic write-rename
        res = processor.save(str(self.processor_path))
        if res.is_err():
            logger.warning(f"[{self.interval}] Failed to save processor: {res.err_value}")
            return Err(res.err_value)

        # 3. Transform Data
        res = processor.transform(preprocessed)
        if res.is_err():
            return Err(res.err_value)
        transformed_data = res.unwrap()

        # 4. Initialize and Train Model (Shadow Instance)
        feature_cols = extract_feature_columns(transformed_data)

        # Persist schema
        res = self.state_manager.set_feature_columns(feature_cols)
        if res.is_err():
            logger.error(f"[{self.interval}] Failed to save feature columns: {res.err_value}")
            return Err(res.err_value)

        model = self._create_model(len(feature_cols))

        # Persist config snapshot
        config_dict = asdict(self.config.models)
        res = self.state_manager.set_model_config(config_dict)
        if res.is_err():
            logger.error(f"[{self.interval}] Failed to save model config: {res.err_value}")
            return Err(res.err_value)

        # 5. Backup & Save Model
        res = self.versioner.backup_artifacts([f"{self.interval}_*"])
        if res.is_err():
            logger.warning(f"[{self.interval}] Failed to create backup, proceeding anyway: {res.err_value}")

        res = model.fit(transformed_data)
        if res.is_err():
            return Err(res.err_value)

        res = model.save(str(self.ensemble_path))
        if res.is_err():
            return Err(res.err_value)

        # 6. Compute Thresholds (on new model)
        # We run prediction on the training data to establish baseline scores for POT
        res = model.predict(transformed_data)
        if res.is_err():
            return Err(res.err_value)

        res = self.thresholds.compute_pot_threshold(res.unwrap())
        if res.is_err():
            return Err(res.err_value)

        # 7. Hot Swap (Atomic Assignment)
        # This is the point of no return. The new model goes live.
        self.processor = processor
        self.model = model
        self._trained_cache = True

        logger.info(f"[{self.interval}] Pipeline training completed & hot-swapped")
        return Ok(None)

    def detect(self) -> Result[List[Dict]]:
        """
        Run anomaly detection on new data.
        
        Thread Safety:
        - Captures local references (`current_model`, `current_processor`) at start.
        - This guarantees that even if `train()` hot-swaps the model in the middle of this function,
          the detection logic continues with the OLD consistent model state.
          
        Returns:
            Result containing a list of detected anomaly events.
        """
        logger.info(f"[{self.interval}] Running pipeline detection...")

        if not self.is_trained:
            return Err(ErrorCode.MODEL_NOT_FOUND)

        # Capture local references for thread safety
        current_processor = self.processor
        current_model = self.model

        res = self.loader.load_new_data()
        if res.is_err():
            return Ok([])
        data = res.unwrap()

        # Ensure model/processor are loaded if they were None (first run after restart)
        if current_model is None:
            feature_cols = extract_feature_columns(data)
            res_load = self._ensure_model_loaded(len(feature_cols))
            if res_load.is_err():
                return Err(res_load.err_value)
            # Update local references
            current_model = self.model
            current_processor = self.processor

        preprocessed = self.preprocessor_stage1.transform(data)
        res = self._prepare_data(preprocessed, current_processor)

        if res.is_err() and res.err_value == ErrorCode.DATA_INVALID_SCHEMA:
            self._handle_schema_change()
            return Err(ErrorCode.DATA_INVALID_SCHEMA)

        if res.is_err():
            return Err(res.err_value)
        normalized_data = res.unwrap()

        feature_cols = extract_feature_columns(data)

        # Single-pass: get scores + contributions together (avoids double inference)
        res = current_model.predict_with_contributions(normalized_data)
        if res.is_err():
            return Err(res.err_value)

        prediction_result = res.unwrap()
        scores = prediction_result["consensus"]
        details = prediction_result["details"]
        contribs = prediction_result["contributions"]
        model_contribs = prediction_result.get("model_contributions", {})

        if len(scores) == 0:
            return Ok([])

        threshold = self.thresholds.get_threshold_or_default(scores)
        raw_anomalies = scores > threshold

        # Post-Processing: Apply business rules (Amplitude, Frequency)
        # Note: We pass raw data 'data' which contains original values before normalization
        anomalies = self.post_processor.process(raw_anomalies, data, feature_cols)

        model_names = list(details.keys())

        # Build Events
        events = build_events(interval=self.interval, df=data, scores=scores, anomalies=anomalies,
            feature_cols=feature_cols, contributions=contribs, top_k=self.config.detection.top_k,
            model_names=model_names)

        self._save_details(data, anomalies, scores, feature_cols, model_contribs)

        # Decay adapted thresholds toward POT baseline (prevents long-term recall loss)
        self.thresholds.decay_threshold()

        res = self.loader.commit(data['timestamp'].max())
        if res.is_err():
            logger.error(f"[{self.interval}] Failed to commit state: {res.err_value}")

        return Ok(events)

    def _handle_schema_change(self) -> None:
        """
        Handle schema mismatch (e.g., column count changed) by safely resetting the model.
        
        Actions:
        1. Backup current artifacts (retention policy: 1 version).
        2. Acquire FILE LOCK to prevent race conditions with training/saving.
        3. Delete incompatible model/processor files.
        4. Reset in-memory state.
        
        This forces the system to trigger a fresh `train()` on the next schedule.
        """
        logger.warning(f"[{self.interval}] Schema changed, resetting model state...")

        # 1. Backup (keep 1 version)
        res = self.versioner.backup_artifacts([f"{self.interval}_*"], max_versions=1)
        if res.is_err():
            logger.error(f"[{self.interval}] Failed to backup artifacts: {res.err_value}")

        # 2. Delete files with lock safety
        # We lock to ensure we don't delete files while a concurrent training job is writing them
        lock_path = self.model_dir / f".{self.interval}.lock"
        with FileLock(str(lock_path)):
            # Delete ensemble
            if self.ensemble_path.exists():
                self.ensemble_path.unlink()

            # Delete processor
            if self.processor_path.exists():
                self.processor_path.unlink()

            # Delete sub-models
            for p in self.model_dir.glob(f"{self.interval}_ensemble_*.pth"):
                p.unlink()

        # Reset runtime state
        self.model = None
        self.processor = DataProcessor(method='standard')
        self._trained_cache = None

        # Clear threshold state as well since model is gone
        self.state_manager.set_threshold(None)

    def incremental_train(self, start_time: int, end_time: int) -> Result[None]:
        """
        Perform incremental training (fine-tuning) on a specific data range.
        Typically triggered by user feedback (False Alarms).
        
        Args:
            start_time: Start timestamp (ms).
            end_time: End timestamp (ms).
            
        Returns:
            Result: Ok(None) on success.
        """
        logger.info(f"[{self.interval}] Incremental pipeline training: {start_time}-{end_time}")

        # Load specific data range
        data_path = self.loader.subdir_path if self.loader.use_subdir else self.loader.root_path
        df = (pl.scan_csv(str(data_path / "*.csv")).filter(
            (pl.col('timestamp') >= start_time) & (pl.col('timestamp') <= end_time)).collect())
        if df.is_empty():
            return Err(ErrorCode.DATA_NOT_FOUND)

        # Use current processor (incremental train usually doesn't retrain the scaler)
        preprocessed = self.preprocessor_stage1.transform(df)
        res = self._prepare_data(preprocessed, self.processor)
        if res.is_err():
            return Err(res.err_value)
        normalized_df = res.unwrap()

        feature_cols = extract_feature_columns(df)

        # Ensure we have a base model to fine-tune
        res = self._ensure_model_loaded(len(feature_cols))
        if res.is_err():
            return Err(res.err_value)

        # Shadow Copy for training
        # We create a new instance and load state to avoid mutating self.model in place while detect is running
        model = self._create_model(len(feature_cols))

        # Load current state into shadow model
        if self.ensemble_path.exists():
            res = model.load(str(self.ensemble_path))
            if res.is_err(): return Err(res.err_value)

        # Backup before overwriting
        res = self.versioner.backup_artifacts([f"{self.interval}_*"])
        if res.is_err():
            logger.warning(f"[{self.interval}] Failed to create backup, proceeding anyway: {res.err_value}")

        # Train (Fine-tune)
        # update_normalization=False prevents shifting the Z-score mean/std on small feedback batches
        res: Result[None] = model.fit(normalized_df, update_normalization=False)
        if res.is_err():
            return Err(res.err_value)

        # Save with Lock
        lock_path = self.model_dir / f".{self.interval}.lock"
        with FileLock(str(lock_path)):
            res: Result[None] = model.save(str(self.ensemble_path))
            if res.is_err():
                return Err(res.err_value)

        # Update Thresholds adaptively
        res = model.predict(normalized_df)
        if res.is_ok():
            self.thresholds.adapt_threshold(res.unwrap())
        else:
            logger.warning(f"[{self.interval}] Failed to predict for threshold update: {res.err_value}")

        # Hot Swap
        self.model = model

        logger.info(f"[{self.interval}] Incremental training completed")
        return Ok(None)

    def _prepare_data(self, df: pl.DataFrame, processor: Optional[DataProcessor] = None) -> Result[pl.DataFrame]:
        proc = processor or self.processor
        if not proc.fitted:
            if not self.processor_path.exists():
                return Err(ErrorCode.PROCESSOR_LOAD_FAILED)
            res = proc.load(str(self.processor_path))
            if res.is_err():
                return Err(res.err_value)

        current = set(extract_feature_columns(df))
        trained = set(proc.columns)
        if current != trained:
            return Err(ErrorCode.DATA_INVALID_SCHEMA)

        return proc.transform(df)

    def _create_model(self, input_dim: int, config: Optional[ModelsConfig] = None) -> AnomalyDetector:
        conf = config or self.config.models

        model_map = {"usad": lambda: USAD(f"USAD_{self.interval}", conf, input_dim),
            "lstm": lambda: LSTM(f"LSTM_{self.interval}", conf, input_dim),
            "sr": lambda: SR(f"SR_{self.interval}", conf, input_dim),
            "gsr": lambda: GSR(f"GSR_{self.interval}", conf, input_dim),
            "gsr_ae": lambda: GSR_AE(f"GSR_AE_{self.interval}", conf, input_dim),
            "historical": lambda: HistoricalThresholdModel(f"HISTORICAL_{self.interval}", conf), }

        enabled = conf.enabled_models or list(model_map.keys())
        models = [model_map[name]() for name in enabled if name in model_map]

        if not models:
            logger.warning(f"[{self.interval}] No valid models enabled, using all")
            models = [fn() for fn in model_map.values()]

        return AnomalyDetector(name=f"AnomalyDetector_{self.interval}", models=models, config=conf)

    def _ensure_model_loaded(self, input_dim: int) -> Result[None]:
        if self.model is None:
            if not self.ensemble_path.exists():
                return Err(ErrorCode.MODEL_NOT_FOUND)

            config = self.config.models
            saved_config_dict = self.state_manager.model_config

            if saved_config_dict:
                valid_keys = ModelsConfig.__annotations__.keys()
                filtered = {k: v for k, v in saved_config_dict.items() if k in valid_keys}
                extras = {k: v for k, v in saved_config_dict.items() if k not in valid_keys}

                config = ModelsConfig(**filtered)
                config.extra_params = extras
                logger.debug(f"[{self.interval}] Loaded model config from state")

            self.model = self._create_model(input_dim, config=config)
            res = self.model.load(str(self.ensemble_path))
            if res.is_err():
                return Err(ErrorCode.MODEL_LOAD_FAILED)
        return Ok(None)

    _MAX_DETAILS_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

    def _save_details(self, df: pl.DataFrame, anomalies: np.ndarray, scores: np.ndarray,
                      feature_cols: List[str],
                      model_contributions: Dict[str, np.ndarray]) -> None:
        """
        Save per-algorithm detail files.
        
        Each algorithm gets its own file: {algo}_{interval}_details.csv
        Only anomaly rows are recorded. Columns: timestamp + each feature metric name.
        Values are per-feature anomaly scores (contributions) from that algorithm.
        
        Args:
            df: Source DataFrame with timestamps.
            anomalies: Boolean mask of detected anomalies.
            scores: Consensus anomaly scores (aligned with df tail).
            feature_cols: List of feature/metric names.
            model_contributions: {model_name: (n_samples, n_features) ndarray}
        """
        if not np.any(anomalies):
            return

        summary_path = Path(self.config.detection.summary_file)
        result_dir = summary_path.parent
        result_dir.mkdir(parents=True, exist_ok=True)

        offset = max(0, len(df) - len(scores))
        timestamps = df['timestamp'][offset:].to_numpy()

        # Indices of anomaly rows
        anom_idx = np.where(anomalies)[0]
        anom_timestamps = timestamps[anom_idx]

        for model_name, contrib_matrix in model_contributions.items():
            # contrib_matrix: (n_samples, n_features)
            if contrib_matrix is None or len(contrib_matrix) == 0:
                continue

            # Extract only anomaly rows
            anom_contribs = contrib_matrix[anom_idx]

            # Build DataFrame: timestamp + per-feature scores
            series_list = [pl.Series("timestamp", anom_timestamps)]
            n_feat = min(anom_contribs.shape[1], len(feature_cols))
            for i in range(n_feat):
                series_list.append(pl.Series(feature_cols[i], anom_contribs[:, i]))

            detail_df = pl.DataFrame(series_list)

            file_path = result_dir / f"{model_name}_{self.interval}_details.csv"

            # Rotate if file too large
            if file_path.exists() and file_path.stat().st_size > self._MAX_DETAILS_FILE_SIZE:
                import time
                rotated = result_dir / f"{model_name}_{self.interval}_details_{int(time.time())}.csv"
                file_path.rename(rotated)

            mode = "a" if file_path.exists() else "w"
            try:
                with open(file_path, mode) as f:
                    detail_df.write_csv(f, include_header=(mode == "w"))
            except Exception as e:
                logger.warning(f"[{self.interval}] Failed to write {model_name} details: {e}. Overwriting.")
                with open(file_path, "w") as f:
                    detail_df.write_csv(f, include_header=True)
