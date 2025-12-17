import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import shutil
from typing import Any, Dict, List, Optional

from config import config
from data.loader import DataLoader
from data.processor import DataProcessor
from models import USAD, LSTM, SR, Ensemble
from utils.logger import get_logger
from utils.thresholding import fit_pot

logger = get_logger("core.manager")

class IntervalManager:
    """Manages training and detection for a specific sampling interval."""
    
    # Maximum number of model versions to keep
    MAX_MODEL_VERSIONS = 3
    
    def __init__(self, interval: str):
        self.interval = interval
        self.loader = DataLoader(interval)
        self.processor = DataProcessor(method='standard')
        
        # Model setup
        model_config = config.models
        self.model = None
        self.save_path = Path(model_config.save_path)
        
        # Current model paths
        self.model_path = self.save_path / f"{interval}_ensemble.pth"
        self.processor_path = self.save_path / f"{interval}_processor.joblib"
        self.threshold_path = self.save_path / f"{interval}_threshold.json"
        
        # Version history directory
        self.versions_path = self.save_path / "versions" / interval

    def _save_threshold(self, threshold: float) -> None:
        """Save threshold value to JSON file."""
        try:
            self.threshold_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.threshold_path, 'w') as f:
                json.dump({"threshold": float(threshold), "updated_at": datetime.now().isoformat()}, f)
        except Exception as e:
            logger.error(f"Failed to save threshold for {self.interval}: {e}")

    def _load_threshold(self) -> Optional[float]:
        if not self.threshold_path.exists():
            return None
        try:
            with open(self.threshold_path, 'r') as f:
                data = json.load(f)
                return data.get("threshold")
        except Exception as e:
            logger.error(f"Failed to load threshold for {self.interval}: {e}")
            return None

    def _create_version_backup(self) -> Optional[str]:
        """Create a versioned backup of current model files before updating.
        
        Returns:
            Version string (timestamp) if backup was created, None otherwise.
        """
        if not self.model_path.exists():
            return None
            
        try:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_dir = self.versions_path / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy current files to version directory
            if self.model_path.exists():
                # Copy all ensemble-related files
                for f in self.save_path.glob(f"{self.interval}_*"):
                    if f.is_file():
                        shutil.copy2(f, version_dir / f.name)
            
            logger.info(f"Created model backup version: {version}")
            
            # Cleanup old versions
            self._cleanup_old_versions()
            
            return version
        except Exception as e:
            logger.warning(f"Failed to create version backup: {e}")
            return None
    
    def _cleanup_old_versions(self) -> None:
        """Remove old versions beyond MAX_MODEL_VERSIONS."""
        if not self.versions_path.exists():
            return
            
        try:
            versions = sorted([d for d in self.versions_path.iterdir() if d.is_dir()], reverse=True)
            for old_version in versions[self.MAX_MODEL_VERSIONS:]:
                shutil.rmtree(old_version)
                logger.debug(f"Removed old model version: {old_version.name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup old versions: {e}")
    
    def list_versions(self) -> List[str]:
        """List available model versions.
        
        Returns:
            List of version strings (timestamps), newest first.
        """
        if not self.versions_path.exists():
            return []
        return sorted([d.name for d in self.versions_path.iterdir() if d.is_dir()], reverse=True)
    
    def rollback_to_version(self, version: str) -> bool:
        """Rollback model to a specific version.
        
        Args:
            version: Version string (timestamp) to rollback to.
            
        Returns:
            True if rollback succeeded, False otherwise.
        """
        version_dir = self.versions_path / version
        if not version_dir.exists():
            logger.error(f"Version {version} not found for {self.interval}")
            return False
            
        try:
            # Backup current before rollback
            self._create_version_backup()
            
            # Restore files from version
            for f in version_dir.glob(f"{self.interval}_*"):
                if f.is_file():
                    shutil.copy2(f, self.save_path / f.name)
            
            # Reload model
            self.model = None  # Force re-initialization
            logger.info(f"Rolled back {self.interval} to version {version}")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback to version {version}: {e}")
            return False

    def _init_model(self, input_dim: int, load_weights: bool = False) -> None:
        """Initializes the Ensemble model if not already initialized."""
        if self.model is None:
            # TODO: consider making this configurable or dynamic
            m_usad = USAD(name=f"USAD_{self.interval}", config=config.models, input_dim=input_dim)
            m_lstm = LSTM(name=f"LSTM_{self.interval}", config=config.models, input_dim=input_dim)
            m_sr = SR(name=f"SR_{self.interval}", config=config.models, input_dim=input_dim)
            # TODO：改个名字
            self.model = Ensemble(
                name=f"Ensemble_{self.interval}", 
                models=[m_usad, m_lstm, m_sr],
                config=config.models
            )
            
        if load_weights and self.model_path.exists():
            # TODO：改为错误码，不要直接抛异常
            try:
                self.model.load(str(self.model_path))
            except Exception as e:
                logger.error(f"Failed to load model for {self.interval}: {e}")

    @property
    def is_trained(self) -> bool:
        """Checks if the model and threshold exist."""
        return self.model_path.exists() and self.threshold_path.exists()

    def train(self) -> bool:
        """
        Trains the model if requirements are met. 
        Returns True if model is ready (was already trained or just trained), False otherwise.
        """
        if self.is_trained:
            return True

        logger.info(f"Checking training requirements for interval {self.interval}...")
        
        try:
            # 2. Check data sufficiency (Must have enough compliant files)
            duration_str = config.training.data_window
            df = self.loader.load_training_data(duration_str)
            
            if df is None or df.is_empty():
                logger.debug(f"Insufficient data for training interval {self.interval}")
                return False

            # Preprocess
            logger.info(f"Preprocessing data for {self.interval}...")
            try:
                self.processor.fit(df)
                self.processor.save(str(self.processor_path))
                df_norm = self.processor.transform(df)
            except Exception as e:
                logger.error(f"Preprocessing failed for {self.interval}: {e}")
                return False
            
            # Initialize model
            numeric_cols = [c for c in df.columns if c not in ['timestamp', 'label', 'time']]
            if not numeric_cols:
                logger.error(f"No numeric columns found for training in {self.interval}")
                return False
            
            input_dim = len(numeric_cols)
            self._init_model(input_dim)
            
            # Train
            try:
                # Backup existing model before overwriting (for version management)
                self._create_version_backup()
                
                # Fit ensemble
                self.model.fit(df_norm)
                self.model.save(str(self.model_path))
                
                # Calculate Threshold with POT
                logger.info(f"Calculating dynamic threshold (POT) for {self.interval}...")
                scores = self.model.predict(df_norm)
                
                pot_risk = config.models.get("pot_risk", 1e-4)
                pot_level = config.models.get("pot_level", 0.98)
                
                threshold = fit_pot(scores, risk=pot_risk, level=pot_level)
                self._save_threshold(threshold)
                
                logger.info(f"Training completed for {self.interval}")
                return True
            except Exception as e:
                logger.error(f"Model training failed for {self.interval}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error during training setup for {self.interval}: {e}")
            return False

    def incremental_train(self, start_time: int, end_time: int) -> bool:
        """
        Retrains the model on a specific data range marked as normal (false alarm).
        
        Returns:
            True if incremental training succeeded, False otherwise.
        """
        logger.info(f"Incremental training for {self.interval} on range {start_time}-{end_time}")
        
        # Load specific data range
        # Use the correct path based on loader's configuration
        try:
            data_path = self.loader.subdir_path if self.loader.use_subdir else self.loader.root_path
            q = pl.scan_csv(str(data_path / "*.csv"))
            q = q.filter((pl.col('timestamp') >= start_time) & (pl.col('timestamp') <= end_time))
            df = q.collect()
            
            if df.is_empty():
                logger.warning(f"No data found for incremental training in {self.interval}")
                return False

            # Preprocess
            if not self.processor.fitted:
                self.processor.load(str(self.processor_path))
                
            df_norm = self.processor.transform(df)
            
            # Ensure model is initialized
            input_dim = len([c for c in df.columns if c not in ['timestamp', 'label', 'time']])
            self._init_model(input_dim, load_weights=True)
            
            # Backup before incremental training
            self._create_version_backup()
            
            # Train (Fine-tune)
            # Ensemble fit with update_stats=False to preserve global stats.
            # We don't want to skew Z-score normalization with statistics from a small feedback batch.
            self.model.fit(df_norm, update_stats=False)
            self.model.save(str(self.model_path))
            
            # Update threshold
            # These are false alarms, so they should NOT be anomalies anymore.
            # Check scores of these instances.
            scores = self.model.predict(df_norm)
            max_new_score = float(scores.max()) if len(scores) > 0 else 0.0
            
            current_threshold = self._load_threshold()
            
            # If current threshold detects these as anomalies, bump it up
            if current_threshold is not None:
                if max_new_score > current_threshold:
                    # Add a small margin
                    new_threshold = max_new_score * 1.05
                    logger.info(f"Updating threshold from {current_threshold:.6f} to {new_threshold:.6f} to suppress false alarms.")
                    self._save_threshold(new_threshold)
            else:
                 # Should not happen if trained, but safety fallback
                 self._save_threshold(max_new_score * 1.05)

            logger.info(f"Incremental training completed for {self.interval}")
            return True
            
        except Exception as e:
            logger.error(f"Error during incremental training: {e}")
            return False

    def detect(self) -> List[Dict]:
        """
        Runs detection on new data.
        Returns list of anomaly events (dicts).
        """
        logger.info(f"Running detection for interval {self.interval}...")
        
        try:
            # Load model/processor if not loaded
            if self.model is None and not self.model_path.exists():
                 logger.warning(f"Model not found for {self.interval}, skipping detection.")
                 return []
    
            # Load new data
            df = self.loader.load_new_data()
            if df is None or df.is_empty():
                logger.debug(f"No new data for detection in {self.interval}")
                return []
    
            # Load processor
            if not self.processor.fitted:
                if self.processor_path.exists():
                    try:
                        self.processor.load(str(self.processor_path))
                    except Exception as e:
                        logger.error(f"Failed to load processor for {self.interval}: {e}")
                        return []
                else:
                    logger.warning(f"Processor not found at {self.processor_path}, cannot normalize data.")
                    return []
    
            # Transform
            try:
                df_norm = self.processor.transform(df)
            except Exception as e:
                logger.error(f"Data transformation failed for {self.interval}: {e}")
                return []
            
            # Ensure model is initialized and loaded
            numeric_cols = [c for c in df.columns if c not in ['timestamp', 'label', 'time']]
            input_dim = len(numeric_cols)
            
            self._init_model(input_dim, load_weights=True)
    
            # Predict
            try:
                scores = self.model.predict(df_norm)
            except Exception as e:
                logger.error(f"Prediction failed for {self.interval}: {e}")
                return []
                
            if len(scores) == 0:
                 logger.warning("Model returned empty scores.")
                 return []
            
            # Thresholding
            try:
                threshold = self._load_threshold()
                if threshold is None:
                    logger.warning(f"No stored threshold for {self.interval}, falling back to 3-sigma.")
                    mean_score = scores.mean()
                    std_score = scores.std()
                    threshold = mean_score + 3 * std_score
                
                anomalies = scores > threshold
            except Exception as e:
                logger.error(f"Thresholding failed for {self.interval}: {e}")
                return []
            
            # Identify Intervals
            events = self._find_anomaly_intervals(df, anomalies, scores)
            
            if not events:
                return []
                
            # For each event, calculate Top K contributions
            try:
                contributions = self.model.get_contribution(df_norm) # (n_samples, n_features)
            except Exception as e:
                logger.error(f"Failed to get contributions: {e}")
                contributions = None

            final_events = []
            
            # We need to align contributions/scores with df
            # Model prediction has window_size offset (len(scores) < len(df))
            # We aligned events to score indices.
            # Let's align dataframe timestamps to score indices.
            
            # The model output corresponds to the end of window?
            # Assuming stride 1.
            # If df has N points, scores has N - W + 1 points.
            # The score at index i corresponds to window df[i : i+W].
            # Usually we assign the anomaly to the last point timestamp df[i+W-1].
            
            offset = len(df) - len(scores)
            
            for evt in events:
                # evt has start_idx, end_idx relative to scores array
                
                # Map to timestamps
                # df index = score_index + offset (if we map to end of window)
                ts_start = df['timestamp'][evt['start_idx'] + offset]
                ts_end = df['timestamp'][evt['end_idx'] + offset]
                
                # Get top k metrics
                top_k_str = ""
                if contributions is not None and len(contributions) == len(scores):
                    # Slice contributions for this event
                    evt_contrib = contributions[evt['start_idx'] : evt['end_idx'] + 1]
                    # Average over the event
                    avg_contrib = np.mean(evt_contrib, axis=0)
                    
                    # Get top k
                    k = config.detection.top_k
                    # Get indices of top k
                    top_indices = np.argsort(avg_contrib)[-k:][::-1]
                    
                    top_metrics = [numeric_cols[idx] for idx in top_indices]
                    top_k_str = ";".join(top_metrics)
                
                final_events.append({
                    "interval": self.interval,
                    "start_time": ts_start,
                    "end_time": ts_end,
                    "top_k_metrics": top_k_str,
                    "is_false_alarm": False,
                    "processed": False
                })

            # Save detailed results (full timeseries with anomalies marked)
            try:
                self._save_details(df, anomalies, scores)
            except Exception as e:
                 logger.error(f"Failed to save details for {self.interval}: {e}")
            
            return final_events
            
        except Exception as e:
            logger.error(f"Unexpected error during detection for {self.interval}: {e}")
            return []

    def _find_anomaly_intervals(self, df, anomalies, scores) -> List[Dict]:
        """
        Finds continuous anomaly segments.
        Returns list of dicts: {'start_idx': int, 'end_idx': int} (indices into scores array)
        """
        events = []
        in_event = False
        start_idx = 0
        
        for i, is_anom in enumerate(anomalies):
            if is_anom and not in_event:
                in_event = True
                start_idx = i
            elif not is_anom and in_event:
                in_event = False
                events.append({'start_idx': start_idx, 'end_idx': i - 1})
                
        if in_event:
            events.append({'start_idx': start_idx, 'end_idx': len(anomalies) - 1})
            
        return events

    def _save_details(self, df: pl.DataFrame, anomalies, scores):
        try:
            result_dir = Path("results")
            result_dir.mkdir(parents=True, exist_ok=True)
            file_path = result_dir / f"{self.interval}_details.csv"
            
            # Align df with scores (due to windowing)
            # scores length = N - window_size + 1 (assuming stride 1)
            # We align to the last timestamp of the window
            if len(df) > len(scores):
                offset = len(df) - len(scores)
                df_subset = df.slice(offset)
            else:
                df_subset = df

            # Create result DF
            res_df = df_subset.select(['timestamp']).with_columns([
                pl.Series("score", scores),
                pl.Series("is_anomaly", anomalies)
            ])
            
            # Append to CSV
            if not file_path.exists():
                res_df.write_csv(file_path)
            else:
                with open(file_path, "a") as f:
                    res_df.write_csv(f, include_header=False)
        except Exception as e:
             raise IOError(f"Failed to save details to {file_path}: {e}")


class Manager:
    def __init__(self):
        # Auto-infer intervals from data source directory
        try:
            source_path = Path(config.data.source_path)
            if source_path.exists() and source_path.is_dir():
                # 1. Try subdirectories
                found_intervals = [d.name for d in source_path.iterdir() if d.is_dir()]
                
                # 2. If no subdirectories, try parsing filenames (e.g., *_metrics-10s.csv)
                if not found_intervals:
                    import re
                    # Pattern: ...-{interval}.csv
                    files = [f.name for f in source_path.glob("*.csv")]
                    inferred = set()
                    for f in files:
                        match = re.search(r'[-_]([0-9]+[a-zA-Z]+)\.csv$', f)
                        if match:
                            inferred.add(match.group(1))
                    if inferred:
                        found_intervals = sorted(list(inferred))

                if found_intervals:
                    self.intervals = sorted(found_intervals)
                    logger.info(f"Auto-inferred sampling intervals: {self.intervals}")
                else:
                    logger.warning(f"No interval directories or matching files found in {source_path}. No intervals to process.")
                    self.intervals = []
            else:
                logger.warning(f"Source path {source_path} not found. No intervals to process.")
                self.intervals = []
        except Exception as e:
            logger.error(f"Error auto-inferring intervals: {e}. No intervals to process.")
            self.intervals = []

        self.managers = {i: IntervalManager(i) for i in self.intervals}
        self.summary_file = Path(config.detection.summary_file)
        
    def train_all(self) -> Dict[str, bool]:
        """Train all interval managers.
        
        Returns:
            Dict mapping interval to training success status.
        """
        results = {}
        for interval, mgr in self.managers.items():
            try:
                results[interval] = mgr.train()
            except Exception as e:
                logger.error(f"Failed to train {interval}: {e}")
                results[interval] = False
        return results

    def detect_all(self) -> List[Dict[str, Any]]:
        """Run detection on all interval managers.
        
        Returns:
            List of all detected anomaly events.
        """
        summaries: List[Dict[str, Any]] = []
        for interval, mgr in self.managers.items():
            try:
                events = mgr.detect()
                if events:
                    summaries.extend(events)
            except Exception as e:
                logger.error(f"Failed to detect {interval}: {e}")
        
        if summaries:
            self._update_summary(summaries)
        
        return summaries

    def run_pipeline(self) -> List[Dict[str, Any]]:
        """
        Runs the full anomaly detection pipeline for all intervals.
        1. Check if training is needed/possible.
        2. If trained (or just finished), run detection.
        
        Returns:
            List of all detected anomaly events.
        """
        summaries: List[Dict[str, Any]] = []
        for interval, mgr in self.managers.items():
            try:
                # Attempt training (will skip if already trained)
                # train() now returns True if model is ready, False otherwise
                is_ready = mgr.train()
                
                if is_ready:
                    # Run detection
                    # TODO：根据模型保存目录是否有来进行检测
                    events = mgr.detect()
                    if events:
                        summaries.extend(events)
                else:
                    # Log at debug level to avoid spamming if data is simply insufficient
                    logger.debug(f"Skipping detection for {interval} as model is not ready.")
            except Exception as e:
                logger.error(f"Pipeline failed for {interval}: {e}")
        
        if summaries:
            self._update_summary(summaries)
        
        return summaries

    def _update_summary(self, summaries: List[Dict[str, Any]]) -> None:
        try:
            self.summary_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Columns: interval, start_time, end_time, top_k_metrics, is_false_alarm, processed
            new_df = pl.DataFrame(summaries)
            
            # Ensure columns order
            cols = ['interval', 'start_time', 'end_time', 'top_k_metrics', 'is_false_alarm', 'processed']
            
            # Check if columns exist and add missing if any (for robustness)
            for c in cols:
                if c not in new_df.columns:
                    new_df = new_df.with_columns(pl.lit(None).alias(c))
            
            new_df = new_df.select(cols)
            
            if not self.summary_file.exists():
                new_df.write_csv(self.summary_file)
            else:
                with open(self.summary_file, "a") as f:
                    new_df.write_csv(f, include_header=False)
        except Exception as e:
            logger.error(f"Failed to update summary file {self.summary_file}: {e}")

    def process_feedback(self) -> int:
        """
        Reads summary file, checks for false alarms that haven't been processed.
        If is_false_alarm is True and processed is False -> Incremental Train -> Mark processed.
        
        Returns:
            Number of feedback items processed.
        """
        if not self.summary_file.exists():
            return 0

        try:
            # Read all summaries
            df = pl.read_csv(self.summary_file)
            
            # Check schema (loose check to allow migration)
            if 'is_false_alarm' not in df.columns or 'processed' not in df.columns:
                logger.error("Summary file missing required columns for feedback processing.")
                return 0

            # Filter for pending feedback
            pending = df.filter((pl.col('is_false_alarm') == True) & (pl.col('processed') == False))
            
            if pending.is_empty():
                return 0

            processed_count = 0
            processed_indices = []
            
            logger.info(f"Found {len(pending)} pending feedback items.")
            
            # Get row indices for pending items
            df_with_idx = df.with_row_index("_idx")
            pending_with_idx = df_with_idx.filter(
                (pl.col('is_false_alarm') == True) & (pl.col('processed') == False)
            )
            
            for row in pending_with_idx.iter_rows(named=True):
                interval = row['interval']
                start = row['start_time']
                end = row['end_time']
                idx = row['_idx']
                
                if interval in self.managers:
                    logger.info(f"Processing feedback for {interval} (row {idx})...")
                    success = self.managers[interval].incremental_train(start, end)
                    if success:
                        processed_indices.append(idx)
                        processed_count += 1
                else:
                    logger.warning(f"Unknown interval '{interval}' in feedback, skipping.")
            
            # Update only successfully processed rows
            if processed_indices:
                df = df.with_row_index("_idx").with_columns(
                    pl.when(pl.col('_idx').is_in(processed_indices))
                    .then(True)
                    .otherwise(pl.col('processed'))
                    .alias('processed')
                ).drop("_idx")
                
                # Rewrite summary file
                df.write_csv(self.summary_file)
                logger.info(f"Feedback processed: {processed_count} items updated.")
            
            return processed_count

        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return 0

# Lazy initialization pattern for Manager singleton
# This avoids initialization at import time and allows for better testing/mocking
_manager_instance: Optional[Manager] = None


def get_manager() -> Manager:
    """Get or create the global Manager instance.
    
    Uses lazy initialization to avoid issues with import-time execution
    and to allow for easier testing/mocking.
    
    Returns:
        The global Manager instance.
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = Manager()
    return _manager_instance


def reset_manager() -> None:
    """Reset the global Manager instance.
    
    Useful for testing or when configuration changes require re-initialization.
    """
    global _manager_instance
    _manager_instance = None


# Backward compatibility alias (deprecated, use get_manager() instead)
# This property-like access will still work but logs a deprecation warning on first use
class _ManagerProxy:
    """Proxy class for backward compatibility with 'manager' global access."""
    
    _warned = False
    
    def __getattr__(self, name):
        if not self._warned:
            logger.debug("Direct 'manager' access is deprecated. Use get_manager() instead.")
            _ManagerProxy._warned = True
        return getattr(get_manager(), name)


manager = _ManagerProxy()
