import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.errors import ErrorCode, Result, Ok, Err
from utils.logger import get_logger

logger = get_logger("core.state")


class StateManager:
    """
    Centralized state management for a specific detection interval.
    
    Purpose:
    - Persists critical runtime metadata that must survive service restarts.
    - Serves as the source of truth for "watermarks" (last processed timestamp).
    - Stores model-related metadata (thresholds, feature schema, config snapshots) to ensure
      consistency between training and detection phases.
      
    Concurrency:
    - Thread-safe: Uses `threading.RLock` to protect shared in-memory state.
    - Process-safe (mostly): Uses atomic file writes to prevents file corruption, 
      though not strictly multi-process safe without external file locks (which Pipeline handles).
    """

    def __init__(self, interval: str, save_directory: Path):
        """
        Initialize the StateManager.
        
        Args:
            interval: Unique identifier for the interval (e.g., '1min', '15min').
            save_directory: Directory where the state JSON file will be persisted.
        """
        self.interval = interval
        self.state_file_path = save_directory / f"{interval}_state.json"

        # In-memory cache of the state
        self._state_data: Dict[str, Any] = {}

        # Reentrant lock for thread safety
        self._lock = threading.RLock()

        # Load existing state immediately
        self._load_state()

    def _load_state(self) -> None:
        """
        Load state from disk if it exists.
        Handles JSON parsing errors gracefully by resetting to empty state.
        """
        with self._lock:
            if not self.state_file_path.exists():
                self._state_data = {}
                return

            try:
                content = self.state_file_path.read_text()
                self._state_data = json.loads(content)
            except Exception as e:
                logger.error(f"Failed to load state file {self.state_file_path}: {e}")
                # Fallback to empty state to allow service to start, 
                # though this means re-processing might happen.
                self._state_data = {}

    def save(self) -> Result[None]:
        """
        Atomically save the current state to disk.
        
        Strategy:
        1. Write to a temporary file (`.tmp`).
        2. Use `os.replace` (atomic on POSIX) to rename temp file to target file.
        
        This prevents data corruption if the process crashes during write.
        """
        with self._lock:
            try:
                self.state_file_path.parent.mkdir(parents=True, exist_ok=True)

                state_to_save = self._state_data.copy()
                state_to_save["last_updated"] = datetime.now().isoformat()

                # Atomic write sequence
                temp_path = self.state_file_path.with_suffix('.tmp')
                temp_path.write_text(json.dumps(state_to_save, indent=2))

                # Atomic replacement
                temp_path.replace(self.state_file_path)

                return Ok(None)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                return Err(ErrorCode.IO_WRITE_FAILED)

    # --- Type-Safe Accessors ---
    # Each accessor manages its own locking to ensure atomic read/write of individual fields.

    @property
    def last_timestamp(self) -> int:
        """
        The timestamp (ms epoch) of the last successfully processed data point.
        Used by DataLoader to determine where to start reading new data.
        """
        with self._lock:
            return self._state_data.get("last_timestamp", 0)

    def update_last_timestamp(self, value: int) -> Result[None]:
        """Update last_timestamp and immediately persist to disk."""
        with self._lock:
            self._state_data["last_timestamp"] = value
            return self.save()

    @property
    def threshold(self) -> Optional[float]:
        """
        The current global anomaly score threshold.
        Scores above this value are considered anomalies.
        """
        with self._lock:
            return self._state_data.get("threshold")

    def set_threshold(self, value: Optional[float]) -> Result[None]:
        """Update the anomaly threshold and persist."""
        with self._lock:
            self._state_data["threshold"] = value
            return self.save()

    @property
    def feature_columns(self) -> List[str]:
        """
        List of feature column names used by the model.
        Essential for validating that inference data matches training schema.
        """
        with self._lock:
            return self._state_data.get("feature_columns", [])

    def set_feature_columns(self, cols: List[str]) -> Result[None]:
        """Update feature columns and persist."""
        with self._lock:
            self._state_data["feature_columns"] = cols
            return self.save()

    @property
    def model_config(self) -> Dict[str, Any]:
        """
        Snapshot of the model configuration used for training.
        Used to reconstruct the model architecture during loading.
        """
        with self._lock:
            return self._state_data.get("model_config", {})

    def set_model_config(self, config: Dict[str, Any]) -> Result[None]:
        """Update model config snapshot and persist."""
        with self._lock:
            self._state_data["model_config"] = config
            return self.save()

    # --- Generic Accessors ---

    def get_section(self, key: str) -> Any:
        """Retrieve a raw section of the state dictionary."""
        with self._lock:
            return self._state_data.get(key)

    def set_section(self, key: str, value: Any) -> Result[None]:
        """Set a raw section of the state and persist."""
        with self._lock:
            self._state_data[key] = value
            return self.save()

    def clear(self) -> None:
        """
        Destructive: Clear the in-memory state and delete the persistent file.
        Used during hard resets or schema changes.
        """
        with self._lock:
            self._state_data = {}
            try:
                if self.state_file_path.exists():
                    self.state_file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete state file {self.state_file_path}: {e}")
