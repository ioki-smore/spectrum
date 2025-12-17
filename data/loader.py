import polars as pl
from pathlib import Path
from typing import List, Optional
import re
import json
from datetime import datetime, timedelta
import logging

from config import config

# Use standard getLogger to rely on parent "spectrum" logger configuration (from utils.logger)
# to avoid duplicate logs.
logger = logging.getLogger("spectrum.data.loader")

class DataLoader:
    """Handles loading and validation of time-series data for a specific interval."""
    # TODO：模型参数，指标名称，数据读取持久化
    def __init__(self, interval: str, state_dir: Optional[Path] = None):
        self.interval = interval
        self.root_path = Path(config.data.source_path)
        # Check if we should use subdirectory or flat file structure
        self.subdir_path = self.root_path / interval
        self.use_subdir = self.subdir_path.exists() and self.subdir_path.is_dir()
        self.interval_ms = self._parse_interval_ms(interval)
        
        # Directory for persisting state (configurable for testing)
        self._state_dir = state_dir if state_dir is not None else Path(config.models.save_path)
        
        # State file for persisting last_timestamp across restarts
        self._state_file = self._state_dir / f"{interval}_loader_state.json"
        self._last_timestamp = self._load_state()

    def _parse_interval_ms(self, interval: str) -> int:
        """Parses interval string (e.g., '10s', '1min') to milliseconds."""
        try:
            if interval.endswith("min"):
                return int(interval[:-3]) * 60 * 1000
            elif interval.endswith("s"):
                return int(interval[:-1]) * 1000
            elif interval.endswith("h"):
                return int(interval[:-1]) * 3600 * 1000
            elif interval.endswith("ms"):
                return int(interval[:-2])
            else:
                # Default fallback or error
                logger.warning(f"Unknown interval format {interval}, defaulting to 1000 ms check.")
                return 1000
        except ValueError:
            logger.error(f"Failed to parse interval {interval}, defaulting to 1000 ms")
            return 1000

    def _get_files(self) -> List[Path]:
        try:
            if not self.root_path.exists():
                logger.warning(f"Source path {self.root_path} does not exist.")
                return []
                
            if self.use_subdir:
                # Subdirectory mode
                files = sorted(self.subdir_path.glob("*.csv"))
            else:
                # Flat file mode: look for *_{interval}.csv or *-{interval}.csv
                files = sorted([
                    p for p in self.root_path.glob(f"*{self.interval}.csv") 
                    if p.name.endswith(f"-{self.interval}.csv") or p.name.endswith(f"_{self.interval}.csv")
                ])
                
            if not files:
                logger.debug(f"No CSV files found for interval {self.interval} in {self.root_path}")
            return files
        except PermissionError:
            logger.error(f"Permission denied accessing {self.root_path}")
            return []
        except Exception as e:
            logger.error(f"Error accessing source path {self.root_path}: {e}")
            return []

    @property
    def _points_per_24h(self) -> int:
        if self.interval_ms <= 0:
            return 0
        ms_in_day = 24 * 3600 * 1000
        return int(ms_in_day / self.interval_ms)
    # TODO：修改参数，传点数
    def _validate_and_fix_daily_data(self, df: pl.DataFrame, filename: str, check_full_day: bool = True) -> Optional[pl.DataFrame]:
        """
        Validates data density and fixes gaps.
        
        Args:
            df: DataFrame to validate.
            filename: Name of the file (for logging).
            check_full_day: 
                If True, compares count against a full 24h expected count (strict for training).
                If False, compares against expected count for the data's own duration (density check for detection).
        """
        if df.is_empty():
            return None
            
        actual = len(df)
        
        if check_full_day:
            # Expect ~24h worth of data
            expected = self._points_per_24h
        else:
            # Expect density based on time range covered
            min_ts = df['timestamp'].min()
            max_ts = df['timestamp'].max()
            if min_ts is None or max_ts is None or min_ts == max_ts:
                # Single point or invalid
                return df
            
            duration_ms = max_ts - min_ts
            # expected points = duration / interval + 1 (fencepost)
            expected = int(duration_ms / self.interval_ms) + 1
            
        if expected == 0:
            return df
            
        missing = expected - actual
        loss_ratio = missing / expected
        
        # Check compliance (allow 1% loss)
        if loss_ratio > 0.01:
            if check_full_day:
                logger.warning(f"File {filename} non-compliant (Full Day): {loss_ratio:.2%} data loss (Actual: {actual}, Expected: {expected}). Discarding.")
            else:
                logger.warning(f"Data chunk {filename} non-compliant (Density): {loss_ratio:.2%} data loss (Actual: {actual}, Expected: {expected}). Discarding.")
            return None
            
        # Interpolate if needed (missing > 0 but <= 1%)
        if loss_ratio > 0:
            logger.info(f"File {filename} has {loss_ratio:.2%} gaps. Interpolating...")
            try:
                df_interp = (
                    df.with_columns(pl.col('timestamp').cast(pl.Datetime("ms")).alias('datetime'))
                    .sort('datetime')
                    .upsample(time_column='datetime', every=f"{self.interval_ms}ms")
                    .interpolate()
                    .fill_null(strategy="forward")
                    .with_columns(pl.col('datetime').cast(pl.Int64).alias('timestamp'))
                    .drop('datetime')
                )
                return df_interp
            except Exception as e:
                logger.error(f"Interpolation failed for {filename}: {e}")
                return None
                
        return df

    def _standardize_schema(self, q: pl.LazyFrame) -> Optional[pl.LazyFrame]:
        """Helper to standardize schema on a LazyFrame"""
        try:
            schema = q.collect_schema()
            if 'timestamp' not in schema.names():
                if 'datetime' in schema.names():
                    q = q.with_columns(
                        pl.col('datetime')
                        .str.strptime(pl.Datetime, format="%Y-%m-%d_%H:%M:%S")
                        .cast(pl.Datetime("ms"))
                        .cast(pl.Int64)
                        .alias('timestamp')
                    ).drop('datetime')
                else:
                    return None
            return q
        except Exception:
            return None

    def load_training_data(self, duration: str = "7d") -> Optional[pl.DataFrame]:
        """
        Load data for training.
        Requires 'duration' compliant daily files (e.g., 7 days -> 7 compliant files).
        Files are compliant if data loss <= 1% of a full 24h cycle.
        """
        files = self._get_files()
        if not files:
            return None
            
        # Parse requirement
        required_count = 7
        # TODO：简化，直接默认天
        if duration.endswith('d'):
            try:
                required_count = int(duration[:-1])
            except ValueError:
                pass
        
        valid_dfs = []
        
        for f in files:
            try:
                q = pl.scan_csv(str(f))
                q = self._standardize_schema(q)
                if q is None:
                    logger.warning(f"Skipping {f.name}: Invalid schema")
                    continue
                    
                df = q.collect()
                # Sort to ensure check works correctly
                if 'timestamp' in df.columns:
                    df = df.sort('timestamp')
                    
                # Enforce Full Day Check for Training
                processed_df = self._validate_and_fix_daily_data(df, f.name, check_full_day=True)
                if processed_df is not None:
                    valid_dfs.append(processed_df)
                    
            except Exception as e:
                logger.error(f"Error processing file {f.name}: {e}")
                continue
        # TODO: 
        # TODO: 只能连续的文件，不根据文件数量判断，直接报错，log中error缺失哪天
        # TODO：删除无用info，改为debug
        if len(valid_dfs) < required_count:
            logger.debug(f"Insufficient compliant files for {self.interval}. Found {len(valid_dfs)}, required {required_count}.")
            return None
            
        logger.debug(f"Found {len(valid_dfs)} compliant files for training (Required: {required_count}).")
        
        try:
            full_df = pl.concat(valid_dfs)
            return full_df.sort('timestamp')
        except Exception as e:
            logger.error(f"Error concatenating training data: {e}")
            return None

    def load_new_data(self) -> Optional[pl.DataFrame]:
        """
        Load data that hasn't been processed yet.
        Applies strict validation and interpolation to ensure data quality.
        Uses density check (not full day) since new data might be partial.
        """
        last_timestamp = self.last_timestamp
        files = self._get_files()
        if not files:
            return None

        valid_dfs = []
        
        # We process files to find new data.
        # Optimization: We could filter files based on modification time or name if possible,
        # but safely scanning all and filtering by timestamp is robust for now.
        
        for f in files:
            try:
                q = pl.scan_csv(str(f))
                q = self._standardize_schema(q)
                if q is None:
                    continue
                
                df = q.collect()
                if 'timestamp' in df.columns:
                    df = df.sort('timestamp')
                    max_ts = df['timestamp'].max()
                    if max_ts is not None and max_ts <= last_timestamp:
                        continue # Skip file if all data is old
                        
                # Validate and fix
                processed_df = self._validate_and_fix_daily_data(df, f.name, check_full_day=False)
                
                if processed_df is not None:
                    # Filter new data
                    new_data = processed_df.filter(pl.col('timestamp') > last_timestamp)
                    if not new_data.is_empty():
                        valid_dfs.append(new_data)
                        
            except Exception as e:
                logger.error(f"Error processing file {f.name} for new data: {e}")
                continue

        if not valid_dfs:
            return None
            
        try:
            full_df = pl.concat(valid_dfs).sort('timestamp')
            
            if not full_df.is_empty():
                new_max = full_df['timestamp'].max()
                if new_max is not None:
                    self.last_timestamp = new_max
            
            return full_df
            
        except Exception as e:
            logger.error(f"Error concatenating new data: {e}")
            return None

    @property
    def last_timestamp(self) -> int:
        """Get the last processed timestamp."""
        return self._last_timestamp
    
    @last_timestamp.setter
    def last_timestamp(self, value: int) -> None:
        """Set and persist the last processed timestamp."""
        self._last_timestamp = value
        self._save_state()
    
    def _load_state(self) -> int:
        """Load persisted state from file."""
        if self._state_file.exists():
            try:
                with open(self._state_file, 'r') as f:
                    data = json.load(f)
                    ts = data.get("last_timestamp", 0)
                    logger.debug(f"Loaded state for {self.interval}: last_timestamp={ts}")
                    return ts
            except Exception as e:
                logger.warning(f"Failed to load state for {self.interval}: {e}")
        return 0
    
    def _save_state(self) -> None:
        """Persist state to file."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, 'w') as f:
                json.dump({
                    "last_timestamp": self._last_timestamp,
                    "updated_at": datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save state for {self.interval}: {e}")
    
    def reset_state(self) -> None:
        """Reset the last_timestamp state (useful for reprocessing)."""
        self._last_timestamp = 0
        if self._state_file.exists():
            try:
                self._state_file.unlink()
            except Exception:
                pass
