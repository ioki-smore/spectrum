"""Data loading and validation for time-series anomaly detection."""

from pathlib import Path
from typing import List, Optional

import polars as pl

from config import DataConfig
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.state import StateManager
from utils.errors import ErrorCode, Result, Ok, Err
from utils.logger import get_logger

logger = get_logger("data.loader")

_MAX_LOSS = 0.001  # Max 0.1% data loss allowed


def _parse_interval_ms(interval: str) -> int:
    """Parse interval string (e.g., '5s', '1min') to milliseconds."""
    units = {'ms': 1, 's': 1000, 'min': 60_000, 'h': 3_600_000}
    for suffix, mult in units.items():
        if interval.endswith(suffix):
            return int(interval[:-len(suffix)]) * mult
    return 1000


class DataLoader:
    """Loads and validates time-series data for a specific sampling interval."""

    def __init__(self, interval: str, data_config: DataConfig, state_manager: 'StateManager'):
        self.interval = interval
        self.interval_ms = _parse_interval_ms(interval)
        self._points_per_day = 86_400_000 // self.interval_ms

        # Paths
        self._root = Path(data_config.source_path)
        self._subdir = self._root / interval
        self._use_subdir = self._subdir.is_dir()

        # State
        self.state_manager = state_manager

    # ==================== Properties ====================

    @property
    def last_timestamp(self) -> int:
        return self.state_manager.last_timestamp

    @property
    def subdir_path(self) -> Path:
        return self._subdir

    @property
    def use_subdir(self) -> bool:
        return self._use_subdir

    @property
    def root_path(self) -> Path:
        return self._root

    # ==================== Public API ====================

    def load_training_data(self, duration: int = 7) -> Result[pl.DataFrame]:
        """Load consecutive compliant daily files for training, scanning backwards."""
        files = self._get_files()
        if not files:
            return Err(ErrorCode.DATA_NOT_FOUND)

        # Scan backwards from latest
        valid_data: List[pl.DataFrame] = []

        for f in reversed(files):
            df = self._load_and_validate(f, self._points_per_day)

            if df is None:
                # If we hit a bad file and we already have some data, 
                # we can't extend the consecutive sequence further back.
                # But maybe we haven't found *any* valid block yet, so we just skip this bad file.
                # However, if we already have a block, a bad file breaks the chain.
                if valid_data:
                    # Chain broken. If we have enough, we are good. If not, we discard the current block and restart.
                    # Actually, simplest strategy: Just keep looking for the *latest* valid block of size N.
                    if len(valid_data) >= duration:
                        break
                    valid_data = []  # Reset
                continue

            # Check consecutive with the previously added (newer) file
            if valid_data:
                # Check Schema Consistency
                if df.columns != valid_data[-1].columns:
                    # Schema mismatch breaks chain
                    if len(valid_data) >= duration:
                        break
                    valid_data = [df] # Start over with this schema
                    continue

                prev_start = valid_data[-1]['timestamp'].min()
                curr_end = df['timestamp'].max()
                gap = prev_start - curr_end

                if gap > 2 * self.interval_ms:
                    # Gap detected.
                    if len(valid_data) >= duration:
                        break
                    # Current block is insufficient and broken. Start new block with this file.
                    valid_data = [df]
                else:
                    valid_data.append(df)
            else:
                valid_data.append(df)

            if len(valid_data) == duration:
                break

        if len(valid_data) < duration:
            return Err(ErrorCode.DATA_INSUFFICIENT)

        # valid_data is in reverse order (newest to oldest)
        # We need to return sorted: oldest to newest
        valid_data.reverse()

        logger.info(f"Loaded {len(valid_data)} files for training (required: {duration})")
        return Ok(pl.concat(valid_data).sort('timestamp'))

    def load_new_data(self) -> Result[pl.DataFrame]:
        """Load unprocessed data since last_timestamp."""
        files = self._get_files()
        if not files:
            return Err(ErrorCode.DATA_NOT_FOUND)

        logger.info(f"load_new_data: last_ts={self.last_timestamp}, found {len(files)} files")

        dfs = []
        current_cols = None
        
        for f in files:
            # We don't enforce strict density for incremental detection, just valid schema
            df = self._load_and_validate(f)
            if df is None:
                continue

            last_timestamp = df['timestamp'].max()
            logger.info(f"Checking {f.name}: max_ts={last_timestamp}, cols={df.columns}, width={df.width}")

            if last_timestamp <= self.last_timestamp:
                logger.info(f"  -> Skipped (max <= last_ts)")
                continue

            new = df.filter(pl.col('timestamp') > self.last_timestamp)
            if not new.is_empty():
                # Enforce schema consistency within the batch
                if current_cols is None:
                    current_cols = new.columns
                elif new.columns != current_cols:
                    logger.warning(f"  -> Schema mismatch in batch ({new.columns} != {current_cols}). Stopping batch here.")
                    # Return what we have so far
                    break
                
                logger.info(f"  -> Loaded {len(new)} new rows")
                dfs.append(new)
            else:
                logger.info(f"  -> Skipped (empty after filter)")

        if not dfs:
            return Err(ErrorCode.DATA_NOT_FOUND)

        try:
            result = pl.concat(dfs).sort('timestamp')
            return Ok(result)
        except Exception as e:
            logger.error(f"Concat failed: {e}")
            for i, d in enumerate(dfs):
                logger.error(f"DF[{i}]: width={d.width}, cols={d.columns}")
            raise e

    def commit(self, timestamp: int) -> Result[None]:
        """Update last processed timestamp and save state."""
        if timestamp > self.last_timestamp:
            return self.state_manager.update_last_timestamp(timestamp)
        return Ok(None)

    def reset_state(self) -> None:
        """Reset processing state."""
        self.state_manager.clear()

    def skip_to_latest(self) -> None:
        """
        Fast-forward the processing state to the start of the latest available data file.
        This ensures that detection starts from current data instead of processing old backlogs.
        """
        files = self._get_files()
        if not files:
            return

        latest_file = files[-1]
        df = self._load_and_validate(latest_file)
        if df is not None and not df.is_empty():
            start_ts = df['timestamp'].min()
            # Set to start_ts - 1 so > comparison works for the first point
            target_ts = start_ts - 1

            # Only update if we are behind
            if self.last_timestamp < target_ts:
                logger.info(
                    f"Fast-forwarding state from {self.last_timestamp} to {target_ts} (start of {latest_file.name})")
                self.state_manager.update_last_timestamp(target_ts)

    def _get_files(self) -> List[Path]:
        """Get sorted CSV files."""
        if not self._root.exists():
            return []

        if self._use_subdir:
            files = list(self._subdir.glob("*.csv"))
        else:
            pattern = f"*{self.interval}.csv"
            files = [f for f in self._root.glob(pattern) if
                f.stem.endswith(f"-{self.interval}") or f.stem.endswith(f"_{self.interval}")]

        return sorted(files, key=lambda p: p.name)

    def _load_and_validate(self, path: Path, expected_count: Optional[int] = None) -> Optional[pl.DataFrame]:
        """
        Load CSV, standardize schema, validate quality.
        Unexpected errors (IO, parsing) propagate to main.
        """
        df = pl.read_csv(path)

        # Standardize
        df = self._standardize_schema(df, path.name)
        if df is None:
            return None

        # Density Check
        if expected_count is not None:
            if not self._check_density(df, expected_count, path.name):
                return None

        # Interpolate if needed
        if expected_count is None:
            ts_range = df['timestamp'].max() - df['timestamp'].min()
            local_expected = ts_range // self.interval_ms + 1 if ts_range > 0 else len(df)
        else:
            local_expected = expected_count

        loss = 0.0
        if local_expected > 0:
            loss = (local_expected - len(df)) / local_expected

        if loss > _MAX_LOSS:
            logger.warning(f"{path.name}: {loss:.1%} loss (got {len(df)}, need {local_expected})")
            return None

        if loss > 0:
            df = self._interpolate(df)

        return df

    def _standardize_schema(self, df: pl.DataFrame, filename: str) -> Optional[pl.DataFrame]:
        """Ensure dataframe has timestamp column and is sorted."""
        if 'timestamp' not in df.columns:
            if 'datetime' in df.columns:
                # Let parse errors bubble up
                df = df.with_columns(
                    pl.col('datetime').str.strptime(pl.Datetime, "%Y-%m-%d_%H:%M:%S", strict=False).dt.epoch(
                        "ms").alias('timestamp')).drop('datetime')

                # Check for nulls if strict=False allowed bad formats
                if df['timestamp'].null_count() > 0:
                    logger.warning(f"Failed to parse some timestamps in {filename}")
                    return None
            else:
                logger.warning(f"No timestamp/datetime column in {filename}")
                return None

        df = df.sort('timestamp')
        if df.is_empty():
            return None
        return df

    def _check_density(self, df: pl.DataFrame, expected: int, filename: str) -> bool:
        """Validate data density."""
        if expected <= 0:
            return True

        loss = (expected - len(df)) / expected
        if loss > _MAX_LOSS:
            logger.warning(f"{filename}: High data loss ({loss:.1%}). Expected ~{expected}, got {len(df)}")
            return False
        return True

    def _interpolate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill small gaps via interpolation."""
        return (df.with_columns(pl.col('timestamp').cast(pl.Datetime("ms")).alias('_dt')).upsample(time_column='_dt',
                                                                                                   every=f"{self.interval_ms}ms").interpolate().fill_null(
            strategy="forward").with_columns(pl.col('_dt').dt.epoch("ms").alias('timestamp')).drop('_dt'))
