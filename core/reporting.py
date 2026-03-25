"""
Reporting and feedback management.
"""

import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

import polars as pl

from config import AppConfig
from utils.logger import get_logger

logger = get_logger("core.reporting")


class ReportHandler:
    """
    Manages writing detection results and reading user feedback.
    
    Persistence:
    - Uses a CSV summary file as the single source of truth for events and feedback.
    - Columns: interval, start_time, end_time, top_k_metrics, is_false_alarm, processed.
    
    Thread Safety:
    - Uses `threading.Lock` to serialize writes to the CSV file, preventing data corruption
      when multiple detection jobs try to append results simultaneously.
    """

    _SUMMARY_COLS = ['interval', 'model', 'start_time', 'end_time', 'top_k_metrics', 'is_false_alarm', 'processed']

    def __init__(self, config: AppConfig):
        self.summary_file = Path(config.detection.summary_file)
        self._lock = threading.Lock()

    def append(self, events: List[Dict[str, Any]]) -> None:
        """
        Thread-safe append of new detection events to the summary CSV.
        
        Args:
            events: List of event dictionaries to append.
        """
        if not events:
            return

        with self._lock:
            self.summary_file.parent.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame(events)
            # Ensure all standard columns exist (fill missing with None)
            for col in self._SUMMARY_COLS:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None).alias(col))

            df = df.select(self._SUMMARY_COLS)
            mode = "a" if self.summary_file.exists() else "w"

            with open(self.summary_file, mode) as f:
                df.write_csv(f, include_header=(mode == "w"))

    def read_pending_feedback(self) -> Optional[pl.DataFrame]:
        """
        Read pending feedback items from the summary file.
        
        Criteria for pending:
        - is_false_alarm == True
        - processed == False
        
        Returns:
            DataFrame with a special `_idx` column containing the original row index, 
            or None if no pending items found.
        """
        if not self.summary_file.exists():
            return None

        try:
            df = pl.read_csv(self.summary_file)
            pending_mask = (pl.col('is_false_alarm') & ~pl.col('processed'))
            pending_df = df.filter(pending_mask)

            if pending_df.is_empty():
                return None

            # Return with row index for accurate update tracking
            return df.with_row_index("_idx").filter(pending_mask)

        except Exception as e:
            logger.error(f"Failed to read summary file: {e}")
            return None

    def mark_processed(self, processed_indices: List[int]) -> None:
        """
        Mark specific rows as processed in the summary file.
        
        Args:
            processed_indices: List of 0-based row indices to mark as processed=True.
        """
        if not processed_indices or not self.summary_file.exists():
            return

        with self._lock:
            try:
                df = pl.read_csv(self.summary_file)
                df_with_idx = df.with_row_index("_idx")

                # Update 'processed' column for matching indices
                updated_df = df_with_idx.with_columns(
                    pl.when(pl.col('_idx').is_in(processed_indices)).then(True).otherwise(pl.col('processed')).alias(
                        'processed')).drop("_idx")

                updated_df.write_csv(self.summary_file)
            except Exception as e:
                logger.error(f"Failed to update summary file: {e}")
