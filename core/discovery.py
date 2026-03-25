"""
Interval discovery components.
"""

import re
from pathlib import Path
from typing import List

from config import AppConfig
from utils.logger import get_logger

logger = get_logger("core.discovery")


class IntervalDiscovery:
    """
    Discovers data intervals (time granularities) from the source directory.
    
    Strategies:
    1. Subdirectories: Looks for folders like '1min', '5min', '1h'.
    2. Filename Patterns: Looks for files ending with '_{interval}.csv' or '-{interval}.csv'.
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def discover(self) -> List[str]:
        """
        Scan source path for valid intervals.
        
        Logic:
        - If subdirectories exist, they are treated as interval names.
        - If no subdirectories, scans CSV files for interval suffixes.
        
        Returns:
            List[str]: A sorted list of unique interval IDs found (e.g., ['1min', '5min']).
        """
        source = Path(self.config.data.source_path)
        if not source.exists():
            return []

        # Priority 1: Subdirectories (Preferred structure)
        # e.g. /data/1min/, /data/5min/
        if dirs := [d.name for d in source.iterdir() if d.is_dir()]:
            return sorted(dirs)

        # Priority 2: Filename patterns (Flat structure)
        # e.g. data_1min.csv, metrics-5min.csv
        # Regex captures the suffix between [- or _] and .csv
        pattern = re.compile(r'[-_]([0-9]+[a-zA-Z]+)\.csv$')
        intervals = {m.group(1) for f in source.glob("*.csv") if (m := pattern.search(f.name))}
        return sorted(intervals)
