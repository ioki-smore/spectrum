"""
Artifact versioning and backup management.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from utils.errors import ErrorCode, Result, Ok, Err
from utils.logger import get_logger

logger = get_logger("core.artifacts")


class ArtifactVersioner:
    """
    Handles backups, versioning, and rollbacks of model artifacts.
    
    Structure:
    - Base Directory: Contains the active (live) model files.
    - Versions Directory: Contains timestamped subdirectories with backups.
    
    Retention Policy:
    - Controlled by `max_versions` (default 1). Oldest versions are pruned automatically after backup.
    """

    def __init__(self, interval: str, base_dir: Path, max_versions: int = 1):
        """
        Args:
            interval: Interval name (e.g., '1min').
            base_dir: Directory containing the active artifacts.
            max_versions: Number of backup versions to retain.
        """
        self.interval = interval
        self.base_dir = base_dir
        self.versions_dir = base_dir / "versions" / interval
        self.max_versions = max_versions

    def list_versions(self) -> List[str]:
        """
        List available backup versions sorted by newest first.
        
        Returns:
            List of version strings (timestamps).
        """
        if not self.versions_dir.exists():
            return []
        return sorted([d.name for d in self.versions_dir.iterdir() if d.is_dir()], reverse=True)

    def backup_artifacts(self, patterns: List[str], max_versions: int = None) -> Result[None]:
        """
        Create a backup of current artifacts matching the given glob patterns.
        
        Args:
            patterns: List of glob patterns to match files in base_dir.
            max_versions: Optional override for retention limit for this operation.
        """
        version_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir = self.versions_dir / version_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        files_backed_up = False
        for pattern in patterns:
            for file_path in self.base_dir.glob(pattern):
                if file_path.is_file():
                    shutil.copy2(file_path, dest_dir / file_path.name)
                    files_backed_up = True

        if files_backed_up:
            logger.debug(f"[{self.interval}] Backup created: {version_name}")
            self._prune_old_versions(max_versions)

        return Ok(None)

    def rollback(self, version_name: str) -> Result[None]:
        """
        Restore artifacts from a specific backup version.
        Automatically creates a backup of the *current* state before rolling back.
        """
        source_dir = self.versions_dir / version_name
        if not source_dir.exists():
            return Err(ErrorCode.MODEL_NOT_FOUND)

        # Safety backup of current state before overwriting
        self.backup_artifacts([f"{self.interval}_*"])

        for file_path in source_dir.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, self.base_dir / file_path.name)

        logger.info(f"[{self.interval}] Rolled back to {version_name}")
        return Ok(None)

    def _prune_old_versions(self, max_versions: int = None) -> None:
        """Delete old backups exceeding the retention limit."""
        if not self.versions_dir.exists():
            return

        limit = max_versions if max_versions is not None else self.max_versions
        versions = self.list_versions()

        for old_version in versions[limit:]:
            shutil.rmtree(self.versions_dir / old_version, ignore_errors=True)
