"""
Logging utilities for RueAI.

Usage:
    from utils.logger import get_logger
    logger = get_logger("module.name")  # -> RueAI.module.name
"""

import logging
import logging.handlers
import os
import re
from pathlib import Path
from typing import List, TYPE_CHECKING

from utils.errors import Result, Ok

if TYPE_CHECKING:
    from config import LoggingConfig

# Constants
_ROOT = "RueAI"


def setup(
        log_dir: str,
        log_file: str,
        rotation: str,
        retention: int,
        max_total_size: int,
        level: int
) -> Result[None]:
    """
    Configure the root RueAI logger.
    
    Args:
        rotation: Rotation interval ('midnight', 'H', 'D', 'M', 'S')
        retention: Days of logs to keep.
        max_total_size: Max total size of all logs in MB.
    """
    root = logging.getLogger(_ROOT)

    # Clear existing handlers
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)

    root.setLevel(level)

    # Setup file handler
    log_path = Path(log_dir) / log_file
    # NOTE: Will raise OSError if failed
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate backupCount based on retention and rotation
    backup_count = retention
    rotation_upper = rotation.upper()

    if rotation_upper == 'H':
        backup_count = retention * 24
    elif rotation_upper == 'M':
        backup_count = retention * 24 * 60
    elif rotation_upper == 'S':
        backup_count = retention * 24 * 60 * 60

    file_handler = logging.handlers.TimedRotatingFileHandler(
        str(log_path),
        when=rotation,
        interval=1,
        backupCount=backup_count,
        encoding='utf-8'
    )

    # Configure naming format: filename.YYYY-MM-DD.log
    # Default suffix is %Y-%m-%d_%H-%M-%S or similar depending on 'when'
    # We enforce %Y-%m-%d for 'midnight' or 'D'
    if rotation_upper in ('MIDNIGHT', 'D'):
        file_handler.suffix = "%Y-%m-%d"
        # Regex to match the suffix for deletion
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    # Custom namer to change filename.log.YYYY-MM-DD -> filename.YYYY-MM-DD.log
    def custom_namer(default_name: str) -> str:
        # default_name is the full path of the rotated file (e.g., /path/to/RueAI.log.2025-01-01)
        default_path = Path(default_name)
        dir_path = default_path.parent
        filename = default_path.name

        # log_file is the configured log filename (e.g., RueAI.log)
        # We need to extract the suffix added by TimedRotatingFileHandler
        # The pattern is usually: log_file + "." + suffix
        base_filename = Path(log_file).name

        if filename.startswith(base_filename + "."):
            suffix = filename[len(base_filename) + 1:]
            # If original file ends with .log, insert suffix before it
            if base_filename.endswith(".log"):
                new_filename = f"{base_filename[:-4]}.{suffix}.log"
                return str(dir_path / new_filename)
            else:
                return str(dir_path / f"{base_filename}.{suffix}")

        return default_name

    file_handler.namer = custom_namer

    # Custom rotator to enforce total size limit
    def custom_rotator(source: str, dest: str) -> None:
        # 1. Perform the default rotation (rename)
        if os.path.exists(source):
            os.rename(source, dest)

        # 2. Enforce total size limit
        if max_total_size > 0:
            limit_bytes = max_total_size * 1024 * 1024
            base_name = Path(log_file).stem  # e.g. RueAI

            # Find all relevant log files
            # Matches: RueAI.log, RueAI.YYYY-MM-DD.log...
            # We look for files starting with the stem
            files: List[Path] = []
            for f in log_path.parent.glob(f"{base_name}*"):
                # Double check it belongs to this log file series
                # Should start with base_name and contain log_file suffix or be the log file
                if f.name.startswith(base_name):
                    files.append(f)

            # Sort by modification time (oldest first)
            files.sort(key=lambda file: file.stat().st_mtime)

            total_size = sum(file.stat().st_size for file in files)

            if total_size <= limit_bytes:
                return

            # Delete the oldest files until under limit
            # Exclude the current active log file (dest is the newly rotated file, source was the active one)
            # The current active file is actually recreated empty after rotation by the handler, 
            # but 'log_path' points to it.
            current_log = log_path.resolve()

            deleted_size = 0
            for f in files:
                # SAFETY: don't delete current active log
                if f.resolve() == current_log:
                    continue
                # NOTE: Will raise OSError if failed
                size = f.stat().st_size
                try:
                    f.unlink()
                except OSError:
                    f.rename(f.parent / f"deprecated_{f.name}.old")
                deleted_size += size
                if total_size - deleted_size <= limit_bytes:
                    break

    file_handler.rotator = custom_rotator
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(file_handler)

    return Ok(None)


def configure_logging(cfg: "LoggingConfig") -> Result[None]:
    """Reconfigure logging from LoggingConfig dataclass."""
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    return setup(
        log_dir=cfg.log_dir,
        log_file=cfg.log_file,
        rotation=cfg.rotation,
        retention=cfg.retention,
        max_total_size=cfg.max_total_size,
        level=level,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger under the RueAI namespace.
    
    Args:
        name: Logger name (e.g., "core.manager" -> "RueAI.core.manager")
    """
    if not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)
