import logging
import logging.handlers
import sys
import os
import time
from pathlib import Path
from typing import Optional

# Flag to track if root logger is configured
_root_configured = False

# Default values (will be overridden by config if available)
_DEFAULT_LOG_DIR = "logs"
_DEFAULT_LOG_FILE = "spectrum.log"
# TODO：7天一轮转
_DEFAULT_RETENTION_DAYS = 15
_DEFAULT_MAX_BYTES = 20 * 1024 * 1024  # 20MB
_DEFAULT_BACKUP_COUNT = 5


class TimedRotatingFileHandlerWithCleanup(logging.handlers.TimedRotatingFileHandler):
    """
    Extended TimedRotatingFileHandler that also cleans up old log files
    based on retention_days setting.
    """
    # TODO：文件名加时间戳
    def __init__(self, filename, when='midnight', interval=1, backupCount=0,
                 encoding=None, delay=False, utc=False, atTime=None,
                 retention_days: int = 15):
        super().__init__(filename, when, interval, backupCount, encoding, delay, utc, atTime)
        self.retention_days = retention_days
    
    def doRollover(self):
        super().doRollover()
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove log files older than retention_days."""
        if self.retention_days <= 0:
            return
            
        log_dir = Path(self.baseFilename).parent
        log_name = Path(self.baseFilename).name
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)
        
        try:
            for f in log_dir.glob(f"{log_name}*"):
                if f.is_file() and f.stat().st_mtime < cutoff_time:
                    f.unlink()
        except Exception:
            pass  # Silently ignore cleanup errors


def setup_logger(
    name: str = "spectrum",
    log_dir: str = _DEFAULT_LOG_DIR,
    log_file: str = _DEFAULT_LOG_FILE,
    retention_days: int = _DEFAULT_RETENTION_DAYS,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a logger with the given name.
    
    The root 'spectrum' logger is configured once with:
    - File output only (no console)
    - Daily rotation at midnight
    - Automatic cleanup of logs older than retention_days
    
    Child loggers (e.g., 'spectrum.data.loader') inherit from root and use propagation.
    """
    global _root_configured
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only configure the root 'spectrum' logger once
    if name == "spectrum" and not _root_configured:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create log directory
        log_path = Path(log_dir) / log_file
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use TimedRotatingFileHandler for daily rotation with cleanup
            file_handler = TimedRotatingFileHandlerWithCleanup(
                filename=str(log_path),
                when='midnight',
                interval=1,
                backupCount=backup_count,
                encoding='utf-8',
                retention_days=retention_days
            )
            file_handler.setFormatter(formatter)
            file_handler.suffix = "%Y-%m-%d"  # Log file suffix format
            logger.addHandler(file_handler)
            
        except PermissionError:
            sys.stderr.write(f"Permission denied: Cannot write to log file {log_path}. Logging disabled.\n")
        except Exception as e:
            sys.stderr.write(f"Failed to setup file logging to {log_path}: {e}. Logging disabled.\n")
        
        _root_configured = True
    
    return logger


def configure_logging(log_config) -> None:
    """
    Reconfigure the root logger with settings from LoggingConfig.
    Should be called after config is loaded.
    
    Args:
        log_config: LoggingConfig instance with log_dir, log_file, retention_days, etc.
    """
    global _root_configured
    
    root_logger = logging.getLogger("spectrum")
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    _root_configured = False
    
    # Re-setup with new config
    setup_logger(
        name="spectrum",
        log_dir=log_config.log_dir,
        log_file=log_config.log_file,
        retention_days=log_config.retention_days,
        max_bytes=log_config.max_bytes,
        backup_count=log_config.backup_count
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the 'spectrum' namespace.
    
    Usage:
        logger = get_logger(__name__)  # e.g., 'spectrum.data.loader'
    
    If name doesn't start with 'spectrum.', it will be prefixed.
    """
    if not name.startswith("spectrum"):
        name = f"spectrum.{name}"
    return logging.getLogger(name)


# Initialize root logger with defaults (will be reconfigured when config loads)
logger = setup_logger()
