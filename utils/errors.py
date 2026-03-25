"""Error codes and result types for RueAI."""

import logging
import sys
from enum import IntEnum
from typing import TypeVar, Optional, Tuple

from result import Result as Result, Ok, Err

__all__ = [
    'ErrorCode',
    'handle_exception',
    'status_to_exit_code',
    'status_from_exception',
    'Result',
    'Ok',
    'Err'
]


class ErrorCode(IntEnum):
    """Error codes for RueAI operations."""

    # Configuration errors (-1xx)
    CONFIG_INVALID = -100
    CONFIG_MISSING = -101

    # Data errors (-2xx)
    DATA_NOT_FOUND = -200
    DATA_INSUFFICIENT = -201
    DATA_INVALID_SCHEMA = -202
    DATA_NON_COMPLIANT = -203
    DATA_NON_CONSECUTIVE = -204
    DATA_PERMISSION_DENIED = -205
    DATA_LOAD_FAILED = -206
    DATA_TRANSFORM_FAILED = -207

    # Model errors (-3xx)
    MODEL_NOT_FOUND = -300
    MODEL_LOAD_FAILED = -301
    MODEL_SAVE_FAILED = -302
    MODEL_TRAIN_FAILED = -303
    MODEL_PREDICT_FAILED = -304
    MODEL_NOT_READY = -305

    # Processor errors (-4xx)
    PROCESSOR_NOT_FITTED = -400
    PROCESSOR_FIT_FAILED = -401
    PROCESSOR_LOAD_FAILED = -402
    PROCESSOR_SAVE_FAILED = -403
    PROCESSOR_MISSING_COLUMNS = -404

    # Threshold errors (-5xx)
    THRESHOLD_NOT_FOUND = -500
    THRESHOLD_SAVE_FAILED = -501
    THRESHOLD_LOAD_FAILED = -502

    # IO errors (-6xx)
    IO_READ_FAILED = -600
    IO_WRITE_FAILED = -601
    IO_PERMISSION_DENIED = -602

    # Logging errors (-7xx)
    LOG_SETUP_FAILED = -700

    # Service errors (-8xx)
    SERVICE_INIT_FAILED = -800
    SERVICE_NO_INTERVALS = -801
    SERVICE_SCHEDULER_FAILED = -802

    # General errors (-9xx)
    TIMEOUT_ERROR = -900
    UNKNOWN_ERROR = -999

    @property
    def message(self) -> str:
        """Default message for this error code."""
        return _ERROR_MESSAGES.get(self, f"Error {self.name}")

    def __str__(self) -> str:
        return f"{self.name}({self.value})"


# Default error messages
_ERROR_MESSAGES = {
    ErrorCode.CONFIG_INVALID: "Configuration is invalid",
    ErrorCode.CONFIG_MISSING: "Configuration file not found",
    ErrorCode.DATA_NOT_FOUND: "Data not found",
    ErrorCode.DATA_INSUFFICIENT: "Insufficient data for operation",
    ErrorCode.DATA_INVALID_SCHEMA: "Invalid data schema",
    ErrorCode.DATA_NON_COMPLIANT: "Data does not meet quality requirements",
    ErrorCode.DATA_NON_CONSECUTIVE: "Non-consecutive data detected",
    ErrorCode.DATA_PERMISSION_DENIED: "Permission denied accessing data",
    ErrorCode.DATA_LOAD_FAILED: "Failed to load data",
    ErrorCode.DATA_TRANSFORM_FAILED: "Failed to transform data",
    ErrorCode.MODEL_NOT_FOUND: "Model not found",
    ErrorCode.MODEL_LOAD_FAILED: "Failed to load model",
    ErrorCode.MODEL_SAVE_FAILED: "Failed to save model",
    ErrorCode.MODEL_TRAIN_FAILED: "Model training failed",
    ErrorCode.MODEL_PREDICT_FAILED: "Model prediction failed",
    ErrorCode.MODEL_NOT_READY: "Model is not ready for inference",
    ErrorCode.PROCESSOR_NOT_FITTED: "Processor not fitted",
    ErrorCode.PROCESSOR_FIT_FAILED: "Failed to fit processor",
    ErrorCode.PROCESSOR_LOAD_FAILED: "Failed to load processor",
    ErrorCode.PROCESSOR_SAVE_FAILED: "Failed to save processor",
    ErrorCode.PROCESSOR_MISSING_COLUMNS: "Missing required columns",
    ErrorCode.THRESHOLD_NOT_FOUND: "Threshold not found",
    ErrorCode.THRESHOLD_SAVE_FAILED: "Failed to save threshold",
    ErrorCode.THRESHOLD_LOAD_FAILED: "Failed to load threshold",
    ErrorCode.IO_READ_FAILED: "Failed to read file",
    ErrorCode.IO_WRITE_FAILED: "Failed to write file",
    ErrorCode.IO_PERMISSION_DENIED: "Permission denied",
    ErrorCode.LOG_SETUP_FAILED: "Failed to setup logging",
    ErrorCode.SERVICE_INIT_FAILED: "Service initialization failed",
    ErrorCode.SERVICE_NO_INTERVALS: "No data intervals discovered",
    ErrorCode.SERVICE_SCHEDULER_FAILED: "Scheduler failed to start",
    ErrorCode.TIMEOUT_ERROR: "Operation timed out",
    ErrorCode.UNKNOWN_ERROR: "Unknown error",
}

T = TypeVar('T')
# Define Result as a type alias for Result[T, ErrorCode]
Result = Result[T, ErrorCode]


def status_to_exit_code(code: int | ErrorCode) -> int:
    """
    Convert internal ErrorCode to system exit code (0-255).
    """
    return abs(int(code))


def status_from_exception(exception: BaseException) -> Tuple[ErrorCode, str]:
    """Map exception to ErrorCode and message."""
    msg = str(exception) if str(exception) else None

    if isinstance(exception, FileNotFoundError):
        return ErrorCode.DATA_NOT_FOUND, msg or ErrorCode.DATA_NOT_FOUND.message
    if isinstance(exception, PermissionError):
        return ErrorCode.IO_PERMISSION_DENIED, msg or ErrorCode.IO_PERMISSION_DENIED.message
    if isinstance(exception, (IOError, OSError)):
        return ErrorCode.IO_READ_FAILED, msg or ErrorCode.IO_READ_FAILED.message
    if isinstance(exception, ValueError):
        return ErrorCode.CONFIG_INVALID, msg or ErrorCode.CONFIG_INVALID.message
    if isinstance(exception, TimeoutError):
        return ErrorCode.TIMEOUT_ERROR, msg or ErrorCode.TIMEOUT_ERROR.message
    if isinstance(exception, RuntimeError):
        return ErrorCode.SERVICE_INIT_FAILED, msg or ErrorCode.SERVICE_INIT_FAILED.message

    return ErrorCode.UNKNOWN_ERROR, msg or ErrorCode.UNKNOWN_ERROR.message


def handle_exception(
        exception: BaseException,
        *,
        logger: Optional[logging.Logger] = None,
        print_to_stderr: bool = False,
) -> int:
    """
    Centralized exception handler.
    Logs the exception and returns an appropriate exit code.
    If logging is not configured (no handlers), automatically prints to stderr.
    """
    if isinstance(exception, KeyboardInterrupt):
        if logger is not None:
            logger.info("Interrupted by user")
        # Ensure interruption is visible even if logging fails
        if print_to_stderr or (logger and not logger.hasHandlers()):
            print("Interrupted by user", file=sys.stderr)
        return 130

    if isinstance(exception, SystemExit):
        return exception.code if isinstance(exception.code, int) else 1

    log = logger or logging.getLogger(__name__)
    code, message = status_from_exception(exception)

    # Check if logger has any handlers configured (handling propagation)
    has_handlers = False
    c = log
    while c:
        if c.handlers:
            has_handlers = True
            break
        if not c.propagate:
            break
        c = c.parent

    # Force stderr if no handlers or explicitly requested
    should_print = print_to_stderr or not has_handlers

    log.exception(f"Unhandled exception: {message}")

    if should_print:
        # Avoid duplicate printing if we just logged to a console handler (not the case here usually)
        print(f"Error: {message} ({code.name})", file=sys.stderr)

    return status_to_exit_code(code)
