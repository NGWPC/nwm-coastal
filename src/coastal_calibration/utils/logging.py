"""Logging and monitoring utilities for coastal calibration workflow."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

    from coastal_calibration.config.schema import MonitoringConfig


__all__ = [
    "ProgressBar",
    "StageProgress",
    "StageStatus",
    "WorkflowMonitor",
    "configure_logger",
    "generate_log_path",
    "get_log_file_path",
    "logger",
    "silence_third_party_loggers",
]

# Module-level logger
logger = logging.getLogger("coastal_calibration")
logger.setLevel(logging.DEBUG)  # Let handlers control filtering
logger.propagate = False

_file_handler: logging.FileHandler | None = None
_console_handler: logging.Handler | None = None

if not logger.handlers:
    _console_handler = RichHandler(
        console=Console(stderr=True, force_jupyter=False, soft_wrap=True),
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        log_time_format="[%Y/%m/%d %H:%M:%S]",
        omit_repeated_times=False,
    )
    _console_handler.setFormatter(logging.Formatter("%(message)s"))

    _console_handler.setLevel(logging.WARNING)
    logger.addHandler(_console_handler)


def generate_log_path(work_dir: Path, prefix: str = "coastal-calibration") -> Path:
    """Generate a timestamped log file path.

    Parameters
    ----------
    work_dir : Path
        Directory where the log file will be created.
    prefix : str, optional
        Prefix for the log file name. Defaults to "coastal-calibration".

    Returns
    -------
    Path
        Path to the log file (e.g., <work_dir>/coastal-calibration-20260206-140112.log).
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return work_dir / f"{prefix}-{timestamp}.log"


def get_log_file_path() -> Path | None:
    """Get the current log file path if file logging is enabled.

    Returns
    -------
    Path or None
        Path to the current log file, or None if file logging is not enabled.
    """
    if _file_handler is not None:
        return Path(_file_handler.baseFilename)
    return None


def _validate_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | int) -> int:
    """Validate and convert logging level to integer.

    Parameters
    ----------
    level : str or int
        Logging level as string or integer.

    Returns
    -------
    int
        Validated logging level as integer constant.

    Raises
    ------
    ValueError
        If the provided level is not valid.
    TypeError
        If the provided level is not a string or integer.
    """
    if isinstance(level, str):
        level_upper = level.upper()
        if level_upper not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            msg = (
                f"Invalid log level: {level}. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
            raise ValueError(msg)
        return getattr(logging, level_upper)

    if isinstance(level, int):
        if level not in (
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ):
            msg = f"Invalid log level: {level}. Must be a valid logging level constant."
            raise ValueError(msg)
        return level

    msg = f"Level must be str or int, got {type(level).__name__}"
    raise TypeError(msg)


def configure_logger(
    *,
    verbose: bool | None = None,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | int | None = None,
    file: str | Path | None = None,
    file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | int | None = None,
    file_mode: Literal["a", "w"] = "a",
    file_only: bool = False,
) -> None:
    """Configure logging settings.

    Parameters
    ----------
    level : str or int, optional
        Set console logging level. Valid options: ``DEBUG``, ``INFO``, ``WARNING``,
        ``ERROR``, ``CRITICAL``, or their integer equivalents.
    verbose : bool, optional
        Shortcut to set console level to DEBUG (True) or WARNING (False).
        If both ``level`` and ``verbose`` are provided, ``level`` takes precedence.
    file : str or Path, optional
        Path to log file. If provided, enables file logging.
        Pass ``None`` to disable file logging.
    file_level : str or int, optional
        Logging level for file handler. Defaults to DEBUG if not specified.
    file_mode : {'a', 'w'}, optional
        File mode: 'a' for append, 'w' for overwrite. Defaults to 'a'.
    file_only : bool, optional
        If True, disable console logging and only log to file.
        Requires ``file`` to be specified. Defaults to False.

    Raises
    ------
    ValueError
        If invalid level or file_level is provided.
    TypeError
        If level or file_level is not a string or integer.

    Notes
    -----
    The logger itself is set to DEBUG level, allowing handlers to independently
    control what messages they receive. This means file logging can capture
    DEBUG messages even when console is set to WARNING.

    Examples
    --------
    >>> # Enable verbose logging
    >>> configure_logger(verbose=True)

    >>> # Set specific level
    >>> configure_logger(level="INFO")

    >>> # Enable file logging (captures all levels by default)
    >>> configure_logger(verbose=True, file="debug.log")

    >>> # Console shows warnings, file captures everything
    >>> configure_logger(verbose=False, file="full.log", file_level="DEBUG")

    >>> # File logging with custom level
    >>> configure_logger(file="errors.log", file_level="ERROR", file_mode="w")

    >>> # Disable file logging
    >>> configure_logger(file=None)
    """
    global _file_handler  # noqa: PLW0603

    if level is not None:
        level_int = _validate_level(level)
        if _console_handler is not None:
            _console_handler.setLevel(level_int)
    elif verbose is not None:
        if _console_handler is not None:
            _console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)

    if file is not None:
        if _file_handler is not None:
            logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None

        file_level_int = _validate_level(file_level) if file_level is not None else logging.DEBUG

        filepath = Path(file)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if file_mode not in ("a", "w"):
            raise ValueError(f"Invalid file_mode: {file_mode}. Must be 'a' or 'w'.")
        _file_handler = logging.FileHandler(filepath, mode=file_mode)
        _file_handler.setLevel(file_level_int)
        _file_handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
            )
        )
        logger.addHandler(_file_handler)

        # Flush after each log message for immediate visibility (useful for tail -f)
        _file_handler.stream.reconfigure(line_buffering=True)  # type: ignore[union-attr]

        # Disable console logging if file_only is True
        if file_only and _console_handler is not None:
            logger.removeHandler(_console_handler)
    elif file is None and _file_handler is not None:
        logger.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None


# ---------------------------------------------------------------------------
# Third-party logger isolation
# ---------------------------------------------------------------------------

#: Logger names whose console output should be suppressed.  Messages from
#: these loggers are still written to the log **file** (if configured) via
#: propagation to the root logger.
_NOISY_LOGGERS: tuple[str, ...] = (
    "hydromt",
    "hydromt_sfincs",
    "xarray",
    "matplotlib",
    "numba",
    "pyproj",
    "rasterio",
    "pyogrio",
    "fsspec",
    "dask",
    "botocore",
    "boto3",
    "s3transfer",
    "urllib3",
    "requests",
)


def silence_third_party_loggers(*, file_level: int = logging.DEBUG) -> None:
    """Prevent third-party loggers from printing to the console.

    * Removes any existing handlers (e.g. the ``StreamHandler`` that
      HydroMT attaches to its ``hydromt`` logger).
    * Adds a ``NullHandler`` so the library's own "no handler found"
      warning is silenced.
    * If a file handler has been configured on **our** logger we also
      attach it to the root logger so that third-party DEBUG/INFO
      messages still end up in the log file for diagnosis.

    Parameters
    ----------
    file_level : int, optional
        Level at which third-party messages are written to the log
        file.  Defaults to ``logging.DEBUG``.
    """
    for name in _NOISY_LOGGERS:
        lg = logging.getLogger(name)
        # Remove any existing handlers (e.g. StreamHandler from hydromt)
        for handler in lg.handlers[:]:
            lg.removeHandler(handler)
        lg.addHandler(logging.NullHandler())
        lg.propagate = True  # allow file capture via root

    # Configure the root logger so propagated messages reach the log
    # file (but never the console).
    root = logging.getLogger()
    # Remove any pre-existing root console handlers
    for handler in root.handlers[:]:
        if isinstance(handler, (logging.StreamHandler, RichHandler)):
            root.removeHandler(handler)
    root.addHandler(logging.NullHandler())

    # If we already have file logging, let root propagate into it.
    if _file_handler is not None:
        if _file_handler not in root.handlers:
            root.addHandler(_file_handler)
        root.setLevel(file_level)


class StageStatus(StrEnum):
    """Workflow stage status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageProgress:
    """Progress information for a workflow stage."""

    name: str
    status: StageStatus = StageStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    message: str = ""
    substeps: list[str] = field(default_factory=list)
    current_substep: int = 0

    @property
    def duration(self) -> timedelta | None:
        """Calculate stage duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        if self.start_time:
            return datetime.now() - self.start_time
        return None

    @property
    def duration_str(self) -> str:
        """Format duration as human-readable string."""
        d = self.duration
        if d is None:
            return "-"
        total_seconds = int(d.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"


class WorkflowMonitor:
    """Monitor and track workflow progress."""

    def _setup_logger(self) -> logging.Logger:
        """Configure logging based on monitoring config.

        * The console handler is kept at WARNING so only problems
          are shown on screen.
        * A log file is always created (either the one specified in
          ``config.log_file`` or an auto-generated one under
          ``config.work_dir``).  All DEBUG-level messages from both
          our own code and third-party libraries end up there.
        * Third-party loggers (HydroMT, xarray, ...) are silenced
          on the console but still write to the log file.
        """
        # Console: only warnings and errors
        if _console_handler is not None and _console_handler in logger.handlers:
            _console_handler.setLevel(logging.WARNING)

        # File: full detail
        if _file_handler is None:
            log_file = self.config.log_file
            if not log_file and hasattr(self, "_work_dir") and self._work_dir:
                log_file = str(generate_log_path(self._work_dir))
            if log_file:
                configure_logger(file=log_file, file_level="DEBUG")

        # Mute noisy third-party console output
        silence_third_party_loggers()

        return logger

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self.stages: dict[str, StageProgress] = {}
        self.workflow_start: datetime | None = None
        self.workflow_end: datetime | None = None
        self.logger = self._setup_logger()

    def register_stages(self, stage_names: list[str]) -> None:
        """Register workflow stages for tracking."""
        for name in stage_names:
            self.stages[name] = StageProgress(name=name)

    def start_workflow(self) -> None:
        """Mark workflow as started."""
        self.workflow_start = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("Coastal Calibration Workflow Started")
        self.logger.info("=" * 60)

    def _log_timing_summary(self) -> None:
        """Log timing summary for all stages."""
        self.logger.info("")
        self.logger.info("Timing Summary:")
        self.logger.info("-" * 40)
        for name, stage in self.stages.items():
            status_sym = {
                StageStatus.COMPLETED: "\u2713",
                StageStatus.FAILED: "\u2717",
                StageStatus.SKIPPED: "-",
                StageStatus.PENDING: "?",
                StageStatus.RUNNING: "...",
            }.get(stage.status, "?")
            self.logger.info(f"  [{status_sym}] {name}: {stage.duration_str}")

    def end_workflow(self, success: bool = True) -> None:
        """Mark workflow as ended."""
        self.workflow_end = datetime.now()
        status = "COMPLETED" if success else "FAILED"
        duration = self.workflow_end - self.workflow_start if self.workflow_start else None
        duration_str = str(duration).split(".")[0] if duration else "-"

        self.logger.info("=" * 60)
        self.logger.info(f"Workflow {status} | Total Duration: {duration_str}")
        self.logger.info("=" * 60)

        if self.config.enable_timing:
            self._log_timing_summary()

    def start_stage(self, name: str, message: str = "") -> None:
        """Mark a stage as started."""
        if name not in self.stages:
            self.stages[name] = StageProgress(name=name)

        stage = self.stages[name]
        stage.status = StageStatus.RUNNING
        stage.start_time = datetime.now()
        stage.message = message

        self.logger.info(f"Stage: {name}")
        if message:
            self.logger.info(f"  {message}")

    def end_stage(
        self,
        name: str,
        status: StageStatus = StageStatus.COMPLETED,
        message: str = "",
    ) -> None:
        """Mark a stage as ended."""
        if name not in self.stages:
            return

        stage = self.stages[name]
        stage.status = status
        stage.end_time = datetime.now()
        if message:
            stage.message = message

        status_icon = {
            StageStatus.COMPLETED: "\u2713",
            StageStatus.FAILED: "\u2717",
            StageStatus.SKIPPED: "-",
        }.get(status, "?")

        self.logger.info(f"  [{status_icon}] {status.value.upper()} ({stage.duration_str})")
        if message:
            self.logger.info(f"  {message}")
        self.logger.info("-" * 40)

    def mark_stage_completed(self, name: str) -> None:
        """Mark a stage as completed without start/end logging.

        Use this for stages whose execution was handled externally
        (e.g., inside a SLURM job) so the timing summary shows a
        checkmark instead of ``?``.
        """
        if name not in self.stages:
            self.stages[name] = StageProgress(name=name)
        self.stages[name].status = StageStatus.COMPLETED

    def update_substep(self, stage_name: str, substep: str) -> None:
        """Update current substep within a stage."""
        if stage_name in self.stages:
            stage = self.stages[stage_name]
            if substep not in stage.substeps:
                stage.substeps.append(substep)
            stage.current_substep = stage.substeps.index(substep)
            self.logger.debug(f"  -> {substep}")

    def log(self, level: str, message: str) -> None:
        """Log a message at the specified level."""
        getattr(self.logger, level.lower())(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    @contextmanager
    def stage_context(self, name: str, message: str = "") -> Iterator[StageProgress]:
        """Context manager for stage execution with automatic status updates."""
        self.start_stage(name, message)
        stage = self.stages[name]
        try:
            yield stage
            self.end_stage(name, StageStatus.COMPLETED)
        except Exception as e:
            self.end_stage(name, StageStatus.FAILED, str(e))
            raise

    def get_progress_dict(self) -> dict[str, Any]:
        """Get progress as a dictionary for serialization."""
        return {
            "workflow_start": self.workflow_start.isoformat() if self.workflow_start else None,
            "workflow_end": self.workflow_end.isoformat() if self.workflow_end else None,
            "stages": {
                name: {
                    "status": stage.status.value,
                    "start_time": stage.start_time.isoformat() if stage.start_time else None,
                    "end_time": stage.end_time.isoformat() if stage.end_time else None,
                    "duration": stage.duration_str,
                    "message": stage.message,
                }
                for name, stage in self.stages.items()
            },
        }

    def save_progress(self, path: Path) -> None:
        """Save progress to JSON file."""
        with path.open("w") as f:
            json.dump(self.get_progress_dict(), f, indent=2)


class ProgressBar:
    """Simple progress bar for long-running operations."""

    def __init__(self, total: int, description: str = "", width: int = 40) -> None:
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()

    def _render(self) -> None:
        """Render progress bar to stdout."""
        if self.total == 0:
            pct = 100
            filled = self.width
        else:
            pct = int(100 * self.current / self.total)
            filled = int(self.width * self.current / self.total)
        bar = "\u2588" * filled + "\u2591" * (self.width - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {int(eta)}s"
        else:
            eta_str = "ETA: --"

        desc = f"{self.description}: " if self.description else ""
        sys.stdout.write(f"\r{desc}|{bar}| {pct}% ({self.current}/{self.total}) {eta_str}")
        sys.stdout.flush()

        if self.current >= self.total:
            sys.stdout.write("\n")

    def update(self, n: int = 1) -> None:
        """Update progress bar."""
        self.current = min(self.current + n, self.total)
        self._render()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        if self.current < self.total:
            sys.stdout.write("\n")
