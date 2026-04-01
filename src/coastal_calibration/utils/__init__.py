"""Utility modules for logging, monitoring, and system info."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coastal_calibration.utils.logging import (
        ProgressBar,
        StageProgress,
        StageStatus,
        WorkflowMonitor,
    )
    from coastal_calibration.utils.system import get_cpu_count

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ProgressBar": ("coastal_calibration.utils.logging", "ProgressBar"),
    "StageProgress": ("coastal_calibration.utils.logging", "StageProgress"),
    "StageStatus": ("coastal_calibration.utils.logging", "StageStatus"),
    "WorkflowMonitor": ("coastal_calibration.utils.logging", "WorkflowMonitor"),
    "get_cpu_count": ("coastal_calibration.utils.system", "get_cpu_count"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "ProgressBar",
    "StageProgress",
    "StageStatus",
    "WorkflowMonitor",
    "get_cpu_count",
]
