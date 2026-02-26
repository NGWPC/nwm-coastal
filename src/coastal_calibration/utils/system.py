"""System information utilities."""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import re
import subprocess

logger = logging.getLogger(__name__)


def get_cpu_count() -> int:
    """Get the number of physical CPU cores, properly handling Apple Silicon.

    On Apple Silicon Macs, returns only the performance (P) core count
    (excluding efficiency cores).  On other platforms, returns the total
    number of logical CPUs reported by the OS.

    Returns
    -------
    int
        Number of physical CPU cores (performance cores on Apple Silicon).
    """
    system = platform.system()
    machine = platform.machine()

    # Check if we're on macOS with Apple Silicon
    if system == "Darwin" and machine == "arm64":
        try:
            cmd = ["sysctl", "-n", "hw.perflevel0.logicalcpu_max"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            perf_cores = int(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            # Fallback to checking sysctl hw.topology output for older macOS versions
            with contextlib.suppress(subprocess.SubprocessError, ValueError):
                cmd = ["sysctl", "hw.topology"]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                match = re.search(r"perfcores:\s*(\d+)", result.stdout)
                if match:
                    return int(match.group(1))
        else:
            return perf_cores

        logger.warning(
            "Could not determine performance cores count on Apple Silicon. "
            "Falling back to total CPU count which may include efficiency cores."
        )

    return os.cpu_count() or 1
