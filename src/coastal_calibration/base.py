"""Base stage class for workflow stages."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coastal_calibration.config.schema import CoastalCalibConfig
    from coastal_calibration.logging import WorkflowMonitor


__all__ = ["WorkflowStage"]


class WorkflowStage(ABC):
    """Abstract base class for workflow stages."""

    name: str = "base"
    description: str = "Base workflow stage"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        self.config = config
        self.monitor = monitor
        self._env: dict[str, str] = {}

    def _log(self, message: str, level: str = "info") -> None:
        """Log message if monitor is available."""
        if self.monitor:
            getattr(self.monitor, level)(f"  {message}")

    def _update_substep(self, substep: str) -> None:
        """Update current substep."""
        if self.monitor:
            self.monitor.update_substep(self.name, substep)

    def build_environment(self) -> dict[str, str]:
        """Build environment variables for the stage.

        Sets only runtime-critical variables: HDF5 file locking and
        model-specific variables (OMP, MPI tuning) added by the
        concrete :class:`~coastal_calibration.config.schema.ModelConfig`
        via its :meth:`build_environment` method.

        All binaries (``mpiexec``, ``pschism``, etc.) are expected on
        ``$PATH`` — pixi installs them into ``$CONDA_PREFIX/bin``.
        """
        env = os.environ.copy()

        # HDF5 file locking (fcntl) is unreliable on NFS-mounted
        # filesystems and causes PermissionError when netCDF4/HDF5
        # tries to create files.  Disable it unless the user has
        # already set the variable.
        env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

        # OpenMP thread pinning — common to all models.
        env["OMP_NUM_THREADS"] = str(self.config.model_config.omp_num_threads)
        env["OMP_PLACES"] = "cores"
        env["OMP_PROC_BIND"] = "close"

        # Delegate model-specific variables (MPI tuning, etc.)
        self.config.model_config.build_environment(env, self.config)

        self._env = env
        return env

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the stage and return results."""

    def validate(self) -> list[str]:
        """Validate stage prerequisites. Return list of errors."""
        return []
