"""Base stage class for workflow stages."""

from __future__ import annotations

import importlib.resources
import os
import subprocess
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coastal_calibration.scripts_path import get_script_environment_vars

if TYPE_CHECKING:
    from coastal_calibration.config.schema import CoastalCalibConfig
    from coastal_calibration.utils.logging import WorkflowMonitor


__all__ = ["WorkflowStage"]


class WorkflowStage(ABC):
    """Abstract base class for workflow stages."""

    name: str = "base"
    description: str = "Base workflow stage"
    requires_container: bool = False

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
            getattr(self.monitor, level)(message)

    def _update_substep(self, substep: str) -> None:
        """Update current substep."""
        if self.monitor:
            self.monitor.update_substep(self.name, substep)

    def _build_date_env(self, env: dict[str, str]) -> None:
        """Add precomputed date variables to the environment."""
        sim = self.config.simulation

        start_dt = datetime.strptime(f"{sim.start_pdy} {sim.start_cyc}", "%Y%m%d %H").replace(
            tzinfo=UTC
        )
        end_dt = start_dt + timedelta(hours=sim.duration_hours + 1)
        env["start_dt"] = start_dt.strftime("%Y-%m-%dT%H-%M-%SZ")
        env["end_dt"] = end_dt.strftime("%Y-%m-%dT%H-%M-%SZ")

        length_hrs = int(sim.duration_hours)
        pdy = sim.start_pdy
        cyc = sim.start_cyc
        pdycyc = f"{pdy}{cyc}"

        env["FORCING_BEGIN_DATE"] = f"{pdycyc}00"
        forcing_end_dt = start_dt + timedelta(hours=length_hrs)
        env["FORCING_END_DATE"] = forcing_end_dt.strftime("%Y%m%d%H00")

        env["END_DATETIME"] = forcing_end_dt.strftime("%Y%m%d%H")

        env["PDY"] = pdy
        env["cyc"] = cyc
        env["FORCING_START_YEAR"] = pdy[:4]
        env["FORCING_START_MONTH"] = pdy[4:6]
        env["FORCING_START_DAY"] = pdy[6:8]
        env["FORCING_START_HOUR"] = cyc

    def build_environment(self) -> dict[str, str]:
        """Build environment variables for the stage.

        Shared variables are set here.  Model-specific variables (compute
        resources, mesh paths, etc.) are added by the concrete
        :class:`~coastal_calibration.config.schema.ModelConfig` via its
        :meth:`build_environment` method.
        """
        sim = self.config.simulation
        paths = self.config.paths

        env = os.environ.copy()

        # HDF5 file locking (fcntl) is unreliable on NFS-mounted
        # filesystems and causes PermissionError when netCDF4/HDF5
        # tries to create files.  Disable it unless the user has
        # already set the variable.
        env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

        env["STARTPDY"] = sim.start_pdy
        env["STARTCYC"] = sim.start_cyc
        env["FCST_LENGTH_HRS"] = str(int(sim.duration_hours))
        env["FCST_TIMESTEP_LENGTH_SECS"] = str(sim.timestep_seconds)
        env["COASTAL_DOMAIN"] = sim.coastal_domain
        env["METEO_SOURCE"] = sim.meteo_source.upper()

        hot_start = paths.hot_start_file
        env["HOT_START_FILE"] = str(hot_start) if hot_start else ""
        env["USE_TPXO"] = "YES" if self.config.boundary.source == "tpxo" else "NO"

        env["RAW_DOWNLOAD_DIR"] = str(paths.download_dir)
        env["COASTAL_WORK_DIR"] = str(paths.work_dir)
        env["DATAexec"] = str(paths.work_dir)
        env["NFS_MOUNT"] = str(paths.nfs_mount)
        env["NWM_DIR"] = str(paths.nwm_dir)
        env["OTPSDIR"] = str(paths.otps_dir)
        env["CONDA_ENV_NAME"] = paths.conda_env_name

        env["NGWPC_COASTAL_PARM_DIR"] = str(paths.parm_dir)
        env["USHnwm"] = str(paths.ush_nwm)
        env["PARMnwm"] = str(paths.parm_nwm)
        env["EXECnwm"] = str(paths.exec_nwm)

        env["CONDA_ENVS_PATH"] = str(paths.conda_envs_path)

        env["INLAND_DOMAIN"] = sim.inland_domain
        env["NWM_DOMAIN"] = sim.nwm_domain
        env["GEO_GRID"] = sim.geo_grid
        env["GEOGRID_FILE"] = str(paths.geogrid_file(sim))

        env["NWM_FORCING_DIR"] = str(paths.meteo_dir(sim.meteo_source))
        env["NWM_CHROUT_DIR"] = str(paths.streamflow_dir(sim.meteo_source))

        stofs_file = self.config.boundary.stofs_file
        env["STOFS_FILE"] = str(stofs_file) if stofs_file else ""

        env["COASTAL_SOURCE"] = "" if self.config.boundary.source == "tpxo" else "stofs"
        env["DATAlogs"] = str(paths.work_dir)

        self._build_date_env(env)

        # Add paths to bundled scripts so bash scripts can find them
        env.update(get_script_environment_vars())

        # Delegate model-specific variables (compute resources, mesh, etc.)
        self.config.model_config.build_environment(env, self.config)

        self._env = env
        return env

    def run_shell_script(
        self,
        script_path: Path,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a shell script with the configured environment."""
        if env is None:
            env = self.build_environment()

        if cwd is None:
            cwd = script_path.parent

        self._log(f"Running script: {script_path}")

        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            self._log(f"Script failed with return code {result.returncode}", "error")
            if result.stderr:
                self._log(f"STDERR: {result.stderr[-2000:]}", "error")
            raise RuntimeError(f"Script {script_path} failed: {result.stderr}")

        return result

    def _get_scripts_dir(self) -> Path:
        """Get the scripts directory path containing bash scripts."""
        return Path(str(importlib.resources.files("coastal_calibration") / "scripts"))

    def _get_default_bindings(self) -> list[str]:
        """Get default Singularity bind paths."""
        paths = self.config.paths
        return [
            str(paths.nfs_mount),
            str(paths.conda_envs_path),
            str(paths.parm_dir),
            str(self._get_scripts_dir()),
            "/usr/bin/bc",
            "/usr/bin/srun",
            "/usr/lib64/libpmi2.so",
            "/usr/lib64/libefa.so",
            "/usr/lib64/libibmad.so",
            "/usr/lib64/libibnetdisc.so",
            "/usr/lib64/libibumad.so",
            "/usr/lib64/libibverbs.so",
            "/usr/lib64/libmana.so",
            "/usr/lib64/libmlx4.so",
            "/usr/lib64/libmlx5.so",
            "/usr/lib64/librdmacm.so",
        ]

    def run_singularity_command(
        self,
        command: list[str],
        sif_path: Path | str,
        bindings: list[str] | None = None,
        pwd: Path | None = None,
        env: dict[str, str] | None = None,
        use_mpi: bool = False,
        mpi_tasks: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command inside the Singularity container.

        Parameters
        ----------
        sif_path : Path or str
            Path to the Singularity SIF image.

        Environment variables are passed to the container using the
        SINGULARITYENV_ prefix, which ensures they are available inside
        the container regardless of Singularity's default environment
        handling.
        """
        if env is None:
            env = self.build_environment()

        sif_path = str(sif_path)

        if bindings is None:
            bindings = self._get_default_bindings()

        bind_str = ",".join(bindings)

        if pwd is None:
            from coastal_calibration.config.schema import DEFAULT_CONTAINER_PWD

            pwd = DEFAULT_CONTAINER_PWD

        sing_cmd = ["singularity", "exec", "-B", bind_str, "--pwd", str(pwd), sif_path]
        sing_cmd.extend(command)

        if use_mpi:
            if mpi_tasks is None:
                msg = "mpi_tasks must be specified when use_mpi=True"
                raise ValueError(msg)
            mpi_cmd = ["mpiexec", "-n", str(mpi_tasks)]
            oversubscribe = getattr(self.config.model_config, "oversubscribe", False)
            if oversubscribe:
                mpi_cmd.append("--oversubscribe")
            sing_cmd = [*mpi_cmd, *sing_cmd]

        self._log(f"Running Singularity command: {' '.join(command[:3])}...")

        # Singularity passes environment variables that are prefixed with
        # SINGULARITYENV_ to the container. We need to add this prefix to
        # all our workflow environment variables so they're available inside.
        # Some variables cannot be overridden (HOME, USER, etc.) and some
        # variable names from the host environment may be invalid.
        singularity_protected = {"HOME", "USER", "SHELL", "TERM", "PATH", "LD_LIBRARY_PATH"}
        singularity_env = env.copy()
        for key, value in env.items():
            # Skip protected variables that Singularity won't allow overriding
            if key in singularity_protected:
                continue
            # Skip variables with invalid names (must be alphanumeric + underscore)
            if not key.replace("_", "").isalnum():
                continue
            singularity_env[f"SINGULARITYENV_{key}"] = value

        # Never use capture_output=True for container commands.
        # MPI ranks, Fortran binaries, and bash scripts with set -x
        # can all produce enough output to fill the OS pipe buffer
        # (64 KB on Linux), causing a deadlock: the child blocks on
        # write while subprocess.run waits for it to exit.
        # Popen.communicate() drains pipes on background threads,
        # avoiding the deadlock.  stdout is discarded; stderr is
        # kept for error reporting.
        proc = subprocess.Popen(
            sing_cmd,
            env=singularity_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        _, stderr = proc.communicate()
        result = subprocess.CompletedProcess(
            sing_cmd, proc.returncode, stdout="", stderr=stderr or ""
        )

        if result.returncode != 0:
            stderr = result.stderr or ""
            self._log(f"Singularity command failed: {stderr[-2000:]}", "error")
            raise RuntimeError(f"Singularity command failed: {stderr}")

        return result

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the stage and return results."""

    def validate(self) -> list[str]:
        """Validate stage prerequisites. Return list of errors."""
        return []
