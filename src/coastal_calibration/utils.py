"""System information and MPI utilities.

Combines CPU detection (:func:`get_cpu_count`), MPI implementation
detection (:func:`detect_mpi`), and MPI environment/command builders
into a single module.
"""

from __future__ import annotations

import contextlib
import os
import platform
import re
import subprocess
from enum import StrEnum
from pathlib import Path

from coastal_calibration.logging import logger

# ---------------------------------------------------------------------------
# CPU detection (was utils/system.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MPI implementation detection and environment (was utils/mpi.py)
# ---------------------------------------------------------------------------

__all__ = [
    "MpiImpl",
    "build_isolated_env",
    "build_mpi_cmd",
    "build_mpi_env",
    "detect_mpi",
    "get_cpu_count",
]


class MpiImpl(StrEnum):
    """Known MPI implementations."""

    OPENMPI = "openmpi"
    MPICH = "mpich"
    UNKNOWN = "unknown"


_cache: dict[str, MpiImpl] = {}


def _mpi_cache_key(env: dict[str, str] | None) -> str:
    """Return a cache key derived from the PATH that will resolve ``mpiexec``."""
    if env is not None:
        return env.get("PATH", "")
    return os.environ.get("PATH", "")


def detect_mpi(env: dict[str, str] | None = None) -> MpiImpl:
    """Detect the active MPI implementation by parsing ``mpiexec --version``.

    Parameters
    ----------
    env : dict[str, str], optional
        Subprocess environment to use for the probe.  When *None*
        (the default), the current process environment is used.  Pass
        an isolated env dict to detect the MPI that a system-compiled
        binary will actually see.

    Returns
    -------
    MpiImpl
        The detected implementation.  Returns :attr:`MpiImpl.UNKNOWN` when
        ``mpiexec`` is not found or produces unrecognised output.
    """
    key = _mpi_cache_key(env)
    if key in _cache:
        return _cache[key]

    try:
        result = subprocess.run(
            ["mpiexec", "--version"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        output = (result.stdout + result.stderr).lower()
    except FileNotFoundError:
        logger.warning("mpiexec not found on PATH — MPI implementation unknown")
        _cache[key] = MpiImpl.UNKNOWN
        return _cache[key]

    if "open mpi" in output or "openrte" in output:
        impl = MpiImpl.OPENMPI
    elif "hydra" in output or "mpich" in output:
        impl = MpiImpl.MPICH
    else:
        logger.warning("Could not identify MPI from mpiexec --version output")
        impl = MpiImpl.UNKNOWN

    logger.debug("Detected MPI implementation: %s", impl)
    _cache[key] = impl
    return impl


def _has_efa() -> bool:
    """Return True when AWS EFA devices are present."""
    sys_ib = Path("/sys/class/infiniband")
    if not sys_ib.exists():
        return False
    try:
        return any(p.name.startswith("efa") for p in sys_ib.iterdir())
    except OSError:
        logger.debug("Unable to read %s; assuming no EFA", sys_ib)
        return False


def build_mpi_env(env: dict[str, str]) -> dict[str, str]:
    """Add MPI-tuning environment variables to *env* (mutating).

    Applies three layers of configuration:

    1. **General** — safe on any cluster (NFS, Lustre, local).
    2. **EFA / OFI fabric** — only when AWS EFA devices are detected
       (``/sys/class/infiniband/efa*``).  These force libfabric as the
       transport and tune buffer sizes for reliable multi-node MPI.
    3. **Implementation-specific** — OpenMPI MCA or MPICH env vars,
       gated on the detected MPI flavour *and* fabric availability.

    Parameters
    ----------
    env : dict[str, str]
        Environment dictionary to update (typically from ``os.environ.copy()``).

    Returns
    -------
    dict[str, str]
        The same *env* dict, updated in place.
    """
    impl = detect_mpi(env)
    efa = _has_efa()

    # ── General (all clusters) ────────────────────────────────────
    if impl is MpiImpl.OPENMPI:
        # Suppress noisy OpenMPI warnings on NFS home directories.
        env.setdefault("OMPI_MCA_mpi_warn_on_fork", "0")
        # Use local disk for shared-memory backing (avoids NFS).
        env.setdefault("OMPI_MCA_orte_tmpdir_base", "/tmp")  # noqa: S108
    elif impl is MpiImpl.MPICH:
        # Cray MPICH / standard MPICH collective tuning — prevents
        # hangs during ESMF initialisation and MPI collectives on
        # any fabric (TCP, OFI, Slingshot).
        env["MPICH_OFI_STARTUP_CONNECT"] = "1"
        env["MPICH_COLL_SYNC"] = "MPI_Bcast"
        env["MPICH_REDUCE_NO_SMP"] = "1"

    # ── EFA / OFI fabric (AWS c5n, hpc6a, etc.) ──────────────────
    if efa:
        # Libfabric tuning for EFA.
        env["FI_OFI_RXM_SAR_LIMIT"] = "3145728"
        env["FI_MR_CACHE_MAX_COUNT"] = "0"
        env["FI_EFA_RECVWIN_SIZE"] = "65536"

        if impl is MpiImpl.OPENMPI:
            # Force OFI transport layer for EFA.
            env["OMPI_MCA_mtl"] = "ofi"
            env["OMPI_MCA_pml"] = "cm"
            env["OMPI_MCA_btl"] = "^openib"

    return env


def build_mpi_cmd(
    ntasks: int,
    *,
    oversubscribe: bool = False,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Build the ``mpiexec`` launcher prefix.

    Parameters
    ----------
    ntasks : int
        Number of MPI ranks.
    oversubscribe : bool
        Allow more ranks than physical cores.  Only has effect with
        OpenMPI (``--oversubscribe``); silently ignored for MPICH.
    env : dict[str, str], optional
        Subprocess environment used to detect the MPI implementation.
        Defaults to the current process environment.

    Returns
    -------
    list[str]
        Command prefix, e.g. ``["mpiexec", "-n", "36"]``.
    """
    cmd = ["mpiexec", "-n", str(ntasks)]
    if oversubscribe and detect_mpi(env) is MpiImpl.OPENMPI:
        cmd.append("--oversubscribe")
    return cmd


def _strip_conda_paths(path_str: str) -> str:
    """Remove ``$CONDA_PREFIX``-rooted entries from a ``PATH``-like string."""
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        return path_str
    prefix = Path(conda_prefix)
    kept = [p for p in path_str.split(os.pathsep) if not Path(p).is_relative_to(prefix)]
    return os.pathsep.join(kept)


def build_isolated_env(
    *,
    omp_num_threads: int,
    runtime_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build an environment isolated from conda for system-compiled binaries.

    Starts from :func:`os.environ`, strips ``$CONDA_PREFIX`` entries from
    ``PATH`` and ``LD_LIBRARY_PATH`` so the subprocess finds system MPI,
    HDF5, and NetCDF libraries, then layers on:

    1. ``HDF5_USE_FILE_LOCKING=FALSE`` (NFS reliability)
    2. OpenMP pinning (``OMP_NUM_THREADS``, ``OMP_PLACES``, ``OMP_PROC_BIND``)
    3. MPI implementation-specific tuning (auto-detected)
    4. User-supplied ``runtime_env`` overrides (applied last)

    Parameters
    ----------
    omp_num_threads : int
        Number of OpenMP threads per process.
    runtime_env : dict[str, str], optional
        Extra variables from the model config.  Applied last, so they
        can override any auto-detected value.

    Returns
    -------
    dict[str, str]
        A new environment dictionary ready for :func:`subprocess.run`.
    """
    env = os.environ.copy()

    # Strip conda libraries so system MPI/HDF5/NetCDF are found.
    for key in ("PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH"):
        if key in env:
            env[key] = _strip_conda_paths(env[key])

    # HDF5 NFS reliability
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    # OpenMP pinning
    env["OMP_NUM_THREADS"] = str(omp_num_threads)
    env["OMP_PLACES"] = "cores"
    env["OMP_PROC_BIND"] = "close"

    # MPI tuning (auto-detected from system mpiexec)
    build_mpi_env(env)

    # User overrides applied last
    if runtime_env:
        env.update(runtime_env)

    return env
