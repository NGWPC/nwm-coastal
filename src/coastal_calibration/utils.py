"""System information, MPI, and time-handling utilities.

Combines CPU detection (:func:`get_cpu_count`), MPI implementation
detection (:func:`detect_mpi`), MPI environment/command builders, and
the project's UTC-normalization helper (:func:`to_naive_utc`).
"""

from __future__ import annotations

import contextlib
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from coastal_calibration.logging import logger

# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------


def to_naive_utc(dt: datetime) -> datetime:
    """Return *dt* as a naive UTC ``datetime``.

    All datetime values inside the package are conventionally **naive UTC**:
    NWM and STOFS data are published on UTC days, NetCDF time axes are
    written in UTC, and the hard-coded ``DateRange.start`` / ``.end``
    constants in :mod:`coastal_calibration.data.downloader` are bare
    ``datetime(YYYY, M, D)`` values that mean "midnight UTC".

    This helper normalizes any user-supplied datetime into that
    convention:

    - tz-aware → converted to UTC, then ``tzinfo`` stripped.
    - tz-naive → returned unchanged (assumed UTC by contract).

    Use this at every public boundary that accepts a ``datetime`` from
    the caller (config loading, data-layer queries, etc.) so the rest
    of the pipeline can compare, subtract, and serialize without ever
    crossing the naive/aware boundary.
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def utc_now() -> datetime:
    """Return ``datetime.now`` as a naive UTC datetime.

    Same convention as :func:`to_naive_utc`. Use for "now" timestamps
    (workflow start/end, log filenames, durations) so they are
    comparable across machines and across the rest of the pipeline.
    """
    return datetime.now(UTC).replace(tzinfo=None)


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


def expand_cpu_affinity_if_constrained() -> None:
    """Expand process CPU affinity to all SLURM-allocated cores.

    ``srun --pty bash`` typically creates a step with one task and
    therefore one CPU, even when the parent allocation has many more.
    A Python process started inside that step inherits the narrow
    affinity mask, and any single-process OpenMP workload it launches
    (e.g. SFINCS) ends up with every thread pinned to the same CPU.

    This helper detects the case (SLURM allocation reports more CPUs
    than the current process is allowed to use) and expands the mask
    to all SLURM-allocated cores. It is a no-op outside SLURM, on
    systems without ``os.sched_setaffinity`` (macOS, Windows), or when
    affinity is already adequate.

    Failures are logged at WARNING but never raised — this is a
    best-effort fix-up, not a precondition.
    """
    if not hasattr(os, "sched_getaffinity"):
        return
    slurm_cpus_str = os.environ.get("SLURM_CPUS_ON_NODE")
    if not slurm_cpus_str:
        return
    try:
        slurm_cpus = int(slurm_cpus_str)
    except ValueError:
        return
    try:
        # sched_*affinity are Linux-only; the hasattr guard above keeps
        # this branch unreachable on macOS / Windows, but pyright doesn't
        # know that and complains about the attribute access.
        allowed = os.sched_getaffinity(0)  # pyright: ignore[reportAttributeAccessIssue]
    except OSError:
        return
    if len(allowed) >= slurm_cpus:
        return
    try:
        os.sched_setaffinity(0, range(slurm_cpus))  # pyright: ignore[reportAttributeAccessIssue]
    except OSError as err:
        logger.warning("Could not expand CPU affinity: %s", err)
        return
    logger.info(
        "Expanded CPU affinity from %d to %d core(s) "
        "(SLURM_CPUS_ON_NODE=%d, was inheriting narrow step mask).",
        len(allowed),
        slurm_cpus,
        slurm_cpus,
    )


# ---------------------------------------------------------------------------
# MPI implementation detection and environment (was utils/mpi.py)
# ---------------------------------------------------------------------------

__all__ = [
    "MpiImpl",
    "build_isolated_env",
    "build_mpi_cmd",
    "build_mpi_env",
    "detect_mpi",
    "expand_cpu_affinity_if_constrained",
    "get_cpu_count",
    "to_naive_utc",
    "utc_now",
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
        ``mpiexec`` is not found or produces unrecognized output.
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


def build_mpi_env(env: dict[str, str]) -> dict[str, str]:
    """Add MPI-tuning environment variables to *env* (mutating).

    Applies only general, implementation-specific tuning that is safe
    on any cluster (NFS, Lustre, local).  Fabric-specific tuning (EFA,
    Slingshot, InfiniBand, etc.) is intentionally NOT auto-detected:
    it varies too widely across clusters and a wrong guess can deadlock
    multi-node MPI in subtle ways.  Users supply the cluster-specific
    transport / OFI / UCX env vars via the ``runtime_env`` field on
    their model config; those overrides are applied after this
    function and so always win.

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
    2. ``OMP_NUM_THREADS`` (thread count only; pinning policy is the
       user's responsibility via ``runtime_env``)
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

    # OpenMP thread count.  Thread-pinning policy (OMP_PROC_BIND,
    # OMP_PLACES) is intentionally NOT set here: users supply it via
    # ``runtime_env`` when their cluster benefits from NUMA pinning.
    env["OMP_NUM_THREADS"] = str(omp_num_threads)

    # MPI tuning (auto-detected from system mpiexec)
    build_mpi_env(env)

    # User overrides applied last
    if runtime_env:
        env.update(runtime_env)

    return env
