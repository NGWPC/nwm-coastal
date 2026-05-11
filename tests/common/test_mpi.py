"""Tests for the MPI detection and environment utilities."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from coastal_calibration.utils import (
    MpiImpl,
    build_isolated_env,
    build_mpi_cmd,
    build_mpi_env,
    detect_mpi,
    expand_cpu_affinity_if_constrained,
)


@pytest.fixture(autouse=True)
def _clear_mpi_cache():
    """Reset the cached MPI detection result between tests."""
    import coastal_calibration.utils as mod

    mod._cache.clear()
    yield
    mod._cache.clear()


# ── detect_mpi ────────────────────────────────────────────────────────


OPENMPI_VERSION = "mpiexec (OpenRTE) 5.0.10\n\nReport bugs to http://www.open-mpi.org/\n"
MPICH_VERSION = "HYDRA build details:\n    Version: 4.2.3\n"
CRAY_MPICH_VERSION = "HYDRA build details:\n    Version: 8.1.9\n    Configure: --prefix=...\n"


class TestDetectMpi:
    def test_openmpi(self):
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=OPENMPI_VERSION, stderr=""
        )
        with patch("subprocess.run", return_value=result):
            assert detect_mpi() is MpiImpl.OPENMPI

    def test_mpich(self):
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout=MPICH_VERSION, stderr="")
        with patch("subprocess.run", return_value=result):
            assert detect_mpi() is MpiImpl.MPICH

    def test_cray_mpich(self):
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=CRAY_MPICH_VERSION, stderr=""
        )
        with patch("subprocess.run", return_value=result):
            assert detect_mpi() is MpiImpl.MPICH

    def test_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert detect_mpi() is MpiImpl.UNKNOWN

    def test_unrecognized_output(self):
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="some custom mpi v1.0", stderr=""
        )
        with patch("subprocess.run", return_value=result):
            assert detect_mpi() is MpiImpl.UNKNOWN

    def test_result_is_cached(self):
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout=MPICH_VERSION, stderr="")
        with patch("subprocess.run", return_value=result) as mock:
            detect_mpi()
            detect_mpi()
            mock.assert_called_once()


# ── build_mpi_env ─────────────────────────────────────────────────────


class TestBuildMpiEnv:
    def test_openmpi_general(self):
        """OpenMPI gets general NFS-safe MCA vars only."""
        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.OPENMPI):
            env: dict[str, str] = {}
            build_mpi_env(env)
            assert env["OMPI_MCA_mpi_warn_on_fork"] == "0"
            assert env["OMPI_MCA_orte_tmpdir_base"] == "/tmp"
            # No fabric-specific tuning is auto-applied; users supply
            # cluster-specific transport vars via runtime_env.
            assert "OMPI_MCA_mtl" not in env
            assert "OMPI_MCA_pml" not in env
            assert "OMPI_MCA_btl" not in env
            assert "FI_OFI_RXM_SAR_LIMIT" not in env
            assert "FI_EFA_RECVWIN_SIZE" not in env
            assert "MPICH_OFI_STARTUP_CONNECT" not in env

    def test_mpich_vars(self):
        """MPICH gets collective-tuning vars; no OpenMPI / fabric vars."""
        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.MPICH):
            env: dict[str, str] = {}
            build_mpi_env(env)
            assert env["MPICH_OFI_STARTUP_CONNECT"] == "1"
            assert env["MPICH_COLL_SYNC"] == "MPI_Bcast"
            assert env["MPICH_REDUCE_NO_SMP"] == "1"
            assert "OMPI_MCA_mtl" not in env
            assert "FI_OFI_RXM_SAR_LIMIT" not in env

    def test_no_fabric_autotuning(self):
        """Regression: no FI_*/EFA env vars are set, regardless of impl.

        These were auto-set by an earlier ``_has_efa()`` probe that
        deadlocked multi-node ESMF allreduce on at least one AWS EFA
        cluster. The fix is to never auto-apply fabric tuning; users
        pass cluster-specific overrides via ``runtime_env`` instead.
        """
        for impl in (MpiImpl.OPENMPI, MpiImpl.MPICH, MpiImpl.UNKNOWN):
            with patch("coastal_calibration.utils.detect_mpi", return_value=impl):
                env: dict[str, str] = {}
                build_mpi_env(env)
                assert "FI_OFI_RXM_SAR_LIMIT" not in env
                assert "FI_MR_CACHE_MAX_COUNT" not in env
                assert "FI_EFA_RECVWIN_SIZE" not in env


# ── build_mpi_cmd ─────────────────────────────────────────────────────


class TestBuildMpiCmd:
    def test_basic(self):
        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.MPICH):
            cmd = build_mpi_cmd(36)
            assert cmd == ["mpiexec", "-n", "36"]

    def test_oversubscribe_openmpi(self):
        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.OPENMPI):
            cmd = build_mpi_cmd(36, oversubscribe=True)
            assert "--oversubscribe" in cmd

    def test_oversubscribe_mpich_ignored(self):
        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.MPICH):
            cmd = build_mpi_cmd(36, oversubscribe=True)
            assert "--oversubscribe" not in cmd


# ── build_isolated_env ────────────────────────────────────────────────


class TestBuildIsolatedEnv:
    def test_strips_conda_paths(self, monkeypatch):
        conda = "/opt/conda/envs/dev"
        monkeypatch.setenv("CONDA_PREFIX", conda)
        monkeypatch.setenv("PATH", f"{conda}/bin:/usr/bin:/usr/local/bin")
        monkeypatch.setenv("LD_LIBRARY_PATH", f"{conda}/lib:/usr/lib")

        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.MPICH):
            env = build_isolated_env(omp_num_threads=4)

        assert conda not in env["PATH"]
        assert "/usr/bin" in env["PATH"]
        assert conda not in env["LD_LIBRARY_PATH"]
        assert "/usr/lib" in env["LD_LIBRARY_PATH"]

    def test_sets_omp_and_hdf5(self):
        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.MPICH):
            env = build_isolated_env(omp_num_threads=8)

        assert env["OMP_NUM_THREADS"] == "8"
        assert env["HDF5_USE_FILE_LOCKING"] == "FALSE"
        # Pinning policy is the user's responsibility, not a default.
        assert env.get("OMP_PLACES") != "cores"
        assert env.get("OMP_PROC_BIND") != "close"

    def test_sets_mpi_tuning(self):
        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.MPICH):
            env = build_isolated_env(omp_num_threads=4)

        assert env["MPICH_OFI_STARTUP_CONNECT"] == "1"

    def test_runtime_env_overrides(self):
        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.MPICH):
            env = build_isolated_env(
                omp_num_threads=4,
                runtime_env={"OMP_NUM_THREADS": "16", "CUSTOM_VAR": "hello"},
            )

        assert env["OMP_NUM_THREADS"] == "16"
        assert env["CUSTOM_VAR"] == "hello"

    def test_no_conda_prefix_is_noop(self, monkeypatch):
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin:/usr/local/bin")

        with patch("coastal_calibration.utils.detect_mpi", return_value=MpiImpl.MPICH):
            env = build_isolated_env(omp_num_threads=4)

        assert env["PATH"] == "/usr/bin:/usr/local/bin"


# ── expand_cpu_affinity_if_constrained ────────────────────────────────


class TestExpandCpuAffinity:
    """Tests for the CPU-affinity expansion helper.

    ``os.sched_*affinity`` only exist on Linux, so the patches use
    ``create=True`` to let these tests run on macOS / Windows too. The
    function itself is hasattr-guarded, so on platforms without those
    APIs it is a real no-op regardless of env vars.
    """

    def test_noop_outside_slurm(self, monkeypatch):
        """No SLURM env vars -> nothing happens, no errors raised."""
        monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
        with patch("os.sched_setaffinity", create=True) as mock_set:
            expand_cpu_affinity_if_constrained()
            mock_set.assert_not_called()

    def test_noop_when_affinity_already_adequate(self, monkeypatch):
        """SLURM allocation matches current mask -> no expansion."""
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
        with (
            patch("os.sched_getaffinity", return_value=set(range(8)), create=True),
            patch("os.sched_setaffinity", create=True) as mock_set,
        ):
            expand_cpu_affinity_if_constrained()
            mock_set.assert_not_called()

    def test_expands_when_constrained(self, monkeypatch):
        """SLURM allocation wider than mask -> expand to allocation size."""
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "18")
        with (
            patch("os.sched_getaffinity", return_value={0}, create=True),
            patch("os.sched_setaffinity", create=True) as mock_set,
        ):
            expand_cpu_affinity_if_constrained()
            mock_set.assert_called_once_with(0, range(18))

    def test_swallows_oserror(self, monkeypatch):
        """Permission errors from cgroup-enforced affinity are logged not raised."""
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "18")
        with (
            patch("os.sched_getaffinity", return_value={0}, create=True),
            patch("os.sched_setaffinity", side_effect=PermissionError, create=True),
        ):
            # Should not raise.
            expand_cpu_affinity_if_constrained()

    def test_handles_malformed_slurm_var(self, monkeypatch):
        """Non-integer SLURM_CPUS_ON_NODE -> silent no-op."""
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "not-a-number")
        with patch("os.sched_setaffinity", create=True) as mock_set:
            expand_cpu_affinity_if_constrained()
            mock_set.assert_not_called()
