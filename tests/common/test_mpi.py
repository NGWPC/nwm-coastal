"""Tests for the MPI detection and environment utilities."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from coastal_calibration.utils.mpi import (
    MpiImpl,
    build_isolated_env,
    build_mpi_cmd,
    build_mpi_env,
    detect_mpi,
)


@pytest.fixture(autouse=True)
def _clear_mpi_cache():
    """Reset the cached MPI detection result between tests."""
    import coastal_calibration.utils.mpi as mod

    mod._cached_impl = None
    yield
    mod._cached_impl = None


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

    def test_unrecognised_output(self):
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
        """OpenMPI sets general NFS-safe vars without EFA."""
        with (
            patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.OPENMPI),
            patch("coastal_calibration.utils.mpi._has_efa", return_value=False),
        ):
            env: dict[str, str] = {}
            build_mpi_env(env)
            assert env["OMPI_MCA_mpi_warn_on_fork"] == "0"
            assert env["OMPI_MCA_orte_tmpdir_base"] == "/tmp"
            # EFA-only vars should NOT be set
            assert "OMPI_MCA_mtl" not in env
            assert "FI_OFI_RXM_SAR_LIMIT" not in env
            assert "MPICH_OFI_STARTUP_CONNECT" not in env

    def test_openmpi_with_efa(self):
        """OpenMPI + EFA sets OFI transport and libfabric tuning."""
        with (
            patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.OPENMPI),
            patch("coastal_calibration.utils.mpi._has_efa", return_value=True),
        ):
            env: dict[str, str] = {}
            build_mpi_env(env)
            assert env["OMPI_MCA_mtl"] == "ofi"
            assert env["OMPI_MCA_pml"] == "cm"
            assert env["OMPI_MCA_btl"] == "^openib"
            assert env["FI_OFI_RXM_SAR_LIMIT"] == "3145728"
            assert env["FI_EFA_RECVWIN_SIZE"] == "65536"

    def test_mpich_vars(self):
        """MPICH always sets collective tuning (any fabric)."""
        with (
            patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.MPICH),
            patch("coastal_calibration.utils.mpi._has_efa", return_value=False),
        ):
            env: dict[str, str] = {}
            build_mpi_env(env)
            assert env["MPICH_OFI_STARTUP_CONNECT"] == "1"
            assert env["MPICH_COLL_SYNC"] == "MPI_Bcast"
            assert env["MPICH_REDUCE_NO_SMP"] == "1"
            assert "OMPI_MCA_mtl" not in env

    def test_efa_libfabric_vars(self):
        """EFA detection sets libfabric tuning regardless of MPI impl."""
        for impl in (MpiImpl.OPENMPI, MpiImpl.MPICH):
            with (
                patch("coastal_calibration.utils.mpi.detect_mpi", return_value=impl),
                patch("coastal_calibration.utils.mpi._has_efa", return_value=True),
            ):
                env: dict[str, str] = {}
                build_mpi_env(env)
                assert "FI_OFI_RXM_SAR_LIMIT" in env
                assert "FI_MR_CACHE_MAX_COUNT" in env
                assert "FI_EFA_RECVWIN_SIZE" in env

    def test_no_efa_no_libfabric_vars(self):
        """Without EFA, no libfabric tuning vars are set."""
        with (
            patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.OPENMPI),
            patch("coastal_calibration.utils.mpi._has_efa", return_value=False),
        ):
            env: dict[str, str] = {}
            build_mpi_env(env)
            assert "FI_OFI_RXM_SAR_LIMIT" not in env


# ── build_mpi_cmd ─────────────────────────────────────────────────────


class TestBuildMpiCmd:
    def test_basic(self):
        with patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.MPICH):
            cmd = build_mpi_cmd(36)
            assert cmd == ["mpiexec", "-n", "36"]

    def test_oversubscribe_openmpi(self):
        with patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.OPENMPI):
            cmd = build_mpi_cmd(36, oversubscribe=True)
            assert "--oversubscribe" in cmd

    def test_oversubscribe_mpich_ignored(self):
        with patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.MPICH):
            cmd = build_mpi_cmd(36, oversubscribe=True)
            assert "--oversubscribe" not in cmd


# ── build_isolated_env ────────────────────────────────────────────────


class TestBuildIsolatedEnv:
    def test_strips_conda_paths(self, monkeypatch):
        conda = "/opt/conda/envs/dev"
        monkeypatch.setenv("CONDA_PREFIX", conda)
        monkeypatch.setenv("PATH", f"{conda}/bin:/usr/bin:/usr/local/bin")
        monkeypatch.setenv("LD_LIBRARY_PATH", f"{conda}/lib:/usr/lib")

        with (
            patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.MPICH),
            patch("coastal_calibration.utils.mpi._has_efa", return_value=False),
        ):
            env = build_isolated_env(omp_num_threads=4)

        assert conda not in env["PATH"]
        assert "/usr/bin" in env["PATH"]
        assert conda not in env["LD_LIBRARY_PATH"]
        assert "/usr/lib" in env["LD_LIBRARY_PATH"]

    def test_sets_omp_and_hdf5(self):
        with (
            patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.MPICH),
            patch("coastal_calibration.utils.mpi._has_efa", return_value=False),
        ):
            env = build_isolated_env(omp_num_threads=8)

        assert env["OMP_NUM_THREADS"] == "8"
        assert env["OMP_PLACES"] == "cores"
        assert env["OMP_PROC_BIND"] == "close"
        assert env["HDF5_USE_FILE_LOCKING"] == "FALSE"

    def test_sets_mpi_tuning(self):
        with (
            patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.MPICH),
            patch("coastal_calibration.utils.mpi._has_efa", return_value=False),
        ):
            env = build_isolated_env(omp_num_threads=4)

        assert env["MPICH_OFI_STARTUP_CONNECT"] == "1"

    def test_runtime_env_overrides(self):
        with (
            patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.MPICH),
            patch("coastal_calibration.utils.mpi._has_efa", return_value=False),
        ):
            env = build_isolated_env(
                omp_num_threads=4,
                runtime_env={"OMP_NUM_THREADS": "16", "CUSTOM_VAR": "hello"},
            )

        assert env["OMP_NUM_THREADS"] == "16"
        assert env["CUSTOM_VAR"] == "hello"

    def test_no_conda_prefix_is_noop(self, monkeypatch):
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin:/usr/local/bin")

        with (
            patch("coastal_calibration.utils.mpi.detect_mpi", return_value=MpiImpl.MPICH),
            patch("coastal_calibration.utils.mpi._has_efa", return_value=False),
        ):
            env = build_isolated_env(omp_num_threads=4)

        assert env["PATH"] == "/usr/bin:/usr/local/bin"
