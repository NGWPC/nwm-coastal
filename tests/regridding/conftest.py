"""Fixtures and skip conditions for regridding tests.

Synthetic fixtures (``synthetic_*``) are always available and require only
ESMF/ESMPy to be installed.  They use tiny grids so regridding completes in
seconds.

Real-data fixtures (``stofs_file``, ``schism_hgrid_nc``, ``ldasin_dir``) are
still provided for optional comparison tests that need large on-disk data.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths to optional real data
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

STOFS_FILE = (
    REPO_ROOT
    / "docs/examples/downloads/coastal/stofs/stofs_2d_glo.20240109"
    / "stofs_2d_glo.t00z.fields.cwl.nc"
)

SCHISM_HGRID_NC = Path("/Volumes/data/schism_models/hawaii/open_bnds_hgrid.nc")

LDASIN_DIR = REPO_ROOT / "docs/examples/downloads/meteo/nwm_ana"


# ---------------------------------------------------------------------------
# MPI subprocess runner (shared by both test modules)
# ---------------------------------------------------------------------------


def run_mpi(nprocs: int, cmd: list[str], env: dict[str, str]) -> None:
    """Run *cmd* under ``mpiexec -np nprocs`` with *env* merged into the environment.

    Raises ``RuntimeError`` with captured stdout/stderr on non-zero exit.
    """
    full_cmd = ["mpiexec", "-np", str(nprocs)] + cmd
    full_env = {**os.environ, **env}
    result = subprocess.run(full_cmd, env=full_env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"MPI job failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


def _esmf_available() -> bool:
    try:
        import ESMF  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        import esmpy  # noqa: F401

        return True
    except ImportError:
        return False


def _esmf_mesh_available(mesh_path: Path) -> bool:
    """Return True if ESMF.Mesh can load the given ESMFMESH file without crashing.

    Probes in a subprocess so a segfault doesn't kill the test session.
    """
    if not mesh_path.exists() or not _esmf_available():
        return False
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
try:
    import ESMF
except ImportError:
    import esmpy as ESMF
ESMF.Manager(debug=False)
ESMF.Mesh(filename={str(mesh_path)!r}, filetype=ESMF.FileFormat.ESMFMESH)
print("ok")
""",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


def _esmf_mpi_available() -> bool:
    """Return True if ESMF/esmpy was compiled with MPI support (pet_count >= 2)."""
    if shutil.which("mpiexec") is None or not _esmf_available():
        return False
    try:
        result = subprocess.run(
            [
                "mpiexec",
                "-np",
                "2",
                sys.executable,
                "-c",
                "import esmpy as ESMF; m = ESMF.Manager(); print(m.pet_count)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            last_line = result.stdout.strip().split("\n")[-1]
            return int(last_line) >= 2
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

have_mpiexec = pytest.mark.skipif(
    shutil.which("mpiexec") is None,
    reason="mpiexec not found on PATH",
)

have_esmf_mpi = pytest.mark.skipif(
    not _esmf_mpi_available(),
    reason="ESMF/esmpy not compiled with MPI support (pet_count=1 under mpiexec -np 2)",
)

_SCHISM_FORCING_MESH = Path("/Volumes/data/schism_models/hawaii/hgrid.nc")
have_esmf_mesh = pytest.mark.skipif(
    not _esmf_mesh_available(_SCHISM_FORCING_MESH),
    reason=(
        f"ESMF.Mesh cannot load {_SCHISM_FORCING_MESH} without crashing "
        "(libesmf_fullylinked crash — likely a macOS esmpy build issue)"
    ),
)

have_esmf = pytest.mark.skipif(
    not _esmf_available(),
    reason="ESMF/ESMPy not importable",
)

have_stofs_data = pytest.mark.skipif(
    not STOFS_FILE.exists(),
    reason=f"STOFS example data not found: {STOFS_FILE}",
)

have_schism_hgrid = pytest.mark.skipif(
    not SCHISM_HGRID_NC.exists(),
    reason=f"SCHISM pacific hgrid not found: {SCHISM_HGRID_NC}",
)

have_ldasin_data = pytest.mark.skipif(
    not LDASIN_DIR.exists() or not any(LDASIN_DIR.glob("*.LDASIN_DOMAIN*")),
    reason=f"LDASIN example data not found: {LDASIN_DIR}",
)

# ---------------------------------------------------------------------------
# Real-data fixtures (used by optional comparison tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def stofs_file() -> Path:
    return STOFS_FILE


@pytest.fixture(scope="session")
def schism_hgrid_nc() -> Path:
    return SCHISM_HGRID_NC


@pytest.fixture(scope="session")
def ldasin_dir() -> Path:
    return LDASIN_DIR


@pytest.fixture(scope="session")
def stofs_cycle_env() -> dict[str, str]:
    """Determine CYCLE_DATE/TIME from the real STOFS file's time metadata."""
    from datetime import datetime, timedelta

    import netCDF4
    from cftime import num2date

    FORECAST_START = 5
    with netCDF4.Dataset(STOFS_FILE) as f:
        tv = f["time"]
        t = num2date(tv[FORECAST_START], units=tv.units)

    if t.minute != 0:
        t = datetime(t.year, t.month, t.day, t.hour) + timedelta(hours=1)

    return {
        "CYCLE_DATE": f"{t.year:04d}{t.month:02d}{t.day:02d}",
        "CYCLE_TIME": f"{t.hour:02d}{t.minute:02d}",
        "LENGTH_HRS": "10",
    }


# ---------------------------------------------------------------------------
# Synthetic fixtures — always available, no external data required
# ---------------------------------------------------------------------------

from .synthetic import (  # noqa: E402
    make_esmfmesh_nc,
    make_geo_em_nc,
    make_hgrid_nc,
    make_ldasin_nc,
    make_stofs_nc,
)


def _synthetic_esmf_mesh_available() -> bool:
    """Return True if ESMF.Mesh can load a small synthetic ESMFMESH file.

    Uses the same subprocess probe as ``_esmf_mesh_available`` so a segfault
    in libesmf_fullylinked (known Python 3.14t free-threaded issue) doesn't
    kill the pytest session.
    """
    if not _esmf_available():
        return False
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        mesh_path = Path(d) / "probe.nc"
        make_esmfmesh_nc(mesh_path)
        return _esmf_mesh_available(mesh_path)


have_synthetic_esmf_mesh = pytest.mark.skipif(
    not _synthetic_esmf_mesh_available(),
    reason=(
        "ESMF.Mesh cannot load a synthetic ESMFMESH file without crashing "
        "(libesmf_fullylinked crash — likely a macOS esmpy build issue)"
    ),
)


@pytest.fixture(scope="session")
def synthetic_stofs_nc(tmp_path_factory) -> Path:
    """ESTOFS-like NetCDF with 50 nodes and 10 hourly timesteps."""
    path = tmp_path_factory.mktemp("stofs") / "synthetic_stofs.nc"
    make_stofs_nc(path)
    return path


@pytest.fixture(scope="session")
def synthetic_stofs_cycle_env() -> dict[str, str]:
    """Environment variables matching ``synthetic_stofs_nc``.

    Time index 5 in the synthetic file equals 2024-01-09 05:00:00 UTC.
    Setting ``CYCLE_DATE=20240109`` and ``CYCLE_TIME=0500`` makes
    ``_determine_time_range`` select ``dt_h=0``, ``start=5``.
    With ``LENGTH_HRS=2`` we read three timesteps (indices 5, 6, 7).
    """
    return {
        "CYCLE_DATE": "20240109",
        "CYCLE_TIME": "0500",
        "LENGTH_HRS": "2",
    }


@pytest.fixture(scope="session")
def synthetic_hgrid_nc(tmp_path_factory) -> Path:
    """SCHISM open-boundary hgrid NetCDF with 20 nodes, 10 boundary nodes."""
    path = tmp_path_factory.mktemp("hgrid") / "synthetic_hgrid.nc"
    make_hgrid_nc(path)
    return path


@pytest.fixture(scope="session")
def synthetic_geo_em_nc(tmp_path_factory) -> Path:
    """Minimal WRF geo_em NetCDF on a 6×5 grid."""
    path = tmp_path_factory.mktemp("geo_em") / "synthetic_geo_em.nc"
    make_geo_em_nc(path)
    return path


@pytest.fixture(scope="session")
def synthetic_ldasin_dir(tmp_path_factory) -> Path:
    """Directory with one synthetic LDASIN forcing file."""
    d = tmp_path_factory.mktemp("ldasin")
    make_ldasin_nc(d / "201603150000.LDASIN_DOMAIN1")
    return d


@pytest.fixture(scope="session")
def synthetic_esmfmesh_nc(tmp_path_factory) -> Path:
    """Minimal ESMFMESH NetCDF: 6 nodes, 4 triangular elements."""
    path = tmp_path_factory.mktemp("esmfmesh") / "synthetic_hgrid.nc"
    make_esmfmesh_nc(path)
    return path
