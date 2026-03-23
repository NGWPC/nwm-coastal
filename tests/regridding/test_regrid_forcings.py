"""Tests for regrid_forcings: sea_level_pressure unit tests and integration tests.

``TestSeaLevelPressure`` runs unconditionally (ESMF is mocked).

Synthetic-data integration tests (``test_synthetic_*``) use tiny in-memory
grids and run whenever ESMF is importable.  They validate the SCHISM
volumetric-flux path (``skip_latlon=True``).

Real-data comparison tests (``test_vsource_matches_original``, etc.) are
guarded by ``have_ldasin_data``, ``have_geo_em``, and ``have_esmf_mesh``.

Run with::

    pytest tests/regridding/test_regrid_forcings.py -v
"""

from __future__ import annotations

import importlib
import os
import sys
import textwrap
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from .conftest import (
    LDASIN_DIR,
    have_esmf,
    have_esmf_mesh,
    have_ldasin_data,
    have_mpiexec,
    have_synthetic_esmf_mesh,
    run_mpi,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ORIGINAL_DRIVER = (
    REPO_ROOT
    / "tests/legacy_scripts/wrf_hydro_workflow_dev/forcings"
    / "WrfHydroFECPP/workflow_driver.py"
)

SCHISM_FORCING_MESH = Path("/Volumes/data/schism_models/hawaii/hgrid.nc")

_GEO_EM_CANDIDATES = [
    Path("/Volumes/data/schism_models/geo_em_HI.nc"),
    *([Path(os.environ["GEOGRID_FILE"])] if "GEOGRID_FILE" in os.environ else []),
]
GEO_EM_FILE = next((p for p in _GEO_EM_CANDIDATES if p.exists()), None)

# ---------------------------------------------------------------------------
# Skip markers for real-data tests
# ---------------------------------------------------------------------------

have_geo_em = pytest.mark.skipif(
    GEO_EM_FILE is None,
    reason=(
        "WRF geogrid file not found at /Volumes/data/schism_models/geo_em_HI.nc "
        "and GEOGRID_FILE env var not set."
    ),
)

have_schism_forcing_mesh = pytest.mark.skipif(
    not SCHISM_FORCING_MESH.exists(),
    reason=f"SCHISM forcing mesh not found: {SCHISM_FORCING_MESH}",
)


# ---------------------------------------------------------------------------
# Helpers for real-data comparison tests
# ---------------------------------------------------------------------------


def _run_original(
    input_dir: Path,
    output_dir: Path,
    geo_em: Path,
    schism_mesh: Path,
    nprocs: int = 1,
) -> None:
    """Run the original workflow_driver.py via mpiexec.

    Patches ``sys.modules`` so that legacy ``import ESMF`` resolves to
    ``esmpy`` on installations where only esmpy (≥ v8.4.0) is available.
    The fecpp package directory is inserted into ``sys.path`` directly rather
    than through PYTHONPATH so no file-based shim is needed.
    """
    original_fecpp_dir = ORIGINAL_DRIVER.parent
    runner = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(original_fecpp_dir)!r})
        sys.path.insert(0, {str(REPO_ROOT / "src")!r})
        try:
            import ESMF
        except ImportError:
            import esmpy as _esmpy
            sys.modules["ESMF"] = _esmpy
            sys.modules["ESMF.constants"] = _esmpy.constants
            _esmpy.Manager(debug=False)
        import runpy
        runpy.run_path({str(ORIGINAL_DRIVER)!r}, run_name="__main__")
    """)
    env = {
        "NWM_FORCING_OUTPUT_DIR": str(input_dir.parent),
        "COASTAL_FORCING_OUTPUT_DIR": str(output_dir),
        "GEOGRID_FILE": str(geo_em),
        "SCHISM_ESMFMESH": str(schism_mesh),
        "FORCING_BEGIN_DATE": input_dir.name,
        "LENGTH_HRS": "0",
    }
    run_mpi(nprocs, [sys.executable, "-c", runner], env)


def _run_new(
    input_dir: Path,
    output_dir: Path,
    geo_em: Path,
    schism_mesh: Path,
    nprocs: int = 1,
) -> None:
    runner = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT / "src")!r})
        try:
            import ESMF
        except ImportError:
            import esmpy as ESMF
        ESMF.Manager(debug=False)
        from coastal_calibration.regridding.regrid_forcings import (
            CoastalForcingRegridder,
        )
        from pathlib import Path
        app = CoastalForcingRegridder(
            input_dir=Path({str(input_dir)!r}),
            output_dir=Path({str(output_dir)!r}),
            geo_em_path=Path({str(geo_em)!r}),
            schism_mesh_path=Path({str(schism_mesh)!r}),
        )
        app.run(file_filter="**/*LDASIN_DOMAIN*", skip_latlon=True)
    """)
    env = {
        "COASTAL_FORCING_OUTPUT_DIR": str(output_dir),
        "LENGTH_HRS": "0",
    }
    run_mpi(nprocs, [sys.executable, "-c", runner], env)


# ---------------------------------------------------------------------------
# Real-data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def geo_em_file() -> Path:
    return GEO_EM_FILE


@pytest.fixture(scope="session")
def ldasin_subdir() -> Path:
    return LDASIN_DIR


# ---------------------------------------------------------------------------
# Unit tests — no external files or ESMF installation required
# ---------------------------------------------------------------------------


def _import_slp():
    """Import sea_level_pressure, mocking ESMF/esmf_utils if not installed."""
    esmf_stub = types.ModuleType("ESMF")
    for attr in (
        "Manager",
        "Grid",
        "Field",
        "Regrid",
        "Mesh",
        "LocStream",
        "RegridMethod",
        "UnmappedAction",
        "ExtrapMethod",
        "MeshLoc",
        "FileFormat",
        "CoordSys",
        "StaggerLoc",
        "local_pet",
        "pet_count",
    ):
        setattr(esmf_stub, attr, mock.MagicMock())

    esmf_utils_stub = types.ModuleType("coastal_calibration.regridding.esmf_utils")
    for name in (
        "build_grid",
        "build_locstream",
        "Regridder",
        "MaskedRegridder",
        "gather_reduce",
        "gatherv_1d",
        "allreduce_minmax",
        "GridBounds",
    ):
        setattr(esmf_utils_stub, name, mock.MagicMock())

    modules_to_patch = {
        "ESMF": esmf_stub,
        "ESMF.constants": types.ModuleType("ESMF.constants"),
        "coastal_calibration.regridding.esmf_utils": esmf_utils_stub,
    }

    parent_key = "coastal_calibration.regridding.regrid_forcings"
    with mock.patch.dict(sys.modules, modules_to_patch):
        if parent_key in sys.modules:
            mod = importlib.reload(sys.modules[parent_key])
        else:
            mod = importlib.import_module(parent_key)
        return mod.sea_level_pressure


class TestSeaLevelPressure:
    """Unit tests for the sea_level_pressure hypsometric formula."""

    @pytest.fixture(autouse=True)
    def _setup_slp(self):
        self._slp = _import_slp()

    def test_zero_height_returns_surface_pressure(self):
        """At height=0, SLP should equal surface pressure."""
        temp = np.array([288.0])
        mixing = np.array([0.01])
        height = np.array([0.0])
        press = np.array([101325.0])

        result = self._slp(temp, mixing, height, press)
        np.testing.assert_allclose(result, press, rtol=1e-10)

    def test_positive_height_increases_pressure(self):
        """SLP > surface pressure when height > 0."""
        temp = np.array([288.0])
        mixing = np.array([0.005])
        height = np.array([500.0])
        press = np.array([95000.0])

        result = self._slp(temp, mixing, height, press)
        assert result[0] > press[0]

    def test_known_value(self):
        """Verify formula against a hand-computed value."""
        g0, r_d, epsilon = 9.80665, 287.058, 0.622

        temp = np.array([288.0])
        mixing = np.array([0.01])
        height = np.array([500.0])
        press = np.array([95000.0])

        t_v = temp * (1 + mixing / epsilon) / (1 + mixing)
        scale_height = r_d * t_v / g0
        expected = press / np.exp(-height / scale_height)

        result = self._slp(temp, mixing, height, press)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_2d_arrays(self):
        """Formula should broadcast correctly over 2D grids."""
        shape = (10, 20)
        temp = np.full(shape, 285.0)
        mixing = np.full(shape, 0.008)
        height = np.full(shape, 200.0)
        press = np.full(shape, 98000.0)

        result = self._slp(temp, mixing, height, press)
        assert result.shape == shape
        assert np.all(result > press)

    def test_negative_height_decreases_pressure(self):
        """SLP < surface pressure when height < 0."""
        temp = np.array([300.0])
        mixing = np.array([0.015])
        height = np.array([-400.0])
        press = np.array([101325.0])

        result = self._slp(temp, mixing, height, press)
        assert result[0] < press[0]


# ---------------------------------------------------------------------------
# Synthetic-data integration tests — run whenever ESMF is importable
# ---------------------------------------------------------------------------


def _run_synthetic_vsource(
    tmp_path, synthetic_ldasin_dir, synthetic_geo_em_nc, synthetic_esmfmesh_nc
):
    """Run CoastalForcingRegridder on synthetic data, skipping on ESMF regrid failures.

    ESMF BILINEAR regridding on small synthetic grids can fail with rc=509
    on certain platforms (e.g., Ubuntu CI) while working on others (macOS).
    """
    from coastal_calibration.regridding.regrid_forcings import (
        CoastalForcingRegridder,
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)

    app = CoastalForcingRegridder(
        input_dir=synthetic_ldasin_dir,
        output_dir=output_dir,
        geo_em_path=synthetic_geo_em_nc,
        schism_mesh_path=synthetic_esmfmesh_nc,
    )
    try:
        app.run(file_filter="**/*LDASIN_DOMAIN*", skip_latlon=True)
    except ValueError as e:
        if "ESMC_FieldRegridStore" in str(e):
            pytest.skip(f"ESMF regridding failed on this platform: {e}")
        raise
    return output_dir


@have_esmf
@have_synthetic_esmf_mesh
def test_synthetic_vsource_output_structure(
    tmp_path,
    synthetic_ldasin_dir,
    synthetic_geo_em_nc,
    synthetic_esmfmesh_nc,
):
    """CoastalForcingRegridder writes a well-formed precip_source.nc (synthetic)."""
    import netCDF4

    output_dir = _run_synthetic_vsource(
        tmp_path, synthetic_ldasin_dir, synthetic_geo_em_nc, synthetic_esmfmesh_nc
    )

    vsource = output_dir / "precip_source.nc"
    assert vsource.exists(), "precip_source.nc was not created"

    with netCDF4.Dataset(vsource) as f:
        assert "time_vsource" in f.dimensions
        assert "nsources" in f.dimensions
        assert "one" in f.dimensions

        assert "vsource" in f.variables
        assert "time_vsource" in f.variables
        assert "source_elem" in f.variables
        assert "time_step_vsource" in f.variables

        vs = f["vsource"][:]
        assert vs.ndim == 2, f"Expected 2D vsource, got shape {vs.shape}"
        assert vs.shape[0] > 0, "No timesteps in vsource"
        assert vs.shape[1] > 0, "No source elements in vsource"

        assert f["time_step_vsource"][0] == 3600.0


@have_esmf
@have_synthetic_esmf_mesh
def test_synthetic_vsource_non_negative(
    tmp_path,
    synthetic_ldasin_dir,
    synthetic_geo_em_nc,
    synthetic_esmfmesh_nc,
):
    """Volumetric flux values are non-negative (synthetic data)."""
    import netCDF4

    output_dir = _run_synthetic_vsource(
        tmp_path, synthetic_ldasin_dir, synthetic_geo_em_nc, synthetic_esmfmesh_nc
    )

    with netCDF4.Dataset(output_dir / "precip_source.nc") as f:
        vs = f["vsource"][:]

    assert np.all(vs >= 0), f"vsource contains negative values: min={vs.min():.4e}"
    assert np.all(vs < 1e11), f"vsource contains implausibly large values: max={vs.max():.4e}"


@have_esmf
@have_synthetic_esmf_mesh
def test_synthetic_source_elem_covers_all_mesh_elements(
    tmp_path,
    synthetic_ldasin_dir,
    synthetic_geo_em_nc,
    synthetic_esmfmesh_nc,
):
    """source_elem should reference all 4 synthetic mesh elements (1-based)."""
    import netCDF4

    output_dir = _run_synthetic_vsource(
        tmp_path, synthetic_ldasin_dir, synthetic_geo_em_nc, synthetic_esmfmesh_nc
    )

    with netCDF4.Dataset(output_dir / "precip_source.nc") as f:
        source_elem = f["source_elem"][:]

    # Synthetic mesh has 4 elements; source_elem should be [1, 2, 3, 4]
    assert len(source_elem) == 4
    np.testing.assert_array_equal(
        np.sort(source_elem),
        np.arange(1, 5),
        err_msg="source_elem should be 1-based indices covering all mesh elements",
    )


# ---------------------------------------------------------------------------
# Optional real-data comparison tests
# ---------------------------------------------------------------------------


@have_esmf
@have_esmf_mesh
@have_mpiexec
@have_ldasin_data
@have_geo_em
@have_schism_forcing_mesh
def test_new_vsource_output_structure(
    tmp_path,
    ldasin_subdir,
    geo_em_file,
):
    """CoastalForcingRegridder writes a well-formed SCHISM precip_source.nc (real data)."""
    import netCDF4

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    _run_new(
        input_dir=ldasin_subdir,
        output_dir=output_dir,
        geo_em=geo_em_file,
        schism_mesh=SCHISM_FORCING_MESH,
    )

    vsource = output_dir / "precip_source.nc"
    assert vsource.exists()

    with netCDF4.Dataset(vsource) as f:
        assert "time_vsource" in f.dimensions
        assert "nsources" in f.dimensions
        assert "one" in f.dimensions

        assert "vsource" in f.variables
        assert "time_vsource" in f.variables
        assert "source_elem" in f.variables
        assert "time_step_vsource" in f.variables

        vs = f["vsource"][:]
        assert vs.ndim == 2
        assert vs.shape[0] > 0
        assert vs.shape[1] > 0
        assert f["time_step_vsource"][0] == 3600.0


@have_esmf
@have_esmf_mesh
@have_mpiexec
@have_ldasin_data
@have_geo_em
@have_schism_forcing_mesh
def test_vsource_matches_original(
    tmp_path,
    ldasin_subdir,
    geo_em_file,
):
    """New CoastalForcingRegridder vsource matches original workflow_driver output."""
    import netCDF4

    orig_out = tmp_path / "orig"
    new_out = tmp_path / "new"
    orig_out.mkdir()
    new_out.mkdir()

    _run_original(
        input_dir=ldasin_subdir,
        output_dir=orig_out,
        geo_em=geo_em_file,
        schism_mesh=SCHISM_FORCING_MESH,
    )
    _run_new(
        input_dir=ldasin_subdir,
        output_dir=new_out,
        geo_em=geo_em_file,
        schism_mesh=SCHISM_FORCING_MESH,
    )

    with netCDF4.Dataset(orig_out / "precip_source.nc") as f:
        orig_vs = f["vsource"][:].data
    with netCDF4.Dataset(new_out / "precip_source.nc") as f:
        new_vs = f["vsource"][:].data

    assert orig_vs.shape == new_vs.shape
    np.testing.assert_allclose(
        orig_vs,
        new_vs,
        rtol=1e-5,
        atol=1e-8,
        err_msg="vsource values differ between original and refactored implementation",
    )
