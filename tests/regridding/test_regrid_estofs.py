"""Tests for regrid_estofs: output structure, physical plausibility, and comparison.

Synthetic-data tests run whenever ESMF/ESMPy is importable; they use tiny
in-memory grids and complete in a few seconds.

Real-data comparison tests are guarded by ``have_stofs_data`` and
``have_schism_hgrid`` -- they are automatically skipped when the large on-disk
datasets are absent.

Run with::

    pytest tests/regridding/test_regrid_estofs.py -v
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import netCDF4
import numpy as np
import pytest

from .conftest import (
    have_esmf,
    have_esmf_mpi,
    have_mpiexec,
    have_schism_hgrid,
    have_stofs_data,
    run_mpi,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ORIGINAL_SCRIPT = REPO_ROOT / "tests/legacy_scripts/wrf_hydro_workflow_dev/coastal/regrid_estofs.py"

NEW_MODULE = "coastal_calibration.regridding.regrid_estofs"


# ---------------------------------------------------------------------------
# Helpers for real-data comparison tests
# ---------------------------------------------------------------------------


def _run_original(
    stofs_nc: Path,
    hgrid_nc: Path,
    output_nc: Path,
    cycle_env: dict[str, str],
    nprocs: int = 1,
) -> None:
    """Run the original regrid_estofs.py via mpiexec.

    Patches ``sys.modules`` so that legacy ``import ESMF`` in the original
    script resolves to ``esmpy`` on installations where only esmpy (≥ v8.4.0)
    is available.  This avoids maintaining a file-based compatibility shim on
    PYTHONPATH.
    """
    runner = textwrap.dedent(f"""
        import sys
        try:
            import ESMF
        except ImportError:
            import esmpy as _esmpy
            sys.modules["ESMF"] = _esmpy
            sys.modules["ESMF.constants"] = _esmpy.constants
            _esmpy.Manager(debug=False)
        import runpy
        sys.argv = {[str(ORIGINAL_SCRIPT), str(stofs_nc), str(hgrid_nc), str(output_nc)]!r}
        runpy.run_path({str(ORIGINAL_SCRIPT)!r}, run_name="__main__")
    """)
    run_mpi(nprocs, [sys.executable, "-c", runner], cycle_env)


def _run_new(
    stofs_nc: Path,
    hgrid_nc: Path,
    output_nc: Path,
    cycle_env: dict[str, str],
    nprocs: int = 1,
) -> None:
    runner = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT / "src")!r})
        from coastal_calibration.regridding.regrid_estofs import regrid_estofs
        regrid_estofs({str(stofs_nc)!r}, {str(hgrid_nc)!r}, {str(output_nc)!r})
    """)
    run_mpi(nprocs, [sys.executable, "-c", runner], cycle_env)


def _load_time_series(nc_path: Path) -> NDArray[np.floating[Any]]:
    with netCDF4.Dataset(nc_path) as f:
        return f["time_series"][:].data


# ---------------------------------------------------------------------------
# Synthetic-data tests — run whenever ESMF is importable
# ---------------------------------------------------------------------------


@have_esmf
def test_synthetic_regrid_estofs_output_structure(
    tmp_path,
    synthetic_stofs_nc,
    synthetic_hgrid_nc,
    synthetic_stofs_cycle_env,
):
    """regrid_estofs writes a well-formed SCHISM elev2D.th.nc using synthetic data."""
    output_nc = tmp_path / "elev2D.th.nc"

    from coastal_calibration.regridding.regrid_estofs import regrid_estofs

    regrid_estofs(
        str(synthetic_stofs_nc),
        str(synthetic_hgrid_nc),
        str(output_nc),
        cycle_date=synthetic_stofs_cycle_env["CYCLE_DATE"],
        cycle_time=synthetic_stofs_cycle_env["CYCLE_TIME"],
        length_hrs=int(synthetic_stofs_cycle_env["LENGTH_HRS"]),
    )

    assert output_nc.exists(), "Output file was not created"

    with netCDF4.Dataset(output_nc) as f:
        assert "time" in f.dimensions
        assert "nOpenBndNodes" in f.dimensions
        assert "nLevels" in f.dimensions
        assert "nComponents" in f.dimensions

        assert "time_step" in f.variables
        assert "time" in f.variables
        assert "time_series" in f.variables

        ts = f["time_series"][:]
        assert ts.ndim == 4, f"Expected 4D time_series, got shape {ts.shape}"
        assert ts.shape[2] == 1, "nLevels should be 1"
        assert ts.shape[3] == 1, "nComponents should be 1"

        expected_nt = int(synthetic_stofs_cycle_env["LENGTH_HRS"]) + 1
        assert ts.shape[0] == expected_nt, f"Expected {expected_nt} timesteps, got {ts.shape[0]}"
        assert ts.shape[1] > 0, "No boundary nodes in output"

        assert f["time_step"][0] == 3600.0


@have_esmf
def test_synthetic_regrid_estofs_no_large_values(
    tmp_path,
    synthetic_stofs_nc,
    synthetic_hgrid_nc,
    synthetic_stofs_cycle_env,
):
    """Regridded water levels are physically plausible (no fill value leakage)."""
    output_nc = tmp_path / "elev2D.th.nc"

    from coastal_calibration.regridding.regrid_estofs import regrid_estofs

    regrid_estofs(
        str(synthetic_stofs_nc),
        str(synthetic_hgrid_nc),
        str(output_nc),
        cycle_date=synthetic_stofs_cycle_env["CYCLE_DATE"],
        cycle_time=synthetic_stofs_cycle_env["CYCLE_TIME"],
        length_hrs=int(synthetic_stofs_cycle_env["LENGTH_HRS"]),
    )

    ts = _load_time_series(output_nc)
    assert np.all(ts > -9999.0), "Output contains fill/missing values (-9999)"
    assert np.all(np.abs(ts) < 100.0), (
        f"Implausibly large water level: min={ts.min():.2f} max={ts.max():.2f}"
    )


@have_esmf
def test_synthetic_regrid_estofs_values_in_input_range(
    tmp_path,
    synthetic_stofs_nc,
    synthetic_hgrid_nc,
    synthetic_stofs_cycle_env,
):
    """Nearest-neighbour regridded values stay within the range of the input field."""
    output_nc = tmp_path / "elev2D.th.nc"

    from coastal_calibration.regridding.regrid_estofs import regrid_estofs

    regrid_estofs(
        str(synthetic_stofs_nc),
        str(synthetic_hgrid_nc),
        str(output_nc),
        cycle_date=synthetic_stofs_cycle_env["CYCLE_DATE"],
        cycle_time=synthetic_stofs_cycle_env["CYCLE_TIME"],
        length_hrs=int(synthetic_stofs_cycle_env["LENGTH_HRS"]),
    )

    with netCDF4.Dataset(synthetic_stofs_nc) as f_in:
        zeta_raw = f_in["zeta"][:]  # masked array
        # Valid (unmasked) values used as source
        valid_values = zeta_raw.data[~zeta_raw.mask]
        src_min, src_max = float(valid_values.min()), float(valid_values.max())

    ts = _load_time_series(output_nc)
    # Values of 0 are written for masked/fill nodes — allow that
    assert np.all((ts >= src_min) | (ts == 0.0)), (
        f"Output below source minimum: min={ts.min():.4f} < src_min={src_min:.4f}"
    )
    assert np.all((ts <= src_max) | (ts == 0.0)), (
        f"Output above source maximum: max={ts.max():.4f} > src_max={src_max:.4f}"
    )


# ---------------------------------------------------------------------------
# Optional comparison tests — require large on-disk datasets
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Real-data MPI test hangs — subprocess.run has no timeout")
@pytest.mark.parametrize(
    "nprocs",
    [
        pytest.param(1, id="serial"),
        pytest.param(2, id="parallel_2", marks=have_esmf_mpi),
    ],
)
@have_esmf
@have_mpiexec
@have_stofs_data
@have_schism_hgrid
def test_regrid_estofs_matches_original(
    tmp_path,
    stofs_file,
    schism_hgrid_nc,
    stofs_cycle_env,
    nprocs,
):
    """New regrid_estofs produces bit-identical output to the original.

    Both implementations use NEAREST_STOD via ESMF so results must match
    exactly (no floating-point accumulation differences).
    """
    orig_out = tmp_path / "orig_elev2D.nc"
    new_out = tmp_path / "new_elev2D.nc"

    _run_original(stofs_file, schism_hgrid_nc, orig_out, stofs_cycle_env, nprocs)
    _run_new(stofs_file, schism_hgrid_nc, new_out, stofs_cycle_env, nprocs)

    orig_ts = _load_time_series(orig_out)
    new_ts = _load_time_series(new_out)

    assert orig_ts.shape == new_ts.shape, (
        f"Shape mismatch: original={orig_ts.shape}, new={new_ts.shape}"
    )
    np.testing.assert_array_equal(
        orig_ts,
        new_ts,
        err_msg="time_series values differ between original and refactored implementation",
    )


# ---------------------------------------------------------------------------
# Source connectivity resolution
#
# The pre-2023 ``estofs`` product ships only time/x/y/zeta, so ``element``
# must come from the companion ``*.maxele.nc`` or be synthesized.
# ---------------------------------------------------------------------------

_BBOX = (-2.0, -2.0, 2.0, 2.0)


def _grid_nodes(n: int = 6) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Return a small regular lon/lat node cloud inside ``_BBOX``."""
    gx, gy = np.meshgrid(np.linspace(-2.0, 2.0, n), np.linspace(-2.0, 2.0, n))
    return gx.ravel(), gy.ravel()


def _write_fields(path: Path, lon, lat, elements=None, start_index: int = 1) -> None:
    """Write a minimal ESTOFS ``fields.cwl.nc``-style file."""
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("node", len(lon))
        ds.createVariable("x", "f8", ("node",))[:] = lon
        ds.createVariable("y", "f8", ("node",))[:] = lat
        if elements is not None:
            ds.createDimension("nele", len(elements))
            ds.createDimension("nvertex", 3)
            var = ds.createVariable("element", "i4", ("nele", "nvertex"))
            var.start_index = start_index
            var[:] = elements


def _write_maxele(path: Path, n_nodes: int, elements, start_index: int = 1) -> None:
    """Write a minimal companion ``maxele`` file carrying connectivity."""
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("node", n_nodes)
        ds.createDimension("nele", len(elements))
        ds.createDimension("nvertex", 3)
        var = ds.createVariable("element", "i4", ("nele", "nvertex"))
        var.start_index = start_index
        var[:] = elements


@have_esmf
def test_resolve_elements_prefers_inline_connectivity(tmp_path):
    """A file carrying ``element`` is used directly, honoring start_index."""
    from coastal_calibration.regridding.regrid_estofs import _resolve_source_elements

    lon, lat = _grid_nodes()
    inline = np.array([[1, 2, 3], [2, 3, 4]])
    nc = tmp_path / "stofs_2d_glo.t00z.fields.cwl.nc"
    _write_fields(nc, lon, lat, elements=inline, start_index=1)

    with netCDF4.Dataset(nc) as ds:
        elements, start_index = _resolve_source_elements(ds, str(nc), lon, lat, _BBOX, 1.0)

    assert start_index == 1
    np.testing.assert_array_equal(elements, inline)


@have_esmf
def test_resolve_elements_reads_companion_maxele(tmp_path):
    """Connectivity is recovered from the sibling maxele file."""
    from coastal_calibration.regridding.regrid_estofs import _resolve_source_elements

    lon, lat = _grid_nodes()
    companion = np.array([[1, 2, 7], [2, 7, 8]])
    nc = tmp_path / "estofs.t00z.fields.cwl.nc"
    _write_fields(nc, lon, lat)
    _write_maxele(tmp_path / "estofs.t00z.fields.cwl.maxele.nc", len(lon), companion)

    with netCDF4.Dataset(nc) as ds:
        elements, start_index = _resolve_source_elements(ds, str(nc), lon, lat, _BBOX, 1.0)

    assert start_index == 1
    np.testing.assert_array_equal(elements, companion)


@have_esmf
def test_resolve_elements_rejects_mismatched_maxele(tmp_path):
    """A companion describing a different mesh is ignored, not misapplied."""
    from coastal_calibration.regridding.regrid_estofs import _resolve_source_elements

    lon, lat = _grid_nodes()
    nc = tmp_path / "estofs.t00z.fields.cwl.nc"
    _write_fields(nc, lon, lat)
    # Node count deliberately disagrees with the fields file.
    _write_maxele(
        tmp_path / "estofs.t00z.fields.cwl.maxele.nc",
        len(lon) + 5,
        np.array([[1, 2, 3]]),
    )

    with netCDF4.Dataset(nc) as ds:
        elements, start_index = _resolve_source_elements(ds, str(nc), lon, lat, _BBOX, 1.0)

    # Fell through to the synthesized triangulation instead of the bad mesh.
    assert start_index == 0
    assert len(elements) > 1


@have_esmf
def test_resolve_elements_synthesizes_when_no_companion(tmp_path):
    """Without any connectivity a valid Delaunay triangulation is built."""
    from coastal_calibration.regridding.regrid_estofs import _resolve_source_elements

    lon, lat = _grid_nodes()
    nc = tmp_path / "estofs.t00z.fields.cwl.nc"
    _write_fields(nc, lon, lat)

    with netCDF4.Dataset(nc) as ds:
        elements, start_index = _resolve_source_elements(ds, str(nc), lon, lat, _BBOX, 1.0)

    assert start_index == 0
    assert elements.shape[1] == 3
    # 0-based indices addressing the full node array, no degenerate triangles.
    assert elements.min() >= 0
    assert elements.max() < len(lon)
    assert (elements[:, 0] != elements[:, 1]).all()


@have_esmf
def test_delaunay_elements_drops_land_spanning_triangles():
    """The max-edge filter removes triangles bridging a gap in the cloud."""
    from coastal_calibration.regridding.esmf_utils import delaunay_elements

    # Two dense clusters separated by a wide empty gap; Delaunay would
    # otherwise bridge them with elements spanning the void.
    left_x, left_y = np.meshgrid(np.linspace(-2.0, -1.8, 6), np.linspace(-1.0, 1.0, 6))
    right_x, right_y = np.meshgrid(np.linspace(1.8, 2.0, 6), np.linspace(-1.0, 1.0, 6))
    lon = np.concatenate([left_x.ravel(), right_x.ravel()])
    lat = np.concatenate([left_y.ravel(), right_y.ravel()])

    unfiltered = delaunay_elements(lon, lat, bbox=_BBOX, max_edge_factor=None)
    filtered = delaunay_elements(lon, lat, bbox=_BBOX, max_edge_factor=3.0)

    assert len(filtered) < len(unfiltered)
    # No surviving triangle spans the gap between the two clusters.
    spans = (lon[filtered].max(axis=1) > 0) & (lon[filtered].min(axis=1) < 0)
    assert not spans.any()
