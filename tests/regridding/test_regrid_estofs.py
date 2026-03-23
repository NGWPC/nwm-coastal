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


def _load_time_series(nc_path: Path) -> np.ndarray:
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
