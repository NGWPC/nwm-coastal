"""Tests for :mod:`coastal_calibration.schism.outputs`.

A synthetic two-block SCHISM output is built on the fly from a tiny
5-node triangular mesh. No external fixtures, no SCHISM binaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from coastal_calibration.schism.outputs import (
    _normalise_face_nodes,
    _parse_base_date,
    load_schism_elevation,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic out2d_<iblock>.nc builder
# ---------------------------------------------------------------------------

# A tiny mesh: 5 nodes, 3 triangles.  Node coords (lon/lat-like).
_NODE_X = np.array([0.0, 1.0, 2.0, 0.5, 1.5], dtype=np.float64)
_NODE_Y = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)

# Triangles, SCHISM convention: 1-based node IDs, 0-fill for the missing 4th
# vertex.  Shape (n_face, 4).
_FACE_NODES_1BASED = np.array(
    [
        [1, 2, 4, 0],
        [2, 5, 4, 0],
        [2, 3, 5, 0],
    ],
    dtype=np.int32,
)


#: Default static bathymetric depth for synthetic blocks (positive = below datum).
_DEPTH = np.array([5.0, 4.5, 4.0, 3.5, 3.0], dtype=np.float64)


def _write_out2d(
    path: Path,
    *,
    seconds: np.ndarray,
    elev: np.ndarray,
    base_date: str = "2020 1 1 0 0",
    depth: np.ndarray | None = None,
    dry_flag: np.ndarray | None = None,
    crs_grid_mapping: str | None = None,
) -> None:
    """Write a minimal out2d_*.nc file mimicking SCHISM's schema."""
    if depth is None:
        depth = _DEPTH
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "elevation": (("time", "nSCHISM_hgrid_node"), elev.astype(np.float32)),
        "depth": (("nSCHISM_hgrid_node",), depth.astype(np.float64)),
        "SCHISM_hgrid_node_x": (("nSCHISM_hgrid_node",), _NODE_X),
        "SCHISM_hgrid_node_y": (("nSCHISM_hgrid_node",), _NODE_Y),
        "SCHISM_hgrid_face_nodes": (
            ("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
            _FACE_NODES_1BASED,
        ),
    }
    if dry_flag is not None:
        data_vars["dryFlagNode"] = (
            ("time", "nSCHISM_hgrid_node"),
            dry_flag.astype(np.int8),
        )
    if crs_grid_mapping is not None:
        data_vars["crs"] = ((), np.int32(0))
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": seconds.astype(np.float64)},
    )
    ds["time"].attrs["base_date"] = base_date
    ds["time"].attrs["units"] = "seconds"
    if crs_grid_mapping is not None:
        ds["crs"].attrs["grid_mapping_name"] = crs_grid_mapping
    ds.to_netcdf(path)


@pytest.fixture
def two_block_run(tmp_path: Path) -> Path:
    """Write two out2d blocks with contiguous times. Returns the run directory."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    # Block 1: t = [0, 3600, 7200]
    t1 = np.array([0.0, 3600.0, 7200.0])
    e1 = np.outer(np.arange(1.0, 4.0), np.arange(1.0, 6.0))  # (3, 5)
    _write_out2d(outputs / "out2d_1.nc", seconds=t1, elev=e1)

    # Block 2: t = [10800, 14400]  (continues after block 1)
    t2 = np.array([10800.0, 14400.0])
    e2 = np.outer(np.arange(4.0, 6.0), np.arange(1.0, 6.0))  # (2, 5)
    _write_out2d(outputs / "out2d_2.nc", seconds=t2, elev=e2)

    return tmp_path


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


class TestNormaliseFaceNodes:
    def test_triangles_get_minus_one(self):
        """Valid vertices become 0-based; padding (0) becomes -1."""
        out = _normalise_face_nodes(_FACE_NODES_1BASED)
        # All non-zero entries shift by -1; zeros become -1.
        expected = np.array(
            [
                [0, 1, 3, -1],
                [1, 4, 3, -1],
                [1, 2, 4, -1],
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(out, expected)

    def test_large_negative_fill(self):
        """Large negative _FillValue should also collapse to -1."""
        raw = np.array([[1, 2, 3, -999999]], dtype=np.int64)
        out = _normalise_face_nodes(raw)
        np.testing.assert_array_equal(out, np.array([[0, 1, 2, -1]]))


class TestParseBaseDate:
    def test_five_fields(self):
        ts = _parse_base_date("2020 3 15 12 30")
        assert ts.year == 2020
        assert ts.month == 3
        assert ts.day == 15
        assert ts.hour == 12
        assert ts.minute == 30

    def test_trailing_utc_offset_ignored(self):
        ts = _parse_base_date("2020 1 1 0 0 0.0")
        assert ts.year == 2020

    def test_float_fields(self):
        """SCHISM sometimes writes e.g. ``"2020. 1. 1. 0. 0."``."""
        ts = _parse_base_date("2020. 1. 1. 0. 0.")
        assert ts.year == 2020
        assert ts.day == 1

    def test_missing_fields_raises(self):
        with pytest.raises(ValueError, match="Unexpected SCHISM base_date"):
            _parse_base_date("2020 1")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestLoadSchismElevation:
    def test_basic_load(self, two_block_run: Path):
        ds = load_schism_elevation(two_block_run)

        # Dimensions.
        assert ds.sizes["time"] == 5
        assert ds.sizes["node"] == 5
        assert ds.sizes["face"] == 3

        # Absolute datetime axis.
        assert ds.time.dtype.kind == "M"
        assert ds.time.to_pandas().iloc[0].year == 2020
        assert ds.time.to_pandas().iloc[0].hour == 0
        assert ds.time.to_pandas().iloc[-1].hour == 4  # 14400 s = 4 h

        # Mesh variables.
        np.testing.assert_array_equal(ds["node_x"].to_numpy(), _NODE_X)
        np.testing.assert_array_equal(ds["node_y"].to_numpy(), _NODE_Y)

        # Bathymetric depth (static).
        np.testing.assert_array_equal(ds["depth"].to_numpy(), _DEPTH)

        # face_nodes: 0-based, -1 fill for triangles.
        fn = ds["face_nodes"].to_numpy()
        assert fn.dtype == np.int64
        assert fn.shape == (3, 4)
        # Each triangle should have exactly one -1 entry.
        assert (fn == -1).sum(axis=1).tolist() == [1, 1, 1]

        # Attributes used by the shared renderer.
        assert ds.attrs["mesh_type"] == "ugrid-triangle-or-quad"
        assert ds.attrs["source_run"] == str(two_block_run)

    def test_water_depth_is_derived(self, two_block_run: Path):
        """``h = elevation + depth``, broadcast across time."""
        ds = load_schism_elevation(two_block_run)
        elev = ds["elevation"].to_numpy()
        depth = ds["depth"].to_numpy()
        np.testing.assert_allclose(ds["h"].to_numpy(), elev + depth[np.newaxis, :])

    def test_dry_flag_passes_through_when_present(self, tmp_path: Path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        n_t, n_n = 2, 5
        elev = np.outer(np.arange(1.0, n_t + 1.0), np.arange(1.0, n_n + 1.0))
        # Dry the last two nodes throughout the run.
        dry = np.zeros((n_t, n_n), dtype=np.int8)
        dry[:, 3:] = 1
        _write_out2d(
            outputs / "out2d_1.nc",
            seconds=np.array([0.0, 3600.0]),
            elev=elev,
            dry_flag=dry,
        )
        ds = load_schism_elevation(tmp_path)
        assert "dryFlagNode" in ds.data_vars
        np.testing.assert_array_equal(ds["dryFlagNode"].to_numpy(), dry)

    def test_dry_flag_absent_when_missing(self, two_block_run: Path):
        """Files without dryFlagNode load fine; the variable is just absent."""
        ds = load_schism_elevation(two_block_run)
        assert "dryFlagNode" not in ds.data_vars

    def test_crs_detected_for_lat_lon(self, tmp_path: Path):
        """A `crs` variable with grid_mapping_name='latitude_longitude' → EPSG:4326."""
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        _write_out2d(
            outputs / "out2d_1.nc",
            seconds=np.array([0.0]),
            elev=np.zeros((1, 5), dtype=np.float32),
            crs_grid_mapping="latitude_longitude",
        )
        ds = load_schism_elevation(tmp_path)
        assert ds.attrs.get("crs") == "EPSG:4326"

    def test_crs_absent_when_no_crs_var(self, two_block_run: Path):
        """The default fixture has no ``crs`` variable; attribute is absent."""
        ds = load_schism_elevation(two_block_run)
        assert "crs" not in ds.attrs

    def test_crs_absent_for_projected_mesh(self, tmp_path: Path):
        """Projected (non-latlon) meshes don't auto-resolve to a single EPSG."""
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        _write_out2d(
            outputs / "out2d_1.nc",
            seconds=np.array([0.0]),
            elev=np.zeros((1, 5), dtype=np.float32),
            crs_grid_mapping="transverse_mercator",
        )
        ds = load_schism_elevation(tmp_path)
        # We don't reconstruct EPSG from CF parameters — leave it for the caller.
        assert "crs" not in ds.attrs

    def test_elevation_values_and_ordering(self, two_block_run: Path):
        ds = load_schism_elevation(two_block_run)
        # Block 1: rows 1..3 by cols 1..5; block 2: rows 4..5 by cols 1..5.
        # Concatenated along time, sorted: row index i has value (i+1)*(node+1).
        expected = np.outer(np.arange(1.0, 6.0), np.arange(1.0, 6.0))
        np.testing.assert_allclose(ds["elevation"].to_numpy(), expected)

    def test_time_slice(self, two_block_run: Path):
        ds = load_schism_elevation(two_block_run, time_slice=slice(1, 4))
        assert ds.sizes["time"] == 3
        # Mesh unchanged.
        assert ds.sizes["node"] == 5
        assert ds.sizes["face"] == 3

    def test_accepts_outputs_dir_directly(self, two_block_run: Path):
        """Passing the outputs/ directory should work identically."""
        ds1 = load_schism_elevation(two_block_run)
        ds2 = load_schism_elevation(two_block_run / "outputs")
        np.testing.assert_array_equal(ds1["elevation"], ds2["elevation"])

    def test_missing_directory(self, tmp_path: Path):
        with pytest.raises(NotADirectoryError):
            load_schism_elevation(tmp_path / "does-not-exist")

    def test_empty_outputs_dir(self, tmp_path: Path):
        (tmp_path / "outputs").mkdir()
        with pytest.raises(FileNotFoundError, match="No SCHISM"):
            load_schism_elevation(tmp_path)

    def test_missing_required_variable(self, tmp_path: Path):
        """A file without ``SCHISM_hgrid_node_x`` should raise KeyError."""
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        bad = xr.Dataset(
            data_vars={
                "elevation": (("time", "nSCHISM_hgrid_node"), np.zeros((1, 5))),
                "SCHISM_hgrid_node_y": (("nSCHISM_hgrid_node",), _NODE_Y),
                "SCHISM_hgrid_face_nodes": (
                    ("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
                    _FACE_NODES_1BASED,
                ),
            },
            coords={"time": np.array([0.0])},
        )
        bad["time"].attrs["base_date"] = "2020 1 1 0 0"
        bad.to_netcdf(outputs / "out2d_1.nc")

        with pytest.raises(KeyError, match="SCHISM_hgrid_node_x"):
            load_schism_elevation(tmp_path)

    def test_missing_base_date_attr(self, tmp_path: Path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        t = np.array([0.0, 3600.0])
        e = np.zeros((2, 5), dtype=np.float32)
        ds = xr.Dataset(
            data_vars={
                "elevation": (("time", "nSCHISM_hgrid_node"), e),
                "depth": (("nSCHISM_hgrid_node",), _DEPTH),
                "SCHISM_hgrid_node_x": (("nSCHISM_hgrid_node",), _NODE_X),
                "SCHISM_hgrid_node_y": (("nSCHISM_hgrid_node",), _NODE_Y),
                "SCHISM_hgrid_face_nodes": (
                    ("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
                    _FACE_NODES_1BASED,
                ),
            },
            coords={"time": t},
        )
        # Deliberately no base_date attribute.
        ds.to_netcdf(outputs / "out2d_1.nc")

        with pytest.raises(KeyError, match="base_date"):
            load_schism_elevation(tmp_path)
