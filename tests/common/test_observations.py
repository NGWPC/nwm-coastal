"""Tests for :mod:`coastal_calibration.observations`.

Exercises the three public helpers on tiny synthetic datasets representing
the three mesh types (regular SFINCS, UGRID quadtree SFINCS, SCHISM
triangular) plus error paths in the CSV loader.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_calibration.observations import (
    extract_water_level_series,
    load_obs_points,
    validate_points_in_domain,
)

if TYPE_CHECKING:
    from pathlib import Path

_HAS_PARQUET = importlib.util.find_spec("pyarrow") is not None

# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


class TestLoadObsPoints:
    def test_reads_well_formed_csv(self, tmp_path: Path):
        path = tmp_path / "points.csv"
        path.write_text("id,lon,lat\nA,-71.4,41.55\nB,-71.5,41.7\n")
        df = load_obs_points(path)
        assert list(df.columns) == ["id", "lon", "lat"]
        assert df["id"].tolist() == ["A", "B"]
        assert df["lon"].dtype == np.float64
        assert df["lat"].dtype == np.float64

    def test_extra_columns_ignored(self, tmp_path: Path):
        path = tmp_path / "points.csv"
        path.write_text("id,lon,lat,name,note\nA,-71.4,41.55,Alice,foo\n")
        df = load_obs_points(path)
        assert list(df.columns) == ["id", "lon", "lat"]

    def test_missing_column_raises(self, tmp_path: Path):
        path = tmp_path / "points.csv"
        path.write_text("id,longitude,latitude\nA,-71.4,41.55\n")  # wrong names
        with pytest.raises(ValueError, match="missing required column"):
            load_obs_points(path)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_obs_points(tmp_path / "no.csv")

    def test_duplicate_ids_raise(self, tmp_path: Path):
        path = tmp_path / "points.csv"
        path.write_text("id,lon,lat\nA,-71.4,41.55\nA,-71.5,41.6\n")
        with pytest.raises(ValueError, match="Duplicate ids"):
            load_obs_points(path)

    def test_non_numeric_coord_raises(self, tmp_path: Path):
        path = tmp_path / "points.csv"
        path.write_text("id,lon,lat\nA,bad,41.55\n")
        with pytest.raises(ValueError, match="Non-numeric"):
            load_obs_points(path)


# ---------------------------------------------------------------------------
# Synthetic dataset fixtures
# ---------------------------------------------------------------------------


def _schism_ds() -> xr.Dataset:
    """4-node SCHISM mesh covering lon ∈ [-71.6, -71.1], lat ∈ [41.3, 41.9]."""
    node_x = np.array([-71.6, -71.1, -71.1, -71.6], dtype=np.float64)
    node_y = np.array([41.3, 41.3, 41.9, 41.9], dtype=np.float64)
    face_nodes = np.array([[0, 1, 2, -1], [0, 2, 3, -1]], dtype=np.int64)
    times = pd.date_range("2024-01-01", periods=3, freq="1h")
    # Give each node a distinctive elevation pattern so we can verify mapping.
    elev = np.array(
        [
            [0.10, 0.20, 0.30, 0.40],  # t=0
            [0.11, 0.21, 0.31, 0.41],  # t=1
            [0.12, 0.22, 0.32, 0.42],  # t=2
        ],
        dtype=np.float32,
    )
    return xr.Dataset(
        data_vars={
            "elevation": (("time", "node"), elev),
            "node_x": (("node",), node_x),
            "node_y": (("node",), node_y),
            "face_nodes": (("face", "face_node"), face_nodes),
        },
        coords={"time": times, "node": np.arange(4), "face": np.arange(2)},
        attrs={"mesh_type": "ugrid-triangle-or-quad", "crs": "EPSG:4326"},
    )


def _sfincs_regular_ds() -> xr.Dataset:
    """Regular-grid SFINCS dataset in WGS84 (CRS attribute intentionally omitted)."""
    x = np.linspace(-71.6, -71.1, 6)
    y = np.linspace(41.3, 41.9, 5)
    times = pd.date_range("2024-01-01", periods=3, freq="1h")
    t_idx = np.arange(3).reshape(-1, 1, 1)
    y_idx = np.arange(5).reshape(1, -1, 1)
    x_idx = np.arange(6).reshape(1, 1, -1)
    zs = (t_idx + 1.0) * (y_idx + x_idx + 1.0)
    return xr.Dataset(
        data_vars={"zs": (("time", "y", "x"), zs.astype(np.float32))},
        coords={"time": times, "y": y, "x": x},
        attrs={"mesh_type": "regular"},
    )


def _sfincs_quadtree_utm_ds() -> xr.Dataset:
    """UGRID quadtree in UTM zone 19N; CRS set so the transformer kicks in."""
    # 9 nodes on a 3x3 grid spanning ~ (300000, 4.59e6) .. (302000, 4.61e6),
    # which covers a small patch of Narragansett Bay.
    x0 = np.linspace(300000.0, 302000.0, 3)
    y0 = np.linspace(4590000.0, 4610000.0, 3)
    xx, yy = np.meshgrid(x0, y0)
    node_x = xx.ravel().astype(np.float64)
    node_y = yy.ravel().astype(np.float64)
    # 4 quads, 0-based.
    face_nodes = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]], dtype=np.int64)
    times = pd.date_range("2024-01-01", periods=3, freq="1h")
    # Distinctive face values so we can verify nearest-face lookup.
    zs = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1, 4.1],
            [1.2, 2.2, 3.2, 4.2],
        ],
        dtype=np.float32,
    )
    return xr.Dataset(
        data_vars={
            "zs": (("time", "face"), zs),
            "node_x": (("node",), node_x),
            "node_y": (("node",), node_y),
            "face_nodes": (("face", "face_node"), face_nodes),
        },
        coords={"time": times, "node": np.arange(9), "face": np.arange(4)},
        attrs={"mesh_type": "ugrid-quadtree", "crs": "EPSG:32619"},
    )


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------


class TestValidatePointsInDomain:
    def test_all_inside(self):
        ds = _schism_ds()
        pts = pd.DataFrame({"id": ["A"], "lon": [-71.3], "lat": [41.6]})
        bbox = validate_points_in_domain(pts, ds)
        # Bbox should be close to the mesh extent.
        minx, _miny, maxx, _maxy = bbox.bounds
        assert minx == pytest.approx(-71.6)
        assert maxx == pytest.approx(-71.1)

    def test_some_outside_raises(self):
        ds = _schism_ds()
        pts = pd.DataFrame({"id": ["ok", "bad"], "lon": [-71.3, -75.0], "lat": [41.5, 41.5]})
        with pytest.raises(ValueError, match="outside the model"):
            validate_points_in_domain(pts, ds)

    def test_projected_ds_rebased_to_wgs84(self):
        """A UTM-19N dataset must still validate WGS84-supplied points.

        The 9-node synthetic mesh spans roughly lon ∈ [-71.40, -71.37],
        lat ∈ [41.44, 41.62] once reprojected.
        """
        ds = _sfincs_quadtree_utm_ds()
        inside = pd.DataFrame({"id": ["A"], "lon": [-71.38], "lat": [41.50]})
        validate_points_in_domain(inside, ds)

        outside = pd.DataFrame({"id": ["X"], "lon": [-60.0], "lat": [41.5]})
        with pytest.raises(ValueError, match="outside the model"):
            validate_points_in_domain(outside, ds)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class TestExtractSchism:
    def test_nearest_node_extraction(self):
        ds = _schism_ds()
        # Near node 0 (-71.6, 41.3) → elev column 0 at t=0 is 0.10.
        pts = pd.DataFrame({"id": ["near_node0"], "lon": [-71.55], "lat": [41.32]})
        series = extract_water_level_series(ds, pts, variable="elevation")
        assert list(series.columns) == ["near_node0"]
        assert series.iloc[0, 0] == pytest.approx(0.10)
        assert series.iloc[2, 0] == pytest.approx(0.12)

    def test_preserves_ids_and_order(self):
        ds = _schism_ds()
        pts = pd.DataFrame(
            {
                "id": ["near_node1", "near_node3"],
                "lon": [-71.12, -71.58],
                "lat": [41.32, 41.88],
            }
        )
        series = extract_water_level_series(ds, pts, variable="elevation")
        assert list(series.columns) == ["near_node1", "near_node3"]
        # Node 1 elevation at t=0 is 0.20; node 3 is 0.40.
        assert series.iloc[0, 0] == pytest.approx(0.20)
        assert series.iloc[0, 1] == pytest.approx(0.40)

    def test_empty_points_returns_time_only(self):
        ds = _schism_ds()
        pts = pd.DataFrame(columns=["id", "lon", "lat"])
        series = extract_water_level_series(ds, pts, variable="elevation")
        assert len(series) == 3
        assert list(series.columns) == []


class TestExtractQuadtree:
    def test_nearest_face_extraction_with_crs_transform(self):
        """Quadtree + UTM CRS: WGS84 query point should resolve correctly."""
        ds = _sfincs_quadtree_utm_ds()
        # Face 0 center at UTM (300500, 4595000) is lon=-71.3895, lat=41.4818.
        pts = pd.DataFrame({"id": ["A"], "lon": [-71.3895], "lat": [41.4818]})
        series = extract_water_level_series(ds, pts, variable="zs")
        # face 0 zs at t=0 is 1.0; verify nearest-face resolved to face 0.
        assert series.iloc[0, 0] == pytest.approx(1.0)
        assert series.iloc[1, 0] == pytest.approx(1.1)


class TestExtractRegular:
    def test_nearest_grid_cell(self):
        ds = _sfincs_regular_ds()
        # Query at y-index 2, x-index 3 → zs = (0+1) * (2 + 3 + 1) = 6.0 at t=0.
        y_target = ds["y"].values[2]
        x_target = ds["x"].values[3]
        pts = pd.DataFrame({"id": ["A"], "lon": [x_target], "lat": [y_target]})
        series = extract_water_level_series(ds, pts, variable="zs")
        assert series.iloc[0, 0] == pytest.approx(6.0)
        assert series.iloc[1, 0] == pytest.approx(12.0)  # t=1 → 2 * 6


# ---------------------------------------------------------------------------
# Round-trip via parquet
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PARQUET, reason="pyarrow not available (test-common env)")
class TestParquetRoundtrip:
    def test_dataframe_round_trips(self, tmp_path: Path):
        ds = _schism_ds()
        pts = pd.DataFrame({"id": ["a", "b"], "lon": [-71.55, -71.15], "lat": [41.32, 41.32]})
        series = extract_water_level_series(ds, pts, variable="elevation")
        out = tmp_path / "obs_water_level.parquet"
        series.to_parquet(out)
        loaded = pd.read_parquet(out)
        pd.testing.assert_frame_equal(series, loaded, check_exact=False)
