"""Tests for :mod:`coastal_calibration.sfincs.outputs`.

Synthetic ``sfincs_map.nc`` files are written on the fly to exercise the
regular-grid path, the quadtree guard, and the variable-missing error.
No external fixtures, no SFINCS binary, no HydroMT.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_calibration.sfincs.outputs import (
    _detect_layout,
    load_sfincs_water_level,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_regular_map(
    path: Path,
    *,
    n_time: int = 4,
    n_y: int = 5,
    n_x: int = 6,
    include_xc_yc: bool = True,
    include_msk: bool = False,
) -> xr.Dataset:
    """Write a synthetic regular-grid ``sfincs_map.nc`` and return the dataset."""
    x = np.linspace(0.0, 500.0, n_x)
    y = np.linspace(0.0, 400.0, n_y)
    times = pd.date_range("2024-01-01", periods=n_time, freq="1h")

    # Deterministic pattern so value-equality tests are easy.
    t_idx = np.arange(n_time).reshape(-1, 1, 1)
    y_idx = np.arange(n_y).reshape(1, -1, 1)
    x_idx = np.arange(n_x).reshape(1, 1, -1)
    zs = (t_idx + 1.0) * (y_idx + x_idx + 1.0)  # (time, y, x)
    # Bed elevation linearly decreasing across the domain so depth h = zs - zb
    # is nontrivial but always positive at t=0.
    yy, xx = np.meshgrid(y, x, indexing="ij")
    zb = -1.0 - 0.001 * (yy + xx)  # negative => below datum

    data_vars: dict[str, tuple[tuple[str, ...], NDArray[Any]]] = {
        "zs": (("time", "y", "x"), zs.astype(np.float32)),
        "zb": (("y", "x"), zb.astype(np.float32)),
    }
    if include_msk:
        data_vars["msk"] = (("y", "x"), np.ones((n_y, n_x), dtype=np.int8))
    if include_xc_yc:
        data_vars["xc"] = (("y", "x"), xx.astype(np.float64))
        data_vars["yc"] = (("y", "x"), yy.astype(np.float64))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": times, "y": y, "x": x},
    )
    ds.to_netcdf(path)
    return ds


def _write_quadtree_map(path: Path) -> None:
    """Write a minimal ``sfincs_map.nc`` with the telltale ``n``/``m`` dims."""
    ds = xr.Dataset(
        data_vars={
            "zs": (("time", "n", "m"), np.zeros((2, 3, 3), dtype=np.float32)),
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=2, freq="1h"),
        },
    )
    ds.to_netcdf(path)


def _write_ugrid_quadtree_map(path: Path, *, include_msk: bool = False) -> xr.Dataset:
    """Write a minimal UGRID-style quadtree ``sfincs_map.nc``.

    Four quads arranged in a 2x2 block (9 nodes), face-valued zs + zb.
    """
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)
    face_nodes = np.array(
        [[1, 2, 5, 4], [2, 3, 6, 5], [4, 5, 8, 7], [5, 6, 9, 8]], dtype=np.float64
    )
    times = pd.date_range("2024-01-01", periods=3, freq="1h")
    zs = np.outer(np.arange(1.0, 4.0), np.arange(1.0, 5.0)).astype(np.float32)
    zb = np.array([-2.0, -1.5, -1.0, -0.5], dtype=np.float32)  # bed below datum

    data_vars: dict[str, tuple[tuple[str, ...], NDArray[Any]]] = {
        "zs": (("time", "nmesh2d_face"), zs),
        "zb": (("nmesh2d_face",), zb),
        "mesh2d_node_x": (("nmesh2d_node",), node_x),
        "mesh2d_node_y": (("nmesh2d_node",), node_y),
        "mesh2d_face_nodes": (
            ("nmesh2d_face", "max_nmesh2d_face_nodes"),
            face_nodes,
        ),
    }
    if include_msk:
        data_vars["msk"] = (("nmesh2d_face",), np.ones(4, dtype=np.int8))

    ds = xr.Dataset(data_vars=data_vars, coords={"time": times})
    ds["mesh2d_face_nodes"].attrs["start_index"] = 1
    ds.to_netcdf(path)
    return ds


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


class TestDetectLayout:
    def test_regular(self, tmp_path: Path):
        p = tmp_path / "sfincs_map.nc"
        _write_regular_map(p)
        with xr.open_dataset(p) as ds:
            assert _detect_layout(ds) == "regular"

    def test_structured_nm(self, tmp_path: Path):
        p = tmp_path / "sfincs_map.nc"
        _write_quadtree_map(p)
        with xr.open_dataset(p) as ds:
            assert _detect_layout(ds) == "structured-nm"

    def test_ugrid_quadtree(self, tmp_path: Path):
        p = tmp_path / "sfincs_map.nc"
        _write_ugrid_quadtree_map(p)
        with xr.open_dataset(p) as ds:
            assert _detect_layout(ds) == "ugrid-quadtree"


# ---------------------------------------------------------------------------
# Integration tests — regular grid
# ---------------------------------------------------------------------------


class TestLoadSfincsRegular:
    def test_basic_load(self, tmp_path: Path):
        _write_regular_map(tmp_path / "sfincs_map.nc")
        ds = load_sfincs_water_level(tmp_path)

        assert ds.attrs["mesh_type"] == "regular"
        assert ds.attrs["source_run"] == str(tmp_path)
        assert ds["zs"].dims == ("time", "y", "x")
        assert ds["h"].dims == ("time", "y", "x")
        assert ds["zb"].dims == ("y", "x")

        # 2-D aux geographic coords preserved.
        assert ds["xc"].dims == ("y", "x")
        assert ds["yc"].dims == ("y", "x")

    def test_water_depth_is_derived(self, tmp_path: Path):
        """``h = zs - zb`` broadcast over time."""
        src = _write_regular_map(tmp_path / "sfincs_map.nc")
        ds = load_sfincs_water_level(tmp_path)
        expected = src["zs"].to_numpy() - src["zb"].to_numpy()[np.newaxis, :, :]
        np.testing.assert_allclose(ds["h"].to_numpy(), expected, rtol=1e-5)

    def test_zs_values_preserved(self, tmp_path: Path):
        src = _write_regular_map(tmp_path / "sfincs_map.nc")
        ds = load_sfincs_water_level(tmp_path)
        np.testing.assert_allclose(ds["zs"].to_numpy(), src["zs"].to_numpy())

    def test_msk_passes_through_when_present(self, tmp_path: Path):
        _write_regular_map(tmp_path / "sfincs_map.nc", include_msk=True)
        ds = load_sfincs_water_level(tmp_path)
        assert "msk" in ds.data_vars

    def test_msk_absent_when_missing(self, tmp_path: Path):
        _write_regular_map(tmp_path / "sfincs_map.nc", include_msk=False)
        ds = load_sfincs_water_level(tmp_path)
        assert "msk" not in ds.data_vars

    def test_no_xc_yc_optional(self, tmp_path: Path):
        _write_regular_map(tmp_path / "sfincs_map.nc", include_xc_yc=False)
        ds = load_sfincs_water_level(tmp_path)
        assert "xc" not in ds.data_vars
        assert "xc" not in ds.coords

    def test_time_slice(self, tmp_path: Path):
        _write_regular_map(tmp_path / "sfincs_map.nc")
        ds = load_sfincs_water_level(tmp_path, time_slice=slice(1, 3))
        assert ds.sizes["time"] == 2

    def test_accepts_direct_file_path(self, tmp_path: Path):
        map_file = tmp_path / "sfincs_map.nc"
        _write_regular_map(map_file)
        ds = load_sfincs_water_level(map_file)
        assert ds.sizes["time"] == 4


# ---------------------------------------------------------------------------
# Integration tests — UGRID quadtree
# ---------------------------------------------------------------------------


class TestLoadSfincsQuadtree:
    def test_basic_load(self, tmp_path: Path):
        _write_ugrid_quadtree_map(tmp_path / "sfincs_map.nc")
        ds = load_sfincs_water_level(tmp_path)

        assert ds.attrs["mesh_type"] == "ugrid-quadtree"
        assert ds["zs"].dims == ("time", "face")
        assert ds["h"].dims == ("time", "face")
        assert ds["zb"].dims == ("face",)
        assert ds.sizes == {"time": 3, "face": 4, "node": 9, "face_node": 4}

        # face_nodes 0-based, no -1 fill (all quads).
        fn = ds["face_nodes"].to_numpy()
        assert fn.dtype == np.int64
        assert fn.min() == 0
        assert fn.max() == 8

    def test_water_depth_is_derived(self, tmp_path: Path):
        src = _write_ugrid_quadtree_map(tmp_path / "sfincs_map.nc")
        ds = load_sfincs_water_level(tmp_path)
        expected = src["zs"].to_numpy() - src["zb"].to_numpy()[np.newaxis, :]
        np.testing.assert_allclose(ds["h"].to_numpy(), expected, rtol=1e-5)

    def test_msk_passes_through_when_present(self, tmp_path: Path):
        _write_ugrid_quadtree_map(tmp_path / "sfincs_map.nc", include_msk=True)
        ds = load_sfincs_water_level(tmp_path)
        assert "msk" in ds.data_vars

    def test_triangle_padding_in_face_nodes(self, tmp_path: Path):
        """A triangle cell (stored with 0 in 1-based) should become -1 fill."""
        node_x = np.array([0.0, 1.0, 0.5], dtype=np.float64)
        node_y = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        face_nodes = np.array([[1, 2, 3, 0]], dtype=np.float64)
        times = pd.date_range("2024-01-01", periods=1, freq="1h")
        ds_src = xr.Dataset(
            data_vars={
                "zs": (("time", "nmesh2d_face"), np.array([[1.0]], dtype=np.float32)),
                "zb": (("nmesh2d_face",), np.array([-2.0], dtype=np.float32)),
                "mesh2d_node_x": (("nmesh2d_node",), node_x),
                "mesh2d_node_y": (("nmesh2d_node",), node_y),
                "mesh2d_face_nodes": (
                    ("nmesh2d_face", "max_nmesh2d_face_nodes"),
                    face_nodes,
                ),
            },
            coords={"time": times},
        )
        ds_src.to_netcdf(tmp_path / "sfincs_map.nc")

        ds = load_sfincs_water_level(tmp_path)
        np.testing.assert_array_equal(ds["face_nodes"].to_numpy()[0], [0, 1, 2, -1])


# ---------------------------------------------------------------------------
# Error paths + namespace
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_directory(self, tmp_path: Path):
        with pytest.raises(NotADirectoryError):
            load_sfincs_water_level(tmp_path / "does-not-exist")

    def test_missing_map_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match=r"sfincs_map\.nc"):
            load_sfincs_water_level(tmp_path)

    def test_missing_zb_raises(self, tmp_path: Path):
        """Without zb we can't derive h; the loader should surface that early."""
        x = np.linspace(0.0, 5.0, 3)
        y = np.linspace(0.0, 4.0, 2)
        times = pd.date_range("2024-01-01", periods=2, freq="1h")
        zs = np.zeros((2, 2, 3), dtype=np.float32)
        ds = xr.Dataset(
            data_vars={"zs": (("time", "y", "x"), zs)},
            coords={"time": times, "y": y, "x": x},
        )
        ds.to_netcdf(tmp_path / "sfincs_map.nc")
        with pytest.raises(KeyError, match=r"'zb'"):
            load_sfincs_water_level(tmp_path)

    def test_structured_nm_not_implemented(self, tmp_path: Path):
        _write_quadtree_map(tmp_path / "sfincs_map.nc")
        with pytest.raises(NotImplementedError, match=r"Structured \(n, m\)"):
            load_sfincs_water_level(tmp_path)


def test_sfincs_namespace_lazy_import():
    """The symbol is exposed on the sfincs subpackage via lazy import."""
    from coastal_calibration import sfincs

    assert callable(sfincs.load_sfincs_water_level)


class TestCrsDetection:
    def test_no_crs_when_inp_and_nc_absent(self, tmp_path: Path):
        """Without sfincs.inp or sfincs.nc nearby, ``crs`` is not populated."""
        _write_regular_map(tmp_path / "sfincs_map.nc")
        ds = load_sfincs_water_level(tmp_path)
        assert "crs" not in ds.attrs

    def test_crs_from_sfincs_inp(self, tmp_path: Path):
        _write_regular_map(tmp_path / "sfincs_map.nc")
        (tmp_path / "sfincs.inp").write_text("epsg = 32614\nother = foo\n")
        ds = load_sfincs_water_level(tmp_path)
        assert ds.attrs["crs"] == "EPSG:32614"

    def test_crs_from_sfincs_nc_wkt(self, tmp_path: Path):
        """When sfincs.inp is absent, fall back to the WKT in sfincs.nc."""
        _write_regular_map(tmp_path / "sfincs_map.nc")
        wkt = (
            'PROJCRS["WGS 84 / UTM zone 19N",BASEGEOGCRS["WGS 84",ID["EPSG",4326]],'
            'CONVERSION["UTM zone 19N"],ID["EPSG",32619]]'
        )
        ds_grid = xr.Dataset(
            data_vars={"crs": ((), np.int32(0))},
        )
        ds_grid["crs"].attrs["crs_wkt"] = wkt
        ds_grid.to_netcdf(tmp_path / "sfincs.nc")
        ds = load_sfincs_water_level(tmp_path)
        assert ds.attrs["crs"] == "EPSG:32619"

    def test_inp_takes_priority_over_nc(self, tmp_path: Path):
        _write_regular_map(tmp_path / "sfincs_map.nc")
        (tmp_path / "sfincs.inp").write_text("epsg = 32614\n")
        wkt = 'PROJCRS["..."],ID["EPSG",32619]]'
        ds_grid = xr.Dataset(data_vars={"crs": ((), np.int32(0))})
        ds_grid["crs"].attrs["crs_wkt"] = wkt
        ds_grid.to_netcdf(tmp_path / "sfincs.nc")
        ds = load_sfincs_water_level(tmp_path)
        # sfincs.inp checked before sfincs.nc.
        assert ds.attrs["crs"] == "EPSG:32614"
