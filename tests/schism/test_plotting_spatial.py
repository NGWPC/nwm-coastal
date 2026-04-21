"""Tests for :mod:`coastal_calibration.plotting.spatial`.

Covers both dispatch paths:

- Unstructured ``ugrid-triangle-or-quad`` (SCHISM) — tripcolor
- Regular ``regular`` (SFINCS) — pcolormesh

Matplotlib runs under the ``Agg`` backend so tests are display-free.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.collections import Collection, QuadMesh

from coastal_calibration.plotting.spatial import (
    _apply_wet_mask,
    _auto_variable,
    _colorbar_extend,
    _extended_cmap,
    _wet_mask,
    plot_water_level,
    triangulate_faces,
    triangulate_faces_indexed,
)

# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------


def _schism_dataset(n_time: int = 3) -> xr.Dataset:
    """5-node triangular mesh (3 triangles)."""
    node_x = np.array([0.0, 1.0, 2.0, 0.5, 1.5])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    face_nodes = np.array(
        [[0, 1, 3, -1], [1, 4, 3, -1], [1, 2, 4, -1]],
        dtype=np.int64,
    )
    times = pd.date_range("2024-01-01", periods=n_time, freq="1h")
    # Deterministic per-node, per-time values.
    elevation = np.outer(np.arange(1.0, n_time + 1.0), np.arange(1.0, 6.0))
    return xr.Dataset(
        data_vars={
            "elevation": (("time", "node"), elevation.astype(np.float32)),
            "node_x": (("node",), node_x),
            "node_y": (("node",), node_y),
            "face_nodes": (("face", "face_node"), face_nodes),
        },
        coords={"time": times, "node": np.arange(5), "face": np.arange(3)},
        attrs={"mesh_type": "ugrid-triangle-or-quad"},
    )


def _schism_mixed_tri_quad_dataset() -> xr.Dataset:
    """6 nodes arranged in a 3x2 grid, 1 quad + 2 triangles."""
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    # Quad: (0, 1, 4, 3); triangles: (1, 2, 4) & (2, 5, 4).
    face_nodes = np.array(
        [[0, 1, 4, 3], [1, 2, 4, -1], [2, 5, 4, -1]],
        dtype=np.int64,
    )
    times = pd.date_range("2024-01-01", periods=2, freq="1h")
    elevation = np.outer(np.arange(1.0, 3.0), np.arange(1.0, 7.0))
    return xr.Dataset(
        data_vars={
            "elevation": (("time", "node"), elevation.astype(np.float32)),
            "node_x": (("node",), node_x),
            "node_y": (("node",), node_y),
            "face_nodes": (("face", "face_node"), face_nodes),
        },
        coords={"time": times, "node": np.arange(6), "face": np.arange(3)},
        attrs={"mesh_type": "ugrid-triangle-or-quad"},
    )


def _sfincs_quadtree_dataset(n_time: int = 3) -> xr.Dataset:
    """9-node, 4-face quadtree (2x2 block of quads). Face-valued zs."""
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)
    # 4 quads, 0-based CCW.
    face_nodes = np.array(
        [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]], dtype=np.int64
    )
    times = pd.date_range("2024-01-01", periods=n_time, freq="1h")
    zs = np.outer(np.arange(1.0, n_time + 1.0), np.arange(1.0, 5.0)).astype(np.float32)
    return xr.Dataset(
        data_vars={
            "zs": (("time", "face"), zs),
            "node_x": (("node",), node_x),
            "node_y": (("node",), node_y),
            "face_nodes": (("face", "face_node"), face_nodes),
        },
        coords={
            "time": times,
            "node": np.arange(9),
            "face": np.arange(4),
        },
        attrs={"mesh_type": "ugrid-quadtree"},
    )


def _sfincs_regular_dataset(n_time: int = 3, n_y: int = 4, n_x: int = 5) -> xr.Dataset:
    x = np.linspace(0.0, 100.0, n_x)
    y = np.linspace(0.0, 80.0, n_y)
    times = pd.date_range("2024-01-01", periods=n_time, freq="1h")
    t_idx = np.arange(n_time).reshape(-1, 1, 1)
    y_idx = np.arange(n_y).reshape(1, -1, 1)
    x_idx = np.arange(n_x).reshape(1, 1, -1)
    zs = (t_idx + 1.0) * (y_idx + x_idx + 1.0)
    return xr.Dataset(
        data_vars={"zs": (("time", "y", "x"), zs.astype(np.float32))},
        coords={"time": times, "y": y, "x": x},
        attrs={"mesh_type": "regular"},
    )


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestTriangulateFaces:
    def test_pure_triangles(self):
        fn = np.array([[0, 1, 2, -1], [1, 3, 2, -1]])
        tri = triangulate_faces(fn)
        assert tri.shape == (2, 3)
        np.testing.assert_array_equal(tri[0], [0, 1, 2])
        np.testing.assert_array_equal(tri[1], [1, 3, 2])

    def test_pure_quads_split_to_two_triangles(self):
        fn = np.array([[0, 1, 4, 3]])
        tri = triangulate_faces(fn)
        # One quad → two triangles.
        assert tri.shape == (2, 3)
        # Both sub-triangles share vertex 0 and use (1, 4) + (4, 3).
        np.testing.assert_array_equal(sorted(tri[0]), sorted([0, 1, 4]))
        np.testing.assert_array_equal(sorted(tri[1]), sorted([0, 4, 3]))

    def test_mixed(self):
        fn = np.array([[0, 1, 4, 3], [1, 2, 4, -1], [2, 5, 4, -1]])
        tri = triangulate_faces(fn)
        # 1 quad (→ 2) + 2 triangles = 4 triangles total.
        assert tri.shape == (4, 3)

    def test_three_col_input(self):
        fn = np.array([[0, 1, 2], [1, 3, 2]])
        tri = triangulate_faces(fn)
        np.testing.assert_array_equal(tri, fn)

    def test_bad_shape(self):
        with pytest.raises(ValueError, match="2-D"):
            triangulate_faces(np.array([0, 1, 2]))
        with pytest.raises(ValueError, match="at least 3 columns"):
            triangulate_faces(np.array([[0, 1]]))


class TestTriangulateFacesIndexed:
    def test_triangle_face_map_pure_triangles(self):
        fn = np.array([[0, 1, 2, -1], [1, 3, 2, -1]])
        tris, tri_face = triangulate_faces_indexed(fn)
        assert tris.shape == (2, 3)
        np.testing.assert_array_equal(tri_face, [0, 1])

    def test_triangle_face_map_pure_quads(self):
        """Each quad gets two triangles, both tagged with the same face index."""
        fn = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
        tris, tri_face = triangulate_faces_indexed(fn)
        assert tris.shape == (4, 3)
        # Two halves of face 0, then two halves of face 1.
        np.testing.assert_array_equal(sorted(tri_face.tolist()), [0, 0, 1, 1])

    def test_face_values_broadcast_via_map(self):
        """Face-valued data should expand to triangle-valued via the map."""
        fn = np.array([[0, 1, 4, 3], [1, 2, 4, -1]])
        tris, tri_face = triangulate_faces_indexed(fn)
        # face 0 → two triangle halves; face 1 → one triangle.
        face_values = np.array([100.0, 200.0])
        tri_values = face_values[tri_face]
        # Triangle for face 1 (the pure triangle) comes first; then 2 halves for face 0.
        assert tri_values.size == tris.shape[0] == 3
        np.testing.assert_array_equal(sorted(tri_values), [100.0, 100.0, 200.0])


class TestColorbarExtend:
    def test_below_only(self):
        vals = np.array([-5.0, 0.5, 1.5])
        assert _colorbar_extend(vals, vmin=0.0, vmax=2.0) == "min"

    def test_above_only(self):
        vals = np.array([0.5, 1.5, 10.0])
        assert _colorbar_extend(vals, vmin=0.0, vmax=2.0) == "max"

    def test_both_ends(self):
        vals = np.array([-1.0, 0.5, 10.0])
        assert _colorbar_extend(vals, vmin=0.0, vmax=2.0) == "both"

    def test_in_range(self):
        vals = np.array([0.1, 0.5, 1.9])
        assert _colorbar_extend(vals, vmin=0.0, vmax=2.0) == "neither"

    def test_all_nan(self):
        vals = np.array([np.nan, np.nan])
        assert _colorbar_extend(vals, vmin=0.0, vmax=2.0) == "neither"


class TestWetMask:
    def test_dry_flag_takes_priority(self):
        """When ``dryFlagNode`` is in the dataset, ``h`` is ignored."""
        ds = xr.Dataset(
            {
                "elevation": (("time", "node"), np.array([[1.0, 2.0]])),
                "h": (("time", "node"), np.array([[10.0, 10.0]])),  # both wet by h
                "dryFlagNode": (("time", "node"), np.array([[0, 1]], dtype=np.int8)),
            },
            coords={"time": [pd.Timestamp("2024-01-01")], "node": [0, 1]},
        )
        wet = _wet_mask(ds, dry_threshold=0.05)
        np.testing.assert_array_equal(wet.to_numpy(), [[True, False]])

    def test_h_threshold_fallback(self):
        """When ``dryFlagNode`` is absent, ``h > threshold`` decides."""
        ds = xr.Dataset(
            {
                "elevation": (("time", "node"), np.array([[1.0, 2.0]])),
                "h": (("time", "node"), np.array([[0.01, 1.0]])),
            },
            coords={"time": [pd.Timestamp("2024-01-01")], "node": [0, 1]},
        )
        wet = _wet_mask(ds, dry_threshold=0.05)
        np.testing.assert_array_equal(wet.to_numpy(), [[False, True]])

    def test_no_mask_source_returns_none(self):
        ds = xr.Dataset(
            {"elevation": (("time", "node"), np.array([[1.0, 2.0]]))},
            coords={"time": [pd.Timestamp("2024-01-01")], "node": [0, 1]},
        )
        assert _wet_mask(ds, dry_threshold=0.05) is None

    def test_apply_wet_mask_does_not_mutate(self):
        ds = xr.Dataset(
            {
                "elevation": (("time", "node"), np.array([[10.0, 20.0]])),
                "h": (("time", "node"), np.array([[0.01, 5.0]])),  # node 0 dry
            },
            coords={"time": [pd.Timestamp("2024-01-01")], "node": [0, 1]},
        )
        out = _apply_wet_mask(ds, "elevation", dry_threshold=0.05)
        # Original untouched.
        np.testing.assert_array_equal(ds["elevation"].to_numpy(), [[10.0, 20.0]])
        # Output has NaN at the dry node.
        masked = out["elevation"].to_numpy()
        assert np.isnan(masked[0, 0])
        assert masked[0, 1] == pytest.approx(20.0)


class TestExtendedCmap:
    def test_under_over_pinned(self):
        cmap = _extended_cmap("viridis")
        # Under colour matches the bottom of the colormap; over matches the top.
        assert tuple(cmap.get_under()) == tuple(cmap(0.0))
        assert tuple(cmap.get_over()) == tuple(cmap(1.0))


class TestAutoVariable:
    def test_prefers_zs(self):
        ds = xr.Dataset(
            data_vars={
                "zs": (("time", "y", "x"), np.zeros((1, 1, 1))),
                "h": (("time", "y", "x"), np.zeros((1, 1, 1))),
            },
            coords={"time": [pd.Timestamp("2024-01-01")]},
        )
        assert _auto_variable(ds) == "zs"

    def test_falls_back_to_elevation(self):
        ds = xr.Dataset(
            data_vars={"elevation": (("time", "node"), np.zeros((1, 1)))},
            coords={"time": [pd.Timestamp("2024-01-01")]},
        )
        assert _auto_variable(ds) == "elevation"

    def test_no_time_var_raises(self):
        ds = xr.Dataset(data_vars={"static": (("node",), np.zeros(3))})
        with pytest.raises(ValueError, match="auto-detect"):
            _auto_variable(ds)


# ---------------------------------------------------------------------------
# Integration tests — unstructured (SCHISM)
# ---------------------------------------------------------------------------


class TestPlotWaterLevelUnstructured:
    def test_returns_ax_and_collection(self):
        ds = _schism_dataset()
        ax, coll = plot_water_level(ds, time=0)
        # tripcolor(shading="gouraud") returns TriMesh (a Collection subclass).
        assert isinstance(coll, Collection)
        assert ax.get_title().startswith("elevation @ 2024-01-01")
        plt.close(ax.get_figure())

    def test_mixed_tri_quad_mesh(self):
        ds = _schism_mixed_tri_quad_dataset()
        ax, coll = plot_water_level(ds, time=0, variable="elevation")
        # gouraud shading carries one value per node.
        arr = coll.get_array()
        assert arr is not None
        assert arr.shape[0] == ds.sizes["node"]
        plt.close(ax.get_figure())

    def test_time_label_selector(self):
        ds = _schism_dataset()
        ax, _ = plot_water_level(ds, time="2024-01-01T01:00")
        assert "01:00" in ax.get_title()
        plt.close(ax.get_figure())

    def test_vmin_vmax_override(self):
        ds = _schism_dataset()
        ax, coll = plot_water_level(ds, time=0, vmin=0.0, vmax=100.0)
        assert coll.norm.vmin == pytest.approx(0.0)
        assert coll.norm.vmax == pytest.approx(100.0)
        plt.close(ax.get_figure())

    def test_percentile_limits_consistent_across_frames(self):
        """Default vmin/vmax should be identical for every frame.

        This is the invariant that keeps animation colours stable across frames.
        """
        ds = _schism_dataset()
        _, c0 = plot_water_level(ds, time=0)
        plt.close("all")
        _, c1 = plot_water_level(ds, time=2)
        assert c0.norm.vmin == pytest.approx(c1.norm.vmin)
        assert c0.norm.vmax == pytest.approx(c1.norm.vmax)
        plt.close("all")

    def test_colorbar_attached(self):
        ds = _schism_dataset()
        ax, _ = plot_water_level(ds, time=0)
        fig = ax.get_figure()
        # Figure should have at least 2 axes: the data axes + the colorbar.
        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_colorbar_disabled(self):
        ds = _schism_dataset()
        ax, _ = plot_water_level(ds, time=0, colorbar=False)
        fig = ax.get_figure()
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_reuses_existing_ax(self):
        ds = _schism_dataset()
        fig, ax = plt.subplots()
        ax2, _ = plot_water_level(ds, time=0, ax=ax)
        assert ax is ax2
        plt.close(fig)

    def test_colorbar_extends_when_data_outside_range(self):
        """User-set limits that exclude real data should trigger extended caps."""
        ds = _schism_dataset()
        ax, coll = plot_water_level(ds, time=0, vmin=2.0, vmax=3.0)
        fig = ax.get_figure()
        assert fig is not None
        # The colorbar axes carry the extend information via `extend` attribute.
        cbar_axes = [a for a in fig.axes if a is not ax]
        assert cbar_axes, "expected a colorbar axes on the figure"
        # colorbar.extend is stored on the ColorbarBase; traverse from the coll's
        # attached colorbar.
        assert coll.colorbar is not None
        assert coll.colorbar.extend == "both"
        plt.close(fig)

    def test_out_of_range_values_use_limit_colors(self):
        """Values outside vmin/vmax render with the under/over colour of the cmap."""
        import matplotlib as mpl

        ds = _schism_dataset()
        _, coll = plot_water_level(ds, time=0, vmin=10.0, vmax=20.0, cmap="viridis")
        cmap = coll.cmap
        # Under colour equals cmap(0); over equals cmap(1).
        np.testing.assert_allclose(cmap.get_under(), cmap(0.0))
        np.testing.assert_allclose(cmap.get_over(), cmap(1.0))
        assert isinstance(cmap, mpl.colors.Colormap)
        plt.close("all")

    def test_mask_dry_uses_dry_flag(self):
        """With a dryFlagNode in the dataset, dry cells become NaN in the plot."""
        ds = _schism_dataset()
        # Add an authoritative dry flag — mark node 0 as dry at t=0.
        dry = np.zeros((ds.sizes["time"], ds.sizes["node"]), dtype=np.int8)
        dry[0, 0] = 1
        ds = ds.assign(dryFlagNode=(("time", "node"), dry))

        _, coll = plot_water_level(ds, time=0, mask_dry=True)
        arr = np.asarray(coll.get_array())
        assert np.isnan(arr[0])
        assert not np.isnan(arr[1])
        plt.close("all")

    def test_mask_dry_off_keeps_all_values(self):
        """With mask_dry=False the renderer doesn't drop any cells."""
        ds = _schism_dataset()
        dry = np.ones((ds.sizes["time"], ds.sizes["node"]), dtype=np.int8)
        ds = ds.assign(dryFlagNode=(("time", "node"), dry))

        _, coll = plot_water_level(ds, time=0, mask_dry=False)
        arr = np.asarray(coll.get_array())
        assert not np.isnan(arr).any()
        plt.close("all")

    def test_basemap_uses_dataset_crs(self, monkeypatch):
        """`basemap=True` calls contextily with the dataset CRS."""
        ds = _schism_dataset()
        ds.attrs["crs"] = "EPSG:4326"

        captured: dict[str, object] = {}

        class _FakeContextily:
            class providers:  # noqa: N801 — mirrors contextily.providers namespace
                class Esri:
                    WorldImagery = "fake_imagery_provider"

            @staticmethod
            def add_basemap(ax, **kwargs):
                captured["ax"] = ax
                captured["kwargs"] = kwargs

        monkeypatch.setitem(__import__("sys").modules, "contextily", _FakeContextily)

        ax, _ = plot_water_level(ds, time=0, basemap=True)
        assert captured["ax"] is ax
        assert captured["kwargs"]["crs"] == "EPSG:4326"
        assert captured["kwargs"]["source"] == "fake_imagery_provider"
        assert "zoom" not in captured["kwargs"]
        plt.close("all")

    def test_basemap_explicit_crs_overrides_dataset(self, monkeypatch):
        ds = _schism_dataset()
        ds.attrs["crs"] = "EPSG:4326"

        captured: dict[str, object] = {}

        class _FakeContextily:
            class providers:  # noqa: N801
                class Esri:
                    WorldImagery = "fake"

            @staticmethod
            def add_basemap(ax, **kwargs):
                captured["kwargs"] = kwargs

        monkeypatch.setitem(__import__("sys").modules, "contextily", _FakeContextily)

        plot_water_level(ds, time=0, basemap=True, crs="EPSG:32619", basemap_zoom=11)
        assert captured["kwargs"]["crs"] == "EPSG:32619"
        assert captured["kwargs"]["zoom"] == 11
        plt.close("all")

    def test_basemap_without_crs_raises(self):
        """basemap=True with no CRS in attrs and no override → ValueError."""
        ds = _schism_dataset()
        ds.attrs.pop("crs", None)
        with pytest.raises(ValueError, match="basemap"):
            plot_water_level(ds, time=0, basemap=True)
        plt.close("all")


# ---------------------------------------------------------------------------
# Integration tests — regular (SFINCS)
# ---------------------------------------------------------------------------


class TestPlotWaterLevelQuadtree:
    def test_returns_collection(self):
        ds = _sfincs_quadtree_dataset()
        ax, coll = plot_water_level(ds, time=0)
        assert isinstance(coll, Collection)
        assert ax.get_title().startswith("zs @ 2024-01-01")
        plt.close(ax.get_figure())

    def test_tri_values_match_face_broadcast(self):
        """tripcolor(shading='flat') receives triangle-length facecolors."""
        ds = _sfincs_quadtree_dataset()
        _, coll = plot_water_level(ds, time=1)
        arr = coll.get_array()
        assert arr is not None
        # 4 quads, split into 2 triangles each = 8 triangle cells.
        assert arr.shape[0] == 8
        plt.close("all")

    def test_triangle_face_map_stashed(self):
        """_plot_quadtree stashes the mapping for animation writers."""
        ds = _sfincs_quadtree_dataset()
        _, coll = plot_water_level(ds, time=0)
        assert hasattr(coll, "triangle_face_map")
        tri_face = coll.triangle_face_map
        # Each original face appears exactly twice (each quad → 2 triangles).
        counts = np.bincount(tri_face, minlength=ds.sizes["face"])
        np.testing.assert_array_equal(counts, [2, 2, 2, 2])
        plt.close("all")


class TestPlotWaterLevelRegular:
    def test_returns_quadmesh(self):
        ds = _sfincs_regular_dataset()
        ax, coll = plot_water_level(ds, time=0)
        assert isinstance(coll, QuadMesh)
        plt.close(ax.get_figure())

    def test_pcolormesh_values_shape(self):
        ds = _sfincs_regular_dataset()
        _, coll = plot_water_level(ds, time=1)
        # QuadMesh.set_array expects flat n_y*n_x values.
        arr = coll.get_array()
        assert arr is not None
        assert arr.size == ds.sizes["y"] * ds.sizes["x"]
        plt.close("all")

    def test_variable_autodetect_prefers_zs(self):
        ds = _sfincs_regular_dataset()
        ax, _ = plot_water_level(ds, time=0)
        # Title should mention zs, not h or anything else.
        assert "zs" in ax.get_title()
        plt.close(ax.get_figure())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_missing_mesh_type(self):
        ds = _schism_dataset()
        ds.attrs.pop("mesh_type")
        with pytest.raises(KeyError, match="mesh_type"):
            plot_water_level(ds, time=0)

    def test_unknown_mesh_type(self):
        ds = _schism_dataset()
        ds.attrs["mesh_type"] = "something-bogus"
        with pytest.raises(ValueError, match="Unknown mesh_type"):
            plot_water_level(ds, time=0)

    def test_variable_not_in_dataset(self):
        ds = _schism_dataset()
        with pytest.raises(KeyError, match="zsmax"):
            plot_water_level(ds, time=0, variable="zsmax")
