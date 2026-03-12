"""Tests for the flood depth map generation module."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
import xarray as xr
import xugrid as xu
from rasterio.transform import from_bounds

from coastal_calibration.utils.floodmap import (
    _ensure_overviews,
    _write_floodmap_cog,
    create_flood_depth_map,
)


# ── helpers ──────────────────────────────────────────────────────────


def _make_quadtree_zsmax(n_faces: int = 100, water_level: float = 1.5) -> xu.UgridDataArray:
    """Build a synthetic UgridDataArray on a simple 1-level quadtree grid.

    Returns a ``(timemax=1, nFaces)`` UgridDataArray where every face
    has ``zsmax = water_level``.
    """
    # Build a simple 10x10 regular quad mesh that xugrid treats as UGRID.
    ncols, nrows = 10, 10
    dx, dy = 100.0, 100.0
    x0, y0 = 0.0, 0.0

    # Node coordinates for a structured quad mesh
    node_x = np.tile(np.arange(ncols + 1) * dx + x0, nrows + 1)
    node_y = np.repeat(np.arange(nrows + 1) * dy + y0, ncols + 1)

    # Face-node connectivity (quads)
    face_nodes = np.full((nrows * ncols, 4), -1, dtype=int)
    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            ll = row * (ncols + 1) + col
            face_nodes[idx] = [ll, ll + 1, ll + ncols + 2, ll + ncols + 1]

    grid = xu.Ugrid2d(
        node_x=node_x,
        node_y=node_y,
        fill_value=-1,
        face_node_connectivity=face_nodes,
    )

    n_faces = nrows * ncols
    vals = np.full((1, n_faces), water_level, dtype="float32")
    da = xr.DataArray(vals, dims=("timemax", grid.face_dimension))
    return xu.UgridDataArray(da, grid)


def _make_dem_tif(path, *, bounds=(0, 0, 1000, 1000), shape=(100, 100), fill=0.5, crs="EPSG:32619"):
    """Write a flat DEM GeoTIFF at ``path`` with the given elevation."""
    transform = from_bounds(*bounds, shape[1], shape[0])
    data = np.full(shape, fill, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=shape[1],
        height=shape[0],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def _make_index_tif(path, *, shape=(100, 100), n_faces=100, crs="EPSG:32619", bounds=(0, 0, 1000, 1000)):
    """Write a synthetic index COG where each pixel maps to a valid face."""
    transform = from_bounds(*bounds, shape[1], shape[0])
    # Map each pixel to a face index (cycling through available faces).
    indices = np.arange(shape[0] * shape[1], dtype="uint32").reshape(shape) % n_faces
    nodata = np.uint32(2147483647)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=shape[1],
        height=shape[0],
        count=1,
        dtype="uint32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(indices, 1)
    return path


# ── _ensure_overviews ────────────────────────────────────────────────


class TestEnsureOverviews:
    def test_builds_overviews_on_small_tif(self, tmp_path):
        """A tiny GeoTIFF gets at least one overview level."""
        tif = tmp_path / "small.tif"
        _make_dem_tif(tif, shape=(50, 50))

        messages: list[str] = []
        _ensure_overviews(tif, messages.append)

        with rasterio.open(tif) as src:
            assert src.overviews(1) == [2]
        assert any("overview" in m.lower() for m in messages)

    def test_skips_when_overviews_exist(self, tmp_path):
        """No double-build if overviews are already present."""
        tif = tmp_path / "has_ovr.tif"
        _make_dem_tif(tif, shape=(50, 50))
        _ensure_overviews(tif, lambda m: None)

        messages: list[str] = []
        _ensure_overviews(tif, messages.append)
        assert len(messages) == 0  # nothing logged → skipped


# ── _write_floodmap_cog ─────────────────────────────────────────────


class TestWriteFloodmapCog:
    def test_with_index(self, tmp_path):
        """Block-by-block downscaling with a pre-built index COG."""
        n_faces = 100
        water_level = 2.0
        dem_elev = 0.5
        expected_depth = water_level - dem_elev  # 1.5

        zsmax = _make_quadtree_zsmax(n_faces=n_faces, water_level=water_level)
        dem = tmp_path / "dem.tif"
        idx = tmp_path / "index.tif"
        out = tmp_path / "flood.tif"

        _make_dem_tif(dem, fill=dem_elev, shape=(100, 100))
        _make_index_tif(idx, shape=(100, 100), n_faces=n_faces)

        _write_floodmap_cog(
            zsmax=zsmax,
            dem_path=dem,
            index_path=idx,
            output_path=out,
            hmin=0.05,
            reproj_method="nearest",
            nrmax=500,
            log_fn=lambda m: None,
        )

        with rasterio.open(out) as src:
            data = src.read(1)
            assert data.shape == (100, 100)
            finite = data[np.isfinite(data)]
            assert len(finite) > 0
            np.testing.assert_allclose(finite, expected_depth, atol=0.01)

    def test_without_index(self, tmp_path):
        """Fallback path (no index) produces finite flood depths."""
        water_level = 2.0
        dem_elev = 0.5

        zsmax = _make_quadtree_zsmax(water_level=water_level)
        dem = tmp_path / "dem.tif"
        out = tmp_path / "flood.tif"

        _make_dem_tif(dem, fill=dem_elev, shape=(100, 100))

        _write_floodmap_cog(
            zsmax=zsmax,
            dem_path=dem,
            index_path=None,
            output_path=out,
            hmin=0.05,
            reproj_method="nearest",
            nrmax=500,
            log_fn=lambda m: None,
        )

        with rasterio.open(out) as src:
            data = src.read(1)
            finite = data[np.isfinite(data)]
            # With rasterize_like fallback, some pixels may not overlap.
            # Just verify we get some finite values and they are positive.
            assert len(finite) > 0
            assert finite.min() > 0

    def test_full_resolution_output(self, tmp_path):
        """Output has the same resolution as the input DEM (no overview shrink)."""
        zsmax = _make_quadtree_zsmax(water_level=2.0)
        dem = tmp_path / "dem.tif"
        out = tmp_path / "flood.tif"

        dem_shape = (200, 150)
        _make_dem_tif(dem, fill=0.5, shape=dem_shape)
        _make_index_tif(tmp_path / "idx.tif", shape=dem_shape, n_faces=100)

        _write_floodmap_cog(
            zsmax=zsmax,
            dem_path=dem,
            index_path=tmp_path / "idx.tif",
            output_path=out,
            hmin=0.05,
            reproj_method="nearest",
            nrmax=500,
            log_fn=lambda m: None,
        )

        with rasterio.open(out) as src:
            assert src.shape == dem_shape

    def test_hmin_filtering(self, tmp_path):
        """Pixels with depth <= hmin are set to NaN."""
        # water_level = 0.54, dem = 0.5 → depth = 0.04 < hmin=0.05
        zsmax = _make_quadtree_zsmax(water_level=0.54)
        dem = tmp_path / "dem.tif"
        out = tmp_path / "flood.tif"

        _make_dem_tif(dem, fill=0.5, shape=(50, 50))
        _make_index_tif(tmp_path / "idx.tif", shape=(50, 50), n_faces=100)

        _write_floodmap_cog(
            zsmax=zsmax,
            dem_path=dem,
            index_path=tmp_path / "idx.tif",
            output_path=out,
            hmin=0.05,
            reproj_method="nearest",
            nrmax=500,
            log_fn=lambda m: None,
        )

        with rasterio.open(out) as src:
            data = src.read(1)
            # All depths should be NaN because 0.04 < 0.05
            assert np.all(np.isnan(data))


# ── create_flood_depth_map (integration) ─────────────────────────────


_NARRAGANSETT = (
    "docs/examples/narragansett-ri/run/sfincs_model",
    "docs/examples/narragansett-ri/output/subgrid/dep_subgrid_lev3.tif",
)


@pytest.mark.skipif(
    not all(map(lambda p: __import__("pathlib").Path(p).exists(), _NARRAGANSETT)),
    reason="Narragansett example model not available",
)
class TestCreateFloodDepthMapIntegration:
    """Integration tests using the Narragansett-RI example model.

    All assertions share a single ``create_flood_depth_map`` call to
    avoid repeated model loading, which triggers netCDF4 segfaults on
    some Python versions.
    """

    def test_quadtree_floodmap(self, tmp_path):
        """End-to-end: index covers all levels, output is full-res with overviews."""
        from pathlib import Path

        model_root = Path(_NARRAGANSETT[0])
        dem_path = Path(_NARRAGANSETT[1])

        idx_path = tmp_path / "index.tif"
        out_path = tmp_path / "flood.tif"

        result = create_flood_depth_map(
            model_root=model_root,
            dem_path=dem_path,
            output_path=out_path,
            index_path=idx_path,
            log=lambda m: None,
        )

        # ── Index covers the full face range (not just coarse levels) ──
        with rasterio.open(idx_path) as src:
            idx = src.read(1)
            nodata = int(src.nodata)
            valid = idx[idx != nodata]
            # Level 4 starts at face ~7090, total faces ~293850
            assert valid.max() > 10000, (
                f"Index max {valid.max()} suggests only coarse levels are mapped"
            )

        # ── Output matches DEM resolution (no overview_level=0 shrink) ──
        with rasterio.open(dem_path) as dem_src:
            dem_shape = dem_src.shape

        with rasterio.open(result) as src:
            assert src.shape == dem_shape
            assert len(src.overviews(1)) >= 1

            data = src.read(1)
            finite = data[np.isfinite(data)]
            assert len(finite) > 0
            # The DEM (dep_subgrid_lev3) includes underwater bathymetry,
            # so depths up to ~55 m are expected.  The old broken index
            # (coarse cells only) produced 131 m — guard against that.
            assert finite.max() < 80.0, (
                f"Max depth {finite.max():.1f} m is unreasonably high; "
                "index likely maps to wrong (coarse) cells"
            )
