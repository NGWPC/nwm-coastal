"""Tests for the shared GDAL CLI helpers in ``_gdal.py``.

Focuses on ``compute_aoi_coverage`` — the other helpers (``build_vrt``,
``clip_to_aoi``) are exercised indirectly by the per-source fetch tests.

Run with::

    pytest tests/test_gdal_helpers.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import shapely
from rasterio.transform import from_bounds

from coastal_calibration.data.transformation import compute_aoi_coverage

if TYPE_CHECKING:
    from pathlib import Path


def _write_raster(
    path: Path,
    data: np.ndarray,
    *,
    bounds: tuple[float, float, float, float] = (-95.3, 29.1, -95.1, 29.4),
    crs: str = "EPSG:4326",
    nodata: float = float("nan"),
) -> Path:
    """Write a tiny single-band GeoTIFF for testing."""
    h, w = data.shape
    transform = from_bounds(*bounds, w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


def _write_zone(
    path: Path, bounds: tuple[float, float, float, float], crs: str = "EPSG:4326"
) -> Path:
    """Write a GeoJSON zone polygon via GeoPandas (produces a FeatureCollection)."""
    gdf = gpd.GeoDataFrame(geometry=[shapely.box(*bounds)], crs=crs)
    gdf.to_file(path, driver="GeoJSON")
    return path


class TestComputeAoiCoverage:
    """Tests for compute_aoi_coverage."""

    def test_all_valid(self, tmp_path: Path) -> None:
        """100 % valid pixels when the entire raster is filled."""
        data = np.ones((10, 10), dtype=np.float32)
        raster = _write_raster(tmp_path / "all_valid.tif", data)
        zone = _write_zone(tmp_path / "zone.geojson", (-95.3, 29.1, -95.1, 29.4))

        pct = compute_aoi_coverage(raster, zone)
        assert pct == pytest.approx(100.0)

    def test_all_nodata(self, tmp_path: Path) -> None:
        """0 % when every pixel is nodata."""
        data = np.full((10, 10), np.nan, dtype=np.float32)
        raster = _write_raster(tmp_path / "all_nodata.tif", data)
        zone = _write_zone(tmp_path / "zone.geojson", (-95.3, 29.1, -95.1, 29.4))

        pct = compute_aoi_coverage(raster, zone)
        assert pct == pytest.approx(0.0)

    def test_half_nodata(self, tmp_path: Path) -> None:
        """~50 % when half the pixels are nodata."""
        data = np.ones((10, 10), dtype=np.float32)
        data[:5, :] = np.nan
        raster = _write_raster(tmp_path / "half.tif", data)
        zone = _write_zone(tmp_path / "zone.geojson", (-95.3, 29.1, -95.1, 29.4))

        pct = compute_aoi_coverage(raster, zone)
        assert pct == pytest.approx(50.0)

    def test_featurecollection_geojson(self, tmp_path: Path) -> None:
        """Zone written by GeoPandas (FeatureCollection) is read correctly."""
        data = np.ones((10, 10), dtype=np.float32)
        raster = _write_raster(tmp_path / "raster.tif", data)

        # GeoPandas always writes FeatureCollections — this is the
        # format produced by the cutline writers in the fetch modules.
        zone = _write_zone(tmp_path / "zone.geojson", (-95.3, 29.1, -95.1, 29.4))
        text = zone.read_text()
        assert "FeatureCollection" in text, "GeoPandas should produce a FeatureCollection"

        pct = compute_aoi_coverage(raster, zone)
        assert pct == pytest.approx(100.0)

    def test_crs_mismatch_reprojection(self, tmp_path: Path) -> None:
        """Zone in EPSG:4326 + raster in EPSG:4269 should still work."""
        data = np.ones((10, 10), dtype=np.float32)
        raster = _write_raster(
            tmp_path / "nad83.tif",
            data,
            crs="EPSG:4269",
        )
        # Zone intentionally in EPSG:4326 (WGS 84), different from raster.
        zone = _write_zone(
            tmp_path / "zone_4326.geojson",
            (-95.3, 29.1, -95.1, 29.4),
            crs="EPSG:4326",
        )

        pct = compute_aoi_coverage(raster, zone)
        assert pct == pytest.approx(100.0)

    def test_zone_outside_raster(self, tmp_path: Path) -> None:
        """Zone that does not overlap the raster returns 0 %."""
        data = np.ones((10, 10), dtype=np.float32)
        raster = _write_raster(tmp_path / "raster.tif", data)
        # Zone far away from the raster extent.
        zone = _write_zone(tmp_path / "zone_far.geojson", (10.0, 50.0, 11.0, 51.0))

        pct = compute_aoi_coverage(raster, zone)
        assert pct == pytest.approx(0.0)
