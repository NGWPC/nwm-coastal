"""Tests for coastal_calibration.data.transformation."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely import box

from coastal_calibration.data.transformation import clip_to_aoi

if TYPE_CHECKING:
    from pathlib import Path

#: TIFF header version word: 42 is classic, 43 is BigTIFF.
_BIGTIFF_VERSION = 43


def _source_raster(path: Path, size: int = 200) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(-96.9, 28.2, -95.9, 28.9, size, size),
    ) as dst:
        dst.write(np.full((size, size), 1.5, dtype="float32"), 1)
    return path


def _cutline(path: Path) -> Path:
    gpd.GeoDataFrame(geometry=[box(-96.8, 28.3, -96.0, 28.8)], crs="EPSG:4326").to_file(
        path, driver="GeoJSON"
    )
    return path


def _tiff_version(path: Path) -> int:
    raw = path.read_bytes()[:4]
    endian = "<" if raw[:2] == b"II" else ">"
    return int(struct.unpack(endian + "H", raw[2:4])[0])


class TestClipToAoi:
    def test_writes_bigtiff(self, tmp_path: Path) -> None:
        """Regression: a clip over a large AOI hit the 4 GB classic TIFF limit.

        GDAL raised "Maximum TIFF file size exceeded" part-way through the
        write, because compression makes the final size unpredictable and it
        had already committed to a classic header.
        """
        out = tmp_path / "out.tif"

        clip_to_aoi(
            _source_raster(tmp_path / "src.tif"),
            _cutline(tmp_path / "cut.geojson"),
            out,
            nodata="nan",
            output_type="Float32",
        )

        assert _tiff_version(out) == _BIGTIFF_VERSION

    def test_output_is_clipped_and_readable(self, tmp_path: Path) -> None:
        """BigTIFF must not cost readability or change the clip itself."""
        out = tmp_path / "out.tif"

        clip_to_aoi(
            _source_raster(tmp_path / "src.tif"),
            _cutline(tmp_path / "cut.geojson"),
            out,
            nodata="nan",
            output_type="Float32",
        )

        with rasterio.open(out) as src:
            assert src.crs.to_epsg() == 4326
            assert src.shape[0] < 200
            assert src.shape[1] < 200
            data = src.read(1)
        assert np.nanmax(data) == np.float32(1.5)
