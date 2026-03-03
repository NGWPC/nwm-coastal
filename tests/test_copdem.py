"""Tests for the Copernicus DEM 30m fetcher.

Covers tile index computation, URL construction, catalog writing,
and the full fetch pipeline with mocked download / GDAL.

Run with::

    pytest tests/test_copdem.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from coastal_calibration.utils.copdem import (
    _tile_indices,
    _tile_url,
    _write_catalog,
)

# ===================================================================
# Tile index tests
# ===================================================================


class TestTileIndices:
    """Test _tile_indices for various bounding boxes."""

    def test_single_tile(self) -> None:
        """A small bbox inside a single 1x1 tile."""
        tiles = _tile_indices((-95.3, 29.1, -95.1, 29.4))
        assert tiles == [(29, -96)]

    def test_multiple_tiles(self) -> None:
        """A bbox spanning multiple tiles."""
        tiles = _tile_indices((-95.5, 28.5, -94.5, 29.5))
        expected = [(28, -96), (28, -95), (29, -96), (29, -95)]
        assert tiles == expected

    def test_exact_degree_boundary(self) -> None:
        """Bbox on exact degree boundaries."""
        tiles = _tile_indices((-96.0, 29.0, -95.0, 30.0))
        assert tiles == [(29, -96)]

    def test_southern_hemisphere(self) -> None:
        """Tile indices for southern hemisphere."""
        tiles = _tile_indices((150.1, -34.9, 150.9, -34.1))
        assert tiles == [(-35, 150)]


# ===================================================================
# URL construction tests
# ===================================================================


class TestTileUrl:
    """Test _tile_url for various lat/lon combinations."""

    def test_northern_eastern(self) -> None:
        url = _tile_url(29, 10)
        assert "N29_00_E010_00" in url
        assert url.endswith(".tif")
        assert "copernicus-dem-30m.s3.amazonaws.com" in url

    def test_northern_western(self) -> None:
        url = _tile_url(29, -96)
        assert "N29_00_W096_00" in url

    def test_southern_eastern(self) -> None:
        url = _tile_url(-35, 150)
        assert "S35_00_E150_00" in url

    def test_southern_western(self) -> None:
        url = _tile_url(-10, -70)
        assert "S10_00_W070_00" in url

    def test_zero_lat_lon(self) -> None:
        url = _tile_url(0, 0)
        assert "N00_00_E000_00" in url


# ===================================================================
# Catalog writer tests
# ===================================================================


class TestWriteCatalog:
    """Test _write_catalog output."""

    def test_writes_valid_yaml(self, tmp_path: Path) -> None:
        cat_path = tmp_path / "copdem30_catalog.yml"
        _write_catalog(cat_path, "copdem30.tif", "copdem30", 4326)

        assert cat_path.exists()
        data = yaml.safe_load(cat_path.read_text())
        assert data["meta"]["name"] == "copdem30"
        assert data["copdem30"]["uri"] == "copdem30.tif"
        assert data["copdem30"]["data_type"] == "RasterDataset"
        assert data["copdem30"]["metadata"]["crs"] == 4326
        assert data["copdem30"]["data_adapter"]["rename"] == {"elevation": "elevtn"}


# ===================================================================
# fetch_copdem30 integration test (mocked I/O)
# ===================================================================


class TestFetchCopdem30:
    """Test fetch_copdem30 with mocked download and GDAL CLI."""

    @pytest.fixture
    def aoi_file(self, tmp_path: Path) -> Path:
        import json

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-95.3, 29.1],
                                [-95.1, 29.1],
                                [-95.1, 29.4],
                                [-95.3, 29.4],
                                [-95.3, 29.1],
                            ]
                        ],
                    },
                    "properties": {},
                }
            ],
        }
        path = tmp_path / "aoi.geojson"
        path.write_text(json.dumps(geojson))
        return path

    def test_fetch_writes_geotiff_and_catalog(self, tmp_path: Path, aoi_file: Path) -> None:
        output_dir = tmp_path / "output"

        def mock_download(urls, file_paths, **kwargs):
            for fp in file_paths if isinstance(file_paths, list) else [file_paths]:
                Path(fp).parent.mkdir(parents=True, exist_ok=True)
                Path(fp).write_bytes(b"\x00" * 100)

        def mock_clip(input_path, cutline_path, output_path, **kwargs):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"\x00" * 200)

        with (
            patch("tiny_retriever.download", side_effect=mock_download) as mock_dl,
            patch(
                "coastal_calibration.utils._gdal.build_vrt",
            ) as mock_vrt,
            patch(
                "coastal_calibration.utils._gdal.clip_to_aoi",
                side_effect=mock_clip,
            ) as mock_clip_fn,
        ):
            from coastal_calibration.utils.copdem import fetch_copdem30

            log_messages: list[str] = []
            geotiff, catalog, name = fetch_copdem30(
                aoi=aoi_file,
                output_dir=output_dir,
                log=log_messages.append,
            )

        assert name == "copdem30"
        assert catalog.name == "copdem30_catalog.yml"
        assert catalog.exists()
        assert geotiff.exists()
        mock_dl.assert_called_once()
        mock_vrt.assert_called_once()
        mock_clip_fn.assert_called_once()
        assert any("tile" in m.lower() for m in log_messages)
