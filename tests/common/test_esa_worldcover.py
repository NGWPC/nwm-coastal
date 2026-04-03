"""Tests for the ESA WorldCover fetcher.

Covers tile grid snapping, tile index computation, URL construction,
catalog writing, and the full fetch pipeline with mocked download / GDAL.

Run with::

    pytest tests/test_esa_worldcover.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from coastal_calibration.data.esa_worldcover import (
    _snap_to_grid,
    _tile_indices,
    _tile_url,
    _write_catalog,
)

# ===================================================================
# Grid snapping tests
# ===================================================================


class TestSnapToGrid:
    """Test _snap_to_grid with various values and modes."""

    def test_floor_positive(self) -> None:
        assert _snap_to_grid(29.1, 3, "floor") == 27

    def test_floor_negative(self) -> None:
        assert _snap_to_grid(-95.3, 3, "floor") == -96

    def test_ceil_positive(self) -> None:
        assert _snap_to_grid(29.1, 3, "ceil") == 30

    def test_ceil_negative(self) -> None:
        assert _snap_to_grid(-95.3, 3, "ceil") == -93

    def test_exact_boundary_floor(self) -> None:
        assert _snap_to_grid(30.0, 3, "floor") == 30

    def test_exact_boundary_ceil(self) -> None:
        assert _snap_to_grid(30.0, 3, "ceil") == 30


# ===================================================================
# Tile index tests
# ===================================================================


class TestTileIndices:
    """Test _tile_indices for various bounding boxes."""

    def test_single_tile(self) -> None:
        """A small bbox inside a single 3x3 tile."""
        tiles = _tile_indices((-95.3, 29.1, -95.1, 29.4))
        assert tiles == [(27, -96)]

    def test_multiple_tiles(self) -> None:
        """A bbox spanning multiple 3x3 tiles."""
        tiles = _tile_indices((-97.0, 28.0, -93.0, 31.0))
        expected = [(27, -99), (27, -96), (30, -99), (30, -96)]
        assert tiles == expected

    def test_southern_hemisphere(self) -> None:
        tiles = _tile_indices((150.0, -34.5, 151.0, -33.5))
        assert tiles == [(-36, 150)]


# ===================================================================
# URL construction tests
# ===================================================================


class TestTileUrl:
    """Test _tile_url for various lat/lon combinations."""

    def test_northern_western(self) -> None:
        url = _tile_url(27, -96)
        assert "N27W096" in url
        assert url.endswith("_Map.tif")
        assert "esa-worldcover" in url

    def test_southern_eastern(self) -> None:
        url = _tile_url(-36, 150)
        assert "S36E150" in url

    def test_zero(self) -> None:
        url = _tile_url(0, 0)
        assert "N00E000" in url


# ===================================================================
# Catalog writer tests
# ===================================================================


class TestWriteCatalog:
    """Test _write_catalog output."""

    def test_writes_valid_yaml(self, tmp_path: Path) -> None:
        cat_path = tmp_path / "esa_worldcover_catalog.yml"
        _write_catalog(cat_path, "esa_worldcover.tif", "esa_worldcover", 4326)

        assert cat_path.exists()
        data = yaml.safe_load(cat_path.read_text())
        assert data["meta"]["name"] == "esa_worldcover"
        assert data["esa_worldcover"]["uri"] == "esa_worldcover.tif"
        assert data["esa_worldcover"]["data_type"] == "RasterDataset"
        assert data["esa_worldcover"]["metadata"]["category"] == "landuse"
        assert data["esa_worldcover"]["metadata"]["crs"] == 4326
        # LULC catalog should NOT have a rename (band used as-is)
        assert "data_adapter" not in data["esa_worldcover"]


# ===================================================================
# fetch_esa_worldcover integration test (mocked I/O)
# ===================================================================


class TestFetchEsaWorldcover:
    """Test fetch_esa_worldcover with mocked download and GDAL CLI."""

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
                "coastal_calibration.data.transformation.build_vrt",
            ) as mock_vrt,
            patch(
                "coastal_calibration.data.transformation.clip_to_aoi",
                side_effect=mock_clip,
            ) as mock_clip_fn,
        ):
            from coastal_calibration.data.esa_worldcover import fetch_esa_worldcover

            log_messages: list[str] = []
            geotiff, catalog, name = fetch_esa_worldcover(
                aoi=aoi_file,
                output_dir=output_dir,
                log=log_messages.append,
            )

        assert name == "esa_worldcover"
        assert catalog.name == "esa_worldcover_catalog.yml"
        assert catalog.exists()
        assert geotiff.exists()
        mock_dl.assert_called_once()
        mock_vrt.assert_called_once()
        mock_clip_fn.assert_called_once()
        assert any("worldcover" in m.lower() for m in log_messages)
