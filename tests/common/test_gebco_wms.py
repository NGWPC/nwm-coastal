"""Tests for the GEBCO fetcher (CEDA tile approach).

Covers tile index computation, URL construction, catalog writing,
and the full fetch pipeline with mocked GDAL CLI.

Run with::

    pytest tests/test_gebco_wms.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from coastal_calibration.data.gebco_wms import (
    _tile_indices,
    _tile_url,
    _write_catalog,
)

# ===================================================================
# Tile index tests
# ===================================================================


class TestTileIndices:
    """Test _tile_indices for various bounding boxes."""

    def test_single_tile_northern_western(self) -> None:
        """Texas coast bbox falls in one tile (W180-W90, N0-N90)."""
        tiles = _tile_indices((-95.5, 29.0, -95.0, 29.5))
        assert tiles == [(0.0, 90.0, -180.0, -90.0)]

    def test_bbox_crossing_equator(self) -> None:
        """A bbox spanning the equator and the prime meridian needs 4 tiles."""
        tiles = _tile_indices((-10.0, -5.0, 10.0, 5.0))
        assert len(tiles) == 4
        # Should include both latitude bands and both longitude bands
        lat_bands = {(t[0], t[1]) for t in tiles}
        assert (-90.0, 0.0) in lat_bands
        assert (0.0, 90.0) in lat_bands
        lon_bands = {(t[2], t[3]) for t in tiles}
        assert (-90.0, 0.0) in lon_bands
        assert (0.0, 90.0) in lon_bands

    def test_bbox_crossing_meridian(self) -> None:
        """A bbox spanning a 90-degree longitude edge."""
        tiles = _tile_indices((-91.0, 29.0, -89.0, 30.0))
        assert len(tiles) == 2
        lon_bands = {(t[2], t[3]) for t in tiles}
        assert (-180.0, -90.0) in lon_bands
        assert (-90.0, 0.0) in lon_bands

    def test_southern_hemisphere(self) -> None:
        tiles = _tile_indices((150.0, -34.5, 151.0, -33.5))
        assert tiles == [(-90.0, 0.0, 90.0, 180.0)]


# ===================================================================
# URL construction tests
# ===================================================================


class TestTileUrl:
    """Test _tile_url for /vsicurl/ URL construction."""

    def test_northern_western(self) -> None:
        url = _tile_url(0.0, 90.0, -90.0, 0.0)
        assert "/vsicurl/" in url
        assert "gebco_2025_n90.0_s0.0_w-90.0_e0.0.tif" in url
        assert "dap.ceda.ac.uk" in url

    def test_southern_eastern(self) -> None:
        url = _tile_url(-90.0, 0.0, 90.0, 180.0)
        assert "gebco_2025_n0.0_s-90.0_w90.0_e180.0.tif" in url


# ===================================================================
# Catalog writer tests
# ===================================================================


class TestWriteCatalog:
    """Test _write_catalog output."""

    def test_writes_valid_yaml(self, tmp_path: Path) -> None:
        cat_path = tmp_path / "gebco_15arcs_catalog.yml"
        _write_catalog(cat_path, "gebco_15arcs.tif", "gebco_15arcs", 4326)

        assert cat_path.exists()
        data = yaml.safe_load(cat_path.read_text())
        assert data["meta"]["name"] == "gebco_15arcs"
        assert data["gebco_15arcs"]["uri"] == "gebco_15arcs.tif"
        assert data["gebco_15arcs"]["data_type"] == "RasterDataset"
        assert data["gebco_15arcs"]["metadata"]["crs"] == 4326
        assert data["gebco_15arcs"]["data_adapter"]["rename"] == {"elevation": "elevtn"}


# ===================================================================
# fetch_gebco integration test (mocked GDAL)
# ===================================================================


class TestFetchGebco:
    """Test fetch_gebco with mocked GDAL CLI."""

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
                                [-95.5, 29.0],
                                [-95.0, 29.0],
                                [-95.0, 29.5],
                                [-95.5, 29.5],
                                [-95.5, 29.0],
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

        def mock_clip(input_path, cutline_path, output_path, **kwargs):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"\x00" * 200)

        with (
            patch(
                "coastal_calibration.data.transformation.build_vrt",
            ) as mock_vrt,
            patch(
                "coastal_calibration.data.transformation.clip_to_aoi",
                side_effect=mock_clip,
            ) as mock_clip_fn,
        ):
            from coastal_calibration.data.gebco_wms import fetch_gebco

            log_messages: list[str] = []
            geotiff, catalog, name = fetch_gebco(
                aoi=aoi_file,
                output_dir=output_dir,
                log=log_messages.append,
            )

        assert name == "gebco_15arcs"
        assert catalog.name == "gebco_15arcs_catalog.yml"
        assert catalog.exists()
        assert geotiff.exists()
        mock_vrt.assert_called_once()
        mock_clip_fn.assert_called_once()
        assert any("gebco" in m.lower() for m in log_messages)

        # Verify build_vrt was called with /vsicurl/ paths
        vrt_call_args = mock_vrt.call_args
        tiff_files = vrt_call_args[0][1]
        assert all("/vsicurl/" in str(f) for f in tiff_files)
