"""Tests for the NOAA DEM discovery and fetch utilities.

Covers index loading, spatial querying, dataset selection, URL
construction, and the fetch workflow with mocked rioxarray I/O.

Run with::

    pytest tests/test_noaa_dem.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coastal_calibration.utils.noaa_dem import (
    _overlap_fraction,
    get_vrt_url,
    load_index,
    query_overlapping,
    select_best,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

TEXAS_BBOX = (-97.0, 26.0, -94.0, 30.0)
MID_OCEAN_BBOX = (-50.0, 30.0, -45.0, 35.0)


@pytest.fixture
def sample_index() -> list[dict[str, Any]]:
    """Return a small synthetic DEM index for testing."""
    return [
        {
            "dataset_name": "TX_Coastal_DEM_2018_8899",
            "title": "Texas Coastal DEM 2018",
            "bbox": [-97.86, 25.84, -93.51, 30.16],
            "resolution_m": 1.0,
            "year": 2018,
            "is_topobathy": False,
            "epsg": 4269,
            "vrt_filename": "TX_Coastal_DEM_2018_EPSG-4269.vrt",
        },
        {
            "dataset_name": "NCEI_ninth_Topobathy_2014_8483",
            "title": "NCEI 1/9 arc-second Coastal Relief Model (CONUS)",
            "bbox": [-123.0, 25.0, -66.75, 49.0],
            "resolution_m": 3.0,
            "year": 2014,
            "is_topobathy": True,
            "epsg": 4269,
            "vrt_filename": "NCEI_ninth_Topobathy_2014_m8483_EPSG-4269_1.vrt",
        },
        {
            "dataset_name": "LA_DEM_2018_9037",
            "title": "Louisiana DEM 2018",
            "bbox": [-94.04, 28.85, -88.76, 30.93],
            "resolution_m": 1.0,
            "year": 2018,
            "is_topobathy": True,
            "epsg": 4269,
            "vrt_filename": "LA_DEM_2018_m9037_EPSG-4269.vrt",
        },
    ]


@pytest.fixture
def aoi_file(tmp_path: Path) -> Path:
    """Create a minimal GeoJSON AOI polygon in the Texas bounding box."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-96.0, 27.0],
                            [-95.0, 27.0],
                            [-95.0, 28.0],
                            [-96.0, 28.0],
                            [-96.0, 27.0],
                        ]
                    ],
                },
                "properties": {},
            }
        ],
    }
    path = tmp_path / "texas_aoi.geojson"
    path.write_text(json.dumps(geojson))
    return path


# ------------------------------------------------------------------
# Index loading
# ------------------------------------------------------------------


class TestLoadIndex:
    """Test that the packaged index loads correctly."""

    def test_load_returns_list(self) -> None:
        index = load_index()
        assert isinstance(index, list)
        assert len(index) > 0

    def test_entries_have_required_keys(self) -> None:
        index = load_index()
        required = {
            "dataset_name",
            "title",
            "bbox",
            "resolution_m",
            "year",
            "is_topobathy",
            "epsg",
            "vrt_filename",
        }
        for entry in index:
            assert required.issubset(entry.keys()), f"Missing keys in {entry['dataset_name']}"

    def test_bbox_has_four_elements(self) -> None:
        for entry in load_index():
            assert len(entry["bbox"]) == 4


# ------------------------------------------------------------------
# Spatial query
# ------------------------------------------------------------------


class TestQueryOverlapping:
    """Test bbox intersection filtering."""

    def test_texas_bbox_finds_texas_dem(self, sample_index: list[dict]) -> None:
        results = query_overlapping(sample_index, TEXAS_BBOX)
        names = {r["dataset_name"] for r in results}
        assert "TX_Coastal_DEM_2018_8899" in names

    def test_texas_bbox_finds_ncei(self, sample_index: list[dict]) -> None:
        results = query_overlapping(sample_index, TEXAS_BBOX)
        names = {r["dataset_name"] for r in results}
        assert "NCEI_ninth_Topobathy_2014_8483" in names

    def test_mid_ocean_returns_empty(self, sample_index: list[dict]) -> None:
        results = query_overlapping(sample_index, MID_OCEAN_BBOX)
        assert results == []

    def test_partial_overlap_with_louisiana(self, sample_index: list[dict]) -> None:
        """Bbox overlapping TX/LA border should include LA DEM."""
        bbox = (-94.5, 29.0, -93.0, 30.0)
        results = query_overlapping(sample_index, bbox)
        names = {r["dataset_name"] for r in results}
        assert "LA_DEM_2018_9037" in names
        assert "TX_Coastal_DEM_2018_8899" in names


# ------------------------------------------------------------------
# Overlap fraction
# ------------------------------------------------------------------


class TestOverlapFraction:
    """Test the _overlap_fraction helper."""

    def test_full_containment(self) -> None:
        rec_bbox = [-100.0, 20.0, -80.0, 40.0]
        query_bbox = (-95.0, 25.0, -85.0, 35.0)
        assert _overlap_fraction(rec_bbox, query_bbox) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        rec_bbox = [0.0, 0.0, 10.0, 10.0]
        query_bbox = (20.0, 20.0, 30.0, 30.0)
        assert _overlap_fraction(rec_bbox, query_bbox) == 0.0

    def test_partial_overlap(self) -> None:
        rec_bbox = [-96.0, 27.0, -94.0, 29.0]
        query_bbox = (-97.0, 26.0, -94.0, 30.0)
        # Intersection: [-96, 27] to [-94, 29] = 2*2=4; query = 3*4=12
        assert _overlap_fraction(rec_bbox, query_bbox) == pytest.approx(4.0 / 12.0)


# ------------------------------------------------------------------
# Selection
# ------------------------------------------------------------------


class TestSelectBest:
    """Test ranking and selection logic."""

    def test_prefers_topobathy(self, sample_index: list[dict]) -> None:
        # NCEI is topobathy; TX is not
        candidates = query_overlapping(sample_index, TEXAS_BBOX)
        best = select_best(candidates, TEXAS_BBOX)
        assert best["is_topobathy"] is True

    def test_prefers_finer_resolution_among_topobathy(self, sample_index: list[dict]) -> None:
        """LA DEM (1m, topobathy) should beat NCEI (3m, topobathy) for LA bbox."""
        bbox = (-93.0, 29.0, -89.0, 30.5)
        candidates = query_overlapping(sample_index, bbox)
        topobathy = [c for c in candidates if c["is_topobathy"]]
        assert len(topobathy) >= 2
        best = select_best(topobathy, bbox)
        assert best["resolution_m"] == 1.0

    def test_empty_candidates_raises(self) -> None:
        with pytest.raises(ValueError, match="No candidate"):
            select_best([], TEXAS_BBOX)


# ------------------------------------------------------------------
# VRT URL construction
# ------------------------------------------------------------------


class TestGetVrtUrl:
    """Test URL generation from index records."""

    def test_constructs_correct_url(self, sample_index: list[dict]) -> None:
        ncei = sample_index[1]
        url = get_vrt_url(ncei)
        assert url.startswith("https://")
        assert "NCEI_ninth_Topobathy_2014_8483" in url
        assert url.endswith("NCEI_ninth_Topobathy_2014_m8483_EPSG-4269_1.vrt")

    def test_texas_url(self, sample_index: list[dict]) -> None:
        tx = sample_index[0]
        url = get_vrt_url(tx)
        assert "TX_Coastal_DEM_2018_8899" in url
        assert url.endswith("TX_Coastal_DEM_2018_EPSG-4269.vrt")


# ------------------------------------------------------------------
# Fetch workflow (mocked I/O)
# ------------------------------------------------------------------


def _make_mock_da(dtype: str = "float32") -> MagicMock:
    """Create a mock DataArray with proper rioxarray attributes."""
    mock_da = MagicMock()
    mock_da.squeeze.return_value = mock_da
    mock_da.rio.clip_box.return_value = mock_da
    mock_da.rio.clip.return_value = mock_da
    mock_da.where.return_value = mock_da
    mock_da.rio.write_transform.return_value = mock_da
    mock_da.rio.write_nodata.return_value = mock_da
    mock_da.dtype = dtype
    mock_da.size = 100
    mock_da.values = np.ones((10, 10), dtype=dtype)
    # Use None so the nodata-masking branch is skipped in mocked tests.
    mock_da.rio.nodata = None
    mock_da.rio.crs.to_epsg.return_value = 4269

    def fake_to_raster(path, **kw):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"\x00" * 100)

    mock_da.rio.to_raster.side_effect = fake_to_raster
    return mock_da


class TestFetchNoaaDem:
    """Test fetch_noaa_dem with mocked rioxarray."""

    def test_fetch_with_auto_discovery(self, aoi_file: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "downloads"
        mock_da = _make_mock_da("float32")

        with patch("rioxarray.open_rasterio", return_value=mock_da):
            from coastal_calibration.utils.noaa_dem import fetch_noaa_dem

            tif, cat, name = fetch_noaa_dem(
                aoi=aoi_file,
                output_dir=output_dir,
            )

        assert tif.exists()
        assert cat.exists()
        assert name == "noaa_topobathy"
        # Catalog YAML should reference the tif
        assert "noaa_topobathy.tif" in cat.read_text()

    def test_fetch_with_explicit_dataset(self, aoi_file: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "downloads"
        mock_da = _make_mock_da("float64")
        mock_da.astype.return_value = mock_da

        with patch("rioxarray.open_rasterio", return_value=mock_da):
            from coastal_calibration.utils.noaa_dem import fetch_noaa_dem

            tif, _cat, name = fetch_noaa_dem(
                aoi=aoi_file,
                output_dir=output_dir,
                dataset_name="NCEI_ninth_Topobathy_2014_8483",
                catalog_name="my_dem",
            )

        assert name == "my_dem"
        assert tif.name == "my_dem.tif"
        # float64 should trigger astype
        mock_da.astype.assert_called_once_with("float32")

    def test_fetch_unknown_dataset_raises(self, aoi_file: Path, tmp_path: Path) -> None:
        from coastal_calibration.utils.noaa_dem import fetch_noaa_dem

        with pytest.raises(ValueError, match="not found in NOAA DEM index"):
            fetch_noaa_dem(
                aoi=aoi_file,
                output_dir=tmp_path,
                dataset_name="NONEXISTENT_DEM",
            )
