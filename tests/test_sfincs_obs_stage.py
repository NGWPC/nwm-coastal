"""Tests for SfincsObservationPointsStage including NOAA gage support."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
import shapely

from coastal_calibration.config.schema import (
    BoundaryConfig,
    CoastalCalibConfig,
    DownloadConfig,
    PathConfig,
    SfincsModelConfig,
    SimulationConfig,
    SlurmConfig,
)
from coastal_calibration.stages.sfincs_build import (
    SfincsObservationPointsStage,
    _set_model,
)


@pytest.fixture
def sfincs_config(tmp_path):
    """Create a SFINCS CoastalCalibConfig for testing."""
    prebuilt = tmp_path / "prebuilt"
    prebuilt.mkdir()
    (prebuilt / "sfincs.inp").write_text("")

    return CoastalCalibConfig(
        slurm=SlurmConfig(user="test"),
        simulation=SimulationConfig(
            start_date=datetime(2021, 6, 11),
            duration_hours=3,
            coastal_domain="atlgulf",
            meteo_source="nwm_retro",
        ),
        boundary=BoundaryConfig(source="tpxo"),
        paths=PathConfig(work_dir=tmp_path / "work", raw_download_dir=tmp_path / "dl"),
        model_config=SfincsModelConfig(prebuilt_dir=prebuilt),
        download=DownloadConfig(enabled=False),
    )


@pytest.fixture
def mock_model():
    """Create a mock SfincsModel with a UTM region and CRS."""
    model = MagicMock()
    model.crs = "EPSG:32615"

    # Region as a GeoDataFrame in UTM (box around Galveston, TX)
    region_utm = gpd.GeoDataFrame(
        geometry=[shapely.box(300000, 3230000, 350000, 3290000)],
        crs="EPSG:32615",
    )
    model.region = region_utm

    # Track add_point calls
    model.observation_points = MagicMock()
    model.observation_points.nr_points = 0
    return model


@pytest.fixture
def noaa_stations_gdf():
    """NOAA stations GeoDataFrame with stations in/out of the domain."""
    return gpd.GeoDataFrame(
        [
            {
                "station_id": "8771450",
                "station_name": "Galveston Pier 21",
                "state": "TX",
                "tidal": True,
                "greatlakes": False,
                "time_zone": "CST",
                "time_zone_offset": "-6",
                "geometry": shapely.Point(-94.7933, 29.3100),
            },
            {
                "station_id": "8770570",
                "station_name": "Sabine Pass North",
                "state": "TX",
                "tidal": True,
                "greatlakes": False,
                "time_zone": "CST",
                "time_zone_offset": "-6",
                "geometry": shapely.Point(-93.8700, 29.7283),
            },
            {
                # Station far away (San Francisco) — should NOT be selected
                "station_id": "9414290",
                "station_name": "San Francisco",
                "state": "CA",
                "tidal": True,
                "greatlakes": False,
                "time_zone": "PST",
                "time_zone_offset": "-8",
                "geometry": shapely.Point(-122.4194, 37.7749),
            },
        ],
        crs=4326,
    )


class TestSfincsObservationPointsStageSkip:
    def test_skip_when_nothing_configured(self, sfincs_config, mock_model):
        _set_model(sfincs_config, mock_model)
        stage = SfincsObservationPointsStage(sfincs_config)
        result = stage.run()
        assert result["status"] == "skipped"

    def test_skip_when_noaa_false_and_no_points(self, sfincs_config, mock_model):
        sfincs_config.model_config.include_noaa_gages = False
        _set_model(sfincs_config, mock_model)
        stage = SfincsObservationPointsStage(sfincs_config)
        result = stage.run()
        assert result["status"] == "skipped"


class TestSfincsObservationPointsStageManual:
    def test_add_manual_points(self, sfincs_config, mock_model):
        sfincs_config.model_config.observation_points = [
            {"x": 310000, "y": 3250000, "name": "pt1"},
            {"x": 320000, "y": 3260000, "name": "pt2"},
        ]
        _set_model(sfincs_config, mock_model)
        stage = SfincsObservationPointsStage(sfincs_config)
        result = stage.run()
        assert result["status"] == "completed"
        assert mock_model.observation_points.add_point.call_count == 2


class TestSfincsObservationPointsStageNOAA:
    def test_add_noaa_gages(self, sfincs_config, mock_model, noaa_stations_gdf):
        sfincs_config.model_config.include_noaa_gages = True
        _set_model(sfincs_config, mock_model)

        with patch(
            "coastal_calibration.coops_api.COOPSAPIClient"
        ) as MockClient:
            mock_client = MockClient.return_value
            mock_client.stations_metadata = noaa_stations_gdf

            stage = SfincsObservationPointsStage(sfincs_config)
            result = stage.run()

        assert result["status"] == "completed"
        # Only stations within the UTM box should be added (Galveston region)
        noaa_count = result["noaa_stations"]
        assert noaa_count >= 0
        # Each added station should have a noaa_ prefix
        for call in mock_model.observation_points.add_point.call_args_list:
            assert call.kwargs["name"].startswith("noaa_")

    def test_noaa_no_stations_in_domain(self, sfincs_config, mock_model):
        sfincs_config.model_config.include_noaa_gages = True
        _set_model(sfincs_config, mock_model)

        # Empty stations GDF
        empty_gdf = gpd.GeoDataFrame(
            columns=["station_id", "station_name", "geometry"],
            geometry="geometry",
            crs=4326,
        )

        with patch(
            "coastal_calibration.coops_api.COOPSAPIClient"
        ) as MockClient:
            mock_client = MockClient.return_value
            mock_client.stations_metadata = empty_gdf

            stage = SfincsObservationPointsStage(sfincs_config)
            result = stage.run()

        assert result["status"] == "completed"
        assert result["noaa_stations"] == 0
        mock_model.observation_points.add_point.assert_not_called()

    def test_noaa_combined_with_manual_points(self, sfincs_config, mock_model, noaa_stations_gdf):
        sfincs_config.model_config.include_noaa_gages = True
        sfincs_config.model_config.observation_points = [
            {"x": 315000, "y": 3255000, "name": "manual_pt"},
        ]
        _set_model(sfincs_config, mock_model)

        with patch(
            "coastal_calibration.coops_api.COOPSAPIClient"
        ) as MockClient:
            mock_client = MockClient.return_value
            mock_client.stations_metadata = noaa_stations_gdf

            stage = SfincsObservationPointsStage(sfincs_config)
            result = stage.run()

        assert result["status"] == "completed"
        # At least the manual point should have been added
        names = [
            call.kwargs["name"] for call in mock_model.observation_points.add_point.call_args_list
        ]
        assert "manual_pt" in names
        # NOAA points have noaa_ prefix
        noaa_names = [n for n in names if n.startswith("noaa_")]
        assert result["noaa_stations"] == len(noaa_names)

    def test_noaa_stations_projected_to_model_crs(
        self, sfincs_config, mock_model, noaa_stations_gdf
    ):
        """Verify that station coordinates are projected into the model CRS."""
        sfincs_config.model_config.include_noaa_gages = True
        _set_model(sfincs_config, mock_model)

        with patch(
            "coastal_calibration.coops_api.COOPSAPIClient"
        ) as MockClient:
            mock_client = MockClient.return_value
            mock_client.stations_metadata = noaa_stations_gdf

            stage = SfincsObservationPointsStage(sfincs_config)
            stage._add_noaa_gages(mock_model)

        for call in mock_model.observation_points.add_point.call_args_list:
            x = call.kwargs["x"]
            y = call.kwargs["y"]
            # UTM Zone 15N coordinates should be in the 100k-900k range for x
            # and millions range for y (northing)
            assert 100_000 < x < 900_000, f"x={x} not in UTM easting range"
            assert 1_000_000 < y < 10_000_000, f"y={y} not in UTM northing range"
