"""Tests for coastal_calibration.coops_api module."""

from __future__ import annotations

from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr

from coastal_calibration.coops_api import (
    COOPSAPIClient,
    DatumValue,
    StationDatum,
    _add_variable_attributes,
    _check_plot_deps,
    _process_responses,
    query_coops_bygeometry,
    query_coops_byids,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_stations_gdf():
    """Minimal stations GeoDataFrame for mocking _get_stations_metadata."""
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
        ],
        crs=4326,
    )


@pytest.fixture
def client(mock_stations_gdf):
    """COOPSAPIClient with _get_stations_metadata mocked out."""
    with patch.object(COOPSAPIClient, "_get_stations_metadata", return_value=mock_stations_gdf):
        yield COOPSAPIClient(timeout=30)


@pytest.fixture
def water_level_response():
    """Realistic single-station water_level API response."""
    return {
        "metadata": {
            "id": "8771450",
            "name": "Galveston Pier 21",
            "lat": "29.3100",
            "lon": "-94.7933",
        },
        "data": [
            {"t": "2023-01-01 00:00", "v": "0.412", "s": "0.003", "f": "0,0,0,0", "q": "v"},
            {"t": "2023-01-01 00:06", "v": "0.418", "s": "0.004", "f": "0,0,0,0", "q": "v"},
            {"t": "2023-01-01 00:12", "v": "0.425", "s": "0.002", "f": "0,0,0,0", "q": "v"},
        ],
    }


@pytest.fixture
def predictions_response():
    """Realistic single-station predictions API response."""
    return {
        "predictions": [
            {"t": "2023-01-01 00:00", "v": "0.500"},
            {"t": "2023-01-01 00:06", "v": "0.510"},
        ],
    }


# ---------------------------------------------------------------------------
# _check_plot_deps
# ---------------------------------------------------------------------------


class TestCheckPlotDeps:
    def test_passes_when_installed(self):
        _check_plot_deps()

    def test_raises_when_missing(self):
        with (
            patch("builtins.__import__", side_effect=ImportError),
            pytest.raises(ImportError, match="Missing optional dependencies"),
        ):
            _check_plot_deps()


# ---------------------------------------------------------------------------
# DatumValue / StationDatum
# ---------------------------------------------------------------------------


class TestDatumValue:
    def test_fields(self):
        dv = DatumValue(name="MLLW", description="Mean Lower Low Water", value=0.0)
        assert dv.name == "MLLW"
        assert dv.description == "Mean Lower Low Water"
        assert dv.value == 0.0


class TestStationDatum:
    @pytest.fixture
    def datum(self):
        return StationDatum(
            station_id="8771450",
            accepted="2021-01-01",
            superseded="",
            epoch="1983-2001",
            units="meters",
            orthometric_datum="NAVD88",
            datums=[
                DatumValue(name="MLLW", description="Mean Lower Low Water", value=0.0),
                DatumValue(name="MSL", description="Mean Sea Level", value=0.162),
                DatumValue(name="NAVD", description="NAVD88", value=0.052),
            ],
            lat=0.0,
            lat_date="",
            lat_time="",
            hat=0.0,
            hat_date="",
            hat_time="",
            min_value=-1.0,
            min_date="",
            min_time="",
            max_value=2.0,
            max_date="",
            max_time="",
            datum_analysis_period=[],
            ngs_link="",
            ctrl_station="",
        )

    def test_get_datum_value_found(self, datum):
        assert datum.get_datum_value("MSL") == 0.162

    def test_get_datum_value_not_found(self, datum):
        assert datum.get_datum_value("NONEXISTENT") is None


# ---------------------------------------------------------------------------
# COOPSAPIClient
# ---------------------------------------------------------------------------


class TestCOOPSAPIClientInit:
    def test_init_sets_timeout(self, client):
        assert client.timeout == 30

    def test_stations_metadata_property(self, client, mock_stations_gdf):
        gdf = client.stations_metadata
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 2
        assert "8771450" in gdf["station_id"].values


class TestValidateParameters:
    def test_valid_parameters(self, client):
        client.validate_parameters("water_level", "MLLW", "metric", "gmt", None)

    def test_invalid_product(self, client):
        with pytest.raises(ValueError, match="Invalid product"):
            client.validate_parameters("bad_product", "MLLW", "metric", "gmt", None)

    def test_invalid_datum(self, client):
        with pytest.raises(ValueError, match="Invalid datum"):
            client.validate_parameters("water_level", "BAD", "metric", "gmt", None)

    def test_invalid_units(self, client):
        with pytest.raises(ValueError, match="Invalid units"):
            client.validate_parameters("water_level", "MLLW", "bad", "gmt", None)

    def test_invalid_timezone(self, client):
        with pytest.raises(ValueError, match="Invalid time_zone"):
            client.validate_parameters("water_level", "MLLW", "metric", "bad", None)

    def test_invalid_predictions_interval(self, client):
        with pytest.raises(ValueError, match="Invalid interval"):
            client.validate_parameters("predictions", "MLLW", "metric", "gmt", "999")

    def test_valid_predictions_interval(self, client):
        client.validate_parameters("predictions", "MLLW", "metric", "gmt", "h")

    def test_interval_ignored_for_non_predictions(self, client):
        client.validate_parameters("water_level", "MLLW", "metric", "gmt", "999")


class TestParseDate:
    def test_yyyymmdd(self, client):
        ts = client.parse_date("20230101")
        assert ts == pd.Timestamp("2023-01-01")

    def test_iso_format(self, client):
        ts = client.parse_date("2023-01-01 12:30")
        assert ts.hour == 12
        assert ts.minute == 30

    def test_invalid_date(self, client):
        with pytest.raises(ValueError, match="Invalid date format"):
            client.parse_date("not-a-date")


class TestBuildUrl:
    def test_basic_url(self, client):
        url = client.build_url(
            station_id="8771450",
            begin_date="20230101 00:00",
            end_date="20230102 00:00",
            product="water_level",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
            interval=None,
        )
        assert "station=8771450" in url
        assert "product=water_level" in url
        assert "datum=MLLW" in url
        assert "format=json" in url
        assert url.startswith(COOPSAPIClient.base_url)

    def test_predictions_with_interval(self, client):
        url = client.build_url(
            station_id="8771450",
            begin_date="20230101 00:00",
            end_date="20230102 00:00",
            product="predictions",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
            interval="h",
        )
        assert "interval=h" in url

    def test_no_interval_when_none(self, client):
        url = client.build_url(
            station_id="8771450",
            begin_date="20230101 00:00",
            end_date="20230102 00:00",
            product="predictions",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
            interval=None,
        )
        assert "interval" not in url


class TestGetDatums:
    @pytest.fixture
    def datum_response(self):
        return {
            "accepted": "2021-01-01",
            "superseded": "",
            "epoch": "1983-2001",
            "units": "meters",
            "OrthometricDatum": "NAVD88",
            "datums": [
                {"name": "MLLW", "description": "Mean Lower Low Water", "value": "0.000"},
                {"name": "MSL", "description": "Mean Sea Level", "value": "0.162"},
            ],
            "LAT": "0.0",
            "LATdate": "",
            "LATtime": "",
            "HAT": "0.0",
            "HATdate": "",
            "HATtime": "",
            "min": "-1.0",
            "mindate": "",
            "mintime": "",
            "max": "2.0",
            "maxdate": "",
            "maxtime": "",
            "DatumAnalysisPeriod": [],
            "NGSLink": "",
            "ctrlStation": "",
        }

    def test_single_station(self, client, datum_response):
        with patch.object(client, "fetch_data", return_value=[datum_response]):
            result = client.get_datums("8771450")
            assert isinstance(result, StationDatum)
            assert result.station_id == "8771450"
            assert result.get_datum_value("MLLW") == 0.0
            assert result.get_datum_value("MSL") == pytest.approx(0.162)

    def test_multiple_stations(self, client, datum_response):
        with patch.object(client, "fetch_data", return_value=[datum_response, datum_response]):
            result = client.get_datums(["8771450", "8770570"])
            assert isinstance(result, list)
            assert len(result) == 2

    def test_error_response_skipped(self, client):
        error_resp = {"error": {"message": "Station not found"}}
        with (
            patch.object(client, "fetch_data", return_value=[error_resp]),
            pytest.raises(ValueError, match="No valid datum data"),
        ):
            client.get_datums("INVALID")

    def test_none_response_skipped(self, client, datum_response):
        with patch.object(client, "fetch_data", return_value=[None, datum_response]):
            result = client.get_datums(["BAD", "8771450"])
            assert len(result) == 1
            assert result[0].station_id == "8771450"


# ---------------------------------------------------------------------------
# _get_stations_metadata (cache behaviour)
# ---------------------------------------------------------------------------


class TestGetStationsMetadata:
    def test_loads_from_cache(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "coops_stations_metadata.json"
        import json

        cache_file.write_text(
            json.dumps(
                [
                    {
                        "id": "9999999",
                        "name": "Test Station",
                        "state": "XX",
                        "tidal": True,
                        "greatlakes": False,
                        "timezone": "EST",
                        "timezonecorr": "-5",
                        "lng": "-70.0",
                        "lat": "40.0",
                    }
                ]
            )
        )
        client = COOPSAPIClient.__new__(COOPSAPIClient)
        client.timeout = 30
        gdf = client._get_stations_metadata()
        assert len(gdf) == 1
        assert gdf.iloc[0]["station_id"] == "9999999"

    def test_fetches_when_no_cache(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_response = {
            "stations": [
                {
                    "id": "9999999",
                    "name": "Test Station",
                    "state": "XX",
                    "tidal": True,
                    "greatlakes": False,
                    "timezone": "EST",
                    "timezonecorr": "-5",
                    "lng": "-70.0",
                    "lat": "40.0",
                }
            ]
        }
        with patch("coastal_calibration.coops_api.fetch", return_value=mock_response):
            client = COOPSAPIClient.__new__(COOPSAPIClient)
            client.timeout = 30
            gdf = client._get_stations_metadata()
            assert len(gdf) == 1
            assert (tmp_path / "cache" / "coops_stations_metadata.json").exists()

    def test_raises_on_empty_response(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("coastal_calibration.coops_api.fetch", return_value=None):
            client = COOPSAPIClient.__new__(COOPSAPIClient)
            client.timeout = 30
            with pytest.raises(ValueError, match="No station data returned"):
                client._get_stations_metadata()


# ---------------------------------------------------------------------------
# _add_variable_attributes
# ---------------------------------------------------------------------------


class TestAddVariableAttributes:
    def test_water_level_attributes_metric(self):
        ds = xr.Dataset(
            {
                "water_level": xr.DataArray([1.0, 2.0], dims=["time"]),
                "sigma": xr.DataArray([0.01, 0.02], dims=["time"]),
                "flags": xr.DataArray(["0,0,0,0", "0,0,0,0"], dims=["time"]),
                "quality": xr.DataArray(["v", "v"], dims=["time"]),
            }
        )
        _add_variable_attributes(ds, "water_level", "metric")
        assert ds["water_level"].attrs["long_name"] == "Water Level"
        assert ds["water_level"].attrs["units"] == "meters"
        assert ds["sigma"].attrs["units"] == "meters"
        assert "description" in ds["flags"].attrs

    def test_water_level_attributes_english(self):
        ds = xr.Dataset({"water_level": xr.DataArray([1.0], dims=["time"])})
        _add_variable_attributes(ds, "water_level", "english")
        assert ds["water_level"].attrs["units"] == "feet"

    def test_predictions_attributes(self):
        ds = xr.Dataset({"water_level": xr.DataArray([1.0], dims=["time"])})
        _add_variable_attributes(ds, "predictions", "metric")
        assert ds["water_level"].attrs["long_name"] == "Predicted Water Level"

    def test_high_low_attributes(self):
        ds = xr.Dataset(
            {
                "water_level": xr.DataArray([1.0], dims=["time"]),
                "tide_type": xr.DataArray(["H"], dims=["time"]),
            }
        )
        _add_variable_attributes(ds, "high_low", "metric")
        assert ds["tide_type"].attrs["description"] == "H = high tide, L = low tide"

    def test_skips_metadata_variables(self):
        ds = xr.Dataset(
            {
                "station_id": xr.DataArray(["S1"], dims=["station"]),
                "water_level": xr.DataArray([1.0], dims=["time"]),
            }
        )
        _add_variable_attributes(ds, "water_level", "metric")
        assert "long_name" not in ds["station_id"].attrs


# ---------------------------------------------------------------------------
# _process_responses
# ---------------------------------------------------------------------------


class TestProcessResponses:
    def test_single_station(self, water_level_response):
        ds = _process_responses(
            [water_level_response],
            ["8771450"],
            product="water_level",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
        )
        assert isinstance(ds, xr.Dataset)
        assert "water_level" in ds
        assert ds.sizes["time"] == 3
        assert ds.sizes["station"] == 1
        assert ds.attrs["product"] == "water_level"
        assert ds.attrs["datum"] == "MLLW"
        assert ds.attrs["source"] == "NOAA CO-OPS API"
        np.testing.assert_allclose(ds["water_level"].values[:, 0], [0.412, 0.418, 0.425], atol=1e-6)

    def test_multiple_stations(self, water_level_response):
        resp2 = {
            "metadata": {
                "id": "8770570",
                "name": "Sabine Pass North",
                "lat": "29.72",
                "lon": "-93.87",
            },
            "data": [
                {"t": "2023-01-01 00:00", "v": "0.100", "s": "0.001", "f": "0,0,0,0", "q": "v"},
                {"t": "2023-01-01 00:06", "v": "0.110", "s": "0.001", "f": "0,0,0,0", "q": "v"},
            ],
        }
        ds = _process_responses(
            [water_level_response, resp2],
            ["8771450", "8770570"],
            product="water_level",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
        )
        assert ds.sizes["station"] == 2
        assert ds.sizes["time"] == 3
        # Station 2 has no data at t=00:12, should be NaN
        assert np.isnan(ds["water_level"].sel(station="8770570", time="2023-01-01 00:12").values)

    def test_predictions_product(self, predictions_response):
        ds = _process_responses(
            [predictions_response],
            ["8771450"],
            product="predictions",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
        )
        assert "water_level" in ds
        assert ds.sizes["time"] == 2

    def test_all_responses_none_raises(self):
        with pytest.raises(ValueError, match="No valid data returned"):
            _process_responses(
                [None],
                ["8771450"],
                product="water_level",
                datum="MLLW",
                units="metric",
                time_zone="gmt",
            )

    def test_error_response_skipped(self, water_level_response):
        error_resp = {"error": {"message": "Station not found"}}
        ds = _process_responses(
            [error_resp, water_level_response],
            ["BAD", "8771450"],
            product="water_level",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
        )
        assert ds.sizes["station"] == 2
        # BAD station should be all NaN
        assert np.all(np.isnan(ds["water_level"].sel(station="BAD").values))

    def test_empty_data_skipped(self, water_level_response):
        empty_resp = {
            "metadata": {"id": "0000000", "name": "Empty", "lat": "0", "lon": "0"},
            "data": [],
        }
        ds = _process_responses(
            [empty_resp, water_level_response],
            ["0000000", "8771450"],
            product="water_level",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
        )
        assert ds.sizes["station"] == 2
        assert np.all(np.isnan(ds["water_level"].sel(station="0000000").values))

    def test_dataset_metadata_variables(self, water_level_response):
        ds = _process_responses(
            [water_level_response],
            ["8771450"],
            product="water_level",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
        )
        assert "station_id" in ds
        assert "station_name" in ds
        assert "latitude" in ds
        assert "longitude" in ds
        assert ds["station_id"].values[0] == "8771450"
        assert ds["latitude"].attrs["standard_name"] == "latitude"

    def test_nan_and_empty_values_handled(self):
        response = {
            "metadata": {"id": "TEST", "name": "Test", "lat": "0", "lon": "0"},
            "data": [
                {"t": "2023-01-01 00:00", "v": "0.5", "s": "0.01", "f": "0,0,0,0", "q": "v"},
                {"t": "2023-01-01 00:06", "v": "", "s": "NaN", "f": "0,0,0,0", "q": "v"},
            ],
        }
        ds = _process_responses(
            [response],
            ["TEST"],
            product="water_level",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
        )
        assert ds["water_level"].values[0, 0] == pytest.approx(0.5)
        assert np.isnan(ds["water_level"].values[1, 0])
        assert np.isnan(ds["sigma"].values[1, 0])


# ---------------------------------------------------------------------------
# query_coops_byids
# ---------------------------------------------------------------------------


class TestQueryCoopsByIds:
    def test_end_before_start_raises(self, mock_stations_gdf):
        with (
            patch.object(COOPSAPIClient, "_get_stations_metadata", return_value=mock_stations_gdf),
            pytest.raises(ValueError, match="end_date must be after begin_date"),
        ):
            query_coops_byids(["8771450"], "20230102", "20230101")

    def test_successful_query(self, mock_stations_gdf, water_level_response):
        with (
            patch.object(COOPSAPIClient, "_get_stations_metadata", return_value=mock_stations_gdf),
            patch.object(COOPSAPIClient, "fetch_data", return_value=[water_level_response]),
        ):
            ds = query_coops_byids(["8771450"], "20230101", "20230102")
            assert isinstance(ds, xr.Dataset)
            assert "water_level" in ds


# ---------------------------------------------------------------------------
# query_coops_bygeometry
# ---------------------------------------------------------------------------


class TestQueryCoopsByGeometry:
    def test_invalid_geometry_raises(self, mock_stations_gdf):
        with patch.object(COOPSAPIClient, "_get_stations_metadata", return_value=mock_stations_gdf):
            bad_geom = shapely.Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])
            with pytest.raises(ValueError, match="Invalid geometry"):
                query_coops_bygeometry(bad_geom, "20230101", "20230102")

    def test_no_stations_in_geometry_raises(self, mock_stations_gdf):
        with patch.object(COOPSAPIClient, "_get_stations_metadata", return_value=mock_stations_gdf):
            # Somewhere in the Pacific with no stations
            geom = shapely.box(170, -50, 175, -45)
            with pytest.raises(ValueError, match="No stations found"):
                query_coops_bygeometry(geom, "20230101", "20230102")

    def test_stations_within_geometry(self, mock_stations_gdf, water_level_response):
        # Box around Galveston
        geom = shapely.box(-95.0, 29.0, -94.5, 29.5)
        with (
            patch.object(COOPSAPIClient, "_get_stations_metadata", return_value=mock_stations_gdf),
            patch("coastal_calibration.coops_api.query_coops_byids") as mock_byids,
        ):
            mock_byids.return_value = xr.Dataset()
            query_coops_bygeometry(geom, "20230101", "20230102")
            mock_byids.assert_called_once()
            call_args = mock_byids.call_args
            assert "8771450" in call_args[0][0]
