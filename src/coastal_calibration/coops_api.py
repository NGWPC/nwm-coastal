"""Fetch NOAA CO-OPS water level data.

This module requires the ``plot`` optional dependencies:

.. code-block:: bash

    pip install coastal-calibration[plot]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, overload

from tiny_retriever import fetch

from coastal_calibration.utils.logging import logger

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd
    import shapely
    import xarray as xr


def _check_plot_deps() -> None:
    """Check that plot optional dependencies are installed."""
    missing = []
    for pkg in ("geopandas", "numpy", "pandas", "shapely", "xarray"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing optional dependencies: {', '.join(missing)}. "
            "Install them with: pip install coastal-calibration[plot]"
        )


__all__ = [
    "COOPSAPIClient",
    "DatumValue",
    "StationDatum",
    "query_coops_bygeometry",
    "query_coops_byids",
]


@dataclass
class DatumValue:
    """Individual datum value information."""

    name: str
    description: str
    value: float


@dataclass
class StationDatum:
    """Complete datum information for a station."""

    station_id: str
    accepted: str
    superseded: str
    epoch: str
    units: str
    orthometric_datum: str
    datums: list[DatumValue]
    lat: float
    lat_date: str
    lat_time: str
    hat: float
    hat_date: str
    hat_time: str
    min_value: float
    min_date: str
    min_time: str
    max_value: float
    max_date: str
    max_time: str
    datum_analysis_period: list[str]
    ngs_link: str
    ctrl_station: str

    def get_datum_value(self, datum_name: str) -> float | None:
        """Get value for a specific datum by name.

        Parameters
        ----------
        datum_name : str
            Name of the datum (e.g., 'MLLW', 'MSL')

        Returns
        -------
        float | None
            Datum value or None if not found
        """
        for datum in self.datums:
            if datum.name == datum_name:
                return datum.value
        return None


class COOPSAPIClient:
    """Client for interacting with NOAA CO-OPS API."""

    base_url: Final[str] = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    valid_products = frozenset(["water_level", "hourly_height", "high_low", "predictions"])
    valid_datums = frozenset(
        [
            "CRD",
            "IGLD",
            "LWD",
            "MHHW",
            "MHW",
            "MTL",
            "MSL",
            "MLW",
            "MLLW",
            "NAVD",
            "STND",
        ]
    )
    valid_units = frozenset(["metric", "english"])
    valid_timezones = frozenset(["gmt", "lst", "lst_ldt"])
    valid_intervals = frozenset(["h", "1", "5", "6", "10", "15", "30", "60", "hilo"])

    def _get_stations_metadata(self) -> gpd.GeoDataFrame:
        """Get metadata for all water level stations as a GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with station metadata and Point geometries.
        """
        import geopandas as gpd
        import numpy as np
        import shapely

        cache_dir = Path("./cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "coops_stations_metadata.json"

        if cache_file.exists():
            logger.info("Loading cached station metadata from %s", cache_file)
            stations = json.loads(cache_file.read_text())
        else:
            metadata_url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=waterlevels"

            logger.info("Fetching metadata for all water level stations")
            response = fetch(
                metadata_url,
                "json",
                request_method="get",
                timeout=self.timeout,
                raise_status=True,
            )

            if not response or "stations" not in response:
                raise ValueError("No station data returned from metadata API")

            stations = response["stations"]
            cache_file.write_text(json.dumps(stations))
            logger.info("Retrieved metadata for %d stations", len(stations))
            logger.info("Saved station metadata to cache: %s", cache_file)

        return gpd.GeoDataFrame(
            (
                {
                    "station_id": station_info.get("id", ""),
                    "station_name": station_info.get("name", ""),
                    "state": station_info.get("state", ""),
                    "tidal": station_info.get("tidal", False),
                    "greatlakes": station_info.get("greatlakes", False),
                    "time_zone": station_info.get("timezone", ""),
                    "time_zone_offset": station_info.get("timezonecorr", ""),
                    "geometry": shapely.Point(
                        float(station_info.get("lng", np.nan)),
                        float(station_info.get("lat", np.nan)),
                    ),
                }
                for station_info in stations
            ),
            crs=4326,
        )

    def __init__(self, timeout: int = 120) -> None:
        """Initialize COOPS API client.

        Parameters
        ----------
        timeout : int, optional
            Request timeout in seconds, by default 120

        Raises
        ------
        ImportError
            If plot optional dependencies are not installed.
        """
        _check_plot_deps()
        self.timeout = timeout
        self._stations_metadata = self._get_stations_metadata()

    @property
    def stations_metadata(self) -> gpd.GeoDataFrame:
        """Get metadata for all water level stations as a GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with station metadata and Point geometries.
        """
        return self._stations_metadata

    def validate_parameters(
        self,
        product: str,
        datum: str,
        units: str,
        time_zone: str,
        interval: str | int | None,
    ) -> None:
        """Validate API parameters.

        Parameters
        ----------
        product : str
            Data product type
        datum : str
            Vertical datum
        units : str
            Unit system
        time_zone : str
            Time zone
        interval : str | int | None
            Time interval for predictions

        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        if product not in self.valid_products:
            raise ValueError(
                f"Invalid product '{product}'. Must be one of: {', '.join(self.valid_products)}"
            )

        if datum.upper() not in self.valid_datums:
            raise ValueError(
                f"Invalid datum '{datum}'. Must be one of: {', '.join(self.valid_datums)}"
            )

        if units not in self.valid_units:
            raise ValueError(
                f"Invalid units '{units}'. Must be one of: {', '.join(self.valid_units)}"
            )

        if time_zone not in self.valid_timezones:
            raise ValueError(
                f"Invalid time_zone '{time_zone}'. Must be one of: {', '.join(self.valid_timezones)}"
            )

        if (
            product == "predictions"
            and interval is not None
            and str(interval) not in self.valid_intervals
        ):
            raise ValueError(
                f"Invalid interval '{interval}' for predictions. "
                f"Must be one of: {', '.join(self.valid_intervals)}"
            )

    def parse_date(self, date_str: str) -> pd.Timestamp:
        """Parse date string to pandas Timestamp.

        Parameters
        ----------
        date_str : str
            Date string in various formats

        Returns
        -------
        pd.Timestamp
            Parsed timestamp

        Raises
        ------
        ValueError
            If date string cannot be parsed
        """
        import pandas as pd

        try:
            return pd.to_datetime(date_str)
        except Exception as e:
            raise ValueError(f"Invalid date format '{date_str}'.") from e

    def build_url(
        self,
        station_id: str,
        begin_date: str,
        end_date: str,
        product: str,
        datum: str,
        units: str,
        time_zone: str,
        interval: str | int | None,
    ) -> str:
        """Build API request URL for a station.

        Parameters
        ----------
        station_id : str
            Station ID
        begin_date : str
            Start date
        end_date : str
            End date
        product : str
            Data product
        datum : str
            Vertical datum
        units : str
            Unit system
        time_zone : str
            Time zone
        interval : str | int | None, optional
            Time interval for predictions

        Returns
        -------
        str
            Complete API request URL
        """
        params = {
            "begin_date": begin_date,
            "end_date": end_date,
            "station": station_id,
            "product": product,
            "datum": datum,
            "units": units,
            "time_zone": time_zone,
            "format": "json",
            "application": "coastal_calibration_coops",
        }

        if product == "predictions" and interval is not None:
            params["interval"] = str(interval)

        query_parts = [f"{k}={v}" for k, v in params.items()]
        return f"{self.base_url}?{'&'.join(query_parts)}"

    def fetch_data(self, urls: list[str]) -> list[dict[str, Any] | None]:
        """Fetch data from API for multiple URLs.

        Parameters
        ----------
        urls : list[str]
            List of API request URLs

        Returns
        -------
        list[dict | None]
            List of JSON responses (None for failed requests)
        """
        logger.info("Fetching data from %d station(s)", len(urls))

        return fetch(
            urls,
            "json",
            request_method="get",
            timeout=self.timeout,
            raise_status=False,
        )

    @overload
    def get_datums(self, station_ids: str) -> StationDatum: ...

    @overload
    def get_datums(self, station_ids: list[str]) -> list[StationDatum]: ...

    def get_datums(self, station_ids: str | list[str]) -> StationDatum | list[StationDatum]:
        """Retrieve datum information for one or more stations.

        Parameters
        ----------
        station_ids : str | list[str]
            Single station ID or list of station IDs

        Returns
        -------
        StationDatum | list[StationDatum]
            Single StationDatum object if input is str,
            list of StationDatum if input is list

        Raises
        ------
        ValueError
            If no valid datum data is returned for any station.
        """
        import numpy as np

        single_input = isinstance(station_ids, str)
        if single_input:
            station_ids = [station_ids]

        datum_base_url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations"
        urls = [f"{datum_base_url}/{sid}/datums.json" for sid in station_ids]
        logger.info("Fetching datum information for %d station(s)", len(station_ids))
        responses = self.fetch_data(urls)
        datum_objects = []
        for station_id, response in zip(station_ids, responses, strict=False):
            if response is None:
                logger.warning("No datum data returned for station %s", station_id)
                continue

            if "error" in response:
                logger.warning(
                    "Datum API error for station %s: %s",
                    station_id,
                    response["error"].get("message", "Unknown error"),
                )
                continue

            datum_values = [
                DatumValue(
                    name=datum_dict.get("name", ""),
                    description=datum_dict.get("description", ""),
                    value=np.float64(datum_dict.get("value", np.nan)),
                )
                for datum_dict in response.get("datums", [])
            ]

            station_datum = StationDatum(
                station_id=station_id,
                accepted=response.get("accepted", ""),
                superseded=response.get("superseded", ""),
                epoch=response.get("epoch", ""),
                units=response.get("units", ""),
                orthometric_datum=response.get("OrthometricDatum", ""),
                datums=datum_values,
                lat=np.float64(response.get("LAT", np.nan)),
                lat_date=response.get("LATdate", ""),
                lat_time=response.get("LATtime", ""),
                hat=np.float64(response.get("HAT", np.nan)),
                hat_date=response.get("HATdate", ""),
                hat_time=response.get("HATtime", ""),
                min_value=np.float64(response.get("min", np.nan)),
                min_date=response.get("mindate", ""),
                min_time=response.get("mintime", ""),
                max_value=np.float64(response.get("max", np.nan)),
                max_date=response.get("maxdate", ""),
                max_time=response.get("maxtime", ""),
                datum_analysis_period=response.get("DatumAnalysisPeriod", []),
                ngs_link=response.get("NGSLink", ""),
                ctrl_station=response.get("ctrlStation", ""),
            )

            datum_objects.append(station_datum)

        if not datum_objects:
            raise ValueError("No valid datum data returned for any station")

        if single_input:
            return datum_objects[0]
        return datum_objects


def _add_variable_attributes(
    ds: xr.Dataset,
    product: str,
    units: str,
) -> None:
    """Add attributes to data variables based on product type.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to add attributes to (modified in place)
    product : str
        Product type
    units : str
        Unit system
    """
    height_unit = "meters" if units == "metric" else "feet"
    product_metadata = {
        "water_level": {
            "water_level": ("Water Level", "sea_surface_height_above_reference_datum"),
            "sigma": ("Standard Deviation", None),
            "flags": ("Data Flags", None),
            "quality": ("Quality Assurance", None),
        },
        "hourly_height": {
            "water_level": (
                "Hourly Height Water Level",
                "sea_surface_height_above_reference_datum",
            ),
            "sigma": ("Standard Deviation", None),
            "flags": ("Data Flags", None),
        },
        "high_low": {
            "water_level": ("High/Low Water Level", "sea_surface_height_above_reference_datum"),
            "tide_type": ("Tide Type", None),
            "flags": ("Data Flags", None),
        },
        "predictions": {
            "water_level": ("Predicted Water Level", "sea_surface_height_above_reference_datum"),
        },
    }

    var_metadata = product_metadata.get(product, {})
    for var_name in ds.data_vars:
        if var_name in ["station_id", "station_name", "latitude", "longitude"]:
            continue

        if var_name in var_metadata:
            long_name, standard_name = var_metadata[str(var_name)]
            ds[var_name].attrs["long_name"] = long_name
            if standard_name is not None:
                ds[var_name].attrs["standard_name"] = standard_name

            if var_name in {"water_level", "sigma"}:
                ds[var_name].attrs["units"] = height_unit

        if var_name == "flags":
            ds[var_name].attrs["description"] = (
                "Data quality flags: 0,0,0,0 = datum, preliminary, inferred, pumping"
            )
        elif var_name == "quality":
            ds[var_name].attrs["description"] = "v = verified, p = preliminary"
        elif var_name == "tide_type":
            ds[var_name].attrs["description"] = "H = high tide, L = low tide"


def _process_responses(  # noqa: PLR0912, PLR0915
    responses: list[dict[str, Any] | None],
    station_ids: list[str],
    product: str,
    datum: str,
    units: str,
    time_zone: str,
) -> xr.Dataset:
    """Process API responses into an xarray Dataset.

    Parameters
    ----------
    responses : list[dict | None]
        List of JSON responses from API
    station_ids : list[str]
        Station IDs corresponding to responses
    product : str
        Product type
    datum : str
        Vertical datum
    units : str
        Unit system
    time_zone : str
        Time zone

    Returns
    -------
    xr.Dataset
        Dataset with water level data and metadata

    Raises
    ------
    ValueError
        If no valid data is returned
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    station_data = {}
    station_metadata = {}
    all_times: list[pd.Timestamp] = []

    for station_id, response in zip(station_ids, responses, strict=False):
        if response is None:
            logger.warning("No data returned for station %s", station_id)
            continue

        if "error" in response:
            error_msg = response["error"].get("message", "Unknown error")
            logger.warning("API error for station %s: %s", station_id, error_msg)
            continue

        data_key = "predictions" if product == "predictions" else "data"
        if data_key not in response:
            logger.warning("No %s in response for station %s", data_key, station_id)
            continue

        data_list = response[data_key]
        if not data_list:
            logger.warning("Empty data for station %s", station_id)
            continue

        df = pd.DataFrame(data_list)
        df["time"] = pd.to_datetime(df["t"])
        all_times.extend(df["time"].tolist())

        metadata = response.get("metadata", {})
        station_metadata[station_id] = {
            "station_id": metadata.get("id", station_id),
            "station_name": metadata.get("name", ""),
            "latitude": float(metadata.get("lat", np.nan)),
            "longitude": float(metadata.get("lon", np.nan)),
        }
        station_data[station_id] = df

    if not station_data:
        raise ValueError("No valid data returned for any station")

    logger.info("Successfully retrieved data for %d station(s)", len(station_data))

    unique_times = sorted(set(all_times))
    time_index = pd.DatetimeIndex(unique_times)

    data_vars = {}

    first_station_df = next(iter(station_data.values()))
    value_columns = [col for col in first_station_df.columns if col not in ["t", "time"]]

    variable_name_mapping = {
        "v": "water_level",
        "s": "sigma",
        "f": "flags",
        "q": "quality",
        "ty": "tide_type",
    }

    for api_col in value_columns:
        var_name = variable_name_mapping.get(api_col, api_col)
        data_array = np.full((len(time_index), len(station_ids)), np.nan, dtype=object)

        for i, station_id in enumerate(station_ids):
            if station_id not in station_data:
                continue

            df = station_data[station_id]
            if api_col not in df.columns:
                continue

            for j, time_val in enumerate(time_index):
                mask = df["time"] == time_val
                if mask.any():
                    value = df.loc[mask, api_col].iloc[0]
                    if pd.notna(value) and value not in {"", "NaN"}:
                        if api_col in ["v", "s"]:
                            try:
                                data_array[j, i] = float(value)
                            except (ValueError, TypeError):
                                data_array[j, i] = np.nan
                        else:
                            data_array[j, i] = str(value)
                    else:
                        data_array[j, i] = np.nan

        if api_col in ["v", "s"]:
            data_array = data_array.astype(float)

        data_vars[var_name] = xr.DataArray(
            data_array,
            dims=["time", "station"],
            coords={"time": time_index, "station": station_ids},
        )

    ds = xr.Dataset(data_vars)
    station_ids_array = np.array(
        [station_metadata.get(sid, {}).get("station_id", sid) for sid in station_ids]
    )
    station_names = np.array(
        [station_metadata.get(sid, {}).get("station_name", "") for sid in station_ids]
    )
    latitudes = np.array(
        [station_metadata.get(sid, {}).get("latitude", np.nan) for sid in station_ids]
    )
    longitudes = np.array(
        [station_metadata.get(sid, {}).get("longitude", np.nan) for sid in station_ids]
    )

    ds["station_id"] = xr.DataArray(
        station_ids_array,
        dims=["station"],
        attrs={"long_name": "Station ID", "cf_role": "timeseries_id"},
    )
    ds["station_name"] = xr.DataArray(
        station_names,
        dims=["station"],
        attrs={"long_name": "Station Name"},
    )
    ds["latitude"] = xr.DataArray(
        latitudes,
        dims=["station"],
        attrs={
            "long_name": "Latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        },
    )
    ds["longitude"] = xr.DataArray(
        longitudes,
        dims=["station"],
        attrs={
            "long_name": "Longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        },
    )

    ds.attrs["product"] = product
    ds.attrs["datum"] = datum
    ds.attrs["units"] = units
    ds.attrs["time_zone"] = time_zone
    ds.attrs["source"] = "NOAA CO-OPS API"
    ds.attrs["retrieved_at"] = datetime.now().isoformat()

    ds["time"].attrs["long_name"] = "Time"
    ds["time"].attrs["standard_name"] = "time"
    ds["station"].attrs["long_name"] = "Station Index"

    _add_variable_attributes(ds, product, units)

    return ds


def query_coops_byids(
    station_ids: list[str],
    begin_date: str,
    end_date: str,
    *,
    product: Literal[
        "water_level",
        "hourly_height",
        "high_low",
        "predictions",
    ] = "water_level",
    datum: str = "MLLW",
    units: Literal["metric", "english"] = "metric",
    time_zone: Literal["gmt", "lst", "lst_ldt"] = "gmt",
    interval: str | int | None = None,
) -> xr.Dataset:
    """Fetch water level data from NOAA CO-OPS API for multiple stations.

    Parameters
    ----------
    station_ids : list[str]
        List of station IDs to retrieve data for.
    begin_date : str
        Start date in format: yyyyMMdd, yyyyMMdd HH:mm, MM/dd/yyyy, or MM/dd/yyyy HH:mm
    end_date : str
        End date in same format as begin_date.
    product : {"water_level", "hourly_height", "high_low", "predictions"}, optional
        Data product to retrieve, by default ``water_level``.
    datum : str, optional
        Vertical datum for water levels, by default "MLLW".
    units : {"metric", "english"}, optional
        Units for data, by default "metric".
    time_zone : {"gmt", "lst", "lst_ldt"}, optional
        Time zone for returned data, by default "gmt".
    interval : str | int | None, optional
        Time interval for predictions product only, by default None.

    Returns
    -------
    xr.Dataset
        Dataset containing water level data with dimensions (time, station).

    Raises
    ------
    ValueError
        If invalid parameters are provided or if API returns errors.
    """
    client = COOPSAPIClient()
    client.validate_parameters(product, datum, units, time_zone, interval)
    begin_dt = client.parse_date(begin_date)
    end_dt = client.parse_date(end_date)

    if end_dt <= begin_dt:
        raise ValueError("end_date must be after begin_date")

    begin_str = begin_dt.strftime("%Y%m%d %H:%M")
    end_str = end_dt.strftime("%Y%m%d %H:%M")

    logger.info(
        "Requesting %s data for %d station(s) from %s to %s",
        product,
        len(station_ids),
        begin_str,
        end_str,
    )

    urls = [
        client.build_url(
            station_id=station_id,
            begin_date=begin_str,
            end_date=end_str,
            product=product,
            datum=datum,
            units=units,
            time_zone=time_zone,
            interval=interval,
        )
        for station_id in station_ids
    ]

    return _process_responses(
        responses=client.fetch_data(urls),
        station_ids=station_ids,
        product=product,
        datum=datum,
        units=units,
        time_zone=time_zone,
    )


def query_coops_bygeometry(
    geometry: shapely.geometry.base.BaseGeometry,
    begin_date: str,
    end_date: str,
    *,
    product: Literal[
        "water_level",
        "hourly_height",
        "high_low",
        "predictions",
    ] = "water_level",
    datum: str = "MLLW",
    units: Literal["metric", "english"] = "metric",
    time_zone: Literal["gmt", "lst", "lst_ldt"] = "gmt",
    interval: str | int | None = None,
) -> xr.Dataset:
    """Fetch water level data from NOAA CO-OPS API for stations within a geometry.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Geometry to select stations within (Point, Polygon, etc.)
    begin_date : str
        Start date in format: yyyyMMdd, yyyyMMdd HH:mm, MM/dd/yyyy, or MM/dd/yyyy HH:mm
    end_date : str
        End date in same format as begin_date.
    product : {"water_level", "hourly_height", "high_low", "predictions"}, optional
        Data product to retrieve, by default ``water_level``.
    datum : str, optional
        Vertical datum for water levels, by default "MLLW".
    units : {"metric", "english"}, optional
        Units for data, by default "metric".
    time_zone : {"gmt", "lst", "lst_ldt"}, optional
        Time zone for returned data, by default "gmt".
    interval : str | int | None, optional
        Time interval for predictions product only, by default None.

    Returns
    -------
    xr.Dataset
        Dataset containing water level data for stations within the geometry.
    """
    import numpy as np
    import shapely

    client = COOPSAPIClient()
    if not all(shapely.is_valid(np.atleast_1d(geometry))):  # pyright: ignore[reportCallIssue,reportArgumentType]
        raise ValueError("Invalid geometry provided.")

    stations_gdf = client.stations_metadata
    selected_stations = stations_gdf[stations_gdf.intersects(geometry)]

    if selected_stations.empty:
        raise ValueError("No stations found within the specified geometry and buffer.")

    station_ids = selected_stations["station_id"].tolist()
    return query_coops_byids(
        station_ids,
        begin_date,
        end_date,
        product=product,
        datum=datum,
        units=units,
        time_zone=time_zone,
        interval=interval,
    )
