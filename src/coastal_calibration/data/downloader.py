"""Async data downloader for coastal model calibration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

import fsspec
import pandas as pd
from tiny_retriever import download

from coastal_calibration.config.schema import (
    BoundarySource,
    CoastalDomain,
    MeteoSource,
    PathConfig,
)
from coastal_calibration.logging import logger
from coastal_calibration.utils import to_naive_utc, utc_now


def _hour_range(start: datetime, end: datetime) -> range:
    """Return a range of integer offsets for hours in ``[start, end)``."""
    return range(int((end - start).total_seconds()) // 3600)


HydroSource = Literal["nwm", "ngen"]
# Coastal boundary source label. ``harmonic`` means "predicted locally
# from a tidal atlas" (no remote download needed); ``stofs`` / ``glofs``
# are downloaded from NOAA. ``tpxo`` is accepted as the deprecated
# upstream alias for ``harmonic`` and normalized at the schema boundary.
CoastalSource = Literal["stofs", "harmonic", "glofs"]
Domain = Literal["conus", "hawaii", "prvi", "atlgulf", "pacific", "alaska"]
GLOFSModel = Literal["leofs", "loofs", "lsofs", "lmhofs"]


@dataclass
class DateRange:
    """Date range for a data source."""

    start: datetime
    end: datetime | None
    description: str

    def validate(self, start: datetime, end: datetime) -> str | None:
        """Validate that the requested period falls within the available range."""
        # Normalize to naive UTC so comparisons against the naive class
        # attributes are well-defined and DST-stable.
        start = to_naive_utc(start)
        end = to_naive_utc(end)
        end_str = self.end.strftime("%Y-%m-%d") if self.end else "present"
        if start < self.start:
            return (
                f"{self.description} data is available from "
                f"{self.start.strftime('%Y-%m-%d')} to {end_str}. "
                f"Requested start date {start.strftime('%Y-%m-%d')} is before "
                f"the earliest available date."
            )
        if self.end is not None and end > self.end:
            return (
                f"{self.description} data is available from "
                f"{self.start.strftime('%Y-%m-%d')} to {end_str}. "
                f"Requested end date {end.strftime('%Y-%m-%d')} is after "
                f"the latest available date."
            )
        # For operational sources (end=None means "present"), check that dates aren't in the future
        if self.end is None:
            now = utc_now()
            if start > now:
                return (
                    f"{self.description} data is available from "
                    f"{self.start.strftime('%Y-%m-%d')} to present. "
                    f"Requested start date {start.strftime('%Y-%m-%d %H:%M')} is in the future "
                    f"(current UTC time: {now.strftime('%Y-%m-%d %H:%M')})."
                )
        return None


DATA_SOURCE_DATE_RANGES: dict[str, dict[str, DateRange]] = {
    "nwm_retro": {
        "conus": DateRange(
            start=datetime(1979, 2, 1),
            end=datetime(2023, 1, 31),
            description="NWM Retrospective 3.0 (CONUS)",
        ),
        "alaska": DateRange(
            start=datetime(1981, 1, 1),
            end=datetime(2019, 12, 31),
            description="NWM Retrospective 3.0 (Alaska)",
        ),
        "hawaii": DateRange(
            start=datetime(1994, 1, 2),
            end=datetime(2013, 12, 31),
            description="NWM Retrospective 3.0 (Hawaii)",
        ),
        "prvi": DateRange(
            start=datetime(2008, 1, 1),
            end=datetime(2023, 6, 30),
            description="NWM Retrospective 3.0 (PR)",
        ),
    },
    "nwm_ana": {
        "conus": DateRange(
            start=datetime(2018, 10, 1),
            end=None,
            description="NWM Analysis and Assimilation (CONUS)",
        ),
        "alaska": DateRange(
            start=datetime(2023, 10, 1),
            end=None,
            description="NWM Analysis and Assimilation (ALASKA)",
        ),
        "hawaii": DateRange(
            start=datetime(2021, 4, 21),
            end=None,
            description="NWM Analysis and Assimilation (HAWAII)",
        ),
        "prvi": DateRange(
            start=datetime(2023, 10, 1),
            end=None,
            description="NWM Analysis and Assimilation (PUERTORICO)",
        ),
    },
    "stofs": {
        "_default": DateRange(
            start=datetime(2020, 12, 30),
            end=None,
            description="STOFS (operational)",
        ),
    },
    "glofs": {
        "_default": DateRange(
            start=datetime(2005, 9, 30),
            end=None,
            description="GLOFS (Great Lakes)",
        ),
    },
}

# Domains that share CONUS data
_CONUS_DOMAINS = {"conus", "atlgulf", "pacific"}


def get_date_range(source: str, domain: str = "conus") -> DateRange | None:
    """Get the date range for a data source and domain.

    Parameters
    ----------
    source : str
        Data source name (e.g., ``nwm_retro``, ``nwm_ana``).
    domain : str
        Model domain (e.g., ``conus``, ``hawaii``, ``prvi``).
        Defaults to ``conus``.

    Returns
    -------
    DateRange or None
        The date range if found, otherwise None.
    """
    source_ranges = DATA_SOURCE_DATE_RANGES.get(source)
    if source_ranges is None:
        return None
    lookup = "conus" if domain in _CONUS_DOMAINS else domain
    return source_ranges.get(lookup) or source_ranges.get("_default")


def get_overlapping_range(
    meteo_source: str,
    coastal_source: str,
    domain: str,
) -> DateRange | None:
    """Get the overlapping date range between a meteo and coastal source.

    Parameters
    ----------
    meteo_source : str
        Meteorological data source (e.g., ``nwm_retro``, ``nwm_ana``).
    coastal_source : str
        Coastal boundary source (e.g., ``stofs``, ``harmonic``).
    domain : str
        Model domain (e.g., ``conus``, ``hawaii``, ``prvi``).

    Returns
    -------
    DateRange or None
        The overlapping range, or None if sources don't overlap or
        aren't found.
    """
    meteo_range = get_date_range(meteo_source, domain)
    if meteo_range is None:
        return None

    if coastal_source == "harmonic":
        return meteo_range

    coastal_range = get_date_range(coastal_source, domain)
    if coastal_range is None:
        return None

    overlap_start = max(meteo_range.start, coastal_range.start)
    overlap_end_meteo = meteo_range.end
    overlap_end_coastal = coastal_range.end

    if overlap_end_meteo is None and overlap_end_coastal is None:
        overlap_end = None
    elif overlap_end_meteo is None:
        overlap_end = overlap_end_coastal
    elif overlap_end_coastal is None:
        overlap_end = overlap_end_meteo
    else:
        overlap_end = min(overlap_end_meteo, overlap_end_coastal)

    if overlap_end is not None and overlap_start >= overlap_end:
        return None

    return DateRange(
        start=overlap_start,
        end=overlap_end,
        description=f"{meteo_range.description} + {coastal_range.description}",
    )


def get_default_sources(
    domain: CoastalDomain,
) -> tuple[MeteoSource, BoundarySource, datetime]:
    """Get default meteo source, boundary source, and start date for a domain.

    Picks source combinations that have overlapping date ranges,
    preferring ``nwm_retro`` + ``stofs`` when available, falling back
    to ``nwm_ana`` + ``stofs``.

    Parameters
    ----------
    domain : CoastalDomain
        Model domain: ``"prvi"``, ``"hawaii"``, ``"atlgulf"``,
        ``"pacific"``, or ``"alaska"``.

    Returns
    -------
    tuple of (MeteoSource, BoundarySource, datetime)
        ``(meteo_source, boundary_source, suggested_start_date)``.

    Raises
    ------
    ValueError
        If no valid source combination exists for the domain.
    """
    # Preferred combinations in priority order.
    # PRVI uses nwm_ana first because SCHISM currently fails with nwm_retro.
    if domain == "prvi":
        combos: list[tuple[MeteoSource, BoundarySource]] = [
            ("nwm_ana", "stofs"),
            ("nwm_ana", "harmonic"),
            ("nwm_retro", "stofs"),
            ("nwm_retro", "harmonic"),
        ]
    else:
        combos: list[tuple[MeteoSource, BoundarySource]] = [
            ("nwm_retro", "stofs"),
            ("nwm_ana", "stofs"),
            ("nwm_retro", "harmonic"),
            ("nwm_ana", "harmonic"),
        ]

    for meteo, coastal in combos:
        overlap = get_overlapping_range(meteo, coastal, domain)
        if overlap is not None:
            # Pick a start date near the beginning of the overlap
            return meteo, coastal, overlap.start
    msg = f"No valid meteo + boundary source combination found for domain '{domain}'"
    raise ValueError(msg)


@dataclass
class DownloadResult:
    """Result of a single download operation."""

    source: str
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    file_paths: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "OK" if not self.errors else "ERRORS"
        lines = [f"  {self.source}: {self.successful}/{self.total_files} files [{status}]"]
        lines.extend(f"    - {err}" for err in self.errors)
        return "\n".join(lines)


@dataclass
class DownloadResults:
    """Results of all download operations."""

    meteo: DownloadResult
    hydro: DownloadResult
    coastal: DownloadResult

    @property
    def has_errors(self) -> bool:
        """Return True if any download result contains errors."""
        return any(r.errors for r in self)

    def __iter__(self) -> Iterator[DownloadResult]:
        """Iterate over all download results."""
        return iter([self.meteo, self.hydro, self.coastal])

    def __str__(self) -> str:
        status = "ERRORS" if self.has_errors else "OK"
        lines: list[str] = [f"DownloadResults: {status}"]
        lines.extend(str(result) for result in self)
        return "\n".join(lines)


# Domain mappings for URL builders
_DOMAIN_MAP_RETRO = {
    "conus": "CONUS",
    "atlgulf": "CONUS",
    "pacific": "CONUS",
    "hawaii": "Hawaii",
    "prvi": "PR",
    "alaska": "Alaska",
}

_DOMAIN_MAP_ANA = {
    "conus": ("", "conus"),
    "atlgulf": ("", "conus"),
    "pacific": ("", "conus"),
    "hawaii": ("_hawaii", "hawaii"),
    "prvi": ("_puertorico", "puertorico"),
    "alaska": ("_alaska", "alaska"),
}

_GLOFS_MODEL_DIRS = {
    "leofs": "lake-erie-operational-forecast-system-leofs",
    "loofs": "lower-ohio-operational-forecast-system-loofs",
    "lsofs": "lake-st-clair-operational-forecast-system-lsofs",
    "lmhofs": "lake-michigan-huron-operational-forecast-system-lmhofs",
}


#: Zarr store per Retrospective domain whose ``crs`` variable carries the
#: authoritative GeoTransform for that domain's 1 km LDASIN grid.
_LDASOUT_ZARR = "s3://noaa-nwm-retrospective-3-0-pds/{domain}/zarr/ldasout.zarr"


def write_nwm_grid_sidecar(out_dir: Path, domain: str) -> Path | None:
    """Record the NWM Retrospective grid for *domain* next to its forcing.

    The PRVI and Alaska Retrospective LDASIN files carry no georeferencing
    at all: no grid-mapping variable, no coordinate variables, no WRF
    global attributes.  NOAA's own ``ldasout.zarr`` for the same domain is
    on the same 1 km grid and does carry a ``crs`` variable, so read its
    ``GeoTransform`` (metadata only, a few kB) and write it as a small
    JSON sidecar that
    :func:`coastal_calibration.data.nwm_forcing.normalize_wrf_forcing`
    picks up when rebuilding x/y coordinates.

    Writing one sidecar per domain lets several domains share a download
    directory; the reader selects by grid shape.

    Parameters
    ----------
    out_dir : pathlib.Path
        Directory holding the Retrospective forcing files.
    domain : str
        Coastal domain (``conus``, ``hawaii``, ``prvi``, ...).

    Returns
    -------
    pathlib.Path or None
        Path to the sidecar, or *None* if the grid could not be read.
        Failure is non-fatal: the domains that need it also have a
        built-in fallback.
    """
    import json

    sidecar = out_dir / f"grid_{domain}.json"
    if sidecar.is_file():
        return sidecar

    url = _LDASOUT_ZARR.format(domain=_DOMAIN_MAP_RETRO.get(domain, "CONUS"))
    # Written via a unique temporary file and renamed, matching
    # ``_execute_download``: a concurrent run or a crash mid-write must never
    # leave a half-parsed sidecar where the preprocessor will read it.
    tmp = sidecar.with_suffix(f".{os.getpid()}.tmp")
    try:
        import fsspec
        import xarray as xr

        with xr.open_zarr(fsspec.get_mapper(url, anon=True), consolidated=True) as ds:
            record = {
                "domain": domain,
                "geotransform": str(ds["crs"].attrs["GeoTransform"]).strip(),
                "shape": [int(ds.sizes["x"]), int(ds.sizes["y"])],
                "source": url,
            }
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(sidecar)
    except Exception as exc:
        # Deliberately broad: s3fs/botocore raise their own hierarchies, and an
        # offline node must still get its (already cached) forcing.
        logger.warning("Could not record the NWM %s grid from %s: %s", domain, url, exc)
        tmp.unlink(missing_ok=True)
        return None

    logger.info("Recorded the NWM %s forcing grid: %s", domain, sidecar)
    return sidecar


def _build_nwm_retro_forcing_urls(
    start: datetime,
    end: datetime,
    output_dir: Path,
    domain: str,
) -> tuple[list[str], list[Path]]:
    """Build URLs for NWM Retrospective forcing (LDASIN) files."""
    base_url = "https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com"
    domain_str = _DOMAIN_MAP_RETRO.get(domain, "CONUS")

    urls: list[str] = []
    paths: list[Path] = []
    out_dir = output_dir / PathConfig.meteo_subdir("nwm_retro", domain)

    for h in _hour_range(start, end):
        dt = start + timedelta(hours=h)
        year = dt.strftime("%Y")
        local_stamp = dt.strftime("%Y%m%d%H")
        remote_stamp = local_stamp + "00" if domain_str == "CONUS" else local_stamp

        url = f"{base_url}/{domain_str}/netcdf/FORCING/{year}/{remote_stamp}.LDASIN_DOMAIN1"
        urls.append(url)
        paths.append(out_dir / f"{local_stamp}.LDASIN_DOMAIN1")

    return urls, paths


def _build_nwm_ana_forcing_urls(
    start: datetime,
    end: datetime,
    output_dir: Path,
    domain: str,
) -> tuple[list[str], list[Path]]:
    """Build URLs for NWM Analysis forcing files from GCS.

    Files are saved locally as ``YYYYMMDDHH.LDASIN_DOMAIN1`` (using the
    *simulation* timestamp ``dt``, not the lagged fetch timestamp) so that:

    1. Multi-day simulations do not overwrite files from different dates
       (the remote filename only contains the hour, not the date).
    2. The ``pre_forcing`` stage can create symlinks with the same
       convention used by ``nwm_retro``, simplifying downstream code.
    """
    base_url = "https://storage.googleapis.com/national-water-model"
    suffix, name = _DOMAIN_MAP_ANA.get(domain, ("", "conus"))

    urls: list[str] = []
    paths: list[Path] = []
    out_dir = output_dir / PathConfig.meteo_subdir("nwm_ana", domain)

    for h in _hour_range(start, end):
        dt = start + timedelta(hours=h)
        # NWM Ana has 2-hour lag
        fetch_dt = dt + timedelta(hours=2)
        date_str = fetch_dt.strftime("%Y%m%d")
        hour_str = f"{fetch_dt.hour:02d}"

        remote_name = f"nwm.t{hour_str}z.analysis_assim.forcing.tm02.{name}.nc"
        url = f"{base_url}/nwm.{date_str}/forcing_analysis_assim{suffix}/{remote_name}"
        urls.append(url)
        # Save with simulation-hour timestamp to avoid overwrites across days.
        local_name = f"{dt.strftime('%Y%m%d%H')}.LDASIN_DOMAIN1"
        paths.append(out_dir / local_name)

    return urls, paths


def _build_nwm_ana_streamflow_urls(
    start: datetime,
    end: datetime,
    output_dir: Path,
    domain: str,
) -> tuple[list[str], list[Path]]:
    """Build URLs for NWM Analysis streamflow (channel_rt) files from GCS."""
    base_url = "https://storage.googleapis.com/national-water-model"
    suffix, name = _DOMAIN_MAP_ANA.get(domain, ("", "conus"))

    urls: list[str] = []
    paths: list[Path] = []
    out_dir = output_dir / PathConfig.streamflow_subdir(domain)

    for h in _hour_range(start, end):
        dt = start + timedelta(hours=h)
        fetch_dt = dt + timedelta(hours=2)
        date_str = fetch_dt.strftime("%Y%m%d")
        hour_str = f"{fetch_dt.hour:02d}"

        if domain == "hawaii":
            # Hawaii sub-hourly naming changed on 2021-04-21:
            #   Before: tm00, tm01, tm02 (3 hourly files)
            #   After:  tm0000..tm0245 (12 fifteen-minute files)
            _hawaii_name_change = datetime(2021, 4, 21)
            if dt < _hawaii_name_change:
                url = (
                    f"{base_url}/nwm.{date_str}/"
                    f"analysis_assim_hawaii/"
                    f"nwm.t{hour_str}z.analysis_assim.channel_rt.tm02.hawaii.nc"
                )
                urls.append(url)
                paths.append(out_dir / f"{dt.strftime('%Y%m%d%H')}00.CHRTOUT_DOMAIN1")
            else:
                for quarter in range(4):
                    minutes = quarter * 15
                    tm_h = 2 - (1 if minutes > 0 else 0)
                    tm_m = (60 - minutes) % 60
                    tm_offset = f"tm{tm_h:02d}{tm_m:02d}"
                    url = (
                        f"{base_url}/nwm.{date_str}/"
                        f"analysis_assim_hawaii/"
                        f"nwm.t{hour_str}z.analysis_assim.channel_rt.{tm_offset}.hawaii.nc"
                    )
                    urls.append(url)
                    paths.append(
                        out_dir / f"{dt.strftime('%Y%m%d%H')}{minutes:02d}.CHRTOUT_DOMAIN1"
                    )
        else:
            url = (
                f"{base_url}/nwm.{date_str}/"
                f"analysis_assim{suffix}/"
                f"nwm.t{hour_str}z.analysis_assim.channel_rt.tm02.{name}.nc"
            )
            urls.append(url)
            paths.append(out_dir / f"{dt.strftime('%Y%m%d%H')}00.CHRTOUT_DOMAIN1")

    return urls, paths


# STOFS naming convention changed on 2023-01-08. The older ``estofs``
# product also stores mesh connectivity in a separate companion file.
STOFS_NAME_CHANGE_DATE = datetime(2023, 1, 8)
STOFS_BASE_URL = "https://noaa-gestofs-pds.s3.amazonaws.com"


def get_stofs_path(start: datetime, output_dir: Path) -> Path:
    """Get the expected local path for a STOFS file.

    Parameters
    ----------
    start : datetime
        Simulation start date.
    output_dir : Path
        Base download directory.

    Returns
    -------
    Path
        Expected path to the STOFS file.
    """
    product = "estofs" if start < STOFS_NAME_CHANGE_DATE else "stofs_2d_glo"
    date_str = start.strftime("%Y%m%d")
    cycle_hour = (start.hour // 6) * 6
    hour_str = f"{cycle_hour:02d}"
    return (
        output_dir
        / PathConfig.COASTAL_SUBDIR
        / "stofs"
        / f"{product}.{date_str}"
        / f"{product}.t{hour_str}z.fields.cwl.nc"
    )


def _build_stofs_urls(
    start: datetime,
    output_dir: Path,
) -> tuple[list[str], list[Path]]:
    """Build URLs for STOFS water level files."""
    product = "estofs" if start < STOFS_NAME_CHANGE_DATE else "stofs_2d_glo"
    date_str = start.strftime("%Y%m%d")
    cycle_hour = (start.hour // 6) * 6
    hour_str = f"{cycle_hour:02d}"

    url = f"{STOFS_BASE_URL}/{product}.{date_str}/{product}.t{hour_str}z.fields.cwl.nc"
    filepath = get_stofs_path(start, output_dir)

    return [url], [filepath]


def _build_stofs_mesh_urls(
    start: datetime,
    output_dir: Path,
) -> tuple[list[str], list[Path]]:
    """Build URLs for the companion file holding STOFS mesh connectivity.

    The pre-2023 ``estofs`` product stores only ``time``/``x``/``y``/
    ``zeta`` in its ``fields.cwl.nc`` file; the ``element`` connectivity
    needed for BILINEAR regridding lives in the companion ``maxele``
    file, which describes the same ADCIRC mesh with identical node
    ordering.  Newer ``stofs_2d_glo`` files carry ``element`` directly,
    so nothing extra is needed and this returns empty lists.
    """
    if start >= STOFS_NAME_CHANGE_DATE:
        return [], []

    date_str = start.strftime("%Y%m%d")
    hour_str = f"{(start.hour // 6) * 6:02d}"
    url = f"{STOFS_BASE_URL}/estofs.{date_str}/estofs.t{hour_str}z.fields.cwl.maxele.nc"
    main = get_stofs_path(start, output_dir)
    return [url], [main.with_name(f"{main.stem}.maxele{main.suffix}")]


def _stofs_cycle_url(cycle: datetime) -> str:
    """Build the STOFS fields-file URL for an exact cycle datetime.

    Unlike :func:`_build_stofs_urls`, *cycle* must already be a valid
    6-hourly cycle time (00/06/12/18Z) -- this does not round.
    """
    product = "estofs" if cycle < STOFS_NAME_CHANGE_DATE else "stofs_2d_glo"
    date_str = cycle.strftime("%Y%m%d")
    hour_str = f"{cycle.hour:02d}"
    return f"{STOFS_BASE_URL}/{product}.{date_str}/{product}.t{hour_str}z.fields.cwl.nc"


def _stofs_cycle_exists(cycle: datetime, timeout: int = 15) -> bool:
    """Check whether a STOFS cycle's fields file actually exists on S3."""
    import requests

    try:
        resp = requests.head(_stofs_cycle_url(cycle), timeout=timeout)
    except requests.RequestException:
        return False
    return resp.status_code == 200


def resolve_stofs_cycle(
    start: datetime,
    *,
    exists: Callable[[datetime], bool] = _stofs_cycle_exists,
    max_lookback_hours: int = 96,
) -> datetime:
    """Find the STOFS cycle closest to (at or before) *start* that actually exists.

    STOFS publishes on a fixed 6-hourly cadence (00/06/12/18Z), but a
    cycle's file isn't necessarily live the instant that cycle time
    passes -- there's a publish lag. Naively rounding *start* down to
    the nearest 6-hourly boundary can therefore name a cycle that
    doesn't exist yet.

    This starts at that naive candidate and walks backward in 6-hour
    steps, checking each one with *exists*, until it finds a cycle that
    is actually published. One rule covers both real usage patterns:

    * **Historical *start*** (well in the past): the naive candidate
      already exists, so this returns immediately on the first check --
      matches the deterministic behavior needed for e.g. calibration
      runs against a known-good archived date.
    * **Live/"now" *start***: the naive candidate is often not published
      yet, so this walks back to whatever the latest actually-available
      cycle is -- exactly "give me the latest forecast."

    Parameters
    ----------
    start : datetime
        Requested simulation start time (naive UTC).
    exists : callable, optional
        ``(datetime) -> bool`` existence check, injectable for testing
        without hitting the network. Defaults to a real HTTP HEAD check
        against S3.
    max_lookback_hours : int, optional
        How far back to search before giving up. Default 96h (16
        cycles / 4 days) comfortably covers normal publish lag while
        still failing fast on a genuine outage or bad date.

    Returns
    -------
    datetime
        The resolved cycle time (always one of 00/06/12/18Z on some day
        at or before *start*).

    Raises
    ------
    ValueError
        No existing cycle was found within *max_lookback_hours*.
    """
    naive_cycle_hour = (start.hour // 6) * 6
    candidate = start.replace(hour=naive_cycle_hour, minute=0, second=0, microsecond=0)
    earliest = candidate - timedelta(hours=max_lookback_hours)

    while candidate >= earliest:
        if exists(candidate):
            return candidate
        candidate -= timedelta(hours=6)

    msg = (
        f"No STOFS cycle found at or before {start:%Y-%m-%d %H:%M} "
        f"within the last {max_lookback_hours}h -- checked back to {earliest:%Y-%m-%d %H:%M}."
    )
    raise ValueError(msg)


# Variables copied verbatim (no time slicing) from the source STOFS file --
# static mesh geometry that regrid_estofs.py needs regardless of which
# timesteps are requested. Kept minimal and matching exactly what
# regrid_estofs._resolve_source_elements / build_unstructured_mesh actually
# read (time and zeta are handled separately since they get sliced).
_STOFS_STATIC_VARS = ("x", "y", "element")


def _download_stofs_time_subset(
    cycle: datetime,
    fetch_start: datetime,
    fetch_end: datetime,
    out_path: Path,
) -> None:
    """Fetch only the needed time window of a STOFS fields file, not the whole thing.

    STOFS fields files are ~12 GB (the full ~186-hour global forecast
    cycle), but a single simulation only ever needs a handful of hours
    out of it. This opens the *remote* file lazily over HTTP range
    requests (the file is HDF5/NETCDF4_CLASSIC, and S3 supports byte
    ranges on any object) and reads only the ``[fetch_start, fetch_end]``
    slice of ``time``/``zeta``, writing a small local NetCDF file with
    that slice plus the static mesh variables (``x``, ``y``, ``element``)
    -- everything ``regrid_estofs.py`` and the SFINCS xarray-based STOFS
    reader actually use. No download of the full remote file ever
    happens; only the bytes needed for this slice cross the network.

    Time values in the written file are the *real* STOFS timestamps
    (not reindexed), so both downstream consumers -- regrid_estofs.py
    (searches for the requested time by value) and the SFINCS
    xarray/HydroMT reader (``.sel(time=...)``) -- work against this
    trimmed file exactly as they would against the full one.
    """
    import h5netcdf
    import netCDF4
    import numpy as np
    from cftime import num2date

    # Plain https:// URL -- fsspec routes this through its generic HTTP
    # backend, not s3fs, so no `anon=True` kwarg here (the bucket is
    # public and needs no credentials either way).
    url = _stofs_cycle_url(cycle)
    with (
        fsspec.open(url) as remote_f,
        h5netcdf.File(remote_f, mode="r", decode_vlen_strings=False) as src,
    ):
        time_var = src.variables["time"]
        units = time_var.attrs["units"].split("!")[0].strip()
        calendar = time_var.attrs.get("calendar", "standard")
        times_raw = time_var[:]
        times = num2date(times_raw, units=units, calendar=calendar)

        # First/last index whose decoded time falls in [fetch_start, fetch_end].
        in_window = [i for i, t in enumerate(times) if fetch_start <= t <= fetch_end]
        if not in_window:
            msg = (
                f"STOFS cycle {cycle:%Y-%m-%d %HZ} has no timesteps in "
                f"[{fetch_start:%Y-%m-%d %H:%M}, {fetch_end:%Y-%m-%d %H:%M}] "
                f"(file covers {times[0]:%Y-%m-%d %H:%M} to {times[-1]:%Y-%m-%d %H:%M})"
            )
            raise ValueError(msg)
        lo, hi = in_window[0], in_window[-1] + 1

        zeta_var = src.variables["zeta"]
        zeta_slice = np.asarray(zeta_var[lo:hi, :])
        time_slice = np.asarray(times_raw[lo:hi])

        static: dict[str, np.ndarray] = {
            name: np.asarray(src.variables[name][:]) for name in _STOFS_STATIC_VARS
        }
        node_count = static["x"].shape[0]

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.unlink(missing_ok=True)
        with netCDF4.Dataset(tmp_path, mode="w", format="NETCDF4_CLASSIC") as dst:
            dst.createDimension("time", None)
            dst.createDimension("node", node_count)
            dst.createDimension("nele", static["element"].shape[0])
            dst.createDimension("nvertex", static["element"].shape[1])

            from coastal_calibration._nc_io import create_var, write_var

            t_out = create_var(dst, "time", "f8", ("time",), attrs=dict(time_var.attrs))
            write_var(t_out, time_slice)

            x_out = create_var(dst, "x", "f8", ("node",), attrs=dict(src.variables["x"].attrs))
            write_var(x_out, static["x"])
            y_out = create_var(dst, "y", "f8", ("node",), attrs=dict(src.variables["y"].attrs))
            write_var(y_out, static["y"])
            elem_out = create_var(
                dst,
                "element",
                "i4",
                ("nele", "nvertex"),
                attrs=dict(src.variables["element"].attrs),
            )
            write_var(elem_out, static["element"])

            # _FillValue must be set only via createVariable's fill_value=
            # kwarg -- netCDF4 rejects a later setncatts() that includes it,
            # so it's popped out of the copied source attrs here.
            zeta_attrs = dict(zeta_var.attrs)
            fill_value = zeta_attrs.pop("_FillValue", None)
            zeta_out = create_var(
                dst,
                "zeta",
                "f4",
                ("time", "node"),
                attrs=zeta_attrs,
                fill_value=fill_value,
            )
            write_var(zeta_out, zeta_slice)

        tmp_path.replace(out_path)


def _build_glofs_urls(
    start: datetime,
    end: datetime,
    output_dir: Path,
    model: str,
) -> tuple[list[str], list[Path]]:
    """Build URLs for GLOFS water level files."""
    base_url = (
        "https://www.ncei.noaa.gov/data/"
        "operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access"
    )
    model_dir = _GLOFS_MODEL_DIRS.get(model, _GLOFS_MODEL_DIRS["leofs"])

    urls: list[str] = []
    paths: list[Path] = []
    out_dir = output_dir / PathConfig.COASTAL_SUBDIR / "glofs"

    for h in _hour_range(start, end):
        dt = start + timedelta(hours=h)
        date_str = dt.strftime("%Y%m%d")
        year = dt.strftime("%Y")
        month = dt.strftime("%m")

        cycle_hour = (dt.hour // 6) * 6
        cycle = f"t{cycle_hour:02d}z"
        suffix = f"n{dt.hour % 6:03d}"

        filename = f"{model}.{cycle}.{date_str}.fields.{suffix}.nc"
        url = f"{base_url}/{model_dir}/{year}/{month}/{filename}"
        urls.append(url)
        paths.append(out_dir / filename)

    return urls, paths


def _execute_download(
    urls: list[str],
    file_paths: list[Path],
    source_name: str,
    timeout: int,
    raise_on_error: bool,
) -> DownloadResult:
    """Download *urls* to *file_paths* atomically via ``.tmp`` rename.

    Each download writes to ``<path>.tmp`` first and is renamed onto the
    final path only after the byte stream completes successfully. A
    crash, network drop, or kill -9 mid-download leaves only ``.tmp``
    debris that the next call cleans up — the canonical ``file_paths``
    are never partial.

    Files that already exist at the final path with non-zero size are
    skipped without contacting the remote. The atomic rename above
    guarantees that a present final file is fully written, so a stat()
    check is sufficient to treat it as cached. This keeps re-runs cheap
    on shared filesystems where every HTTP HEAD costs a NAT round-trip.
    """
    if not urls:
        return DownloadResult(source=source_name)

    result = DownloadResult(
        source=source_name,
        total_files=len(urls),
        file_paths=list(file_paths),
    )

    pending_urls: list[str] = []
    pending_finals: list[Path] = []
    for url, final_path in zip(urls, file_paths, strict=True):
        if final_path.exists() and final_path.stat().st_size > 0:
            result.successful += 1
        else:
            pending_urls.append(url)
            pending_finals.append(final_path)

    if not pending_urls:
        return result

    tmp_paths = [p.with_suffix(p.suffix + ".tmp") for p in pending_finals]
    for tmp_path in tmp_paths:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear any stale .tmp leftover from a previous interrupted run.
        tmp_path.unlink(missing_ok=True)

    # 8 mb chunk size is reasonable for large files like STOFS (~12 GB)
    # while not causing too much overhead for smaller files.
    chunk_size = 8 * 1024 * 1024
    try:
        download(
            pending_urls,
            tmp_paths,
            timeout=timeout,
            raise_status=raise_on_error,
            chunk_size=chunk_size,
        )
    except Exception as e:
        result.errors.append(str(e))

    for url, tmp_path, final_path in zip(pending_urls, tmp_paths, pending_finals, strict=True):
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            result.failed += 1
            if not result.errors:
                result.errors.append(f"Failed to download: {url}")
            tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.replace(final_path)
            result.successful += 1

    return result


def validate_date_ranges(
    start_time: datetime,
    end_time: datetime,
    meteo_source: str,
    coastal_source: str,
    domain: str,
) -> list[str]:
    """Validate that requested dates are within available ranges."""
    errors: list[str] = []

    meteo_range = get_date_range(meteo_source, domain)
    if meteo_range:
        error = meteo_range.validate(start_time, end_time)
        if error:
            errors.append(error)

    if coastal_source != "harmonic":
        coastal_range = get_date_range(coastal_source, domain)
        if coastal_range:
            error = coastal_range.validate(start_time, end_time)
            if error:
                errors.append(error)

    return errors


def _log_summary(results: DownloadResults) -> None:
    """Log download summary."""
    total_files = 0
    total_success = 0
    total_failed = 0

    for result in results:
        status = "OK" if not result.errors else "ERRORS"
        logger.info(
            "  %s: %d/%d [%s]",
            result.source,
            result.successful,
            result.total_files,
            status,
        )
        total_files += result.total_files
        total_success += result.successful
        total_failed += result.failed

        for error in result.errors:
            logger.error("    %s", error)

    logger.info(
        "  Total: %d/%d (failed: %d)",
        total_success,
        total_files,
        total_failed,
    )


def _download_meteo(
    meteo_source: str,
    start: datetime,
    end: datetime,
    out_dir: Path,
    domain: str,
    timeout: int,
    raise_on_error: bool,
) -> DownloadResult:
    """Resolve and execute the meteorological forcing download."""
    if meteo_source == "ngen_forecast":
        # Forcing is pre-generated on disk by the ngen forecast engine and
        # read from paths.forecast_meteo_file — there is nothing to fetch.
        return DownloadResult(source="meteo/ngen_forecast")
    if meteo_source == "nwm_retro":
        urls, paths = _build_nwm_retro_forcing_urls(start, end, out_dir, domain)
        # PRVI and Alaska Retrospective forcing ships with no georeferencing,
        # so record the domain's grid alongside it while we are online.
        write_nwm_grid_sidecar(out_dir / PathConfig.meteo_subdir("nwm_retro", domain), domain)
    else:
        urls, paths = _build_nwm_ana_forcing_urls(start, end, out_dir, domain)
    return _execute_download(urls, paths, f"meteo/{meteo_source}", timeout, raise_on_error)


def _download_hydro(
    meteo_source: str,
    hydro_source: str,
    start: datetime,
    end: datetime,
    out_dir: Path,
    domain: str,
    timeout: int,
    raise_on_error: bool,
) -> DownloadResult:
    """Resolve and execute the hydrological (streamflow) download."""
    if meteo_source == "ngen_forecast":
        # Streamflow for the forecast pipeline comes from t-route output,
        # which is not wired up yet — skip the hydro download for now.
        return DownloadResult(source=f"hydro/{hydro_source}")
    if hydro_source == "ngen":
        return DownloadResult(
            source=f"hydro/{hydro_source}",
            errors=["NGEN hydrology source not yet supported"],
        )
    if meteo_source == "nwm_retro":
        # nwm_retro streamflow is read directly from the S3 Zarr store at
        # runtime — no file download needed.
        return DownloadResult(source=f"hydro/{hydro_source}")
    urls, paths = _build_nwm_ana_streamflow_urls(start, end, out_dir, domain)
    return _execute_download(urls, paths, f"hydro/{hydro_source}", timeout, raise_on_error)


def download_data(
    start_time: datetime | str,
    end_time: datetime | str,
    output_dir: Path | str,
    domain: Domain,
    *,
    meteo_source: MeteoSource = "nwm_retro",
    hydro_source: HydroSource = "nwm",
    coastal_source: CoastalSource = "stofs",
    glofs_model: GLOFSModel = "leofs",
    tidal_atlas_path: Path | str | None = None,
    timeout: int = 600,
    raise_on_error: bool = False,
) -> DownloadResults:
    """Download meteorological, hydrological, and coastal data.

    Parameters
    ----------
    start_time : str or datatime.datetime
        Start of simulation period (datetime or ISO format string).
    end_time : str or datatime.datetime
        End of simulation period (datetime or ISO format string).
    output_dir : str or pathlib.Path
        Root directory for downloaded data.
    domain : {"conus", "hawaii", "prvi", "atlgulf", "pacific"}
        Model domain: ``conus``, ``hawaii``, ``prvi``, ``atlgulf``,
        or ``pacific``.
    meteo_source : {"nwm_retro", "nwm_ana"}, optional
        Meteorological data source: ``nwm_retro`` or ``nwm_ana``.
        Defaults to ``nwm_retro``.
    hydro_source : {"nwm", "ngen"}, optional
        Hydrology data source: ``nwm`` or ``ngen``.
        Defaults to ``nwm``.
    coastal_source : {"harmonic", "stofs", "glofs"}, optional
        Coastal water level source: ``harmonic`` (predict locally from
        a tidal atlas), ``stofs``, or ``glofs``. Defaults to ``stofs``.
    glofs_model : {"leofs", "loofs", "lsofs", "lmhofs"}, optional
        GLOFS model (only used if ``coastal_source`` is ``glofs``):
        ``leofs``, ``loofs``, ``lsofs``, or ``lmhofs``.
        Defaults to ``leofs``.
    tidal_atlas_path : str or pathlib.Path, optional
        Local path to the tidal atlas directory. Required when
        ``coastal_source == "harmonic"``; the atlas cannot be
        downloaded. Defaults to ``None``.
    timeout : int, optional
        Download timeout in seconds, defaults to 600.
    raise_on_error : bool, optional
        Whether to raise exceptions on download failures.
        Defaults to ``False``.

    Returns
    -------
    DownloadResults
        Results for each data source (meteo, hydro, coastal).

    Examples
    --------
    >>> results = download_data(
    ...     "2021-06-11",
    ...     "2021-06-12",
    ...     "./data/downloads",
    ...     "pacific",
    ...     meteo_source="nwm_retro",
    ...     coastal_source="stofs",
    ... )
    """
    start = to_naive_utc(pd.to_datetime(start_time).to_pydatetime())
    end = to_naive_utc(pd.to_datetime(end_time).to_pydatetime())
    out_dir = Path(output_dir)
    atlas_path = Path(tidal_atlas_path) if tidal_atlas_path else None

    errors = validate_date_ranges(start, end, meteo_source, coastal_source, domain)
    if errors:
        raise ValueError("Date range validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    meteo_result = _download_meteo(
        meteo_source, start, end, out_dir, domain, timeout, raise_on_error
    )
    hydro_result = _download_hydro(
        meteo_source, hydro_source, start, end, out_dir, domain, timeout, raise_on_error
    )

    if coastal_source == "harmonic":
        if atlas_path is None:
            coastal_result = DownloadResult(
                source="coastal/harmonic",
                total_files=1,
                failed=1,
                errors=["Tidal atlas requires local installation. Set tidal_atlas_path."],
            )
        elif not atlas_path.exists():
            coastal_result = DownloadResult(
                source="coastal/harmonic",
                total_files=1,
                failed=1,
                errors=[f"Tidal atlas not found: {atlas_path}"],
            )
        else:
            coastal_result = DownloadResult(
                source="coastal/harmonic",
                total_files=1,
                successful=1,
                file_paths=[atlas_path],
            )
    elif coastal_source == "stofs":
        # STOFS fields file is ~12 GB for the full cycle, but a simulation
        # only ever needs a handful of hours out of it -- fetch just the
        # [start, end+1h] window (the +1h matches regrid_estofs.py's own
        # total_hours = length_hrs + 1 convention) from whichever cycle is
        # actually the latest available at/before start, rather than
        # downloading the whole thing. See resolve_stofs_cycle /
        # _download_stofs_time_subset. The local path is unchanged
        # (get_stofs_path(start, ...)) so downstream lookups in
        # schism/boundary.py and sfincs/data_catalog.py need no changes --
        # only *where the data is sourced from* is resolution-aware, not
        # where it lands on disk.
        stofs_out_path = get_stofs_path(start, out_dir)
        if stofs_out_path.exists() and stofs_out_path.stat().st_size > 0:
            coastal_result = DownloadResult(
                source="coastal/stofs",
                total_files=1,
                successful=1,
                file_paths=[stofs_out_path],
            )
        else:
            coastal_result = DownloadResult(
                source="coastal/stofs", total_files=1, file_paths=[stofs_out_path]
            )
            try:
                cycle = resolve_stofs_cycle(start)
                logger.info("STOFS: using cycle %s for start %s", cycle, start)
                _download_stofs_time_subset(
                    cycle, start, end + timedelta(hours=1), stofs_out_path
                )
                coastal_result.successful = 1
            except Exception as e:  # noqa: BLE001 -- surfaced via DownloadResult, not raised here
                coastal_result.failed = 1
                coastal_result.errors.append(str(e))
                if raise_on_error:
                    raise

        # Best effort: regrid_estofs synthesizes a triangulation when this
        # companion mesh file is absent, so a failure is not fatal. The
        # lists are empty for the newer product, where this is a no-op.
        mesh_urls, mesh_paths = _build_stofs_mesh_urls(start, out_dir)
        _execute_download(
            mesh_urls, mesh_paths, "coastal/stofs-mesh", max(timeout, 3600), raise_on_error=False
        )
    else:
        urls, paths = _build_glofs_urls(start, end, out_dir, glofs_model)
        coastal_result = _execute_download(urls, paths, "coastal/glofs", timeout, raise_on_error)

    results = DownloadResults(meteo=meteo_result, hydro=hydro_result, coastal=coastal_result)
    _log_summary(results)
    return results
