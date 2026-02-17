"""Async data downloader for coastal model calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

from tiny_retriever import download

from coastal_calibration.config.schema import (
    BoundarySource,
    CoastalDomain,
    MeteoSource,
    PathConfig,
)
from coastal_calibration.utils.logging import logger
from coastal_calibration.utils.time import iter_hours as _iter_hours
from coastal_calibration.utils.time import parse_datetime as _parse_datetime

HydroSource = Literal["nwm", "ngen"]
CoastalSource = Literal["stofs", "tpxo", "glofs"]
Domain = Literal["conus", "hawaii", "prvi", "atlgulf", "pacific"]
GLOFSModel = Literal["leofs", "loofs", "lsofs", "lmhofs"]


@dataclass
class DateRange:
    """Date range for a data source."""

    start: datetime
    end: datetime | None
    description: str

    def validate(self, start: datetime, end: datetime) -> str | None:
        """Validate that the requested period falls within the available range."""
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
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            if start > today:
                return (
                    f"{self.description} data is available from "
                    f"{self.start.strftime('%Y-%m-%d')} to present. "
                    f"Requested start date {start.strftime('%Y-%m-%d')} is in the future."
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
        Coastal boundary source (e.g., ``stofs``, ``tpxo``).
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

    if coastal_source == "tpxo":
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
        or ``"pacific"``.

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
            ("nwm_ana", "tpxo"),
            ("nwm_retro", "stofs"),
            ("nwm_retro", "tpxo"),
        ]
    else:
        combos: list[tuple[MeteoSource, BoundarySource]] = [
            ("nwm_retro", "stofs"),
            ("nwm_ana", "stofs"),
            ("nwm_retro", "tpxo"),
            ("nwm_ana", "tpxo"),
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


# Domain mappings for URL builders
_DOMAIN_MAP_RETRO = {
    "conus": "CONUS",
    "atlgulf": "CONUS",
    "pacific": "CONUS",
    "hawaii": "Hawaii",
    "prvi": "PR",
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
    out_dir = output_dir / PathConfig.METEO_SUBDIR / "nwm_retro"

    for dt in _iter_hours(start, end):
        year = dt.strftime("%Y")
        local_stamp = dt.strftime("%Y%m%d%H")
        remote_stamp = local_stamp + "00" if domain_str == "CONUS" else local_stamp

        url = f"{base_url}/{domain_str}/netcdf/FORCING/{year}/{remote_stamp}.LDASIN_DOMAIN1"
        urls.append(url)
        paths.append(out_dir / f"{local_stamp}.LDASIN_DOMAIN1")

    return urls, paths


def _build_nwm_retro_streamflow_urls(
    start: datetime,
    end: datetime,
    output_dir: Path,
    domain: str,
) -> tuple[list[str], list[Path]]:
    """Build URLs for NWM Retrospective streamflow (CHRTOUT) files."""
    base_url = "https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com"
    domain_str = _DOMAIN_MAP_RETRO.get(domain, "CONUS")

    urls: list[str] = []
    paths: list[Path] = []
    out_dir = output_dir / PathConfig.STREAMFLOW_SUBDIR / "nwm_retro"

    for dt in _iter_hours(start, end):
        year = dt.strftime("%Y")
        stamp = dt.strftime("%Y%m%d%H") + "00"

        url = f"{base_url}/{domain_str}/netcdf/CHRTOUT/{year}/{stamp}.CHRTOUT_DOMAIN1"
        urls.append(url)
        paths.append(out_dir / f"{stamp}.CHRTOUT_DOMAIN1")

        # Hawaii has 15-minute sub-hourly data
        if domain == "hawaii":
            for minute in (15, 30, 45):
                stamp_sub = dt.strftime("%Y%m%d%H") + f"{minute:02d}"
                url_sub = f"{base_url}/Hawaii/netcdf/CHRTOUT/{year}/{stamp_sub}.CHRTOUT_DOMAIN1"
                urls.append(url_sub)
                paths.append(out_dir / f"{stamp_sub}.CHRTOUT_DOMAIN1")

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
    out_dir = output_dir / PathConfig.METEO_SUBDIR / "nwm_ana"

    for dt in _iter_hours(start, end):
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
    out_dir = output_dir / PathConfig.HYDRO_SUBDIR / "nwm"

    for dt in _iter_hours(start, end):
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
    name_change_date = datetime(2023, 1, 8)
    product = "estofs" if start < name_change_date else "stofs_2d_glo"
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
    base_url = "https://noaa-gestofs-pds.s3.amazonaws.com"
    # STOFS naming convention changed on 2023-01-08
    name_change_date = datetime(2023, 1, 8)

    product = "estofs" if start < name_change_date else "stofs_2d_glo"
    date_str = start.strftime("%Y%m%d")
    cycle_hour = (start.hour // 6) * 6
    hour_str = f"{cycle_hour:02d}"

    url = f"{base_url}/{product}.{date_str}/{product}.t{hour_str}z.fields.cwl.nc"
    filepath = get_stofs_path(start, output_dir)

    return [url], [filepath]


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

    for dt in _iter_hours(start, end):
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
    """Download all *urls* to *file_paths* using tiny_retriever."""
    if not urls:
        return DownloadResult(source=source_name)

    result = DownloadResult(
        source=source_name,
        total_files=len(urls),
        file_paths=list(file_paths),
    )

    for path in file_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    # 8 mb chunk size is reasonable for large files like STOFS (~12 GB)
    # while not causing too much overhead for smaller files.
    chunk_size = 8 * 1024 * 1024
    try:
        download(
            urls, file_paths, timeout=timeout, raise_status=raise_on_error, chunk_size=chunk_size
        )
    except Exception as e:
        result.errors.append(str(e))

    for url, path in zip(urls, file_paths, strict=False):
        if not path.exists() or path.stat().st_size == 0:
            result.failed += 1
            if not result.errors:
                result.errors.append(f"Failed to download: {url}")
            if path.exists():
                path.unlink()
        else:
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

    if coastal_source != "tpxo":
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
            "%s: %d/%d [%s]",
            result.source,
            result.successful,
            result.total_files,
            status,
        )
        total_files += result.total_files
        total_success += result.successful
        total_failed += result.failed

        for error in result.errors:
            logger.error("  %s", error)

    logger.info(
        "Total: %d/%d (failed: %d)",
        total_success,
        total_files,
        total_failed,
    )


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
    tpxo_local_path: Path | str | None = None,
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
    coastal_source : {"tpxo", "stofs", "glofs"}, optional
        Coastal water level source: ``tpxo``, ``stofs``, or ``glofs``.
        Defaults to ``stofs``.
    glofs_model : {"leofs", "loofs", "lsofs", "lmhofs"}, optional
        GLOFS model (only used if ``coastal_source`` is ``glofs``):
        ``leofs``, ``loofs``, ``lsofs``, or ``lmhofs``.
        Defaults to ``leofs``.
    tpxo_local_path : str or pathlib.Path, optional
        Local path to TPXO data (TPXO cannot be downloaded).
        Defaults to ``None``.
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
    start = _parse_datetime(start_time)
    end = _parse_datetime(end_time)
    out_dir = Path(output_dir)
    tpxo_path = Path(tpxo_local_path) if tpxo_local_path else None

    errors = validate_date_ranges(start, end, meteo_source, coastal_source, domain)
    if errors:
        raise ValueError("Date range validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    if meteo_source == "nwm_retro":
        urls, paths = _build_nwm_retro_forcing_urls(start, end, out_dir, domain)
    else:
        urls, paths = _build_nwm_ana_forcing_urls(start, end, out_dir, domain)
    meteo_result = _execute_download(urls, paths, f"meteo/{meteo_source}", timeout, raise_on_error)

    if hydro_source == "ngen":
        hydro_result = DownloadResult(
            source=f"hydro/{hydro_source}",
            errors=["NGEN hydrology source not yet supported"],
        )
    elif meteo_source == "nwm_retro":
        urls, paths = _build_nwm_retro_streamflow_urls(start, end, out_dir, domain)
        hydro_result = _execute_download(
            urls, paths, f"hydro/{hydro_source}", timeout, raise_on_error
        )
    else:
        urls, paths = _build_nwm_ana_streamflow_urls(start, end, out_dir, domain)
        hydro_result = _execute_download(
            urls, paths, f"hydro/{hydro_source}", timeout, raise_on_error
        )

    if coastal_source == "tpxo":
        if tpxo_path is None:
            coastal_result = DownloadResult(
                source="coastal/tpxo",
                total_files=1,
                failed=1,
                errors=["TPXO data requires local installation. Set tpxo_local_path."],
            )
        elif not tpxo_path.exists():
            coastal_result = DownloadResult(
                source="coastal/tpxo",
                total_files=1,
                failed=1,
                errors=[f"TPXO local path not found: {tpxo_path}"],
            )
        else:
            coastal_result = DownloadResult(
                source="coastal/tpxo",
                total_files=1,
                successful=1,
                file_paths=[tpxo_path],
            )
    elif coastal_source == "stofs":
        urls, paths = _build_stofs_urls(start, out_dir)
        # STOFS fields file is ~12 GB -- give it a generous timeout.
        stofs_timeout = max(timeout, 3600)
        coastal_result = _execute_download(
            urls, paths, "coastal/stofs", stofs_timeout, raise_on_error
        )
    else:
        urls, paths = _build_glofs_urls(start, end, out_dir, glofs_model)
        coastal_result = _execute_download(urls, paths, "coastal/glofs", timeout, raise_on_error)

    results = DownloadResults(meteo=meteo_result, hydro=hydro_result, coastal=coastal_result)
    _log_summary(results)
    return results
