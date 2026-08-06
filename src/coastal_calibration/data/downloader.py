"""Async data downloader for coastal model calibration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

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
            today = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
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

    if meteo_source == "nwm_retro":
        urls, paths = _build_nwm_retro_forcing_urls(start, end, out_dir, domain)
        # PRVI and Alaska Retrospective forcing ships with no georeferencing,
        # so record the domain's grid alongside it while we are online.
        write_nwm_grid_sidecar(out_dir / PathConfig.meteo_subdir("nwm_retro", domain), domain)
    else:
        urls, paths = _build_nwm_ana_forcing_urls(start, end, out_dir, domain)
    meteo_result = _execute_download(urls, paths, f"meteo/{meteo_source}", timeout, raise_on_error)

    if hydro_source == "ngen":
        hydro_result = DownloadResult(
            source=f"hydro/{hydro_source}",
            errors=["NGEN hydrology source not yet supported"],
        )
    elif meteo_source == "nwm_retro":
        # nwm_retro streamflow is read directly from the S3 Zarr store
        # at runtime — no file download needed.
        hydro_result = DownloadResult(source=f"hydro/{hydro_source}")
    else:
        urls, paths = _build_nwm_ana_streamflow_urls(start, end, out_dir, domain)
        hydro_result = _execute_download(
            urls, paths, f"hydro/{hydro_source}", timeout, raise_on_error
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
        urls, paths = _build_stofs_urls(start, out_dir)
        # STOFS fields file is ~12 GB -- give it a generous timeout.
        stofs_timeout = max(timeout, 3600)
        coastal_result = _execute_download(
            urls, paths, "coastal/stofs", stofs_timeout, raise_on_error
        )
        # Best effort: regrid_estofs synthesizes a triangulation when this
        # companion mesh file is absent, so a failure is not fatal. The
        # lists are empty for the newer product, where this is a no-op.
        mesh_urls, mesh_paths = _build_stofs_mesh_urls(start, out_dir)
        _execute_download(
            mesh_urls, mesh_paths, "coastal/stofs-mesh", stofs_timeout, raise_on_error=False
        )
    else:
        urls, paths = _build_glofs_urls(start, end, out_dir, glofs_model)
        coastal_result = _execute_download(urls, paths, "coastal/glofs", timeout, raise_on_error)

    results = DownloadResults(meteo=meteo_result, hydro=hydro_result, coastal=coastal_result)
    _log_summary(results)
    return results
