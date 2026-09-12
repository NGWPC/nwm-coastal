"""NOAA Great Lakes Operational Forecast System (GLOFS) water levels.

GLOFS nowcast files carry full 3-D model output and run up to ~180 MB per
hour, of which the boundary forcing needs only the surface elevation. So
rather than downloading whole files, this module reads just ``time``,
``lon``, ``lat`` and ``zeta`` from each remote file, caches that slice
per hour, and merges the slices into one GeoDataset that both the SCHISM
and SFINCS boundary stages read.

The NCEI archive has three layouts, chosen by the *cycle* date in the
filename (all verified against NCEI in September 2026):

====  =============  =====================  ==================================  =========
Era   Lakes          Cycle dates            Filename                            Format
====  =============  =====================  ==================================  =========
POM   loofs, lsofs   before 2022-10-20      glofs.{m}.fields.nowcast.{d}.t{c}z  netCDF3
B     all            before 2024-09-09      nos.{m}.fields.n{h}.{d}.t{c}z       netCDF3
C     all            2024-09-09 onward      {m}.t{c}z.{d}.fields.n{h}           HDF5
====  =============  =====================  ==================================  =========

Nowcast cycles look *back*: cycle ``t06z`` covers 00-06 UTC. An FVCOM
file ``n{h}`` in cycle ``t{c}z`` is valid at ``c - 6 + h``; a POM file
holds the six hours ending at its cycle. Water levels are relative to the
lake's low-water datum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import numpy as np

from coastal_calibration.config.schema import PathConfig
from coastal_calibration.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from numpy.typing import NDArray

    from coastal_calibration.config.schema import GLOFSModel
    from coastal_calibration.data.downloader import DownloadResult

__all__ = [
    "GLOFS_BASE_URL",
    "GlofsSource",
    "ensure_glofs_waterlevel",
    "fetch_glofs",
    "glofs_sources",
    "glofs_waterlevel_path",
]

GLOFS_BASE_URL = (
    "https://www.ncei.noaa.gov/oa/prod-model/"
    "operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access"
)

_LAKE_DIRS: dict[str, str] = {
    "leofs": "lake-erie-operational-forecast-system-leofs",
    "lmhofs": "lake-michigan-and-huron-operational-forecast-system-lmhofs",
    "loofs": "lake-ontario-operational-forecast-system-loofs",
    "lsofs": "lake-superior-operational-forecast-system-lsofs",
}

# Cycle dates at which each layout starts (see the module docstring).
_HDF5_START = date(2024, 9, 9)
_FVCOM_START: dict[str, date] = {"loofs": date(2022, 10, 20), "lsofs": date(2022, 10, 20)}

_HOUR = timedelta(hours=1)
_HTTP_BLOCK_SIZE = 4 * 2**20  # few large range requests beat many small ones
_RETRIES = 3

FileFormat = Literal["pom", "netcdf3", "hdf5"]


@dataclass(frozen=True)
class GlofsSource:
    """Where one hour of GLOFS water level lives."""

    valid_time: datetime
    url: str
    step: int
    fmt: FileFormat


def glofs_sources(model: GLOFSModel, start: datetime, end: datetime) -> list[GlofsSource]:
    """Return the file, and time step within it, for each hour in ``[start, end)``."""
    if model not in _LAKE_DIRS:
        raise ValueError(f"Unknown GLOFS model {model!r}; expected one of {', '.join(_LAKE_DIRS)}")
    start = start.replace(minute=0, second=0, microsecond=0)
    n_hours = int((end - start).total_seconds() // 3600)
    return [_source_for_hour(model, start + k * _HOUR) for k in range(n_hours)]


def _source_for_hour(model: str, hour: datetime) -> GlofsSource:
    offset = hour.hour % 6
    fvcom_cycle = hour - timedelta(hours=offset) + timedelta(hours=6)
    fvcom_start = _FVCOM_START.get(model)

    if fvcom_start is not None and fvcom_cycle.date() < fvcom_start:
        # POM files hold the six hours ending at the cycle.
        cycle = hour if offset == 0 else fvcom_cycle
        name = f"glofs.{model}.fields.nowcast.{cycle:%Y%m%d}.t{cycle:%H}z.nc"
        step = 5 - int((cycle - hour).total_seconds() // 3600)
        return GlofsSource(hour, _url(model, cycle, name), step, "pom")

    cycle = fvcom_cycle
    if cycle.date() < _HDF5_START:
        name = f"nos.{model}.fields.n{offset:03d}.{cycle:%Y%m%d}.t{cycle:%H}z.nc"
        return GlofsSource(hour, _url(model, cycle, name), 0, "netcdf3")
    name = f"{model}.t{cycle:%H}z.{cycle:%Y%m%d}.fields.n{offset:03d}.nc"
    return GlofsSource(hour, _url(model, cycle, name), 0, "hdf5")


def _url(model: str, cycle: datetime, name: str) -> str:
    return f"{GLOFS_BASE_URL}/{_LAKE_DIRS[model]}/{cycle:%Y}/{cycle:%m}/{name}"


# ---------------------------------------------------------------------------
# Remote reads
# ---------------------------------------------------------------------------


def _round_hour(t: datetime) -> datetime:
    return (t + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)


def _to_datetimes(values: NDArray[np.floating], units: str) -> list[datetime]:
    import netCDF4

    dts = netCDF4.num2date(
        values, units, only_use_cftime_datetimes=False, only_use_python_datetimes=True
    )
    return [_round_hour(dt) for dt in np.atleast_1d(dts)]


def _read_remote(
    url: str, fmt: FileFormat, steps: list[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float32], list[datetime]]:
    """Read ``lon``, ``lat``, ``zeta[steps]`` and times from one remote file.

    Returns 1-D coordinates with longitude in [-180, 180) and ``zeta`` of
    shape ``(len(steps), n_points)``. POM land cells are dropped.
    """
    if fmt == "hdf5":
        import fsspec
        import h5py

        with (
            fsspec.open(url, "rb", block_size=_HTTP_BLOCK_SIZE, cache_type="blockcache") as fh,
            h5py.File(fh, "r") as f,
        ):
            lon = np.asarray(f["lon"][:], dtype=np.float64)
            lat = np.asarray(f["lat"][:], dtype=np.float64)
            zeta = np.asarray(f["zeta"][steps, :], dtype=np.float32)
            units = f["time"].attrs["units"]
            units = units.decode() if isinstance(units, bytes) else str(units)
            times = _to_datetimes(f["time"][steps], units)
    else:
        import netCDF4

        with netCDF4.Dataset(f"{url}#mode=bytes") as ds:
            lon = np.asarray(ds["lon"][:], dtype=np.float64).ravel()
            lat = np.asarray(ds["lat"][:], dtype=np.float64).ravel()
            raw = ds["zeta"][steps]
            times = _to_datetimes(ds["time"][steps], ds["time"].units)
        raw = np.ma.masked_invalid(raw).reshape(len(steps), -1)
        if fmt == "pom":
            wet = ~np.ma.getmaskarray(raw).any(axis=0)
            lon, lat, raw = lon[wet], lat[wet], raw[:, wet]
        zeta = np.ma.filled(raw.astype(np.float32), np.nan)

    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, lat, zeta, times


def _read_with_retry(
    url: str, fmt: FileFormat, steps: list[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float32], list[datetime]]:
    for attempt in range(1, _RETRIES + 1):
        try:
            return _read_remote(url, fmt, steps)
        except FileNotFoundError:
            raise
        except OSError as exc:
            # netCDF-C reports a missing remote file as a generic OSError.
            if "404" in str(exc) or "No such file" in str(exc) or attempt == _RETRIES:
                raise
            logger.debug(f"GLOFS read failed ({exc}); retry {attempt}/{_RETRIES - 1}")
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Per-hour cache
# ---------------------------------------------------------------------------


def _cache_dir(download_dir: Path, model: str) -> Path:
    return download_dir / PathConfig.COASTAL_SUBDIR / "glofs" / model


def _hour_path(download_dir: Path, model: str, hour: datetime) -> Path:
    return _cache_dir(download_dir, model) / f"{model}.{hour:%Y%m%d%H}.nc"


def _write_hour(
    path: Path,
    lon: NDArray[np.float64],
    lat: NDArray[np.float64],
    zeta: NDArray[np.float32],
    hour: datetime,
    url: str,
) -> None:
    import netCDF4

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".nc.part")
    with netCDF4.Dataset(tmp, "w") as ds:
        ds.createDimension("node", zeta.size)
        ds.createVariable("lon", "f8", ("node",))[:] = lon
        ds.createVariable("lat", "f8", ("node",))[:] = lat
        ds.createVariable("zeta", "f4", ("node",), zlib=True)[:] = zeta
        ds.valid_time = hour.isoformat()
        ds.source_url = url
    tmp.replace(path)


def _read_hour(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float32]]:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        return ds["lon"][:].data, ds["lat"][:].data, ds["zeta"][:].data


def fetch_glofs(
    model: GLOFSModel,
    start: datetime,
    end: datetime,
    download_dir: Path,
    raise_on_error: bool = True,
) -> DownloadResult:
    """Fetch GLOFS water levels for each hour in ``[start, end)`` into the cache.

    Hours already cached are skipped, so an interrupted fetch resumes.
    Each POM file is read once for all the hours it holds.
    """
    from coastal_calibration.data.downloader import DownloadResult

    sources = glofs_sources(model, start, end)
    result = DownloadResult(source="coastal/glofs", total_files=len(sources))

    pending: dict[str, list[GlofsSource]] = {}
    for src in sources:
        path = _hour_path(download_dir, model, src.valid_time)
        if path.exists():
            result.successful += 1
            result.file_paths.append(path)
        else:
            pending.setdefault(src.url, []).append(src)

    n_pending = sum(len(group) for group in pending.values())
    if n_pending:
        logger.info(f"GLOFS {model}: fetching {n_pending} hour(s) from NCEI")

    done = 0
    for url, group in pending.items():
        try:
            lon, lat, zeta, times = _read_with_retry(url, group[0].fmt, [s.step for s in group])
        except Exception as exc:
            hours = ", ".join(f"{s.valid_time:%Y-%m-%d %H:%M}" for s in group)
            msg = f"GLOFS {model} {hours}: could not read {url} ({exc})"
            result.failed += len(group)
            result.errors.append(msg)
            if raise_on_error:
                raise RuntimeError(msg) from exc
            continue

        for src, z, t in zip(group, zeta, times, strict=True):
            if t != src.valid_time:
                msg = (
                    f"GLOFS {url} step {src.step} is valid at {t:%Y-%m-%d %H:%M}, "
                    f"expected {src.valid_time:%Y-%m-%d %H:%M}"
                )
                raise ValueError(msg)
            path = _hour_path(download_dir, model, src.valid_time)
            _write_hour(path, lon, lat, z, src.valid_time, url)
            result.successful += 1
            result.file_paths.append(path)

        done += len(group)
        if done % 24 < len(group) or done == n_pending:
            logger.info(f"GLOFS {model}: {done}/{n_pending} hour(s) fetched")

    return result


# ---------------------------------------------------------------------------
# Consolidated GeoDataset
# ---------------------------------------------------------------------------


def glofs_waterlevel_path(
    download_dir: Path, model: str, start: datetime, duration_hours: int
) -> Path:
    """Path of the merged water-level file for one simulation window."""
    end = start + timedelta(hours=duration_hours)
    return (
        download_dir
        / PathConfig.COASTAL_SUBDIR
        / "glofs"
        / f"glofs_{model}_{start:%Y%m%d%H}_{end:%Y%m%d%H}.nc"
    )


def ensure_glofs_waterlevel(
    download_dir: Path, model: GLOFSModel, start: datetime, duration_hours: int
) -> Path:
    """Merge the cached hours ``start .. start + duration_hours`` into one file.

    The result is a GeoDataset in the same layout as a STOFS fields file:
    ``zeta(time, node)`` with node coordinates ``x`` (longitude) and ``y``
    (latitude). It is written once and reused.

    Raises
    ------
    FileNotFoundError
        If any hour hasn't been fetched.
    ValueError
        If NOAA changed the model grid inside the window.
    """
    out = glofs_waterlevel_path(download_dir, model, start, duration_hours)
    if out.exists():
        return out

    hours = [start + k * _HOUR for k in range(duration_hours + 1)]
    paths = [_hour_path(download_dir, model, h) for h in hours]
    missing = [h for h, p in zip(hours, paths, strict=True) if not p.exists()]
    if missing:
        msg = (
            f"{len(missing)} of {len(hours)} GLOFS {model} hour(s) are not downloaded "
            f"({missing[0]:%Y-%m-%d %H:%M} … {missing[-1]:%Y-%m-%d %H:%M}) under "
            f"{_cache_dir(download_dir, model)}. Run the download stage for this window."
        )
        raise FileNotFoundError(msg)

    lon0, lat0, _ = _read_hour(paths[0])
    zetas = []
    for hour, path in zip(hours, paths, strict=True):
        lon, lat, zeta = _read_hour(path)
        if lon.shape != lon0.shape or not (np.array_equal(lon, lon0) and np.array_equal(lat, lat0)):
            msg = (
                f"GLOFS {model} changed grids at {hour:%Y-%m-%d %H:%M} (NOAA upgraded "
                "the model mid-window). Split the simulation at that time."
            )
            raise ValueError(msg)
        zetas.append(zeta)

    _write_geodataset(out, hours, lon0, lat0, np.stack(zetas), model)
    logger.info(f"GLOFS {model}: merged {len(hours)} hour(s) into {out.name}")
    return out


def _write_geodataset(
    path: Path,
    hours: Iterable[datetime],
    lon: NDArray[np.float64],
    lat: NDArray[np.float64],
    zeta: NDArray[np.float32],
    model: str,
) -> None:
    import netCDF4

    hours = list(hours)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".nc.part")
    units = f"hours since {hours[0]:%Y-%m-%d %H:%M:%S}"
    with netCDF4.Dataset(tmp, "w") as ds:
        ds.createDimension("time", len(hours))
        ds.createDimension("node", lon.size)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = units
        t.calendar = "standard"
        t[:] = netCDF4.date2num(hours, units, calendar="standard")
        x = ds.createVariable("x", "f8", ("node",))
        x.units = "degrees_east"
        x.standard_name = "longitude"
        x[:] = lon
        y = ds.createVariable("y", "f8", ("node",))
        y.units = "degrees_north"
        y.standard_name = "latitude"
        y[:] = lat
        z = ds.createVariable("zeta", "f4", ("time", "node"), zlib=True, fill_value=np.nan)
        z.units = "m"
        z.long_name = "water surface elevation above the lake low-water datum"
        z[:] = zeta
        ds.title = f"GLOFS {model} nowcast water level"
        ds.source = GLOFS_BASE_URL
    tmp.replace(path)
