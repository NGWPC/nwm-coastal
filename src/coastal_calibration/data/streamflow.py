"""Shared streamflow reader for SFINCS and SCHISM workflows.

Provides a single ``read_streamflow`` function that extracts discharge
timeseries for a set of ``feature_id`` values from one of three sources:

* **nwm_retro** — the consolidated NWM Retrospective Zarr store on S3
  (no local files required).
* **nwm_ana** — local NWM CHRTOUT netCDF files, via fast direct
  ``netCDF4.Dataset`` access.
* **ngen_forecast** — a t-route output netCDF (``troute_output_*.nc``)
  produced by the ngen forecast, keyed by NextGen hydrofabric
  ``feature_id``.

All three return the same ``DataFrame`` shape (``DatetimeIndex`` ×
``feature_id`` columns, m³/s) so downstream discharge stages are agnostic
to the source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from coastal_calibration.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Zarr store URLs for NWM Retrospective v3.0 on S3
# ---------------------------------------------------------------------------
_ZARR_STORES: dict[str, str] = {
    "conus": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
    "atlgulf": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
    "pacific": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
    "hawaii": "s3://noaa-nwm-retrospective-3-0-pds/Hawaii/zarr/chrtout.zarr",
    "prvi": "s3://noaa-nwm-retrospective-3-0-pds/PR/zarr/chrtout.zarr",
    "alaska": "s3://noaa-nwm-retrospective-3-0-pds/Alaska/zarr/chrtout.zarr",
}


# ---------------------------------------------------------------------------
# Zarr path  (nwm_retro)
# ---------------------------------------------------------------------------


def _read_from_zarr(
    feature_ids: Sequence[int],
    start: datetime,
    end: datetime,
    *,
    domain: str,
) -> pd.DataFrame:
    """Read streamflow from the NWM Retrospective Zarr store on S3."""
    import fsspec
    import xarray as xr

    url = _ZARR_STORES.get(domain)
    if url is None:
        raise ValueError(
            f"No Zarr store configured for domain {domain!r}. Available: {sorted(_ZARR_STORES)}"
        )

    logger.info("Reading streamflow from Zarr store: %s", url)

    mapper = fsspec.get_mapper(url, anon=True)
    ds = xr.open_zarr(mapper, consolidated=True, chunks="auto")  # pyright: ignore[reportArgumentType]

    available = set(ds["feature_id"].values.tolist())
    # Dedupe: the same feature_id can appear twice in feature_ids (e.g. a reach
    # listed as both a source and a sink), which would select duplicate columns
    # and break label-based access downstream. Matches the CHRTOUT path.
    keep = sorted(set(feature_ids) & available)

    if not keep:
        logger.warning("None of the requested feature_ids found in Zarr store")
        return pd.DataFrame()

    sf = ds["streamflow"].sel(feature_id=keep, time=slice(start, end)).load()

    df = sf.to_pandas()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.fillna(0.0)

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    return df  # pyright: ignore[reportReturnType]


# ---------------------------------------------------------------------------
# netCDF4 direct-read path  (nwm_ana / local CHRTOUT files)
# ---------------------------------------------------------------------------


def _extract_timestamp(ds: Any, fpath: Path) -> pd.Timestamp:
    """Extract timestamp from an open netCDF4 Dataset or fall back to filename."""
    import netCDF4

    for tvar_name in ("time", "model_output_valid_time"):
        if tvar_name in ds.variables:
            tvar = ds.variables[tvar_name]
            t_val = netCDF4.num2date(
                tvar[:].item(),
                units=tvar.units,
                calendar=getattr(tvar, "calendar", "standard"),
            )
            return pd.Timestamp(str(t_val))

    # Fallback: parse timestamp from filename (YYYYMMDDHHMM.CHRTOUT_DOMAIN1)
    stem = fpath.stem.split(".")[0]
    fmt = {8: "%Y%m%d", 10: "%Y%m%d%H", 12: "%Y%m%d%H%M", 14: "%Y%m%d%H%M%S"}
    n = len(stem)
    return pd.to_datetime(stem, format=fmt.get(n, "%Y%m%d%H%M"))


def _read_from_chrtout(
    chrtout_files: list[Path],
    feature_ids: list[int],
) -> pd.DataFrame:
    """Read streamflow from local CHRTOUT netCDF files.

    Uses direct ``netCDF4.Dataset`` access for speed — avoids xarray/dask
    overhead.  Each file's ``feature_id`` array is mapped independently so
    files from different NWM domains (with different feature layouts) are
    handled correctly.
    """
    import netCDF4

    if not chrtout_files:
        return pd.DataFrame()

    fid_list = sorted(set(feature_ids))
    fid_to_col = {f: i for i, f in enumerate(fid_list)}
    n_fids = len(fid_list)

    rows: list[tuple[pd.Timestamp, NDArray[np.floating[Any]]]] = []

    for fpath in chrtout_files:
        with netCDF4.Dataset(str(fpath), "r") as ds:
            all_fids = ds.variables["feature_id"][:]
            sf = np.ma.filled(ds.variables["streamflow"][:], 0.0)
            if sf.ndim > 1:
                sf = sf.squeeze()

            # Per-file index mapping — safe even if files have different
            # feature_id layouts (e.g. CONUS vs Hawaii CHRTOUT).
            keep_mask = np.isin(all_fids, fid_list)
            keep_idx = np.where(keep_mask)[0]
            if keep_idx.size == 0:
                continue

            vals = np.zeros(n_fids, dtype=np.float64)
            for pos in keep_idx:
                fid = int(all_fids[pos])
                vals[fid_to_col[fid]] = sf[pos]

            rows.append((_extract_timestamp(ds, fpath), vals))

    if not rows:
        logger.warning("None of the requested feature_ids found in CHRTOUT files")
        return pd.DataFrame()

    timestamps, data_rows = zip(*rows, strict=True)
    df = pd.DataFrame(
        np.array(data_rows),
        index=pd.DatetimeIndex(timestamps, name="time"),
        columns=fid_list,
    )

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    return df


def _read_from_troute(
    troute_file: Path,
    feature_ids: list[int],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Read discharge from a t-route output netCDF (``troute_output_*.nc``).

    The file is written by the ngen forecast's routing step with layout:

    * ``feature_id(feature_id)`` — int64 NextGen hydrofabric reach IDs,
    * ``time(time)`` — ``seconds since <file_reference_time>``,
    * ``flow(feature_id, time)`` — discharge in m³/s.

    Note the ``(feature_id, time)`` axis order is **transposed** relative
    to NWM CHRTOUT (``(time, feature_id)``); it is transposed here so the
    returned frame is time-indexed like the other readers.  Selection is
    by exact ``feature_id`` match — the coastal discharge crosswalk is
    expected to use the same NextGen hydrofabric IDs (no translation).
    """
    import netCDF4

    with netCDF4.Dataset(str(troute_file), "r") as ds:
        all_fids = np.asarray(ds.variables["feature_id"][:])
        tvar = ds.variables["time"]
        # ``np.atleast_1d`` guards the single-timestep case so the result is
        # always an iterable array of cftime/datetime values.
        raw_times = np.atleast_1d(
            netCDF4.num2date(
                tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard")
            )
        )
        # flow is (feature_id, time); read fully then subset by row.
        flow = np.ma.filled(ds.variables["flow"][:], 0.0).astype(np.float64)

    times = pd.DatetimeIndex([pd.Timestamp(str(t)) for t in raw_times], name="time")

    # Dedupe requested ids and map to their row in the file (skip missing).
    fid_to_row = {int(f): i for i, f in enumerate(all_fids)}
    keep = [f for f in sorted(set(feature_ids)) if f in fid_to_row]
    if not keep:
        logger.warning("None of the requested feature_ids found in t-route output")
        return pd.DataFrame()

    rows = [fid_to_row[f] for f in keep]
    data = flow[rows, :].T  # (time, keep) — transpose to time-major

    df = pd.DataFrame(data, index=times, columns=keep)
    # Inclusive window filter (t-route time axis is monotonic hourly).
    df = df.loc[(df.index >= start) & (df.index <= end)]
    df = df.fillna(0.0)

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    return df


def read_streamflow(
    feature_ids: Sequence[int],
    start: datetime,
    end: datetime,
    *,
    meteo_source: Literal["nwm_retro", "nwm_ana", "ngen_forecast"] = "nwm_retro",
    domain: str = "conus",
    chrtout_files: Sequence[Path] | None = None,
    troute_file: Path | None = None,
) -> pd.DataFrame:
    """Read NWM streamflow for *feature_ids* over *[start, end]*.

    Returns a :class:`~pandas.DataFrame` with a :class:`~pandas.DatetimeIndex`
    and integer ``feature_id`` columns.  Values are in m³/s; fill/masked
    values are replaced with ``0``.

    Parameters
    ----------
    feature_ids
        NWM channel reach identifiers to extract.
    start, end
        Inclusive time bounds. Tz-aware values are converted to UTC and
        stripped (NWM data is on UTC days); tz-naive values are passed
        through unchanged (assumed UTC).
    meteo_source
        ``"nwm_retro"`` reads from the S3 Zarr store (no local files
        needed).  ``"nwm_ana"`` requires *chrtout_files*.
        ``"ngen_forecast"`` requires *troute_file*.
    domain
        Coastal domain key (``"conus"``, ``"atlgulf"``, ``"pacific"``,
        ``"hawaii"``, ``"prvi"``).  Only used for the Zarr path.
    chrtout_files
        Sorted list of local CHRTOUT netCDF paths.  Required when
        *meteo_source* is ``"nwm_ana"``.
    troute_file
        Path to a t-route output netCDF (``troute_output_*.nc``).
        Required when *meteo_source* is ``"ngen_forecast"``.
    """
    from coastal_calibration.utils import to_naive_utc

    if not feature_ids:
        return pd.DataFrame()

    start = to_naive_utc(start)
    end = to_naive_utc(end)

    if meteo_source == "nwm_retro":
        return _read_from_zarr(feature_ids, start, end, domain=domain)

    if meteo_source == "ngen_forecast":
        if troute_file is None:
            raise ValueError("troute_file is required when meteo_source is 'ngen_forecast'")
        return _read_from_troute(troute_file, list(feature_ids), start, end)

    if chrtout_files is None:
        raise ValueError("chrtout_files is required when meteo_source is 'nwm_ana'")
    return _read_from_chrtout(list(chrtout_files), list(feature_ids))
