"""Shared NWM streamflow reader for SFINCS and SCHISM workflows.

Provides a single ``read_streamflow`` function that extracts discharge
timeseries from NWM CHRTOUT data for a set of ``feature_id`` values.

For **nwm_retro** data the function reads directly from the consolidated
Zarr store on S3 — no file download required.  For **nwm_ana** (operational)
data it reads from local CHRTOUT netCDF files using fast direct
``netCDF4.Dataset`` access.
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


def read_streamflow(
    feature_ids: Sequence[int],
    start: datetime,
    end: datetime,
    *,
    meteo_source: Literal["nwm_retro", "nwm_ana"] = "nwm_retro",
    domain: str = "conus",
    chrtout_files: Sequence[Path] | None = None,
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
    domain
        Coastal domain key (``"conus"``, ``"atlgulf"``, ``"pacific"``,
        ``"hawaii"``, ``"prvi"``).  Only used for the Zarr path.
    chrtout_files
        Sorted list of local CHRTOUT netCDF paths.  Required when
        *meteo_source* is ``"nwm_ana"``.
    """
    from coastal_calibration.utils import to_naive_utc

    if not feature_ids:
        return pd.DataFrame()

    start = to_naive_utc(start)
    end = to_naive_utc(end)

    if meteo_source == "nwm_retro":
        return _read_from_zarr(feature_ids, start, end, domain=domain)

    if chrtout_files is None:
        raise ValueError("chrtout_files is required when meteo_source is 'nwm_ana'")
    return _read_from_chrtout(list(chrtout_files), list(feature_ids))
