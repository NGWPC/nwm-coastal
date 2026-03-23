"""Shared NWM streamflow reader for SFINCS and SCHISM workflows.

Provides a single ``read_streamflow`` function that extracts discharge
timeseries from NWM CHRTOUT data for a set of ``feature_id`` values.

For **nwm_retro** data the function reads directly from the consolidated
Zarr store on S3 — no file download required.  For **nwm_ana** (operational)
data it reads from local CHRTOUT netCDF files using fast direct
``netCDF4.Dataset`` access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from coastal_calibration.utils.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

# ---------------------------------------------------------------------------
# Zarr store URLs for NWM Retrospective v3.0 on S3
# ---------------------------------------------------------------------------
_ZARR_STORES: dict[str, str] = {
    "conus": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
    "atlgulf": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
    "pacific": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
    "hawaii": "s3://noaa-nwm-retrospective-3-0-pds/Hawaii/zarr/chrtout.zarr",
    "prvi": "s3://noaa-nwm-retrospective-3-0-pds/PR/zarr/chrtout.zarr",
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
    ds = xr.open_zarr(mapper, consolidated=True, chunks="auto")

    fids = list(feature_ids)
    available = ds["feature_id"].values
    keep = [f for f in fids if f in available]

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

    return df


# ---------------------------------------------------------------------------
# netCDF4 direct-read path  (nwm_ana / local CHRTOUT files)
# ---------------------------------------------------------------------------


def _read_from_chrtout(
    chrtout_files: list[Path],
    feature_ids: list[int],
) -> pd.DataFrame:
    """Read streamflow from local CHRTOUT netCDF files.

    Uses direct ``netCDF4.Dataset`` access with pre-computed integer
    indices for speed — avoids xarray/dask overhead.
    """
    import netCDF4 as nc  # noqa: N813

    if not chrtout_files:
        return pd.DataFrame()

    # Build integer index mapping from the first file
    with nc.Dataset(str(chrtout_files[0]), "r") as ds0:
        all_fids = ds0.variables["feature_id"][:]

    fid_set = set(feature_ids)
    keep_mask = np.isin(all_fids, list(fid_set))
    keep_idx = np.where(keep_mask)[0]
    keep_fids = all_fids[keep_idx].tolist()

    if not keep_idx.size:
        logger.warning("None of the requested feature_ids found in CHRTOUT files")
        return pd.DataFrame()

    n_files = len(chrtout_files)
    n_features = len(keep_idx)
    data = np.empty((n_files, n_features), dtype=np.float64)
    timestamps: list[pd.Timestamp] = []

    for i, fpath in enumerate(chrtout_files):
        with nc.Dataset(str(fpath), "r") as ds:
            sf = ds.variables["streamflow"][:].filled(0.0)
            if sf.ndim > 1:
                sf = sf.squeeze()
            data[i, :] = sf[keep_idx]

            # Extract timestamp
            ts: pd.Timestamp | None = None
            for tvar_name in ("time", "model_output_valid_time"):
                if tvar_name in ds.variables:
                    tvar = ds.variables[tvar_name]
                    t_val = nc.num2date(
                        tvar[:].item(),
                        units=tvar.units,
                        calendar=getattr(tvar, "calendar", "standard"),
                    )
                    ts = pd.Timestamp(str(t_val))
                    break
            if ts is None:
                # Fallback: parse timestamp from filename (YYYYMMDDHHMM.CHRTOUT_DOMAIN1)
                stem = fpath.stem.split(".")[0]
                ts = pd.Timestamp(stem[:10])
            timestamps.append(ts)

    df = pd.DataFrame(
        data,
        index=pd.DatetimeIndex(timestamps, name="time"),
        columns=keep_fids,
    )

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

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
        Inclusive time bounds.
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
    if not feature_ids:
        return pd.DataFrame()

    if meteo_source == "nwm_retro":
        return _read_from_zarr(feature_ids, start, end, domain=domain)

    if chrtout_files is None:
        raise ValueError("chrtout_files is required when meteo_source is 'nwm_ana'")
    return _read_from_chrtout(list(chrtout_files), list(feature_ids))
