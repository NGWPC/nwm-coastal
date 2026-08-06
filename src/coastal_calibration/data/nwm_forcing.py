"""Normalization for NWM LDASIN meteorological forcing.

The NWM Retrospective serves its non-CONUS forcing in the raw WRF layout,
and two of those domains ship it broken:

* PRVI and Alaska files carry no grid-mapping variable, no coordinate
  variables, no ``XLAT``/``XLONG``, and no global attributes, so there is
  nothing in the file to rebuild a transform from.  The grid comes from the
  sidecar :func:`coastal_calibration.data.downloader.write_nwm_grid_sidecar`
  records from NOAA's own ``ldasout.zarr``, or from a built-in table.
* PRVI pins ``valid_time`` to 00Z of the day in every hourly file and leaves
  ``Times`` empty, so the timestamp has to come from the filename.

None of this is a shortcoming of any reader: the information is absent from
the files.  This module therefore stands on its own, separate from the
hydromt stopgaps in :mod:`coastal_calibration.sfincs._hydromt_compat`.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from coastal_calibration.logging import logger as _log

if TYPE_CHECKING:
    import xarray as xr

__all__ = ["GRID_SIDECAR_GLOB", "normalize_wrf_forcing"]


# Raw WRF dimension names used by the non-CONUS NWM Retrospective forcing.
_WRF_DIMS = {"west_east": "x", "south_north": "y", "Time": "time"}

# Last-resort GDAL-order GeoTransforms keyed by grid shape ``(n_x, n_y)``,
# for download directories that predate the grid sidecar (see
# ``coastal_calibration.data.downloader.write_nwm_grid_sidecar``) or were
# populated without network access.  Copied from the ``crs`` variable of
# each domain's ``ldasout.zarr``, which is also where the sidecar reads
# them.  A shape is a weak domain identifier, so this table is only
# consulted when no sidecar is present and its use is logged.
_WRF_GEOTRANSFORM = {
    (300, 110): "-149999.716023 1000.0 0 55001.2695425 0 -1000.0",
    (879, 459): "-1130889.442087718 1000 0 -2704514.413754986 0 -1000",
}

#: Sidecar files the downloader writes next to the Retrospective forcing.
GRID_SIDECAR_GLOB = "grid_*.json"

#: A GDAL GeoTransform is six numbers: x_origin, x_res, x_rot, y_origin,
#: y_rot, y_res.
_GEOTRANSFORM_TERMS = 6


def _parse_geotransform(text: str, origin: str) -> tuple[float, ...]:
    """Parse a GDAL-order GeoTransform string into six finite floats.

    *origin* names where the string came from, for the error message.
    """
    import math

    try:
        values = tuple(float(v) for v in text.split())
    except ValueError as exc:
        msg = f"GeoTransform from {origin} is not numeric: {text!r}"
        raise ValueError(msg) from exc
    if len(values) != _GEOTRANSFORM_TERMS or not all(math.isfinite(v) for v in values):
        msg = (
            f"GeoTransform from {origin} must be six finite numbers "
            f"(x_origin, x_res, x_rot, y_origin, y_rot, y_res); got {text!r}"
        )
        raise ValueError(msg)
    return values


def _sidecar_geotransform(source: str, n_x: int, n_y: int) -> tuple[float, ...] | None:
    """Return the GeoTransform the downloader recorded for this grid.

    The sidecars live next to the forcing files and each names the domain
    it came from, so several domains can share one download directory.
    Selection is by grid shape, which is unambiguous unless two domains in
    the same directory happen to share one.

    Comparison is on parsed numbers, not the raw text: NOAA writes the same
    resolution as ``1000`` for one domain and ``1000.0`` for another, and
    those must not read as a conflict.
    """
    if not source:
        return None

    matches: dict[str, tuple[float, ...]] = {}
    for path in sorted(Path(source).parent.glob(GRID_SIDECAR_GLOB)):
        try:
            record = json.loads(path.read_text())
            if record.get("shape") != [n_x, n_y] or not record.get("geotransform"):
                continue
            geotransform = _parse_geotransform(str(record["geotransform"]), path.name)
        except (OSError, ValueError, AttributeError):
            _log.warning("Ignoring unusable NWM grid sidecar: %s", path)
            continue
        matches[str(record.get("domain", path.stem))] = geotransform

    if len(set(matches.values())) > 1:
        msg = (
            f"NWM grid sidecars for domains ({', '.join(sorted(matches))}) all describe a "
            f"{n_x}x{n_y} grid but disagree on its GeoTransform, so the forcing in "
            f"{Path(source).parent} cannot be georeferenced unambiguously. Download each "
            "domain into its own directory."
        )
        raise ValueError(msg)
    return next(iter(matches.values()), None)


def normalize_wrf_forcing(ds: xr.Dataset) -> xr.Dataset:
    """Convert raw WRF-style NWM forcing to a CF ``x``/``y``/``time`` layout.

    The NWM Retrospective serves its non-CONUS forcing (e.g. Hawaii) in the
    raw WRF layout: dimensions ``west_east``/``south_north``/``Time``, no
    ``x``/``y`` coordinate variables at all, and the timestamp in a
    character array (``Times``) plus a CF-encoded ``valid_time`` data
    variable.  CONUS Retrospective and every Analysis file instead use the
    ``x``/``y``/``time`` layout that hydromt's raster accessor expects.

    Without coordinates hydromt raises "x dimension not found", so rebuild
    them from a ``GeoTransform``, take the time coordinate from the
    ``YYYYMMDDHH`` filename stamp, and drop the leftover grid-mapping and
    character-timestamp variables.

    The GeoTransform is resolved in three steps.  The file's own
    grid-mapping variable wins (Hawaii and CONUS carry one).  PRVI and
    Alaska carry no georeferencing whatsoever, so next comes the grid
    sidecar the downloader writes from NOAA's own ``ldasout.zarr``.  The
    built-in :data:`_WRF_GEOTRANSFORM` table is the last resort, for
    download directories predating the sidecar.

    Datasets that already use the CF layout are returned unchanged.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset as opened from an LDASIN file.

    Returns
    -------
    xarray.Dataset
        Dataset with ``x``/``y``/``time`` dimensions and coordinates.
    """
    import numpy as np

    if not {"west_east", "south_north"} <= set(ds.dims):
        return ds

    # The grid-mapping variable is named after the projection
    # (``lambert_conformal_conic``, ``polar_stereographic``, ...), so find
    # it by the attribute that actually matters rather than by name.
    grid_mapping = next(
        (name for name, var in ds.variables.items() if "GeoTransform" in var.attrs),
        None,
    )
    n_x = ds.sizes["west_east"]
    n_y = ds.sizes["south_north"]
    source = str(ds.encoding.get("source", ""))

    if grid_mapping is not None:
        geotransform = _parse_geotransform(
            str(ds[grid_mapping].attrs["GeoTransform"]), f"variable '{grid_mapping}'"
        )
    elif (sidecar := _sidecar_geotransform(source, n_x, n_y)) is not None:
        geotransform = sidecar
    elif (n_x, n_y) in _WRF_GEOTRANSFORM:
        geotransform = _parse_geotransform(
            _WRF_GEOTRANSFORM[n_x, n_y], f"the built-in {n_x}x{n_y} entry"
        )
        _log.warning(
            "No grid sidecar for the %dx%d NWM forcing grid; falling back to the built-in "
            "GeoTransform. Re-run the download stage to record the grid from NOAA's "
            "ldasout.zarr instead.",
            n_x,
            n_y,
        )
    else:
        raise ValueError(
            "WRF-style NWM forcing has no grid-mapping variable carrying a "
            f"GeoTransform attribute, no grid sidecar alongside it, and its {n_x}x{n_y} "
            "grid is not a known NWM domain, so x/y coordinates cannot be rebuilt."
        )

    # Origins are the outer edge of the first pixel, so offset by half a cell
    # to get centers.
    x_origin, x_res, _, y_origin, _, y_res = geotransform

    # The GeoTransform is a leftover from the GDAL conversion these files went
    # through and declares a north-up raster (negative y_res), but the rows are
    # actually stored south to north, as ``south_north`` implies. Both axes
    # therefore ascend; deriving y from the sign of y_res instead flips the
    # field by the height of the domain. Verified against the same NWM Hawaii
    # grid in Analysis form: ascending y puts the orographic surface-pressure
    # low over Mauna Loa (19.5N), descending y puts it 2.2 degrees out to sea.
    x_start = min(x_origin, x_origin + n_x * x_res)
    y_start = min(y_origin, y_origin + n_y * y_res)

    ds = ds.rename({old: new for old, new in _WRF_DIMS.items() if old in ds.dims})
    ds = ds.assign_coords(
        x=x_start + (np.arange(n_x) + 0.5) * abs(x_res),
        y=y_start + (np.arange(n_y) + 0.5) * abs(y_res),
    )

    # PRVI pins ``valid_time`` to 00Z of the day in every hourly file and
    # leaves ``Times`` empty, so all files claim the same timestamp and
    # ``open_mfdataset`` cannot order them.  The downloader always names
    # files ``YYYYMMDDHH.LDASIN_DOMAIN1``, so take the timestamp from the
    # filename and only fall back to ``valid_time`` (CF-encoded, but
    # shipped as a data variable so xarray never promotes it) for datasets
    # that were not opened from a file.
    stamp = re.fullmatch(
        r"(\d{10})\.LDASIN_DOMAIN1(\.nc)?", Path(ds.encoding.get("source", "")).name
    )
    if stamp is not None:
        ds = ds.assign_coords(
            time=("time", np.array([datetime.strptime(stamp[1], "%Y%m%d%H")], "datetime64[ns]"))
        )
    elif "valid_time" in ds.variables:
        import xarray as xr_mod

        ds = ds.assign_coords(time=xr_mod.decode_cf(ds[["valid_time"]])["valid_time"].variable)

    # ``Times`` carries a DateStrLen dimension that hydromt cannot handle,
    # and the grid-mapping variable is redundant once the CRS is set from
    # the catalog metadata.
    drop = ["Times", "valid_time", grid_mapping]
    return ds.drop_vars([v for v in drop if v is not None and v in ds.variables])
