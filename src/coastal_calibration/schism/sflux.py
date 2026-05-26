"""SCHISM sflux atmospheric forcing generation.

Replaces the legacy ``makeAtmo.py`` script.  All logic is expressed as a
plain Python function with explicit parameters — no environment-variable
reading, no subprocess invocation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import netCDF4
import numpy as np

from coastal_calibration.logging import logger

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from numpy.typing import NDArray


def _round_down(n: float, decimals: int = 0) -> float:
    multiplier = 10**decimals
    return math.floor(n * multiplier) / multiplier


def _compute_subset_indices(
    lats: NDArray[np.floating[Any]],
    lons: NDArray[np.floating[Any]],
    mesh_bbox: tuple[float, float, float, float],
    buffer_deg: float,
) -> tuple[int, int, int, int]:
    """Return ``(j0, j1, i0, i1)`` slice bounds covering *mesh_bbox* on the geogrid.

    The bounds are expanded by *buffer_deg* on all sides and returned as a
    contiguous rectangle in geogrid index space.  Raises ``ValueError`` when
    the buffered bbox has no overlap with the geogrid.
    """
    lon_min, lat_min, lon_max, lat_max = mesh_bbox
    mask = (
        (lats >= lat_min - buffer_deg)
        & (lats <= lat_max + buffer_deg)
        & (lons >= lon_min - buffer_deg)
        & (lons <= lon_max + buffer_deg)
    )
    if not mask.any():
        geo_lon_min, geo_lon_max = float(lons.min()), float(lons.max())
        geo_lat_min, geo_lat_max = float(lats.min()), float(lats.max())
        raise ValueError(
            f"Mesh bbox (lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]) "
            f"with buffer {buffer_deg} deg has no overlap with the geogrid "
            f"(lon=[{geo_lon_min}, {geo_lon_max}], lat=[{geo_lat_min}, {geo_lat_max}]). "
            "Wrong geogrid file for this mesh?"
        )
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _pressure_to_msl(
    temp: NDArray[np.floating[Any]],
    mixing: NDArray[np.floating[Any]],
    height: NDArray[np.floating[Any]],
    press: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Reduce surface pressure to mean sea level.

    Parameters
    ----------
    temp : array
        2-m air temperature (K).
    mixing : array
        2-m specific humidity (kg/kg).
    height : array
        Terrain height (m).
    press : array
        Surface pressure (Pa).

    Returns
    -------
    numpy.ndarray
        Sea-level pressure (Pa).
    """
    g0 = 9.80665
    Rd = 287.058  # noqa: N806
    epsilon = 0.622

    Tv = temp * (1 + (mixing / epsilon)) / (1 + mixing)  # noqa: N806
    H = Rd * Tv / g0  # noqa: N806
    return press / np.exp(-height / H)


def make_atmo_sflux(  # noqa: PLR0915
    forcing_input_dir: Path,
    work_dir: Path,
    start_dt: datetime,
    geogrid_file: Path,
    *,
    mesh_bbox: tuple[float, float, float, float] | None = None,
    bbox_buffer_deg: float = 0.5,
) -> None:
    """Create SCHISM sflux atmospheric forcing from NWM LDASIN files.

    Produces ``<work_dir>/sflux/sflux_air_1.0001.nc`` from the LDASIN
    files found in *forcing_input_dir*. The last timestep is duplicated
    so that SCHISM always has a value at the end of the simulation
    window. The simulation length is inferred from the number of files
    on disk; the caller does not need to pass it.

    When *mesh_bbox* is provided, the geogrid and every LDASIN slab are
    cropped to the smallest contiguous index rectangle covering the
    buffered bbox before being written to disk.  This avoids carrying
    the full CONUS forcing grid into a sflux file when the SCHISM mesh
    only spans a small subdomain, which dominates I/O on multi-node MPI
    runs that read the file concurrently from NFS.  Pass ``None`` to
    write the full geogrid (the historical behavior).

    Parameters
    ----------
    forcing_input_dir : Path
        Directory containing ``*LDASIN_DOMAIN1`` input files.
    work_dir : Path
        SCHISM working directory. The ``sflux/`` sub-directory will be
        created if it does not exist.
    start_dt : datetime
        Simulation start (UTC).
    geogrid_file : Path
        WRF geogrid file containing ``HGT_M``, ``XLAT_M``, ``XLONG_M``.
    mesh_bbox : tuple of float, optional
        ``(lon_min, lat_min, lon_max, lat_max)`` of the SCHISM mesh in
        degrees.  When set, the output is cropped to this bbox plus
        *bbox_buffer_deg* on each side.  Raises ``ValueError`` when the
        buffered bbox has no overlap with the geogrid.
    bbox_buffer_deg : float, optional
        Pad applied to *mesh_bbox* on each side, in degrees.  Defaults to
        0.5 degrees so coastal cells just outside the mesh extent are
        retained for safety.  Ignored when *mesh_bbox* is ``None``.
    """
    logger.debug("    Loading geogrid data from %s", geogrid_file)
    with netCDF4.Dataset(geogrid_file) as geo:
        height = np.asarray(geo["HGT_M"][0, :])
        lats = np.asarray(geo["XLAT_M"][0, :])
        lons = np.asarray(geo["XLONG_M"][0, :])

    if mesh_bbox is not None:
        j0, j1, i0, i1 = _compute_subset_indices(lats, lons, mesh_bbox, bbox_buffer_deg)
        logger.info(
            "    Subsetting geogrid to mesh bbox: ny %d→%d, nx %d→%d",
            lats.shape[0],
            j1 - j0,
            lons.shape[1],
            i1 - i0,
        )
        height = height[j0:j1, i0:i1]
        lats = lats[j0:j1, i0:i1]
        lons = lons[j0:j1, i0:i1]
    else:
        j0, j1 = 0, lats.shape[0]
        i0, i1 = 0, lons.shape[1]

    files = sorted(str(p) for p in forcing_input_dir.glob("*LDASIN_DOMAIN1"))
    if not files:
        msg = f"No LDASIN_DOMAIN1 files found in {forcing_input_dir}"
        raise FileNotFoundError(msg)
    logger.info("    Creating sflux from %d LDASIN files in %s", len(files), forcing_input_dir)

    sflux_dir = work_dir / "sflux"
    sflux_dir.mkdir(parents=True, exist_ok=True)
    out_path = sflux_dir / "sflux_air_1.0001.nc"

    from coastal_calibration._nc_io import create_var, write_var

    base_date_str = start_dt.strftime("%Y-%m-%d")
    base_date = [
        np.int32(start_dt.year),
        np.int32(start_dt.month),
        np.int32(start_dt.day),
        np.int32(0),
    ]
    field_dims = ("time", "ny_grid", "nx_grid")

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ncout:
        ncout.createDimension("time", len(files) + 1)
        ncout.createDimension("ny_grid", lats.shape[0])
        ncout.createDimension("nx_grid", lons.shape[1])

        nctime = create_var(
            ncout,
            "time",
            "f4",
            ("time",),
            attrs={
                "long_name": "Time",
                "standard_name": "time",
                "units": f"days since {base_date_str}",
                "base_date": base_date,
            },
        )
        time = np.arange(0, (1 / 24) * (len(files) + 1), 1 / 24)
        time += start_dt.hour / 24.0
        time[0] = _round_down(time[0], 7)
        write_var(nctime, time)

        nclon = create_var(
            ncout,
            "lon",
            "f4",
            ("ny_grid", "nx_grid"),
            attrs={
                "long_name": "Longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
            },
        )
        write_var(nclon, lons)

        nclat = create_var(
            ncout,
            "lat",
            "f4",
            ("ny_grid", "nx_grid"),
            attrs={
                "long_name": "Latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
            },
        )
        write_var(nclat, lats)

        nct = create_var(
            ncout,
            "stmp",
            "f4",
            field_dims,
            attrs={
                "long_name": "Surface Air Temperature (2m AGL)",
                "standard_name": "air_temperature",
                "units": "K",
            },
        )
        ncq = create_var(
            ncout,
            "spfh",
            "f4",
            field_dims,
            attrs={
                "long_name": "Surface Specific Humidity (2m AGL)",
                "standard_name": "specific_humidity",
                "units": "kg/kg",
            },
        )
        ncu = create_var(
            ncout,
            "uwind",
            "f4",
            field_dims,
            attrs={
                "long_name": "Surface Eastward Air Velocity (10m AGL)",
                "standard_name": "eastward_wind",
                "units": "m/s",
            },
        )
        ncv = create_var(
            ncout,
            "vwind",
            "f4",
            field_dims,
            attrs={
                "long_name": "Surface Northward Air Velocity (10m AGL)",
                "standard_name": "northward_wind",
                "units": "m/s",
            },
        )
        ncp = create_var(
            ncout,
            "prmsl",
            "f4",
            field_dims,
            attrs={
                "long_name": "Pressure reduced to MSL",
                "standard_name": "air_pressure_at_sea_level",
                "units": "Pa",
            },
        )

        for i, file in enumerate(files):
            with netCDF4.Dataset(file) as data:
                t2d = np.asarray(data.variables["T2D"][0])[j0:j1, i0:i1]
                q2d = np.asarray(data.variables["Q2D"][0])[j0:j1, i0:i1]
                psfc = np.asarray(data.variables["PSFC"][0])[j0:j1, i0:i1]
                u2d = np.asarray(data.variables["U2D"][0])[j0:j1, i0:i1]
                v2d = np.asarray(data.variables["V2D"][0])[j0:j1, i0:i1]
                write_var(nct, t2d, index=i)
                write_var(ncq, q2d, index=i)
                write_var(ncu, u2d, index=i)
                write_var(ncv, v2d, index=i)
                write_var(ncp, _pressure_to_msl(t2d, q2d, height, psfc), index=i)

        # Duplicate last timestep so SCHISM always has a trailing value
        write_var(ncu, np.asarray(ncu[-2]), index=-1)
        write_var(ncv, np.asarray(ncv[-2]), index=-1)
        write_var(ncp, np.asarray(ncp[-2]), index=-1)
        write_var(nct, np.asarray(nct[-2]), index=-1)
        write_var(ncq, np.asarray(ncq[-2]), index=-1)
