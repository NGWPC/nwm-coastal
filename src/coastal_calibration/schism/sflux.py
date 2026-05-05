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


def make_atmo_sflux(
    forcing_input_dir: Path,
    work_dir: Path,
    start_dt: datetime,
    geogrid_file: Path,
) -> None:
    """Create SCHISM sflux atmospheric forcing from NWM LDASIN files.

    Produces ``<work_dir>/sflux/sflux_air_1.0001.nc`` from the LDASIN
    files found in *forcing_input_dir*. The last timestep is duplicated
    so that SCHISM always has a value at the end of the simulation
    window. The simulation length is inferred from the number of files
    on disk; the caller does not need to pass it.

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
    """
    logger.debug("    Loading geogrid data from %s", geogrid_file)
    with netCDF4.Dataset(geogrid_file) as geo:
        height = geo["HGT_M"][0, :]
        lats = geo["XLAT_M"][0, :]
        lons = geo["XLONG_M"][0, :]

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
                t2d = np.asarray(data.variables["T2D"][0])
                q2d = np.asarray(data.variables["Q2D"][0])
                psfc = np.asarray(data.variables["PSFC"][0])
                write_var(nct, t2d, index=i)
                write_var(ncq, q2d, index=i)
                write_var(ncu, np.asarray(data.variables["U2D"][0]), index=i)
                write_var(ncv, np.asarray(data.variables["V2D"][0]), index=i)
                write_var(ncp, _pressure_to_msl(t2d, q2d, height, psfc), index=i)

        # Duplicate last timestep so SCHISM always has a trailing value
        write_var(ncu, np.asarray(ncu[-2]), index=-1)
        write_var(ncv, np.asarray(ncv[-2]), index=-1)
        write_var(ncp, np.asarray(ncp[-2]), index=-1)
        write_var(nct, np.asarray(nct[-2]), index=-1)
        write_var(ncq, np.asarray(ncq[-2]), index=-1)
