"""SCHISM sflux atmospheric forcing generation.

Replaces the legacy ``makeAtmo.py`` script.  All logic is expressed as a
plain Python function with explicit parameters — no environment-variable
reading, no subprocess invocation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import netCDF4 as nc  # noqa: N813
import numpy as np

from coastal_calibration.utils.logging import logger

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path


def _round_down(n: float, decimals: int = 0) -> float:
    multiplier = 10**decimals
    return math.floor(n * multiplier) / multiplier


def _slp(
    temp: np.ndarray,
    mixing: np.ndarray,
    height: np.ndarray,
    press: np.ndarray,
) -> np.ndarray:
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
    duration_hours: int,
    geogrid_file: Path,
) -> None:
    """Create SCHISM sflux atmospheric forcing from NWM LDASIN files.

    Produces ``<work_dir>/sflux/sflux_air_1.0001.nc`` from the LDASIN
    files found in *forcing_input_dir*.  The last timestep is duplicated
    so that SCHISM always has a value at the end of the simulation window.

    Parameters
    ----------
    forcing_input_dir : Path
        Directory containing ``*LDASIN_DOMAIN1`` input files.
    work_dir : Path
        SCHISM working directory.  The ``sflux/`` sub-directory will be
        created if it does not exist.
    start_dt : datetime
        Simulation start (UTC).
    duration_hours : int
        Simulation length in hours.  Negative values indicate analysis
        mode (sign is stripped; only the magnitude is used here).
    geogrid_file : Path
        WRF geogrid file containing ``HGT_M``, ``XLAT_M``, ``XLONG_M``.
    """
    length_hours = abs(duration_hours)

    # Load geospatial data
    logger.debug("    Loading geogrid data from %s", geogrid_file)
    with nc.Dataset(geogrid_file) as geo:
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

    ncout = nc.Dataset(out_path, "w", format="NETCDF4")
    try:
        ncout.createDimension("time", len(files) + 1)
        ncout.createDimension("ny_grid", lats.shape[0])
        ncout.createDimension("nx_grid", lons.shape[1])

        nctime = ncout.createVariable("time", "f4", ("time",))
        nclon = ncout.createVariable("lon", "f4", ("ny_grid", "nx_grid"))
        nclat = ncout.createVariable("lat", "f4", ("ny_grid", "nx_grid"))
        ncu = ncout.createVariable("uwind", "f4", ("time", "ny_grid", "nx_grid"))
        ncv = ncout.createVariable("vwind", "f4", ("time", "ny_grid", "nx_grid"))
        ncp = ncout.createVariable("prmsl", "f4", ("time", "ny_grid", "nx_grid"))
        nct = ncout.createVariable("stmp", "f4", ("time", "ny_grid", "nx_grid"))
        ncq = ncout.createVariable("spfh", "f4", ("time", "ny_grid", "nx_grid"))

        # Time axis
        time = np.arange(0, (1 / 24) * (len(files) + 1), 1 / 24)
        time += start_dt.hour / 24.0
        time[0] = _round_down(time[0], 7)

        base_date_str = start_dt.strftime("%Y-%m-%d")
        nctime.long_name = "Time"
        nctime.standard_name = "time"
        nctime.units = f"days since {base_date_str}"
        nctime.base_date = [
            np.int32(start_dt.year),
            np.int32(start_dt.month),
            np.int32(start_dt.day),
            np.int32(0),
        ]
        nctime[:] = time

        # Coordinate metadata
        nclon.long_name = "Longitude"
        nclon.standard_name = "longitude"
        nclon.units = "degrees_east"
        nclon[:] = lons

        nclat.long_name = "Latitude"
        nclat.standard_name = "latitude"
        nclat.units = "degrees_north"
        nclat[:] = lats

        ncu.long_name = "Surface Eastward Air Velocity (10m AGL)"
        ncu.standard_name = "eastward_wind"
        ncu.units = "m/s"

        ncv.long_name = "Surface Northward Air Velocity (10m AGL)"
        ncv.standard_name = "northward_wind"
        ncv.units = "m/s"

        ncp.long_name = "Pressure reduced to MSL"
        ncp.standard_name = "air_pressure_at_sea_level"
        ncp.units = "Pa"

        nct.long_name = "Surface Air Temperature (2m AGL)"
        nct.standard_name = "air_temperature"
        nct.units = "K"

        ncq.long_name = "Surface Specific Humidity (2m AGL)"
        ncq.standard_name = "specific_humidity"
        ncq.units = "kg/kg"

        for i, file in enumerate(files):
            with nc.Dataset(file) as data:
                nct[i, :] = data.variables["T2D"][:]
                ncq[i, :] = data.variables["Q2D"][:]
                ncu[i, :] = data.variables["U2D"][:]
                ncv[i, :] = data.variables["V2D"][:]
                ncp[i, :] = _slp(
                    np.array(nct[i, :]),
                    np.array(ncq[i, :]),
                    height,
                    np.array(data.variables["PSFC"][:]),
                )

        # Duplicate last timestep so SCHISM always has a trailing value
        ncu[-1] = ncu[-2]
        ncv[-1] = ncv[-2]
        ncp[-1] = ncp[-2]
        nct[-1] = nct[-2]
        ncq[-1] = ncq[-2]
    finally:
        ncout.close()

    _ = length_hours  # consumed by caller when deciding file count
