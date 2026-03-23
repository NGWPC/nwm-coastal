"""OTPS (OTPSnc) input/output helpers.

Refactored from ``tpxo_to_open_bnds_hgrid/make_otps_input.py`` and
``tpxo_to_open_bnds_hgrid/otps_to_open_bnds_hgrid.py``.  All
environment-variable and ``argparse`` boilerplate has been replaced with
explicit function parameters.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

from coastal_calibration.tides._tpxo_out import TPXOOut
from coastal_calibration.utils.logging import logger

_SCHISM_COORD_NAME = "nodeCoords"
_SCHISM_OPEN_BOUNDARY_NAME = "openBndNodes"
_MISSING = -9999.0
_TIME_STEP_S = 3600


def make_otps_input(
    grid_file: Path,
    output_file: Path,
    start_dt: datetime,
    end_dt: datetime,
    timestep_s: int,
) -> None:
    """Write an OTPSnc input file for TPXO tidal predictions.

    Parameters
    ----------
    grid_file : Path
        SCHISM grid netCDF4 file with ``nodeCoords`` and ``openBndNodes``.
    output_file : Path
        Destination path for the OTPSnc input text file.
    start_dt : datetime
        Start of the prediction window (UTC).
    end_dt : datetime
        End of the prediction window (UTC).
    timestep_s : int
        Output time step in seconds.
    """
    logger.info("Writing OTPS input: %s (grid=%s)", output_file, grid_file)
    time_step = timedelta(seconds=timestep_s)

    with netCDF4.Dataset(grid_file) as f_in:
        coords = f_in[_SCHISM_COORD_NAME][:]
        valid_indices = f_in[_SCHISM_OPEN_BOUNDARY_NAME][:]
        coords = [coords[i].tolist() for i in valid_indices]

    with open(output_file, "w") as fout:
        for c in coords:
            current = start_dt
            while current <= end_dt:
                fout.write(f"{c[1]}  {c[0]}  {current.strftime('%Y %m %d %H %M %S')}\n")
                current += time_step


def otps_to_open_bnds(
    otps_output_file: Path,
    grid_file: Path,
    elev_output_file: Path,
) -> None:
    """Convert OTPSnc predict_tide output to SCHISM elev2D.th.nc format.

    Parameters
    ----------
    otps_output_file : Path
        predict_tide output text file produced by OTPS.
    grid_file : Path
        SCHISM grid netCDF4 file with ``nodeCoords`` and ``openBndNodes``.
    elev_output_file : Path
        Destination path for the SCHISM boundary forcing netCDF4 file.
    """
    logger.info("Converting OTPS output to SCHISM boundary: %s", elev_output_file)
    tpxo = TPXOOut(str(otps_output_file))

    with netCDF4.Dataset(grid_file) as f_in:
        coords = f_in[_SCHISM_COORD_NAME][:]
        valid_indices = f_in[_SCHISM_OPEN_BOUNDARY_NAME][:]
        coords = [coords[i].tolist() for i in valid_indices]

    start = datetime.strptime(
        f"{tpxo.df['mm.dd.yyyy'].iloc[0]} {tpxo.df['hh:mm:ss'].iloc[0]}", "%m.%d.%Y %H:%M:%S"
    )
    end = datetime.strptime(
        f"{tpxo.df['mm.dd.yyyy'].iloc[-1]} {tpxo.df['hh:mm:ss'].iloc[-1]}", "%m.%d.%Y %H:%M:%S"
    )
    nsteps = math.floor((end - start).total_seconds() / _TIME_STEP_S) + 1

    with netCDF4.Dataset(elev_output_file, "w", format="NETCDF4") as f_out:
        f_out.createDimension("time", None)
        f_out.createDimension("nOpenBndNodes", len(coords))
        f_out.createDimension("nLevels", 1)
        f_out.createDimension("nComponents", 1)
        f_out.createDimension("one", 1)

        time_step_var = f_out.createVariable("time_step", "f8", ("one",))
        time_var = f_out.createVariable("time", "f8", ("time",))
        time_series_var = f_out.createVariable(
            "time_series",
            "f8",
            ("time", "nOpenBndNodes", "nLevels", "nComponents"),
            fill_value=_MISSING,
            zlib=True,
        )

        time_var.long_name = "model time"
        time_var.standard_name = "time"
        time_var.units = (
            f"seconds since {start.strftime('%Y-%m-%d %H:%M:%S')}        ! NCDASE - BASE_DAT"
        )
        time_var.base_date = (
            f"{start.strftime('%Y-%m-%d %H:%M:%S')}        ! NCDASE - BASE_DATE"
        )
        time_var.start_time = 0.0

        time_step_var[:] = np.array([_TIME_STEP_S])
        time_var[:] = np.arange(0, nsteps * _TIME_STEP_S, _TIME_STEP_S)

        for c in range(len(coords)):
            df_selected = tpxo.df[
                tpxo.df["Lat"].between(coords[c][1] - 0.0001, coords[c][1] + 0.0001)
                & tpxo.df["Lon"].between(coords[c][0] - 0.0001, coords[c][0] + 0.0001)
            ]
            if not df_selected.empty:
                data = df_selected["z(m)"]
                data = np.where(data > _MISSING, data, 0)
                time_series_var[:, c, 0, 0] = data[0:nsteps]
            else:
                time_series_var[:, c, 0, 0] = 0
