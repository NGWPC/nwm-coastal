"""OTPS (OTPSnc) input/output helpers.

Refactored from ``tpxo_to_open_bnds_hgrid/make_otps_input.py`` and
``tpxo_to_open_bnds_hgrid/otps_to_open_bnds_hgrid.py``.  All
environment-variable and ``argparse`` boilerplate has been replaced with
explicit function parameters.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import netCDF4
import numpy as np

from coastal_calibration.logging import logger
from coastal_calibration.tides._tpxo_out import TPXOOut

if TYPE_CHECKING:
    from pathlib import Path

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

    with output_file.open("w") as fout:
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

    from coastal_calibration._nc_io import write_elev2d_th

    # Build the (nt, n_open_bnd_nodes) elevation matrix from the parsed
    # OTPS DataFrame, one column per boundary node.
    series = np.zeros((nsteps, len(coords)))
    for c, coord in enumerate(coords):
        lon_c, lat_c = coord[0], coord[1]
        df_selected = tpxo.df[
            tpxo.df["Lat"].between(lat_c - 0.0001, lat_c + 0.0001)
            & tpxo.df["Lon"].between(lon_c - 0.0001, lon_c + 0.0001)
        ]
        if not df_selected.empty:
            series[:, c] = df_selected["z(m)"].to_numpy()[:nsteps]

    base = start.strftime("%Y-%m-%d %H:%M:%S")
    write_elev2d_th(
        elev_output_file,
        n_open_bnd_nodes=len(coords),
        time_seconds=np.arange(0, nsteps * _TIME_STEP_S, _TIME_STEP_S),
        time_step_seconds=_TIME_STEP_S,
        time_series=series,
        time_attrs={
            "long_name": "model time",
            "standard_name": "time",
            "units": f"seconds since {base}        ! NCDASE - BASE_DAT",
            "base_date": f"{base}        ! NCDASE - BASE_DATE",
            "start_time": 0.0,
        },
        missing=_MISSING,
    )
