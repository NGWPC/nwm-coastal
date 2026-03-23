r"""Regrid ESTOFS water level data to SCHISM open boundary nodes using ESMF.

This module regrids ESTOFS ``zeta`` (water surface elevation) from an
unstructured node grid to SCHISM open boundary node locations using
nearest-source-to-destination interpolation. The output is written in
SCHISM's ``elev2D.th.nc`` format.

MPI-parallel: ESMF decomposes the LocStreams across ranks; results
are gathered to rank 0 for writing.

Usage::

    mpirun -np 4 python -m coastal_calibration.regridding.regrid_estofs \\
        estofs.t00z.fields.cwl.nc schism.hgrid.nc output.nc \\
        --cycle-date 20240101 --cycle-time 0000 --length-hrs 180

NOTE: The SCHISM hgrid boundary file must be created using gr3_2_esmf.py
with the ``--filter_open_bnds`` flag::

    ./gr3_2_esmf.py --filter_open_bnds hgrid.gr3 schism.hgrid.nc
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import esmpy as ESMF
import netCDF4
import numpy as np
from cftime import num2date

from coastal_calibration.utils.logging import logger

from .esmf_utils import MaskedRegridder, build_locstream, gather_reduce

local_pet = ESMF.local_pet()

FORECAST_START = 5
TIME_STEP = 3600
MISSING = -9999.0


def _determine_time_range(
    f_in,
    forecast_start: int,
    cycle_date: str,
    cycle_time: str,
    length_hrs: int,
) -> tuple[int, int, np.ndarray, dict]:
    """Determine the time slice to extract from the ESTOFS input.

    Parameters
    ----------
    f_in
        Open netCDF4 Dataset handle for the ESTOFS file.
    forecast_start
        Index of the first forecast timestep.
    cycle_date
        Cycle date as ``YYYYMMDD``.
    cycle_time
        Cycle time as ``HHMM``.
    length_hrs
        Total forecast length in hours.

    Returns
    -------
    start
        Index of the first timestep to read.
    nt
        Number of timesteps.
    times
        Time values for the selected range.
    time_atts
        Attributes of the time variable (for output metadata).
    """
    etime_var = f_in["time"]
    estart = num2date(etime_var[forecast_start], units=etime_var.units)

    # Round up non-hourly ESTOFS start times
    if estart.minute != 0:
        estart = datetime(estart.year, estart.month, estart.day, estart.hour)
        estart += timedelta(hours=1)

    total_hours = length_hrs + 1
    fdate = datetime.strptime(cycle_date + cycle_time, "%Y%m%d%H%M")

    dt_h = int((fdate - estart).total_seconds() / 3600)
    start = forecast_start + dt_h

    logger.debug("ESTOFS time range: start=%d, total_hours=%d", start, total_hours)
    times = f_in["time"][start : start + total_hours]
    time_atts = f_in["time"].__dict__

    return start, len(times), times, time_atts


def _write_schism_output(
    nc_out: str,
    output: np.ndarray,
    times: np.ndarray,
    time_atts: dict,
    n_bnd_nodes: int,
):
    """Write regridded data in SCHISM elev2D.th.nc format."""
    nt = len(times)
    with netCDF4.Dataset(nc_out, "w", format="NETCDF4") as f_out:
        f_out.createDimension("time", None)
        f_out.createDimension("nOpenBndNodes", n_bnd_nodes)
        f_out.createDimension("nLevels", 1)
        f_out.createDimension("nComponents", 1)
        f_out.createDimension("one", 1)

        time_step_var = f_out.createVariable("time_step", "f8", ("one",))
        time_var = f_out.createVariable("time", "f8", ("time",))
        time_var.setncatts(time_atts)
        time_var.start_time = times[0]
        time_series_var = f_out.createVariable(
            "time_series",
            "f8",
            ("time", "nOpenBndNodes", "nLevels", "nComponents"),
            fill_value=MISSING,
            zlib=True,
        )

        time_step_var[:] = np.array([TIME_STEP])
        time_var[:] = np.arange(0, nt * TIME_STEP, TIME_STEP)
        for t in range(nt):
            data = np.where(output[t] > MISSING, output[t], 0)
            time_series_var[t, :, 0, 0] = data


def regrid_estofs(
    nc_in: str,
    nc_grid: str,
    nc_out: str,
    cycle_date: str,
    cycle_time: str,
    length_hrs: int,
    regrid_field: str = "zeta",
) -> None:
    """Regrid an ESTOFS field to SCHISM open boundary nodes.

    Parameters
    ----------
    nc_in
        Path to the ESTOFS input NetCDF file.
    nc_grid
        Path to the SCHISM hgrid NetCDF file (with open boundary info).
    nc_out
        Path to the output NetCDF file.
    regrid_field
        Name of the ESTOFS variable to regrid (default: ``"zeta"``).

    Notes
    -----
    Requires ``netcdf4`` built with parallel HDF5 support (the
    ``mpi_openmpi`` conda-forge variant) so that all MPI ranks can read
    from the same file independently.
    """
    # Read SCHISM boundary coordinates — every rank reads independently
    with netCDF4.Dataset(nc_grid) as f_grid:
        all_coords = f_grid["nodeCoords"][:]
        valid_indices = f_grid["openBndNodes"][:]

    bnd_coords = np.array([all_coords[i].tolist() for i in valid_indices])
    bnd_lons = np.asarray([c[0] for c in bnd_coords])
    bnd_lats = np.asarray([c[1] for c in bnd_coords])

    with netCDF4.Dataset(nc_in) as f_in:
        start, nt, times, time_atts = _determine_time_range(
            f_in, FORECAST_START, cycle_date, cycle_time, length_hrs
        )

        src_lon = f_in["x"][:]
        src_lat = f_in["y"][:]

        # Build ESMF LocStreams
        locstream_in = build_locstream(src_lon, src_lat)
        locstream_out = build_locstream(bnd_lons, bnd_lats)

        field_in = ESMF.Field(locstream_in, name="EstofsIn")
        field_out = ESMF.Field(locstream_out, name="OpenBoundary")

        # Global index range for slicing data arrays
        i_lo = locstream_in._global_lower
        i_hi = locstream_in._global_upper
        o_lo = locstream_out._global_lower
        o_hi = locstream_out._global_upper

        regridder = MaskedRegridder(
            method=ESMF.RegridMethod.NEAREST_STOD,
            unmapped_action=ESMF.UnmappedAction.IGNORE,
            src_mask_values=[1],
        )

        output = np.zeros((nt, len(bnd_lons)))
        for t in range(start, start + nt):
            data = f_in[regrid_field][t][i_lo:i_hi]
            field_in.data[...] = data
            locstream_in["ESMF:Mask"] = (
                data.mask.astype("i4")
                if hasattr(data, "mask") and np.ndim(data.mask) > 0
                else np.zeros(len(data), dtype="i4")
            )

            field_out.data[...] = MISSING
            field_regridded = regridder(field_in, field_out)
            output[t - start, o_lo:o_hi] = field_regridded.data[...]

        field_in.destroy()
        field_out.destroy()

    # Gather distributed results to root
    output = gather_reduce(output, global_shape=(nt, len(bnd_coords)))

    # Write output on root rank
    if local_pet == 0:
        _write_schism_output(nc_out, output, times, time_atts, len(bnd_coords))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regrid ESTOFS data to SCHISM open boundary format."
    )
    parser.add_argument("estofs_input", type=str, help="Input ESTOFS .nc file")
    parser.add_argument("schism_grid", type=str, help="SCHISM hgrid .nc file")
    parser.add_argument("regrid_output", type=str, help="Output .nc file")
    parser.add_argument("--cycle-date", required=True, help="Cycle date (YYYYMMDD)")
    parser.add_argument("--cycle-time", required=True, help="Cycle time (HHMM)")
    parser.add_argument("--length-hrs", type=int, default=180, help="Forecast length in hours")
    args = parser.parse_args()

    regrid_estofs(
        args.estofs_input,
        args.schism_grid,
        args.regrid_output,
        cycle_date=args.cycle_date,
        cycle_time=args.cycle_time,
        length_hrs=args.length_hrs,
    )


if __name__ == "__main__":
    main()
