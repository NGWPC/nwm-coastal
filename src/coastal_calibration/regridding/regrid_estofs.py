r"""Regrid ESTOFS water level data to SCHISM open boundary nodes using ESMF.

This module regrids ESTOFS ``zeta`` (water surface elevation) from an
unstructured node grid to SCHISM open boundary node locations.  The
default interpolation is BILINEAR (barycentric within source
triangles), with a NEAREST_STOD fallback for destination points that
fall outside any valid source element.  The output is written in
SCHISM's ``elev2D.th.nc`` format.

The BILINEAR + Mesh approach replaces an earlier LocStream-only
NEAREST_STOD setup that produced spatially discontinuous boundary
forcing when the STOFS source resolution was much coarser than the
SCHISM destination spacing — most notably on short cut-line boundaries
created by polygon-based mesh extraction.  See the project README for
the full problem statement.

MPI-parallel: ESMF decomposes the source mesh and destination
LocStream across ranks; results are gathered to rank 0 for writing.

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
from typing import TYPE_CHECKING, Any

import esmpy as ESMF
import netCDF4
import numpy as np
from cftime import num2date

from coastal_calibration.logging import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .esmf_utils import (
    Regridder,
    build_locstream,
    build_unstructured_mesh,
    gather_reduce,
)

local_pet = ESMF.local_pet()

FORECAST_START = 5
TIME_STEP = 3600
MISSING = -9999.0


def _determine_time_range(
    f_in: netCDF4.Dataset,
    forecast_start: int,
    cycle_date: str,
    cycle_time: str,
    length_hrs: int,
) -> tuple[int, int, NDArray[Any], dict[str, Any]]:
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


def _regrid_timeseries(
    *,
    f_in: netCDF4.Dataset,
    regrid_field: str,
    src_keep_idx: NDArray[np.integer[Any]],
    field_in: Any,
    field_out: Any,
    field_fallback: Any,
    bilinear: Any,
    nearest: Any,
    o_lo: int,
    o_hi: int,
    start: int,
    nt: int,
    n_dst: int,
) -> NDArray[np.float64]:
    """Run the BILINEAR + NEAREST_STOD per-timestep loop.

    Returns the ``(nt, n_dst)`` output array filled in the local
    boundary slice ``[o_lo:o_hi]``. Fill-valued source nodes are zeroed
    pre-interpolation, then any destination point that ends up
    fill-valued or non-finite from BILINEAR is back-filled from the
    NEAREST_STOD result.
    """
    output = np.zeros((nt, n_dst))
    for t in range(start, start + nt):
        raw = f_in[regrid_field][t]
        local_data = np.asarray(raw)[src_keep_idx].astype(np.float64, copy=False)
        # Replace fill values with 0 so they don't pollute bilinear sums
        local_data[local_data <= MISSING + 1.0] = 0.0
        field_in.data[...] = local_data

        field_out.data[...] = MISSING
        field_fallback.data[...] = MISSING

        bilinear(field_in, field_out)
        nearest(field_in, field_fallback)

        primary = np.asarray(field_out.data[...])
        backup = np.asarray(field_fallback.data[...])
        invalid = (primary <= MISSING + 1.0) | ~np.isfinite(primary)
        primary[invalid] = backup[invalid]
        output[t - start, o_lo:o_hi] = primary
    return output


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

    # Bbox of destination boundary nodes — used to subset the very large
    # global STOFS mesh down to the relevant region before handing it to
    # ESMF.  This keeps memory and setup cost manageable.
    dst_bbox = (
        float(bnd_lons.min()),
        float(bnd_lats.min()),
        float(bnd_lons.max()),
        float(bnd_lats.max()),
    )

    with netCDF4.Dataset(nc_in) as f_in:
        start, nt, times, time_atts = _determine_time_range(
            f_in, FORECAST_START, cycle_date, cycle_time, length_hrs
        )

        src_lon = f_in["x"][:]
        src_lat = f_in["y"][:]
        src_elements = f_in["element"][:]
        src_start_index = int(getattr(f_in["element"], "start_index", 1))

        # Build the source as an ESMF.Mesh (with element connectivity)
        # so we can do BILINEAR / barycentric interpolation.  The mesh
        # is built once and reused across all timesteps.
        src_mesh, src_keep_idx = build_unstructured_mesh(
            src_lon,
            src_lat,
            src_elements,
            start_index=src_start_index,
            bbox=dst_bbox,
            bbox_buffer_deg=2.0,
        )
        n_src_kept = len(src_keep_idx)
        if local_pet == 0:
            logger.info(
                "STOFS source mesh: %d/%d nodes kept after bbox filter",
                n_src_kept,
                len(src_lon),
            )

        # Destination remains a LocStream — boundary points have no
        # connectivity.  Mesh→LocStream BILINEAR is supported by ESMF.
        locstream_out = build_locstream(bnd_lons, bnd_lats)

        # Source field on the mesh (defined at NODE locations).
        field_in = ESMF.Field(src_mesh, name="EstofsIn", meshloc=ESMF.MeshLoc.NODE)
        field_out = ESMF.Field(locstream_out, name="OpenBoundary")
        field_fallback = ESMF.Field(locstream_out, name="OpenBoundaryFallback")

        o_lo = locstream_out._global_lower  # pyright: ignore[reportAttributeAccessIssue]
        o_hi = locstream_out._global_upper  # pyright: ignore[reportAttributeAccessIssue]

        # Build the BILINEAR regridder once (reused across timesteps).
        # The fallback NEAREST_STOD handles any destination point that
        # falls outside the source mesh.  Masked source nodes (dry land
        # in STOFS) are simply set to 0 in the source data — distant
        # masked nodes get small bilinear weights at boundary destinations
        # so this introduces only a small bias that is dwarfed by the
        # nearby wet-ocean values dominating the interpolation.
        bilinear = Regridder(
            field_in,
            field_out,
            method=ESMF.RegridMethod.BILINEAR,  # pyright: ignore[reportArgumentType]
            unmapped_action=ESMF.UnmappedAction.IGNORE,  # pyright: ignore[reportArgumentType]
        )
        nearest = Regridder(
            field_in,
            field_fallback,
            method=ESMF.RegridMethod.NEAREST_STOD,  # pyright: ignore[reportArgumentType]
            unmapped_action=ESMF.UnmappedAction.IGNORE,  # pyright: ignore[reportArgumentType]
        )

        output = _regrid_timeseries(
            f_in=f_in,
            regrid_field=regrid_field,
            src_keep_idx=src_keep_idx,
            field_in=field_in,
            field_out=field_out,
            field_fallback=field_fallback,
            bilinear=bilinear,
            nearest=nearest,
            o_lo=o_lo,
            o_hi=o_hi,
            start=start,
            nt=nt,
            n_dst=len(bnd_lons),
        )

        bilinear.destroy()
        nearest.destroy()
        field_in.destroy()
        field_out.destroy()
        field_fallback.destroy()
        src_mesh.destroy()

    # Gather distributed results to root
    output = gather_reduce(output, global_shape=(nt, len(bnd_coords)))

    # Write output on root rank
    if local_pet == 0:
        if output is None:
            msg = "gather_reduce returned None on root rank"
            raise RuntimeError(msg)
        from coastal_calibration._nc_io import write_elev2d_th

        nt = len(times)
        write_elev2d_th(
            nc_out,
            n_open_bnd_nodes=len(bnd_coords),
            time_seconds=np.arange(0, nt * TIME_STEP, TIME_STEP),
            time_step_seconds=TIME_STEP,
            time_series=output,  # (nt, n_bnd_nodes); helper reshapes to 4-D
            time_attrs={**time_atts, "start_time": times[0]},
            missing=MISSING,
        )


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
