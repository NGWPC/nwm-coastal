#!/usr/bin/env python3
"""Utility to use ESMF to regrid ESTOFS data files to the SCHISM elev2D.th.nc format.

NOTE: input schism hgrid boundary file created using gr3_2_esmf.py with the --filter_open_bnds flag
./gr3_2_esmf.py --filter_open_bnds hgrid.gr3 schism.hgrid.nc

Usage: ./regrid_estofs.py  /glade/scratch/rcabell/coastal/estofs.t00z.fields.cwl.nc \
    /glade/scratch/bpetzke/ForcingEngine/gr3_2_esmf/prvi.schism.hgrid.nc \
    /glade/scratch/bpetzke/ForcingEngine/regrid_estofs/prvi.estofs.t00z.fields.cwl.regrid.nc
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from time import monotonic

import ESMF

# import esmpy
import netCDF4
import numpy as np
from cftime import num2date
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)

# ESMF.Manager(debug=True)


local_pet = ESMF.local_pet()
pet_count = ESMF.pet_count()

schism_coord_name = "nodeCoords"
schism_open_boundary_name = "openBndNodes"
x_name = "x"
y_name = "y"
time_name = "time"
elem_name = "element"
time_step = 3600
missing = -9999.0

TEST = False
BND_BUFFER = 0.25


def create_locstream(lons, lats, name=""):
    # https://earthsystemmodeling.org/esmpy_doc/release/ESMF_8_1_0/html/examples.html#locstream-create

    assert len(lons) == len(lats)

    count = len(lons) // pet_count
    if count * pet_count < len(lons) and local_pet == (pet_count - 1):
        count += len(lons) - (count * pet_count)

    locstream = ESMF.LocStream(count, coord_sys=ESMF.CoordSys.SPH_DEG)

    lbounds = locstream.lower_bounds[0]
    ubounds = locstream.upper_bounds[0]

    locstream["ESMF:Lon"] = lons[lbounds:ubounds]
    locstream["ESMF:Lat"] = lats[lbounds:ubounds]

    return locstream


def regrid_chunk(beg_t, end_t, x, y, nt, in_field, coords):
    # print("Starting regrid_chunk kernel", flush=True)

    bnd_lons, bnd_lats = [[c[n] for c in coords] for n in (0, 1)]  # coords[:, 0], coords[:, 1]
    regrid = None

    mesh_in = create_locstream(x, y, "mesh_in")
    locstream_out = create_locstream(bnd_lons, bnd_lats, "bnd_out")

    field_from = ESMF.Field(mesh_in, name="EstofsIn")
    field_to = ESMF.Field(locstream_out, name="OpenBoundary")

    output = np.empty((nt, len(coords)))
    output[:] = 0

    i_lbounds = mesh_in.lower_bounds[0]
    i_ubounds = mesh_in.upper_bounds[0]

    o_lbounds = locstream_out.lower_bounds[0]
    o_ubounds = locstream_out.upper_bounds[0]

    for t in range(beg_t, end_t):
        monotonic()
        if local_pet == 0:
            pass

        data = in_field[t][i_lbounds:i_ubounds]  # only read once
        field_from.data[...] = data
        mesh_in["ESMF:Mask"] = data.mask.astype("i4")  # 0 is unmasked, 1 is masked

        field_to.data[...] = missing

        method = ESMF.RegridMethod.NEAREST_STOD
        regrid = ESMF.Regrid(
            srcfield=field_from,
            dstfield=field_to,
            regrid_method=method,
            unmapped_action=ESMF.UnmappedAction.IGNORE,
            src_mask_values=[1],
        )

        field_regridded = regrid(field_from, field_to)
        output[t - beg_t][o_lbounds:o_ubounds] = field_regridded.data[...]
        if local_pet == 0:
            pass
    # mesh_in.destroy()
    # locstream_out.destroy()

    return output


def regrid(nc_in, nc_grid, nc_out, regrid_field):
    FORECAST_START = 5

    if local_pet == 0:
        pass
    with netCDF4.Dataset(nc_grid) as f_in:
        coords = f_in[schism_coord_name][:]
        valid_indices = f_in[schism_open_boundary_name][:]

    if local_pet == 0:
        pass
    with netCDF4.Dataset(nc_in) as f_in:
        start = FORECAST_START  # ignore spinup

        etime_var = f_in[time_name]
        estart = num2date(etime_var[FORECAST_START], units=etime_var.units)

        # round up non-hourly ESTOFS start times
        if estart.minute != 0:
            estart = datetime(estart.year, estart.month, estart.day, estart.hour)
            estart += timedelta(hours=1)
        if local_pet == 0:
            pass

        cdate = os.environ["CYCLE_DATE"]
        ctime = os.environ["CYCLE_TIME"]
        total_hours = int(os.environ.get("LENGTH_HRS", 180)) + 1
        fdate = datetime.strptime(cdate + ctime, "%Y%m%d%H%M")
        if local_pet == 0:
            pass

        dt_h = int((fdate - estart).total_seconds() / 3600)
        start += dt_h
        if local_pet == 0:
            pass

        x = f_in[x_name][:]
        y = f_in[y_name][:]
        times = f_in[time_name][start : start + total_hours]
        t = len(times)
        input_time_atts = f_in[time_name].__dict__
        in_field = f_in[regrid_field]

        coords = [coords[i].tolist() for i in valid_indices]
        output_local = regrid_chunk(start, t + start, x, y, t, in_field, coords)

    output = np.zeros((t, len(coords))) if local_pet == 0 else None
    comm.Reduce(output_local, output, op=MPI.SUM)

    if local_pet == 0:
        with netCDF4.Dataset(nc_out, "w", format="NETCDF4") as f_out:
            f_out.createDimension("time", None)
            f_out.createDimension("nOpenBndNodes", len(coords))
            f_out.createDimension("nLevels", 1)
            f_out.createDimension("nComponents", 1)
            f_out.createDimension("one", 1)

            time_step_var = f_out.createVariable("time_step", "f8", ("one",))
            time_var = f_out.createVariable("time", "f8", ("time",))
            time_var.setncatts(input_time_atts)
            time_var.start_time = times[0]
            time_series_var = f_out.createVariable(
                "time_series",
                "f8",
                ("time", "nOpenBndNodes", "nLevels", "nComponents"),
                fill_value=missing,
                zlib=True,
            )

            time_step_var[:] = np.array([time_step])
            time_var[:] = np.arange(0, len(times) * time_step, time_step)
            # print(time_var)
            # for chunk in regridded_chunks:
            for t in range(len(times)):
                # t, data = output
                data = output[t]
                data = np.where(data > missing, data, 0)
                # print(f"Outputting t = {t}")
                time_series_var[t, :, 0, 0] = data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("estofs_input", type=str, help="Input .nc file")
    parser.add_argument("schism_grid", type=str, help=".nc file containing schism coordinates")
    parser.add_argument("regrid_output", type=str, help="Output .nc file")
    args = parser.parse_args()

    if local_pet == 0:
        pass
    regrid(args.estofs_input, args.schism_grid, args.regrid_output, "zeta")
    if local_pet == 0:
        pass


if __name__ == "__main__":
    main()
