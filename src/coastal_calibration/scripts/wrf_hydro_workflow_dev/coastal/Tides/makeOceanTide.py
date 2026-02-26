#!/usr/bin/env python3
"""
Created on Fri Sep 17 18:47:06 2021.

@author: Camaron.George
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import netCDF4 as nc
import numpy as np
import pytides.constituent as con
from pytides.tide import Tide
from scipy.interpolate import griddata

# path = '/scratch2/NCEPDEV/ohd/Camaron.George/'
# grid = 'schismRuns/EastGulf/hgrid.gr3'
# out = 'schismRuns/EastGulf/tides.nc'
# startPred = datetime(2020,9,2,13)#time zero for the prediction
# hours = np.arange(60.0)#number of hours in the prediction
# predTimes = Tide._times(startPred,hours)


def generate_tidal_levels(consts_path, grid_file, output_file, start_time, total_hours):
    lon = []
    lat = []
    elems = []
    bnodes = []
    with open(grid_file) as f:
        next(f)
        line = f.readline()
        ne = int(line.split()[0])
        nn = int(line.split()[1])
        for i in range(nn):
            line = f.readline()
            lon.append(float(line.split()[1]))
            lat.append(float(line.split()[2]))
        for i in range(ne):
            line = f.readline()
            elems.append(line)
        num_open_boundaries = int(f.readline().split()[0])
        total_open_boundary_nodes = int(f.readline().split()[0])
        for _ in range(num_open_boundaries):
            num_boundary_nodes = int(f.readline().split()[0])
            for _ in range(num_boundary_nodes):
                bnodes.append(int(f.readline()))
        assert len(bnodes) == total_open_boundary_nodes

    lo = [lon[b - 1] for b in bnodes]
    la = [lat[b - 1] for b in bnodes]

    tide_constants = ["k1", "k2", "m2", "n2", "o1", "p1", "q1", "s2"]
    consfiles = [os.path.join(consts_path, x) + ".nc" for x in tide_constants]
    pha = np.zeros((len(lo), len(consfiles)))
    amp = np.zeros((len(lo), len(consfiles)))

    col = 0
    for file in consfiles:
        data = nc.Dataset(file, "r")
        if col == 0:
            x = data.variables["lon"][:]
            y = data.variables["lat"][:]
            x, y = np.meshgrid(x, y)
            i = np.where(x > 180.0)
            x[i] = x[i] - 360.0
            i = np.where(
                (x < max(lo) + 1) & (x > min(lo) - 1) & (y < max(la) + 1) & (y > min(la) - 1)
            )
            x = x[i]
            y = y[i]

        a = data.variables["amplitude"][:]
        a = a[i]
        p = data.variables["phase"][:]
        p = p[i]

        xI = x[not a.mask]
        yI = y[not a.mask]
        p = p[not a.mask]
        a = a[not a.mask]

        amp[:, col] = griddata((xI, yI), a, (lo, la), method="linear")
        pha[:, col] = griddata((xI, yI), p, (lo, la), method="linear")
        col += 1

    pred_times = Tide._times(start_time, total_hours)

    wl = np.zeros((len(pred_times), amp.shape[0]))
    cons = [c for c in con.noaa if c != con._Z0]
    n = [3, 34, 0, 2, 5, 29, 25, 1]  # corresponds to position in list in constituent.py
    cons = [cons[c] for c in n]
    model = np.zeros(len(cons), dtype=Tide.dtype)
    model["constituent"] = cons
    for i in range(amp.shape[0]):
        model["amplitude"] = amp[i, :]
        model["phase"] = pha[i, :]

        tide = Tide(model=model, radians=False)
        wl[:, i] = (tide.at(pred_times)) / 100.0

    mode = "a" if os.path.exists(output_file) else "w"

    # open a netCDF file to write
    ncout = nc.Dataset(output_file, mode, format="NETCDF4")

    if mode == "w":
        # define axis size
        ncout.createDimension("time", None)
        ncout.createDimension("nOpenBndNodes", amp.shape[0])

        # create time axis
        nctime = ncout.createVariable("time", "f8", ("time",))

        # create water level time series
        ncwl = ncout.createVariable(
            "time_series",
            "f8",
            (
                "time",
                "nOpenBndNodes",
            ),
        )
        start = 0

        # copy axis from original dataset
        nctime[:] = np.arange(651600.0, 865000, 3600.0)
        ncwl[:] = wl

    else:
        nctime = ncout["time"]
        ncwl = ncout["time_series"]
        start = 181

    # generate timestamps  # TODO: parameterize these!
    t_step = 3600
    t_start = start * t_step
    t_end = (start + total_hours.stop) * t_step
    new_times = np.arange(t_start, t_end, t_step)  # 181 hours to total_hours hours
    nctime[start:] = new_times
    ncwl[start:] = wl

    ncout.close()


def test():
    params_path = "/glade/work/rcabell/ecflow/hydro-workflow/coastal/Tides/TidalConst"
    grid_file = "/glade/work/rcabell/ecflow/hydro-workflow/coastal/prvi/hgrid.gr3"
    output_file = "/glade/u/home/rcabell/work/ecflow/hydro-workflow/jobdir/prvi/medium_range_mem1/coastal/prvi/elev2D.th.nc"
    start_time = datetime(2020, 9, 1, 6, 0) + timedelta(hours=182)
    total_hours = range(60)
    generate_tidal_levels(params_path, grid_file, output_file, start_time, total_hours)


def workflow_driver():
    """Extend ESTOFS boundary conditions with tidal predictions.

    In the NWM operational medium-range forecast, ESTOFS/STOFS water
    level data covers only the first 180 hours.  For longer forecasts
    (up to 241 h) this function fills hours 181+ with tidal predictions
    generated from harmonic constituents via pytides.  The result is
    appended to the existing ``elev2D.th.nc`` produced by the ESTOFS
    regridding step.

    The constants 181 and 182 originate from NWM's medium-range
    configuration and should be parameterized when this script is
    rewritten.
    """
    consts_path = os.environ["TIDAL_CONSTANTS_DIR"]
    grid_file = os.environ["COASTAL_DOMAIN_GR3"]
    output_file = os.environ["SCHISM_OUTPUT_FILE"]
    cdate = os.environ["CYCLE_DATE"]
    ctime = os.environ["CYCLE_TIME"]
    total_hours = int(os.environ.get("LENGTH_HRS", 241))
    # ESTOFS covers hours 0-180; nothing to fill for short forecasts
    if total_hours < 182:
        return

    # Start tidal prediction at hour 181 (first hour beyond ESTOFS)
    start_time = datetime.strptime(cdate + ctime, "%Y%m%d%H%M") + timedelta(hours=181)
    hour_range = range(total_hours - 181)
    generate_tidal_levels(consts_path, grid_file, output_file, start_time, hour_range)


if __name__ == "__main__":
    workflow_driver()
