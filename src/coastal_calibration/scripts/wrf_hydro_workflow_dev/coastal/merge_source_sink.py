#!/usr/bin/env python3
"""
Created on Mon Mar 29 22:15:15 2021.

@author: Camaron.George
"""

from __future__ import annotations

from os import environ, path

import netCDF4 as nc
import numpy as np


def log(*args, **kwargs):
    kwargs.update({"flush": True})


# output_path to where the discharge files are stored
output_path = environ["COASTAL_WORK_DIR"]
area_path = environ["COASTAL_ROOT_DIR"]

# read in discharge files from combine_sink_source.py
log("Reading source_sink_in.1")
soel1 = []
siel = []
with open(path.join(output_path, "source_sink.in.1")) as f:
    nsoel1 = int(f.readline())
    for i in range(nsoel1):
        soel1.append(int(f.readline()))
    next(f)
    nsiel = int(f.readline())
    for i in range(nsiel):
        siel.append(int(f.readline()))

log("Reading vsink.th.1")
with open(path.join(output_path, "vsink.th.1")) as f:
    for count, line in enumerate(f):
        pass
count += 1

si = np.zeros((count, nsiel + 1))
with open(path.join(output_path, "vsink.th.1")) as f:
    for j in range(count):
        line = f.readline()
        if len(line):
            si[j, :] = np.fromstring(line, dtype=float, sep=" ")
time = si[:, 0]
si = si[:, 1:]

log("Reading vsource.th.1")
so1 = np.zeros((count, nsoel1 + 1))
with open(path.join(output_path, "vsource.th.1")) as f:
    for j in range(count):
        line = f.readline()
        if len(line):
            so1[j, :] = np.fromstring(
                line, dtype=float, sep=" "
            )  # list(map(float, line.split()[1:]))
so1 = so1[:, 1:]

log("Reading precip_source.nc")
precip = nc.Dataset(path.join(output_path, "precip_source.nc"), "r")
so2 = precip.variables["vsource"][:]

# The discharge files (vsource.th.1 / vsink.th.1) may contain one more
# timestep than the precipitation forcing (precip_source.nc) because
# makeDischarge writes an extra trailing step.  Truncate the discharge
# arrays to match the precipitation time dimension so the merge works.
ntime = so2.shape[0]
if so1.shape[0] > ntime:
    so1 = so1[:ntime, :]
    si = si[:ntime, :]
    time = time[:ntime]

log("Merging sources...")
for i in range(len(soel1)):
    so2[:, soel1[i] - 1] = so2[:, soel1[i] - 1] + so1[:, i]

log("Applying minimum value threshold")
# read in element areas and calculate the threshold for each element
#  (threshold results in less than 1 cm change in water level for element)
threshold = np.genfromtxt(path.join(area_path, "element_areas.txt"))
threshold = (0.01 * threshold) / (3600.0 * (len(so2) - 1))

# find the max discharge for all elements and remove those
# elements where the max discharge is below the threshold
# md = [so2[:, i].max() for i in range(so2.shape[1])]
# keep = [idx for idx, val in enumerate(md) if val > threshold[idx]]
md = np.max(so2, axis=0)
keep = np.argwhere(md > threshold).ravel()

so2 = so2[:, keep]

# add one to each index in keep variable to find element numbers of sources
# keep = [keep[i] + 1 for i in range(len(keep))]
keep += 1

# mso = np.zeros((count, 2, (len(keep))))
# mso[:, 0, :] = int(-9999)

log("Writing source.nc output")
# write source.nc file
ncout = nc.Dataset(path.join(output_path, "source.nc"), "w", format="NETCDF4")
ncout.set_fill_off()

ncout.createDimension("time_vsource", len(time))
ncout.createDimension("time_vsink", len(time))
ncout.createDimension("time_msource", len(time))
ncout.createDimension("nsources", len(keep))
ncout.createDimension("nsinks", nsiel)
ncout.createDimension("ntracers", 2)
ncout.createDimension("one", 1)

ncso = ncout.createVariable("source_elem", "i4", ("nsources",))
ncsi = ncout.createVariable("sink_elem", "i4", ("nsinks",))
ncvso = ncout.createVariable(
    "vsource",
    "f8",
    (
        "time_vsource",
        "nsources",
    ),
    zlib=True,
)
ncvsi = ncout.createVariable(
    "vsink",
    "f8",
    (
        "time_vsink",
        "nsinks",
    ),
    zlib=True,
)
ncvmo = ncout.createVariable(
    "msource",
    "i4",
    (
        "time_msource",
        "ntracers",
        "nsources",
    ),
    zlib=True,
)
nctso = ncout.createVariable("time_vsource", "f8", ("time_vsource",))
nctsi = ncout.createVariable("time_vsink", "f8", ("time_vsink",))
nctmo = ncout.createVariable("time_msource", "f8", ("time_msource",))
ncvsos = ncout.createVariable("time_step_vsource", "f4", ("one",))
ncvsis = ncout.createVariable("time_step_vsink", "f4", ("one",))
ncvmos = ncout.createVariable("time_step_msource", "f4", ("one",))

ncso[:] = keep
ncsi[:] = siel
ncvso[:] = so2
ncvsi[:] = si
nctso[:] = time
nctsi[:] = time
nctmo[:] = time
ncvsos[:] = time[1] - time[0]
ncvsis[:] = time[1] - time[0]
ncvmos[:] = time[1] - time[0]

log("Filling msource")
fill_val = np.zeros((len(time), len(keep))) + -9999
ncvmo[:, 0, :] = fill_val
ncout.sync()

fill_val.fill(0)
ncvmo[:, 1, :] = fill_val
ncout.sync()

log("Processing complete")

ncout.close()
