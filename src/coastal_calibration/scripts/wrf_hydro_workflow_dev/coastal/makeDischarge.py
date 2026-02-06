#!/usr/bin/env python3
"""
Created on Fri Feb 19 21:49:26 2021.

@authors: Camaron.George@noaa.gov, rcabell@ucar.edu
"""

from __future__ import annotations

import glob
import os
from os import path

import netCDF4 as nc
import numpy as np


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


# path to nwmdata
nwm_output_dir = os.environ["WRF_HYDRO_ROOT"]
nwm_ana_dir = os.environ.get("NWM_ANA_DIR")
is_ana = "analysis" in os.environ.get(
    "NWM_CYCLE", "analysis"
)  # don't add extra CHRTOUT for reanalysis and analysis_assim

# path to reaches list file
params_dir = os.environ["COASTAL_WORK_DIR"]
tstep = 3600.0

# read in list of source and sink reaches
soelems = []
soids = []
sielems = []
siids = []
with open(path.join(params_dir, "nwmReaches.csv")) as f:
    nso = int(f.readline())
    for i in range(nso):
        line = f.readline()
        soelems.append(int(line.split()[0]))
        soids.append(int(line.split()[1]))
    next(f)
    nsi = int(f.readline())
    for i in range(nsi):
        line = f.readline()
        sielems.append(int(line.split()[0]))
        siids.append(int(line.split()[1]))

# build vsource and vsink arrays
chrtout_files = [sorted(glob.glob(path.join(nwm_ana_dir, "*CHRTOUT*")))[-1]] if not is_ana else []
chrtout_files.extend(sorted(glob.glob(path.join(nwm_output_dir, "*CHRTOUT*"))))

[print(f"\t{e}") for e in chrtout_files]

sub_hourly = any(
    i for i in chrtout_files if "15.CHRTOUT" in i
)  # assume only 15-minute output or hourly for now
chunk_size = 4 if sub_hourly else 1
rows = int(len(chrtout_files) / chunk_size)
if sub_hourly:
    vsource = np.zeros((rows + 1, len(soids)))
    vsink = np.zeros((rows + 1, len(siids)))
else:
    vsource = np.zeros((rows, len(soids)))
    vsink = np.zeros((rows, len(siids)))

# USE FIRST OUTPUT FILE
data = nc.Dataset(chrtout_files[0], "r")
featureID = data.variables["feature_id"][:]
streamflow = data.variables["streamflow"][:].filled(
    0
)  # replace missing values with zero contribution

# find locations of each read in featureID list
source = []
for i in soids:
    source.append(np.where(featureID == i)[0][0])
sink = []
for i in siids:
    sink.append(np.where(featureID == i)[0][0])

# replace first row of zeros in vsource and vsink arrays with streamflow data for time 0
vsource[0, :] = streamflow[source]
vsink[0, :] = -1 * streamflow[sink]

# get streamflow data for each reach at timestep and replace each row of zeros in vsource and vsink arrays
for row, files in enumerate(chunker(chrtout_files[1:], chunk_size)):
    streamflow = None
    for file in files:
        data = nc.Dataset(file, "r")
        if streamflow is None:
            streamflow = data.variables["streamflow"][:].filled(
                0
            )  # replace missing values with zero contribution
        else:
            streamflow += data.variables["streamflow"][:].filled(0)

    streamflow /= len(files)

    vsource[row + 1, :] = streamflow[source]
    vsink[row + 1, :] = -1 * streamflow[sink]

# write initial set of discharge files
t = 0.0
o = open(path.join(params_dir, "vsource.th"), "w")
for i in range(vsource.shape[0]):
    o.write(str(t) + "\t")
    for j in range(vsource.shape[1]):
        if j != vsource.shape[1] - 1:
            o.write(str(vsource[i, j]) + "\t")
        else:
            o.write(str(vsource[i, j]) + "\n")
    t += tstep
o.close()


t = 0.0
o = open(path.join(params_dir, "vsink.th"), "w")
for i in range(vsink.shape[0]):
    o.write(str(t) + "\t")
    for j in range(vsink.shape[1]):
        if j != vsink.shape[1] - 1:
            o.write(str(vsink[i, j]) + "\t")
        else:
            o.write(str(vsink[i, j]) + "\n")
    t += tstep
o.close()

o = open(path.join(params_dir, "source_sink.in"), "w")
o.write(str(len(soelems)) + "\n")
for i in range(len(soelems)):
    o.write(str(soelems[i]) + "\n")
o.write("\n")
o.write(str(len(sielems)) + "\n")
for i in range(len(sielems)):
    o.write(str(sielems[i]) + "\n")
o.close()
