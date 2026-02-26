from __future__ import annotations

import argparse

import numpy as np
from netCDF4 import Dataset


def correct_elevation(elev_file, correction_file):
    elev_correct = np.loadtxt(correction_file, delimiter=",", skiprows=1, usecols=5)

    with Dataset(elev_file, "r+") as ds:
        elev_var = ds["time_series"]
        nt = elev_var.shape[0]
        for t in range(nt):
            elev_var[t] = elev_var[t].ravel() - elev_correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("schism_elev2D", type=str, help="Input elev2D.th.nc file")
    parser.add_argument(
        "correction_file", type=str, help="elevation_correction.csv file for domain"
    )
    args = parser.parse_args()

    try:
        correct_elevation(args.schism_elev2D, args.correction_file)
    except ValueError:
        pass
    except OSError:
        pass
