#!/usr/bin/env python3
"""
Created on Wed Mar 31 18:43:00 2021.

@author: Camaron.George@noaa.gov / rcabell@ucar.edu
"""

from __future__ import annotations

import glob

# from pyproj import CRS, Transformer
import os
from os import path

import netCDF4 as nc
import numpy as np


def round_down(n, decimals=0):
    import math

    multiplier = 10**decimals
    return math.floor(n * multiplier) / multiplier


def slp(temp, mixing, height, press):
    g0 = 9.80665
    Rd = 287.058
    epsilon = 0.622

    Tv = temp * (1 + (mixing / epsilon)) / (1 + mixing)
    H = Rd * Tv / g0

    press_sl = press / np.exp(-height / H)

    return press_sl


class App:
    def __init__(self) -> None:
        # paths to where the forcing data is loaded / saved
        forcing_input_path = os.environ["COASTAL_FORCING_INPUT_DIR"]
        output_path = os.environ["COASTAL_WORK_DIR"]

        # file for precipitation only; will be used to combine with discharge later
        self.precip_out_nc = path.join(output_path, "precip.nc")

        # file for rest of atmo variables, which will be used in the simulation so this should be saved
        # in the folder where it will be used in the run
        self.sflux_out_nc = path.join(output_path, "sflux", "sflux_air_1.0001.nc")

        # start times, sim length
        self.start_year = os.environ["FORCING_START_YEAR"]
        self.start_month = os.environ["FORCING_START_MONTH"]
        self.start_day = os.environ["FORCING_START_DAY"]
        self.start_hour = int(os.environ["FORCING_START_HOUR"])
        self.length_hours = int(os.environ.get("LENGTH_HRS", 0))
        self.ana_flag = self.length_hours < 0
        self.length_hours = abs(self.length_hours)

        # # analysis file/data that will be used for zero time step (will be handled by FE)
        # file = path.join(forcing_input_path, 'nwm.t00z.analysis_assim.forcing.tm00.conus.latlon.nc')
        # if self.ana_flag:
        #     self.dir_date = os.environ["FORCING_END_DATE"]
        # else:
        #     self.dir_date = os.environ["FORCING_BEGIN_DATE"]
        # if len(self.dir_date) == 12:
        #     # remove minutes
        #     self.dir_date = self.dir_date[:-2]

        # TODO: compute actual file names to detect missing data
        self.forcing_input_glob = path.join(forcing_input_path, "*LDASIN_DOMAIN1")

        # Load geospatial data (lats, lons, terrain heights)
        geo_file = os.environ["GEOGRID_FILE"]
        with nc.Dataset(geo_file) as geo:
            self.height = geo["HGT_M"][0, :]
            self.lats = geo["XLAT_M"][0, :]
            self.lons = geo["XLONG_M"][0, :]

    def create_output(self):
        ncout = nc.Dataset(self.sflux_out_nc, "w", format="NETCDF4")

        files = sorted(glob.glob(self.forcing_input_glob))
        data = nc.Dataset(files[0], "r")

        # get lon/lat locations of data
        # cf_dict = data['crs'].__dict__
        # proj_crs = CRS.from_cf(cf_dict)
        # xformer = Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)
        # x, y = np.meshgrid(data['x'][:], data['y'][:])
        # lons, lats = xformer.transform(x, y)

        ncout.createDimension("time", len(files) + 1)
        ncout.createDimension("ny_grid", self.lats.shape[0])
        ncout.createDimension("nx_grid", self.lons.shape[1])

        nctime = ncout.createVariable("time", "f4", ("time",))
        nclon = ncout.createVariable(
            "lon",
            "f4",
            (
                "ny_grid",
                "nx_grid",
            ),
        )
        nclat = ncout.createVariable(
            "lat",
            "f4",
            (
                "ny_grid",
                "nx_grid",
            ),
        )
        ncu = ncout.createVariable(
            "uwind",
            "f4",
            (
                "time",
                "ny_grid",
                "nx_grid",
            ),
        )
        ncv = ncout.createVariable(
            "vwind",
            "f4",
            (
                "time",
                "ny_grid",
                "nx_grid",
            ),
        )
        ncp = ncout.createVariable(
            "prmsl",
            "f4",
            (
                "time",
                "ny_grid",
                "nx_grid",
            ),
        )
        nct = ncout.createVariable(
            "stmp",
            "f4",
            (
                "time",
                "ny_grid",
                "nx_grid",
            ),
        )
        ncq = ncout.createVariable(
            "spfh",
            "f4",
            (
                "time",
                "ny_grid",
                "nx_grid",
            ),
        )

        nctime.long_name = "Time"
        nctime.standard_name = "time"

        time = np.arange(0, (1 / 24) * (len(files) + 1), 1 / 24)
        time += int(self.start_hour) / 24.0
        time[0] = round_down(time[0], 7)
        nctime[:] = time

        data.close()

        # set up field variables

        baseDateStr = f"{self.start_year}-{self.start_month}-{self.start_day}"  # "2020-08-26 0'
        baseDate = list(map(np.int32, [self.start_year, self.start_month, self.start_day, 0]))
        nctime.units = "days since " + baseDateStr
        nctime.base_date = baseDate

        nclon.long_name = "Longitude"
        nclon.standard_name = "longitude"
        nclon.units = "degrees_east"
        nclon[:] = self.lons

        nclat.long_name = "Latitude"
        nclat.standard_name = "latitude"
        nclat.units = "degrees_north"
        nclat[:] = self.lats

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
            data = nc.Dataset(file)

            nct[i, :] = data.variables["T2D"][:]
            ncq[i, :] = data.variables["Q2D"][:]
            ncu[i, :] = data.variables["U2D"][:]
            ncv[i, :] = data.variables["V2D"][:]
            ncp[i, :] = slp(nct[i, :], ncq[i, :], self.height, data.variables["PSFC"][:])
            # ncr[i, :] = data.variables['RAINRATE'][:]

            data.close()

        # copy last data
        ncu[-1] = ncu[-2]
        ncv[-1] = ncv[-2]
        ncp[-1] = ncp[-2]
        nct[-1] = nct[-2]
        ncq[-1] = ncq[-2]

        ncout.close()


if __name__ == "__main__":
    App().create_output()
