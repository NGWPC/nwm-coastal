r"""Regrid WRF-Hydro forcing data to lat-lon grids and SCHISM mesh elements.

This module provides ``CoastalForcingRegridder``, which regrids NWM/WRF-Hydro
LDASIN forcing files to:

1. **Lat-lon grid**: Atmospheric variables (U2D, V2D, LWDOWN, T2D, Q2D,
   PSFC/SLP, SWDOWN, LQFRAC) are bilinearly interpolated from the WRF-Hydro
   curvilinear grid to a regular 0.01-degree lat-lon grid.

2. **SCHISM mesh**: RAINRATE is bilinearly interpolated to SCHISM mesh elements,
   then converted to volumetric flux (m³/s) using element areas.

MPI-parallel: ESMF decomposes grids/meshes across ranks; results are
gathered to rank 0 for writing.

Usage::

    mpirun -np 4 python -m coastal_calibration.regridding.regrid_forcings \\
        --input-dir /path/to/nwm --output-dir /path/to/output \\
        --geogrid-file geo_em.nc --schism-mesh hgrid.nc \\
        --length-hrs 180 --forcing-begin-date 2024010100 \\
        --job-index 0 --job-count 1
"""

from __future__ import annotations

import math
from pathlib import Path

import esmpy as ESMF
import numpy as np
from netCDF4 import Dataset

from coastal_calibration.utils.logging import logger

from .esmf_utils import (
    Regridder,
    allreduce_minmax,
    build_grid,
    gather_reduce,
    gatherv_1d,
)


def _pick_time_var(ds: Dataset) -> str:
    """Return the name of the time variable in *ds*.

    WRF-Hydro LDASIN files use ``"time"``; ERA5/HRRR-derived files may use
    ``"valid_time"``.  Raises ``KeyError`` with a descriptive message if
    neither is present.
    """
    for name in ("time", "valid_time"):
        if name in ds.variables:
            return name
    raise KeyError(
        f"Expected 'time' or 'valid_time' in {ds.filepath()}, found: {list(ds.variables)}"
    )


def sea_level_pressure(
    temp: np.ndarray,
    mixing: np.ndarray,
    height: np.ndarray,
    press: np.ndarray,
) -> np.ndarray:
    """Compute sea-level pressure from surface pressure via hypsometric equation.

    Parameters
    ----------
    temp
        Temperature (K), e.g. T2D.
    mixing
        Water vapor mixing ratio (kg/kg), e.g. Q2D.
    height
        Surface elevation (m), e.g. HGT_M.
    press
        Surface pressure (Pa), e.g. PSFC.

    Returns
    -------
    Sea-level pressure (Pa).
    """
    g0 = 9.80665
    Rd = 287.058
    epsilon = 0.622

    Tv = temp * (1 + (mixing / epsilon)) / (1 + mixing)
    H = Rd * Tv / g0
    return press / np.exp(-height / H)


class CoastalForcingRegridder:
    """Regrids WRF-Hydro forcing data to lat-lon and SCHISM mesh.

    Parameters
    ----------
    input_dir
        Directory containing WRF-Hydro LDASIN forcing files.
    output_dir
        Directory for regridded output files.
    geo_em_path
        Path to the WRF geogrid file (for source grid definition).
    schism_mesh_path
        Path to the SCHISM mesh in ESMFMESH format.
    """

    #: Atmospheric variables to regrid to lat-lon
    LATLON_VARS = ("U2D", "V2D", "LWDOWN", "T2D", "Q2D", "PSFC", "SWDOWN", "LQFRAC")

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        geo_em_path: Path,
        schism_mesh_path: Path,
        *,
        job_index: int | None = None,
        job_count: int | None = None,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.root = ESMF.local_pet() == 0

        self.job_idx = job_index
        self.job_count = job_count

        # Load SCHISM mesh
        self.schism_mesh = ESMF.Mesh(
            filename=str(schism_mesh_path), filetype=ESMF.FileFormat.ESMFMESH
        )
        with Dataset(schism_mesh_path, "r") as smesh:
            self.total_elements = smesh.dimensions["elementCount"].size

        # Determine output lat/lon range from mesh node coords (MPI-aware)
        node_lons = self.schism_mesh.coords[0][0]
        node_lats = self.schism_mesh.coords[0][1]
        lon_min, lon_max = allreduce_minmax(node_lons)
        lat_min, lat_max = allreduce_minmax(node_lats)

        # Read source grid and height from geogrid file
        ds = Dataset(geo_em_path, "r")
        self.src_height = ds.variables["HGT_M"][:]

        xlat = ds.variables["XLAT_M"][0, :].T
        xlon = ds.variables["XLONG_M"][0, :].T
        clat = ds.variables["XLAT_C"][0, :].T
        clon = ds.variables["XLONG_C"][0, :].T
        ds.close()

        # Build source (WRF-Hydro curvilinear) grid
        self.in_grid, self.in_bounds = build_grid(xlat, xlon, clat, clon)

        # Build destination regular lat-lon grid
        dlat = dlon = 0.01
        self.lats = np.arange(
            math.floor(lat_min / dlat) * dlat,
            (math.ceil(lat_max / dlat) * dlat) + dlat,
            dlat,
        )
        self.lons = np.arange(
            math.floor(lon_min / dlon) * dlon,
            (math.ceil(lon_max / dlon) * dlon) + dlon,
            dlon,
        )
        longitudes, latitudes = np.meshgrid(self.lons, self.lats, indexing="ij")
        self.out_grid, self.out_bounds = build_grid(latitudes, longitudes)

        # Regridder handles (lazy, built on first use)
        self._latlon_regridder = None
        self._schism_regridder = None

        self.schism_first_timestep = None

    def _read_start_time(self, ds: Dataset) -> float:
        """Extract the forcing start time from an input dataset."""
        if "time" in ds.variables:
            return ds["time"][0] * 60  # minutes -> seconds
        if "valid_time" in ds.variables:
            return ds["valid_time"][0]
        raise KeyError("Input file has neither 'time' nor 'valid_time' variable")

    def _init_vsource_nc(self, ds: Dataset, ntimes: int):
        """Create dimensions and variables for the SCHISM vsource file."""
        ds.createDimension("time_vsource", ntimes)
        ds.createDimension("nsources", self.total_elements)
        ds.createDimension("one", 1)

        eso = ds.createVariable("source_elem", "i4", ("nsources",))
        ds.createVariable("vsource", "f8", ("time_vsource", "nsources"), zlib=True)
        ds.createVariable("time_vsource", "f8", ("time_vsource",))
        vts = ds.createVariable("time_step_vsource", "f4", ("one",))

        eso[:] = np.arange(1, self.total_elements + 1)
        vts[:] = 3600

    def _regrid_to_schism(self, input_file: Path, vsource_ds: Dataset | None):
        """Regrid RAINRATE to SCHISM mesh elements and write to vsource."""
        input_ds = Dataset(input_file)

        # Populate source field
        in_field = ESMF.Field(grid=self.in_grid, name="rainrate-in")
        b = self.in_bounds
        in_field.data[...] = input_ds.variables["RAINRATE"][0, :].T[
            b.x_lo : b.x_hi, b.y_lo : b.y_hi
        ]

        # Populate destination field on mesh elements
        out_field = ESMF.Field(
            grid=self.schism_mesh, meshloc=ESMF.MeshLoc.ELEMENT, name="rainrate-out"
        )
        # Initialise to 0 so unmapped elements (IGNORE action) are left at 0.
        out_field.data[...] = 0.0

        # Build regridder once, reuse for subsequent files
        if self._schism_regridder is None:
            self._schism_regridder = Regridder(
                in_field,
                out_field,
                method=ESMF.RegridMethod.BILINEAR,
                unmapped_action=ESMF.UnmappedAction.IGNORE,
                extrap_method=ESMF.ExtrapMethod.NONE,
            )

        out_field = self._schism_regridder(in_field, out_field)
        # Clamp any negative bilinear interpolation artefacts to zero.
        np.clip(out_field.data, 0.0, None, out=out_field.data)

        # Convert to volumetric flux (m^3/s)
        R0_SCHISM = 6378206.4  # earth radius in meters used by SCHISM
        DENSITY_FACTOR = 1000

        unit_areas = ESMF.Field(self.schism_mesh, meshloc=ESMF.MeshLoc.ELEMENT, name="areafield")
        unit_areas.get_area()
        areas_m2 = unit_areas.data[...] * (R0_SCHISM * R0_SCHISM)
        out_field.data[...] *= areas_m2 / DENSITY_FACTOR
        unit_areas.destroy()

        # Gather distributed data to root
        local_count = self.schism_mesh.size[1]
        all_elements = gatherv_1d(out_field.data, local_count)

        if all_elements is not None and len(all_elements) != self.total_elements:
            msg = (
                f"Gathered element count {len(all_elements)} != "
                f"mesh dimension {self.total_elements} - dimension mismatch would "
                "corrupt the vsource output file"
            )
            raise ValueError(msg)

        # Write on root rank
        if self.root and vsource_ds is not None:
            step_time = self._read_start_time(input_ds)
            output_ts = int(step_time - self.schism_first_timestep)
            output_idx = output_ts // 3600
            vsource_ds["time_vsource"][output_idx] = output_ts
            vsource_ds["vsource"][output_idx, :] = all_elements
            vsource_ds.sync()

        in_field.destroy()
        out_field.destroy()
        input_ds.close()

    def _init_latlon_nc(self, output_ds: Dataset, nlats: int, nlons: int, input_ds: Dataset):
        """Create dimensions, coordinates, and time variable for lat-lon output."""
        output_ds.createDimension(dimname="lat", size=nlats)
        output_ds.createDimension(dimname="lon", size=nlons)
        output_ds.createDimension(dimname="time", size=0)

        lat_coord = output_ds.createVariable(
            varname="lat", dimensions=("lat",), datatype=self.lats.dtype
        )
        lat_coord.long_name = "latitude"
        lat_coord.units = "degrees_north"
        lat_coord.standard_name = "latitude"
        lat_coord.axis = "Y"

        lon_coord = output_ds.createVariable(
            varname="lon", dimensions=("lon",), datatype=self.lons.dtype
        )
        lon_coord.long_name = "longitude"
        lon_coord.units = "degrees_east"
        lon_coord.standard_name = "longitude"
        lon_coord.axis = "X"

        in_time = input_ds.variables[_pick_time_var(input_ds)]
        time_coord = output_ds.createVariable(
            varname="time", dimensions=("time",), datatype=in_time.datatype
        )
        time_coord.long_name = "valid output time"
        time_coord.units = in_time.units
        time_coord.calendar = "standard"
        time_coord.standard_name = "time"

    def _regrid_to_latlon(self, input_file: Path, apply_slp: bool = True):  # noqa: PLR0912
        """Regrid atmospheric variables to a regular lat-lon grid."""
        input_ds = Dataset(input_file)
        nlons, nlats = self.out_grid.max_index

        # Prepare output dataset on root
        if self.root:
            output_path = self.output_dir / (input_file.stem + ".latlon.nc")
            output_ds = Dataset(output_path, "w")
            self._init_latlon_nc(output_ds, nlats, nlons, input_ds)
        else:
            output_ds = None

        for variable in self.LATLON_VARS:
            if variable not in input_ds.variables:
                continue

            # Read and optionally transform the variable
            data = input_ds.variables[variable][0, :].T
            var_name = variable
            var_attrs = {}
            for attr in ("standard_name", "long_name", "units"):
                if attr in input_ds.variables[variable].ncattrs():
                    var_attrs[attr] = getattr(input_ds.variables[variable], attr)

            if apply_slp and variable == "PSFC":
                data = sea_level_pressure(
                    temp=input_ds.variables["T2D"][0, :].T,
                    mixing=input_ds.variables["Q2D"][0, :].T,
                    height=self.src_height[0, :].T,
                    press=data,
                )
                var_name = "SLP"
                var_attrs = {
                    "standard_name": "air_pressure_at_mean_sea_level",
                    "long_name": "Air pressure reduced to mean sea level",
                    "units": "Pa",
                }

            # Create output variable on root
            if self.root:
                new_var = output_ds.createVariable(
                    varname=var_name, datatype="f4", dimensions=("time", "lat", "lon")
                )
                for attr, val in var_attrs.items():
                    setattr(new_var, attr, val)

            # Populate source field with local partition slice
            in_field = ESMF.Field(grid=self.in_grid, name=f"{variable}-in")
            b = self.in_bounds
            in_field.data[...] = data[b.x_lo : b.x_hi, b.y_lo : b.y_hi]

            out_field = ESMF.Field(grid=self.out_grid, name=f"{variable}-out")
            out_field.data[...] = 0.0

            # Build regridder once, reuse for subsequent variables/files
            if self._latlon_regridder is None:
                self._latlon_regridder = Regridder(
                    in_field,
                    out_field,
                    method=ESMF.RegridMethod.BILINEAR,
                    unmapped_action=ESMF.UnmappedAction.IGNORE,
                )
            else:
                self._latlon_regridder(
                    in_field, out_field, zero_region=ESMF.constants.Region.SELECT
                )

            # Assemble global output from all partitions
            global_output = np.zeros((nlons, nlats))
            ob = self.out_bounds
            global_output[ob.x_lo : ob.x_hi, ob.y_lo : ob.y_hi] = out_field.data[...]

            final_output = gather_reduce(global_output, global_shape=(nlons, nlats))

            if self.root:
                output_ds.variables[var_name][0, :] = final_output.T

            in_field.destroy()
            out_field.destroy()

        # Write coordinates and close
        if self.root:
            output_ds.variables["lat"][:] = self.lats
            output_ds.variables["lon"][:] = self.lons
            output_ds.variables["time"][:] = input_ds.variables[_pick_time_var(input_ds)][:]
            output_ds.close()
        input_ds.close()

    def run(
        self,
        file_filter: str = "**/*LDASIN_DOMAIN*",
        skip_latlon: bool = False,
        apply_slp: bool = True,
    ):
        """Process all forcing files: regrid to lat-lon and/or SCHISM mesh.

        Parameters
        ----------
        file_filter
            Glob pattern for input files within ``input_dir``.
        skip_latlon
            If True, skip the lat-lon regridding step.
        apply_slp
            If True, convert PSFC to sea-level pressure in lat-lon output.
        """
        input_files = sorted(self.input_dir.glob(file_filter))
        if not input_files:
            raise FileNotFoundError(f"No files match '{file_filter}' in {self.input_dir}")

        # Job array partitioning for lat-lon regridding
        if self.job_idx is not None and self.job_count is not None:
            idx = self.job_idx
            count = math.ceil(len(input_files) / self.job_count)
            sub_input_files = input_files[idx * count : idx * count + count]
        else:
            idx = 0
            sub_input_files = input_files

        # Determine first timestep for SCHISM time offsets
        if self.root:
            with Dataset(input_files[0]) as ds0:
                self.schism_first_timestep = self._read_start_time(ds0)

        # Initialize SCHISM vsource output on idx=0
        schism_vsource = None
        if idx == 0 and self.root:
            schism_vsource = Dataset(self.output_dir / "precip_source.nc", "w", format="NETCDF4")
            self._init_vsource_nc(schism_vsource, len(input_files))

        # Process files
        for file in input_files:
            if not skip_latlon and file in sub_input_files:
                self._regrid_to_latlon(file, apply_slp=apply_slp)
            if idx == 0:
                self._regrid_to_schism(file, schism_vsource)

        if schism_vsource is not None:
            schism_vsource.sync()
            schism_vsource.close()


def main() -> None:
    """Entry point: reads config from CLI args and runs regridding."""
    import argparse

    parser = argparse.ArgumentParser(description="Regrid WRF-Hydro forcing to SCHISM mesh")
    parser.add_argument("--input-dir", required=True, help="NWM forcing output directory")
    parser.add_argument("--output-dir", required=True, help="Coastal forcing output directory")
    parser.add_argument("--geogrid-file", required=True, help="WRF geogrid file path")
    parser.add_argument("--schism-mesh", required=True, help="SCHISM ESMF mesh file path")
    parser.add_argument("--length-hrs", type=int, required=True, help="Forecast length in hours")
    parser.add_argument(
        "--forcing-begin-date", default=None, help="Forcing begin date (YYYYMMDDHHmm)"
    )
    parser.add_argument("--forcing-end-date", default=None, help="Forcing end date (YYYYMMDDHHmm)")
    parser.add_argument(
        "--job-index", type=int, default=None, help="Job array index (for HPC parallelism)"
    )
    parser.add_argument(
        "--job-count", type=int, default=None, help="Total job array size (for HPC parallelism)"
    )
    args = parser.parse_args()

    ESMF.Manager(debug=False)

    dir_date = args.forcing_end_date if args.length_hrs < 0 else args.forcing_begin_date
    if dir_date and len(dir_date) == 12:
        dir_date = dir_date[:-2]  # remove minutes

    input_path = Path(args.input_dir) / dir_date
    output_path = Path(args.output_dir)
    schism_mesh = Path(args.schism_mesh)
    geogrid = Path(args.geogrid_file)

    logger.info("Regridding forcings: %s -> %s", input_path, output_path)
    app = CoastalForcingRegridder(
        input_path,
        output_path,
        geogrid,
        schism_mesh,
        job_index=args.job_index,
        job_count=args.job_count,
    )
    app.run(file_filter="**/*LDASIN_DOMAIN*", skip_latlon=True)


if __name__ == "__main__":
    main()
