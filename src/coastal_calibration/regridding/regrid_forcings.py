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
from typing import TYPE_CHECKING, Any

import esmpy as ESMF
import netCDF4
import numpy as np

from coastal_calibration.logging import logger

from .esmf_utils import (
    Regridder,
    allreduce_minmax,
    build_grid,
    gather_reduce,
    gatherv_1d,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _pick_time_var(ds: netCDF4.Dataset) -> str:
    """Return the name of the time variable in *ds*.

    WRF-Hydro LDASIN files use ``"time"``; ERA5/HRRR-derived files may use
    ``"valid_time"``; the ngen forecast forcing engine writes ``"Time"``
    (a multi-timestep ``minutes since 1970`` axis, same units as ``time``).
    Raises ``KeyError`` with a descriptive message if none is present.
    """
    for name in ("time", "valid_time", "Time"):
        if name in ds.variables:
            return name
    raise KeyError(
        f"Expected 'time', 'valid_time' or 'Time' in {ds.filepath()}, found: {list(ds.variables)}"
    )


def _time_slabs(files: list[Path]) -> list[tuple[Path, int]]:
    """Enumerate ``(file, time_index)`` slabs across *files*.

    Canonical WRF-Hydro LDASIN files carry a single timestep, yielding one
    slab per file.  The ngen forecast forcing engine packs several
    timesteps into one file, yielding one slab per index.  This lets the
    regridder treat both layouts uniformly without copying multi-timestep
    files into per-hour files first.
    """
    slabs: list[tuple[Path, int]] = []
    for f in files:
        with netCDF4.Dataset(f) as ds:
            n_times = ds.variables[_pick_time_var(ds)].shape[0]
        slabs.extend((f, t) for t in range(n_times))
    return slabs


def sea_level_pressure(
    temp: NDArray[np.floating[Any]],
    mixing: NDArray[np.floating[Any]],
    height: NDArray[np.floating[Any]],
    press: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
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
        sim_start_epoch: float | None = None,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.root = ESMF.local_pet() == 0

        self.job_idx = job_index
        self.job_count = job_count
        # Explicit simulation start (Unix epoch seconds), when known -- used
        # to anchor precip_source.nc's vsource row 0 to SCHISM's own
        # start_date, matching SCHISM's read (schism_init.F90/schism_step.F90):
        # it indexes "vsource" purely positionally (row i == elapsed i*dt from
        # its own start_date), never reading time_vsource's stored values, so
        # the row that lands at index 0 MUST really be valid at start_date.
        # Falls back to the first slab's own timestamp (old behavior) only
        # when the caller doesn't know the true start (should not happen via
        # the CLI entry point below, which always has it).
        self.sim_start_epoch = sim_start_epoch

        # Load SCHISM mesh
        self.schism_mesh = ESMF.Mesh(
            filename=str(schism_mesh_path), filetype=ESMF.FileFormat.ESMFMESH
        )
        with netCDF4.Dataset(schism_mesh_path, "r") as smesh:
            self.total_elements = smesh.dimensions["elementCount"].size

        # Determine output lat/lon range from mesh node coords (MPI-aware)
        node_lons = self.schism_mesh.coords[0][0]  # pyright: ignore[reportOptionalSubscript]
        node_lats = self.schism_mesh.coords[0][1]  # pyright: ignore[reportOptionalSubscript]
        lon_min, lon_max = allreduce_minmax(node_lons)  # pyright: ignore[reportArgumentType]
        lat_min, lat_max = allreduce_minmax(node_lats)  # pyright: ignore[reportArgumentType]

        # Read source grid and height from geogrid file
        with netCDF4.Dataset(geo_em_path, "r") as ds:
            self.src_height = ds.variables["HGT_M"][:]
            xlat = ds.variables["XLAT_M"][0, :].T
            xlon = ds.variables["XLONG_M"][0, :].T
            clat = ds.variables["XLAT_C"][0, :].T
            clon = ds.variables["XLONG_C"][0, :].T

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

    def _read_start_time(self, ds: netCDF4.Dataset, time_index: int = 0) -> float:
        """Return the valid time (seconds) of slab *time_index* in *ds*.

        ``time`` and ``Time`` are stored as minutes (WRF-Hydro / ngen
        forecast convention) and converted to seconds; ``valid_time`` is
        already in seconds.  The value is absolute, so differences between
        slabs give the hourly output offsets used to index ``vsource``.
        """
        name = _pick_time_var(ds)
        value = float(np.asarray(ds.variables[name][time_index]))
        if name == "valid_time":
            return value
        return value * 60  # 'time'/'Time' are minutes -> seconds

    def _init_vsource_nc(self, ds: netCDF4.Dataset, ntimes: int):
        """Create dimensions and variables for the SCHISM vsource file."""
        from coastal_calibration._nc_io import create_var, write_var

        ds.createDimension("time_vsource", ntimes)
        ds.createDimension("nsources", self.total_elements)
        ds.createDimension("one", 1)

        eso = create_var(ds, "source_elem", "i4", ("nsources",))
        create_var(ds, "vsource", "f8", ("time_vsource", "nsources"), zlib=True)
        create_var(ds, "time_vsource", "f8", ("time_vsource",))
        vts = create_var(ds, "time_step_vsource", "f4", ("one",))

        write_var(eso, np.arange(1, self.total_elements + 1))
        vts[:] = 3600

    def _regrid_to_schism(
        self,
        input_file: Path,
        vsource_ds: netCDF4.Dataset | None,
        time_index: int = 0,
    ):
        """Regrid RAINRATE to SCHISM mesh elements and write to vsource.

        *time_index* selects the timestep to read; canonical LDASIN files
        have a single step (index 0) while forecast files hold several.
        """
        with netCDF4.Dataset(input_file) as input_ds:
            # Populate source field
            in_field = ESMF.Field(grid=self.in_grid, name="rainrate-in")
            b = self.in_bounds
            in_field.data[...] = input_ds.variables["RAINRATE"][time_index, :].T[  # pyright: ignore[reportOptionalSubscript]
                b.x_lo : b.x_hi, b.y_lo : b.y_hi
            ]

            # Populate destination field on mesh elements
            out_field = ESMF.Field(
                grid=self.schism_mesh, meshloc=ESMF.MeshLoc.ELEMENT, name="rainrate-out"
            )
            # Initialise to 0 so unmapped elements (IGNORE action) are left at 0.
            out_field.data[...] = 0.0  # pyright: ignore[reportOptionalSubscript]

            # Build regridder once, reuse for subsequent files.
            # CONSERVE is required for Grid -> Mesh(ELEMENT) regridding;
            # BILINEAR only supports Mesh(NODE) destinations.
            if self._schism_regridder is None:
                self._schism_regridder = Regridder(
                    in_field,
                    out_field,
                    method=ESMF.RegridMethod.CONSERVE,  # pyright: ignore[reportArgumentType]
                    unmapped_action=ESMF.UnmappedAction.IGNORE,  # pyright: ignore[reportArgumentType]
                )

            out_field = self._schism_regridder(in_field, out_field)
            # Clamp any negative bilinear interpolation artefacts to zero.
            np.clip(out_field.data, 0.0, None, out=out_field.data)  # pyright: ignore[reportCallIssue, reportArgumentType]

            # Convert to volumetric flux (m^3/s)
            R0_SCHISM = 6378206.4  # earth radius in meters used by SCHISM
            DENSITY_FACTOR = 1000

            unit_areas = ESMF.Field(
                self.schism_mesh, meshloc=ESMF.MeshLoc.ELEMENT, name="areafield"
            )
            unit_areas.get_area()
            areas_m2 = unit_areas.data[...] * (R0_SCHISM * R0_SCHISM)  # pyright: ignore[reportOptionalSubscript]
            out_field.data[...] *= areas_m2 / DENSITY_FACTOR  # pyright: ignore[reportOptionalSubscript]
            unit_areas.destroy()

            # Gather distributed data to root
            local_count: int = self.schism_mesh.size[1]  # pyright: ignore[reportAssignmentType]
            all_elements = gatherv_1d(out_field.data, local_count)  # pyright: ignore[reportArgumentType]

            if all_elements is not None and len(all_elements) != self.total_elements:
                msg = (
                    f"Gathered element count {len(all_elements)} != "
                    f"mesh dimension {self.total_elements} - dimension mismatch would "
                    "corrupt the vsource output file"
                )
                raise ValueError(msg)

            # Write on root rank
            if self.root and vsource_ds is not None:
                step_time = self._read_start_time(input_ds, time_index)
                output_ts = int(step_time - self.schism_first_timestep)  # pyright: ignore[reportOperatorIssue]
                output_idx = output_ts // 3600
                ntimes = vsource_ds.dimensions["time_vsource"].size
                if 0 <= output_idx < ntimes:
                    vsource_ds["time_vsource"][output_idx] = output_ts
                    vsource_ds["vsource"][output_idx, :] = all_elements
                    vsource_ds.sync()
                else:
                    logger.warning(
                        "    _regrid_to_schism: slab at %s (elapsed %ds) falls outside "
                        "the SCHISM window [0, %ds] -- dropping",
                        input_ds.filepath(),
                        output_ts,
                        (ntimes - 1) * 3600,
                    )

            in_field.destroy()
            out_field.destroy()

    def _init_latlon_nc(
        self, output_ds: netCDF4.Dataset, nlats: int, nlons: int, input_ds: netCDF4.Dataset
    ):
        """Create dimensions, coordinates, and time variable for lat-lon output."""
        from coastal_calibration._nc_io import create_var

        output_ds.createDimension(dimname="lat", size=nlats)
        output_ds.createDimension(dimname="lon", size=nlons)
        output_ds.createDimension(dimname="time", size=0)

        create_var(
            output_ds,
            "lat",
            self.lats.dtype,
            ("lat",),
            attrs={
                "long_name": "latitude",
                "units": "degrees_north",
                "standard_name": "latitude",
                "axis": "Y",
            },
        )
        create_var(
            output_ds,
            "lon",
            self.lons.dtype,
            ("lon",),
            attrs={
                "long_name": "longitude",
                "units": "degrees_east",
                "standard_name": "longitude",
                "axis": "X",
            },
        )
        in_time = input_ds.variables[_pick_time_var(input_ds)]
        create_var(
            output_ds,
            "time",
            in_time.datatype,
            ("time",),
            attrs={
                "long_name": "valid output time",
                "units": in_time.units,
                "calendar": "standard",
                "standard_name": "time",
            },
        )

    def _regrid_to_latlon(  # noqa: PLR0912, PLR0915
        self, input_file: Path, apply_slp: bool = True, time_index: int = 0
    ):
        """Regrid atmospheric variables to a regular lat-lon grid.

        *time_index* selects the timestep to read.  Output files are named
        ``<stem>.latlon.nc`` for the first slab (preserving the one-file-per
        -hour convention) and ``<stem>.t<NNN>.latlon.nc`` for further slabs
        of a multi-timestep forecast file, so slabs never overwrite.
        """
        from coastal_calibration._nc_io import create_var, write_var

        with netCDF4.Dataset(input_file) as input_ds:
            nlons, nlats = self.out_grid.max_index

            # Prepare output dataset on root
            if self.root:
                suffix = f".t{time_index:03d}" if time_index else ""
                output_path = self.output_dir / (input_file.stem + suffix + ".latlon.nc")
                output_ds = netCDF4.Dataset(output_path, "w", format="NETCDF4")
                self._init_latlon_nc(output_ds, nlats, nlons, input_ds)
            else:
                output_ds = None

            try:
                for variable in self.LATLON_VARS:
                    if variable not in input_ds.variables:
                        continue

                    # Read and optionally transform the variable
                    data = input_ds.variables[variable][time_index, :].T
                    var_name = variable
                    var_attrs = {}
                    for attr in ("standard_name", "long_name", "units"):
                        if attr in input_ds.variables[variable].ncattrs():
                            var_attrs[attr] = getattr(input_ds.variables[variable], attr)

                    if apply_slp and variable == "PSFC":
                        data = sea_level_pressure(
                            temp=input_ds.variables["T2D"][time_index, :].T,
                            mixing=input_ds.variables["Q2D"][time_index, :].T,
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
                        if output_ds is None:
                            msg = "output_ds is None on root rank"
                            raise RuntimeError(msg)
                        create_var(
                            output_ds,
                            var_name,
                            "f4",
                            ("time", "lat", "lon"),
                            attrs=var_attrs,
                        )

                    # Populate source field with local partition slice
                    in_field = ESMF.Field(grid=self.in_grid, name=f"{variable}-in")
                    b = self.in_bounds
                    in_field.data[...] = data[b.x_lo : b.x_hi, b.y_lo : b.y_hi]  # pyright: ignore[reportOptionalSubscript]

                    out_field = ESMF.Field(grid=self.out_grid, name=f"{variable}-out")
                    out_field.data[...] = 0.0  # pyright: ignore[reportOptionalSubscript]

                    # Build regridder once, reuse for subsequent variables/files
                    if self._latlon_regridder is None:
                        self._latlon_regridder = Regridder(
                            in_field,
                            out_field,
                            method=ESMF.RegridMethod.BILINEAR,  # pyright: ignore[reportArgumentType]
                            unmapped_action=ESMF.UnmappedAction.IGNORE,  # pyright: ignore[reportArgumentType]
                        )
                    else:
                        self._latlon_regridder(
                            in_field,
                            out_field,
                            zero_region=ESMF.constants.Region.SELECT,  # pyright: ignore[reportArgumentType]
                        )

                    # Assemble global output from all partitions
                    global_output = np.zeros((nlons, nlats))
                    ob = self.out_bounds
                    global_output[ob.x_lo : ob.x_hi, ob.y_lo : ob.y_hi] = out_field.data[...]  # pyright: ignore[reportOptionalSubscript]

                    final_output = gather_reduce(global_output, global_shape=(nlons, nlats))

                    if self.root:
                        if output_ds is None or final_output is None:
                            msg = "output_ds or final_output is None on root rank"
                            raise RuntimeError(msg)
                        output_ds.variables[var_name][0, :] = final_output.T

                    in_field.destroy()
                    out_field.destroy()

                # Write coordinates
                if self.root:
                    if output_ds is None:
                        msg = "output_ds is None on root rank"
                        raise RuntimeError(msg)
                    write_var(output_ds.variables["lat"], self.lats)
                    write_var(output_ds.variables["lon"], self.lons)
                    # Each lat-lon output holds a single slab, so write only
                    # this slab's timestamp (not the file's whole time axis,
                    # which would exceed the single data row for forecast
                    # files that pack multiple timesteps).
                    output_ds.variables["time"][0] = input_ds.variables[_pick_time_var(input_ds)][
                        time_index
                    ]
            finally:
                if output_ds is not None:
                    output_ds.close()

    def run(
        self,
        file_filter: str = "**/*LDASIN_DOMAIN*",
        skip_latlon: bool = False,
        apply_slp: bool = True,
        n_hours: int | None = None,
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
        n_hours
            Number of hours the SCHISM run actually needs (i.e.
            ``duration_hours``). ``precip_source.nc`` is sized to
            ``n_hours + 1`` rows (0..n_hours, matching SCHISM's own
            positional read) regardless of how many slabs the met input
            actually contains -- falls back to ``len(slabs)`` when not
            given (should only happen for callers that predate this fix).
        """
        input_files = sorted(self.input_dir.glob(file_filter))
        if not input_files:
            raise FileNotFoundError(f"No files match '{file_filter}' in {self.input_dir}")

        # Unit of work is a (file, timestep) slab, not a file: canonical
        # LDASIN files yield one slab each, forecast files yield several.
        slabs = _time_slabs(input_files)

        # Job array partitioning for lat-lon regridding
        if self.job_idx is not None and self.job_count is not None:
            idx = self.job_idx
            count = math.ceil(len(slabs) / self.job_count)
            sub_slabs = set(slabs[idx * count : idx * count + count])
        else:
            idx = 0
            sub_slabs = set(slabs)

        # Anchor for SCHISM time offsets. SCHISM reads vsource from
        # precip_source.nc purely positionally (row i == elapsed i*3600s
        # from ITS OWN start_date -- see schism_init.F90/schism_step.F90,
        # neither ever reads time_vsource's stored values), so row 0 must
        # really be valid at start_date or every later row silently
        # misaligns. Prefer the caller-supplied sim_start_epoch (the
        # SCHISM run's true start_date) over the met file's own first
        # slab -- the two only coincide when the met window happens to
        # start exactly at start_date, which isn't guaranteed.
        if self.root:
            if self.sim_start_epoch is not None:
                self.schism_first_timestep = self.sim_start_epoch
            else:
                first_file, first_ti = slabs[0]
                with netCDF4.Dataset(first_file) as ds0:
                    self.schism_first_timestep = self._read_start_time(ds0, first_ti)

        # Initialize SCHISM vsource output on idx=0. Sized to the full
        # required window, not len(slabs) -- a slab short of that window
        # (e.g. the met engine's own window starting later than
        # start_date) must leave real gaps, not shrink the array to fit.
        schism_vsource = None
        if idx == 0 and self.root:
            schism_vsource = netCDF4.Dataset(
                self.output_dir / "precip_source.nc", "w", format="NETCDF4"
            )
            ntimes = (n_hours + 1) if n_hours is not None else len(slabs)
            self._init_vsource_nc(schism_vsource, ntimes)

        try:
            # Process slabs
            for file, time_index in slabs:
                if not skip_latlon and (file, time_index) in sub_slabs:
                    self._regrid_to_latlon(file, apply_slp=apply_slp, time_index=time_index)
                if idx == 0:
                    self._regrid_to_schism(file, schism_vsource, time_index)
        finally:
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

    ESMF.Manager(debug=False)  # pyright: ignore[reportCallIssue]

    dir_date = args.forcing_end_date if args.length_hrs < 0 else args.forcing_begin_date
    if dir_date and len(dir_date) == 12:
        dir_date = dir_date[:-2]  # remove minutes

    input_path = Path(args.input_dir) / dir_date
    output_path = Path(args.output_dir)
    schism_mesh = Path(args.schism_mesh)
    geogrid = Path(args.geogrid_file)

    # forcing_begin_date is always the SCHISM run's own start_date
    # chronologically, even in reanalysis mode (negative length-hrs, where
    # forcing_end_date is the earlier boundary) -- see NWMForcingStage.run()
    # in forcing.py, which derives it directly from sim.start_pdy/start_cyc.
    sim_start_epoch = None
    n_hours = None
    if args.forcing_begin_date:
        from datetime import datetime, timezone

        fmt = "%Y%m%d%H%M" if len(args.forcing_begin_date) == 12 else "%Y%m%d%H"
        sim_start_dt = datetime.strptime(args.forcing_begin_date, fmt).replace(
            tzinfo=timezone.utc
        )
        sim_start_epoch = sim_start_dt.timestamp()
        n_hours = abs(args.length_hrs)

    logger.info("Regridding forcings: %s -> %s", input_path, output_path)
    app = CoastalForcingRegridder(
        input_path,
        output_path,
        geogrid,
        schism_mesh,
        job_index=args.job_index,
        job_count=args.job_count,
        sim_start_epoch=sim_start_epoch,
    )
    app.run(file_filter="**/*LDASIN_DOMAIN*", skip_latlon=True, n_hours=n_hours)


if __name__ == "__main__":
    main()
