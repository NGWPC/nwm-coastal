#!/usr/bin/env python3
"""Cut a small ESMF/WRF-Hydro GEOGRID domain out of an existing one (e.g. CONUS).

Produces `geo_em_<output_name>.nc` and
`GEOGRID_LDASOUT_Spatial_Metadata_<output_name>.nc` from an existing domain's
same-named files, by taking a plain index (i,j) hyperslab of every spatial
variable -- no reprojection, since the source grid's lat/lon are already
plain 2-D arrays on a regular index grid. Every variable in the source
`geo_em_*.nc` shares the `south_north`/`west_east` dims (or their staggered
`_stag` counterparts) with `XLAT_M`/`XLONG_M`, so the extraction is a
straightforward hyperslab of everything sharing those dims -- this is the
same approach `NextGen_Forcings_Engine_BMI`'s `geoMod.py` relies on when it
builds an ESMF grid directly from `XLAT_M`/`XLONG_M`.

The forcing engine still downloads raw CONUS-scale meteorological input
(RAP/HRRR/MRMS) as before -- only the destination grid it regrids onto
shrinks, which is what actually bounds memory/compute per regrid pass.

Usage:
    extract_esmf_domain.py \\
        --domain-dir /path/to/run_ngen/data/esmf_mesh/NWM/domain \\
        --source-domain CONUS \\
        --extract-geojson /path/to/esmf_conus_03s_extract.geojson \\
        --output-name vpu03s

    # domain-dir defaults to $RUN_NGEN_ROOT/data/esmf_mesh/NWM/domain if unset
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np

STAG_DIM_MAP = {
    "south_north": "south_north_stag",
    "west_east": "west_east_stag",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_domain_dir = None
    run_ngen_root = os.environ.get("RUN_NGEN_ROOT")
    if run_ngen_root:
        default_domain_dir = str(Path(run_ngen_root) / "data" / "esmf_mesh" / "NWM" / "domain")

    parser.add_argument(
        "--domain-dir",
        type=Path,
        default=Path(default_domain_dir) if default_domain_dir else None,
        required=default_domain_dir is None,
        help=(
            "Directory holding geo_em_<domain>.nc / "
            "GEOGRID_LDASOUT_Spatial_Metadata_<domain>.nc, and where the "
            "new <output_name> files are written. Defaults to "
            "$RUN_NGEN_ROOT/data/esmf_mesh/NWM/domain if RUN_NGEN_ROOT is set."
        ),
    )
    parser.add_argument(
        "--source-domain",
        required=True,
        help="Existing domain name to extract from, e.g. 'CONUS'.",
    )
    parser.add_argument(
        "--extract-geojson",
        type=Path,
        required=True,
        help="Polygon (any CRS readable by geopandas) defining the extract extent.",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Name for the new domain, e.g. 'vpu03s' -> geo_em_vpu03s.nc etc.",
    )
    parser.add_argument(
        "--buffer-cells",
        type=int,
        default=10,
        help=(
            "Extra grid cells to pad around the polygon's bounding index "
            "range on each side, to avoid regridding edge artifacts right "
            "at the domain boundary. Default: 10."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def clip_to_domain_bounds(
    polygon_bounds: tuple[float, float, float, float],
    xlat: np.ndarray,
    xlong: np.ndarray,
    source_domain: str,
) -> tuple[float, float, float, float]:
    """Clip polygon_bounds (minx, miny, maxx, maxy) to the source domain's own
    actual lat/lon coverage, warning if the requested extract polygon reaches
    beyond it. The source domain's coverage is itself a lon/lat bounding box
    over its XLAT_M/XLONG_M (a conservative superset, since the true LCC-grid
    footprint isn't a rectangle in lon/lat space -- fine here, since it's
    only used to clip an already-a-bounding-box request).
    """
    minx, miny, maxx, maxy = polygon_bounds
    domain_minx, domain_maxx = float(xlong.min()), float(xlong.max())
    domain_miny, domain_maxy = float(xlat.min()), float(xlat.max())

    clipped_minx, clipped_maxx = max(minx, domain_minx), min(maxx, domain_maxx)
    clipped_miny, clipped_maxy = max(miny, domain_miny), min(maxy, domain_maxy)

    if (clipped_minx, clipped_miny, clipped_maxx, clipped_maxy) != (minx, miny, maxx, maxy):
        print(
            f"WARNING: extract polygon bounds {polygon_bounds} extend beyond "
            f"source domain '{source_domain}''s coverage "
            f"({domain_minx}, {domain_miny}, {domain_maxx}, {domain_maxy}). "
            f"Clipping to ({clipped_minx}, {clipped_miny}, {clipped_maxx}, {clipped_maxy})."
        )
    if clipped_minx >= clipped_maxx or clipped_miny >= clipped_maxy:
        raise ValueError(
            f"Extract polygon bounds {polygon_bounds} do not overlap source domain "
            f"'{source_domain}''s coverage at all -- nothing to extract."
        )
    return clipped_minx, clipped_miny, clipped_maxx, clipped_maxy


def find_index_bbox(
    xlat: np.ndarray, xlong: np.ndarray, polygon_bounds: tuple[float, float, float, float], buffer_cells: int
) -> tuple[int, int, int, int]:
    """Return (j0, j1, i0, i1) -- half-open index ranges into (south_north, west_east)
    covering every grid point whose lat/lon falls inside polygon_bounds, padded
    by buffer_cells on each side and clamped to the source grid's extent.
    """
    minx, miny, maxx, maxy = polygon_bounds
    mask = (xlong >= minx) & (xlong <= maxx) & (xlat >= miny) & (xlat <= maxy)
    if not mask.any():
        raise ValueError(
            "No grid points found within the extract polygon's bounding box "
            f"({polygon_bounds}) -- check the geojson's extent and CRS."
        )
    j_idx, i_idx = np.nonzero(mask)
    ny, nx = xlat.shape
    j0 = max(0, int(j_idx.min()) - buffer_cells)
    j1 = min(ny, int(j_idx.max()) + 1 + buffer_cells)
    i0 = max(0, int(i_idx.min()) - buffer_cells)
    i1 = min(nx, int(i_idx.max()) + 1 + buffer_cells)
    return j0, j1, i0, i1


def var_slices(
    var: netCDF4.Variable, j0: int, j1: int, i0: int, i1: int, y_j0: int | None = None, y_j1: int | None = None
) -> tuple:
    """Build a slice tuple for `var` given unstaggered index range (j0:j1, i0:i1),
    extending by one on the appropriate side(s) for any staggered dims.

    `y_j0`/`y_j1` (defaulting to `j0`/`j1`) are used only for the `y` dim --
    GEOGRID_LDASOUT_Spatial_Metadata_*.nc's `y` coordinate is stored
    north-to-south (descending), the OPPOSITE row order of geo_em_*.nc's
    `south_north` (south-to-north, ascending). Applying the same raw index
    range to both silently grabs the wrong (mirrored) rows for `y` --
    callers must pass an already-mirrored `y_j0`/`y_j1` when `y`'s order
    differs from `south_north`'s (see `find_index_bbox`'s caller in `main`).
    """
    if y_j0 is None:
        y_j0 = j0
    if y_j1 is None:
        y_j1 = j1
    slices = []
    for dim in var.dimensions:
        if dim == "south_north":
            slices.append(slice(j0, j1))
        elif dim == "south_north_stag":
            slices.append(slice(j0, j1 + 1))
        elif dim == "west_east":
            slices.append(slice(i0, i1))
        elif dim == "west_east_stag":
            slices.append(slice(i0, i1 + 1))
        elif dim == "y":
            slices.append(slice(y_j0, y_j1))
        elif dim == "x":
            slices.append(slice(i0, i1))
        else:
            slices.append(slice(None))
    return tuple(slices)


def copy_dataset_subset(
    src_path: Path,
    dst_path: Path,
    j0: int,
    j1: int,
    i0: int,
    i1: int,
    extra_global_attrs: dict | None = None,
    y_j0: int | None = None,
    y_j1: int | None = None,
) -> None:
    """Copy src_path -> dst_path, hyperslab-subsetting every variable that
    shares south_north/west_east/y/x (or their staggered variants), keeping
    all other variables/dims untouched. Preserves the source's on-disk
    format (NETCDF3_64BIT_OFFSET for these files).

    See `var_slices` for why `y_j0`/`y_j1` exist separately from `j0`/`j1`.
    """
    if y_j0 is None:
        y_j0 = j0
    if y_j1 is None:
        y_j1 = j1
    with netCDF4.Dataset(src_path, "r") as src, netCDF4.Dataset(dst_path, "w", format=src.data_model) as dst:
        subset_dim_sizes = {
            "south_north": j1 - j0,
            "south_north_stag": (j1 - j0) + 1,
            "west_east": i1 - i0,
            "west_east_stag": (i1 - i0) + 1,
            "y": y_j1 - y_j0,
            "x": i1 - i0,
        }
        for name, dim in src.dimensions.items():
            size = subset_dim_sizes.get(name, len(dim) if not dim.isunlimited() else None)
            dst.createDimension(name, size)

        for name, var in src.variables.items():
            new_var = dst.createVariable(name, var.dtype, var.dimensions)
            new_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            sl = var_slices(var, j0, j1, i0, i1, y_j0=y_j0, y_j1=y_j1)
            new_var[:] = var[sl]

        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        if extra_global_attrs:
            dst.setncatts(extra_global_attrs)


def main() -> int:
    args = parse_args()

    domain_dir: Path = args.domain_dir
    if domain_dir is None:
        print("ERROR: --domain-dir not given and RUN_NGEN_ROOT is not set.", file=sys.stderr)
        return 1

    src_geo_em = domain_dir / f"geo_em_{args.source_domain}.nc"
    src_ldasout = domain_dir / f"GEOGRID_LDASOUT_Spatial_Metadata_{args.source_domain}.nc"
    for p in (src_geo_em, src_ldasout, args.extract_geojson):
        if not p.is_file():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 1

    dst_geo_em = domain_dir / f"geo_em_{args.output_name}.nc"
    dst_ldasout = domain_dir / f"GEOGRID_LDASOUT_Spatial_Metadata_{args.output_name}.nc"
    if not args.overwrite:
        for p in (dst_geo_em, dst_ldasout):
            if p.exists():
                print(f"ERROR: {p} already exists (use --overwrite to replace it).", file=sys.stderr)
                return 1

    print(f"Reading extract polygon: {args.extract_geojson}")
    gdf = gpd.read_file(args.extract_geojson)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    polygon_bounds = tuple(gdf.total_bounds)  # (minx, miny, maxx, maxy) in lon/lat
    print(f"  bounds (lon/lat): {polygon_bounds}")

    print(f"Reading source domain grid: {src_geo_em}")
    with netCDF4.Dataset(src_geo_em, "r") as ds:
        xlat = ds.variables["XLAT_M"][0, :, :].filled(np.nan) if hasattr(ds.variables["XLAT_M"][0, :, :], "filled") else np.asarray(ds.variables["XLAT_M"][0, :, :])
        xlong = ds.variables["XLONG_M"][0, :, :].filled(np.nan) if hasattr(ds.variables["XLONG_M"][0, :, :], "filled") else np.asarray(ds.variables["XLONG_M"][0, :, :])
        src_attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}

    polygon_bounds = clip_to_domain_bounds(polygon_bounds, xlat, xlong, args.source_domain)
    j0, j1, i0, i1 = find_index_bbox(xlat, xlong, polygon_bounds, args.buffer_cells)
    ny, nx = j1 - j0, i1 - i0
    print(f"  index range: south_north[{j0}:{j1}] ({ny} cells), west_east[{i0}:{i1}] ({nx} cells)")
    print(f"  (buffer_cells={args.buffer_cells} already applied, clamped to source grid extent)")

    corner_xlat = xlat[j0:j1, i0:i1]
    corner_xlong = xlong[j0:j1, i0:i1]
    # geo_em's corner_lats/corner_lons are a 16-element array (4 corners x
    # {mass,u,v,unstag} staggering variants); recomputing all 16 precisely
    # isn't necessary for this pipeline's actual usage (geoMod.py builds its
    # ESMF grid straight from XLAT_M/XLONG_M, not these attrs) -- but keep
    # the 4 mass-point corners accurate and repeat them across the 4
    # staggering slots so the attribute shape/type stays valid for any
    # other WRF-Hydro tooling that does read it.
    ul, ur, lr, ll = (
        (float(corner_xlat[-1, 0]), float(corner_xlong[-1, 0])),
        (float(corner_xlat[-1, -1]), float(corner_xlong[-1, -1])),
        (float(corner_xlat[0, -1]), float(corner_xlong[0, -1])),
        (float(corner_xlat[0, 0]), float(corner_xlong[0, 0])),
    )
    corner_lats = np.array([ul[0], ur[0], lr[0], ll[0]] * 4, dtype=np.float32)
    corner_lons = np.array([ul[1], ur[1], lr[1], ll[1]] * 4, dtype=np.float32)

    extra_attrs = {
        "WEST-EAST_GRID_DIMENSION": nx + 1,
        "SOUTH-NORTH_GRID_DIMENSION": ny + 1,
        "WEST-EAST_PATCH_START_UNSTAG": 1,
        "WEST-EAST_PATCH_END_UNSTAG": nx,
        "WEST-EAST_PATCH_START_STAG": 1,
        "WEST-EAST_PATCH_END_STAG": nx + 1,
        "SOUTH-NORTH_PATCH_START_UNSTAG": 1,
        "SOUTH-NORTH_PATCH_END_UNSTAG": ny,
        "SOUTH-NORTH_PATCH_START_STAG": 1,
        "SOUTH-NORTH_PATCH_END_STAG": ny + 1,
        "i_parent_start": 1,
        "j_parent_start": 1,
        "i_parent_end": nx + 1,
        "j_parent_end": ny + 1,
        "corner_lats": corner_lats,
        "corner_lons": corner_lons,
        "region": args.output_name,
    }

    print(f"Writing {dst_geo_em}")
    copy_dataset_subset(src_geo_em, dst_geo_em, j0, j1, i0, i1, extra_global_attrs=extra_attrs)

    print(f"Reading/writing spatial metadata: {src_ldasout} -> {dst_ldasout}")
    with netCDF4.Dataset(src_ldasout, "r") as ds:
        y_full = np.asarray(ds.variables["y"][:])
        # y is stored NORTH-TO-SOUTH (descending) in this file family, the
        # OPPOSITE row order of geo_em_*.nc's south_north (south-to-north,
        # ascending) -- confirmed by direct inspection, not assumed.
        # Applying south_north's [j0:j1] range directly to y would silently
        # grab the wrong (mirrored) rows. Mirror the range when y is
        # descending; use it as-is if some future source domain's y turns
        # out to be ascending instead.
        y_descending = y_full[0] > y_full[-1]
        ny_total = len(y_full)
        if y_descending:
            y_j0, y_j1 = ny_total - j1, ny_total - j0
        else:
            y_j0, y_j1 = j0, j1
        x = np.asarray(ds.variables["x"][i0:i1])
        y = np.asarray(ds.variables["y"][y_j0:y_j1])
        dx = float(x[1] - x[0]) if len(x) > 1 else None
        dy = float(y[1] - y[0]) if len(y) > 1 else None

    ldasout_extra_attrs = {"region": args.output_name}
    if dx is not None and dy is not None:
        # Preserve the half-pixel-offset GeoTransform convention the source
        # file uses (origin = first-cell-center minus half a pixel).
        x_origin = x[0] - dx / 2.0
        y_origin = y[0] - dy / 2.0  # dy is negative (north-up grid), so this steps up to the top edge
        ldasout_extra_attrs["GeoTransform_override"] = f"{x_origin} {dx} 0 {y_origin} 0 {dy} "
    copy_dataset_subset(src_ldasout, dst_ldasout, j0, j1, i0, i1, extra_global_attrs=None, y_j0=y_j0, y_j1=y_j1)
    if "GeoTransform_override" in ldasout_extra_attrs:
        with netCDF4.Dataset(dst_ldasout, "a") as ds:
            if "crs" in ds.variables:
                ds.variables["crs"].GeoTransform = ldasout_extra_attrs["GeoTransform_override"]

    print()
    print(f"Done. New domain '{args.output_name}': {nx} x {ny} cells ({nx * ny:,} points, "
          f"vs source's full extent).")
    print(f"  {dst_geo_em}")
    print(f"  {dst_ldasout}")
    print()
    print(f"To use: pass -gdomain {args.output_name} to ngen_rte.run_coastal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
