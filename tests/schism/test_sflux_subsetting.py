"""End-to-end sflux subsetting test.

Builds a small synthetic SCHISM mesh and a CONUS-sized synthetic
geogrid + LDASIN file, then runs the full :func:`make_sflux` pipeline
and asserts the resulting ``sflux_air_1.0001.nc`` is cropped to the
mesh footprint (much smaller than the geogrid).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import netCDF4
import numpy as np

from coastal_calibration.schism.prep import make_sflux

if TYPE_CHECKING:
    from pathlib import Path


# Synthetic "CONUS" geogrid: not actually CONUS-sized — kept small enough
# for tests to stay fast, but large enough that subsetting is observable.
# Spans lon ∈ [-130, -65], lat ∈ [25, 50] at ~1-deg resolution.
GEO_NY, GEO_NX = 26, 66
GEO_LAT = np.linspace(25.0, 50.0, GEO_NY, dtype=np.float32)
GEO_LON = np.linspace(-130.0, -65.0, GEO_NX, dtype=np.float32)

# Small mesh "near Mendocino" — only ~2x2 degrees.
MESH_LON_MIN, MESH_LON_MAX = -124.5, -122.5
MESH_LAT_MIN, MESH_LAT_MAX = 38.5, 40.5


def _write_geogrid(path: Path) -> None:
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north", GEO_NY)
        ds.createDimension("west_east", GEO_NX)
        ds.createVariable("HGT_M", "f4", ("Time", "south_north", "west_east"))[:] = np.zeros(
            (1, GEO_NY, GEO_NX), dtype=np.float32
        )
        ds.createVariable("XLAT_M", "f4", ("Time", "south_north", "west_east"))[:] = (
            np.broadcast_to(GEO_LAT[:, None], (1, GEO_NY, GEO_NX))
        )
        ds.createVariable("XLONG_M", "f4", ("Time", "south_north", "west_east"))[:] = (
            np.broadcast_to(GEO_LON[None, :], (1, GEO_NY, GEO_NX))
        )


def _write_ldasin(path: Path) -> None:
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north", GEO_NY)
        ds.createDimension("west_east", GEO_NX)
        for name, val in [
            ("T2D", 295.0),
            ("Q2D", 0.005),
            ("U2D", 1.5),
            ("V2D", -0.5),
            ("PSFC", 100000.0),
        ]:
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[:] = np.full(
                (1, GEO_NY, GEO_NX), val, dtype=np.float32
            )


def _write_small_hgrid(work_dir: Path) -> None:
    """Write a minimal ``hgrid.gr3`` containing four corner nodes at the mesh extent."""
    nodes = [
        (1, MESH_LON_MIN, MESH_LAT_MIN, 10.0),
        (2, MESH_LON_MAX, MESH_LAT_MIN, 10.0),
        (3, MESH_LON_MAX, MESH_LAT_MAX, 10.0),
        (4, MESH_LON_MIN, MESH_LAT_MAX, 10.0),
    ]
    lines = ["Mendocino test mesh", f"2 {len(nodes)}"]
    lines.extend(f"{nid} {x:.6f} {y:.6f} {d:.3f}" for nid, x, y, d in nodes)
    # Two triangles; element block is required for a valid .gr3 even though
    # make_sflux only reads the node block.
    lines.append("1 3 1 2 3")
    lines.append("2 3 1 3 4")
    (work_dir / "hgrid.gr3").write_text("\n".join(lines) + "\n")


def test_make_sflux_subsets_to_mesh_footprint(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_small_hgrid(work_dir)

    geogrid_file = tmp_path / "geo_em.d01.nc"
    _write_geogrid(geogrid_file)

    # LDASIN files live under forcing_input_dir / YYYYMMDDHH / (see make_sflux).
    forcing_input_dir = tmp_path / "forcing_input"
    forcing_subdir = forcing_input_dir / "2024010100"
    forcing_subdir.mkdir(parents=True)
    _write_ldasin(forcing_subdir / "2024010100.LDASIN_DOMAIN1")
    _write_ldasin(forcing_subdir / "2024010101.LDASIN_DOMAIN1")

    make_sflux(
        work_dir=work_dir,
        forcing_input_dir=forcing_input_dir,
        start_date=datetime(2024, 1, 1, 0),
        geogrid_file=geogrid_file,
    )

    # make_sflux renames sflux_air_1.0001.nc → sflux_air_1.1.nc.
    out_path = work_dir / "sflux" / "sflux_air_1.1.nc"
    assert out_path.is_file()
    with netCDF4.Dataset(out_path) as ds:
        ny = ds.dimensions["ny_grid"].size
        nx = ds.dimensions["nx_grid"].size
        # Subset must be strictly smaller than the input geogrid.
        assert ny < GEO_NY, f"ny_grid={ny} not smaller than geogrid ny={GEO_NY}"
        assert nx < GEO_NX, f"nx_grid={nx} not smaller than geogrid nx={GEO_NX}"
        # Subset must still cover the mesh extent (and its 0.5-deg buffer
        # gets the cell adjacent to each edge).
        lats = ds["lat"][:]
        lons = ds["lon"][:]
        assert lats.min() <= MESH_LAT_MIN
        assert lats.max() >= MESH_LAT_MAX
        assert lons.min() <= MESH_LON_MIN
        assert lons.max() >= MESH_LON_MAX
        # Forcing variables are written on the subset grid.
        assert ds["stmp"].shape == (3, ny, nx)
