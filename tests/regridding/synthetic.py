"""Factory functions to create small synthetic NetCDF fixtures for regridding tests.

All functions write files that match the exact variable/dimension structure
consumed by the regridding modules under ``src/coastal_calibration/regridding/``.
They use tiny grids (O(10) nodes / O(50) nodes) so ESMF regridding completes in
seconds rather than minutes.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

# ---------------------------------------------------------------------------
# Domain constants — Hawaii-like region shared across all fixtures
# ---------------------------------------------------------------------------

LON_W, LON_E = -157.8, -156.2
LAT_S, LAT_N = 20.2, 21.8

# Tighter LDASIN/geo_em domain — fully covers ESMFMESH nodes
GEO_LON_W, GEO_LON_E = -158.0, -155.5
GEO_LAT_S, GEO_LAT_N = 20.0, 22.0


# ---------------------------------------------------------------------------
# ESTOFS-like file  (for regrid_estofs)
# ---------------------------------------------------------------------------


def make_stofs_nc(path: Path, n_nodes: int = 50, n_times: int = 10) -> None:
    """Write a minimal ESTOFS-like NetCDF file.

    Time layout (seconds since 2024-01-09 00:00:00):
        index 0–9  →  hours 0–9
        index 5    →  05:00 UTC  ← cycle start used in ``synthetic_stofs_cycle_env``

    The ``zeta`` variable has ~20 % of values masked, matching the wet/dry
    mask present in real ESTOFS output.
    """
    rng = np.random.default_rng(42)
    lons = rng.uniform(LON_W, LON_E, n_nodes)
    lats = rng.uniform(LAT_S, LAT_N, n_nodes)

    zeta_data = rng.uniform(-0.5, 1.5, (n_times, n_nodes)).astype("f4")
    mask = rng.random((n_times, n_nodes)) < 0.2

    times = np.arange(n_times, dtype="f8") * 3600.0  # hourly steps

    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n_times)
        ds.createDimension("node", n_nodes)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = "seconds since 2024-01-09 00:00:00"
        tv.calendar = "gregorian"
        tv[:] = times

        xv = ds.createVariable("x", "f8", ("node",))
        xv[:] = lons

        yv = ds.createVariable("y", "f8", ("node",))
        yv[:] = lats

        fill = np.float32(-9999.0)
        zv = ds.createVariable("zeta", "f4", ("time", "node"), fill_value=fill)
        zeta_fill = zeta_data.copy()
        zeta_fill[mask] = fill
        zv[:] = zeta_fill


# ---------------------------------------------------------------------------
# SCHISM open-boundary hgrid NC  (for regrid_estofs)
# ---------------------------------------------------------------------------


def make_hgrid_nc(path: Path, n_nodes: int = 20, n_bnd: int = 10) -> None:
    """Write a minimal SCHISM hgrid NetCDF (``nodeCoords`` + ``openBndNodes``).

    This is *not* a full ESMFMESH file — it only contains the variables read
    by :func:`regrid_estofs` to extract open-boundary node coordinates.
    """
    rng = np.random.default_rng(7)
    lons = rng.uniform(LON_W + 0.2, LON_E - 0.2, n_nodes)
    lats = rng.uniform(LAT_S + 0.2, LAT_N - 0.2, n_nodes)
    bnd_indices = rng.choice(n_nodes, size=n_bnd, replace=False).astype("i4")

    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("nodeCount", n_nodes)
        ds.createDimension("coordDim", 2)
        ds.createDimension("openBndCount", n_bnd)

        coords = ds.createVariable("nodeCoords", "f8", ("nodeCount", "coordDim"))
        coords[:, 0] = lons
        coords[:, 1] = lats

        bnd = ds.createVariable("openBndNodes", "i4", ("openBndCount",))
        bnd[:] = bnd_indices


# ---------------------------------------------------------------------------
# WRF geo_em file  (for regrid_forcings)
# ---------------------------------------------------------------------------


def make_geo_em_nc(path: Path, nx: int = 6, ny: int = 5) -> None:
    """Write a minimal WRF geo_em-style NetCDF file.

    Grid is stored in ``(Time, south_north, west_east)`` order as WRF does.
    The ``CoastalForcingRegridder`` reads ``[0, :].T`` so the effective shape
    passed to :func:`build_grid` is ``(nx, ny)`` (Fortran order).
    """
    lons_1d = np.linspace(GEO_LON_W, GEO_LON_E, nx)
    lats_1d = np.linspace(GEO_LAT_S, GEO_LAT_N, ny)
    lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)  # (ny, nx)

    # Corner coordinates: one extra node in each direction
    lons_c_1d = np.linspace(GEO_LON_W - 0.2, GEO_LON_E + 0.2, nx + 1)
    lats_c_1d = np.linspace(GEO_LAT_S - 0.2, GEO_LAT_N + 0.2, ny + 1)
    lons_c, lats_c = np.meshgrid(lons_c_1d, lats_c_1d)  # (ny+1, nx+1)

    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.createDimension("south_north_stag", ny + 1)
        ds.createDimension("west_east_stag", nx + 1)

        xlat = ds.createVariable("XLAT_M", "f4", ("Time", "south_north", "west_east"))
        xlat[:] = lats_2d[np.newaxis, :, :]

        xlon = ds.createVariable("XLONG_M", "f4", ("Time", "south_north", "west_east"))
        xlon[:] = lons_2d[np.newaxis, :, :]

        cxlat = ds.createVariable("XLAT_C", "f4", ("Time", "south_north_stag", "west_east_stag"))
        cxlat[:] = lats_c[np.newaxis, :, :]

        cxlon = ds.createVariable("XLONG_C", "f4", ("Time", "south_north_stag", "west_east_stag"))
        cxlon[:] = lons_c[np.newaxis, :, :]

        hgt = ds.createVariable("HGT_M", "f4", ("Time", "south_north", "west_east"))
        hgt[:] = np.zeros((1, ny, nx), dtype="f4")


# ---------------------------------------------------------------------------
# LDASIN forcing file  (for regrid_forcings)
# ---------------------------------------------------------------------------

# Time in minutes since 1970-01-01 anchored to a fixed date for repeatability
_LDASIN_TIME_MINUTES = 28512000.0  # 2024-03-15 00:00:00 UTC


def make_ldasin_nc(
    path: Path,
    nx: int = 6,
    ny: int = 5,
    time_minutes: float = _LDASIN_TIME_MINUTES,
) -> None:
    """Write a minimal WRF-Hydro LDASIN forcing NetCDF file.

    Variables are stored in ``(time, south_north, west_east)`` order (C order),
    matching real LDASIN files.  The :class:`CoastalForcingRegridder` reads
    them as ``variable[0, :].T`` to get ``(nx, ny)`` Fortran-order arrays.
    """
    rng = np.random.default_rng(13)
    shape = (1, ny, nx)

    var_ranges = [
        ("U2D", -5.0, 5.0),
        ("V2D", -5.0, 5.0),
        ("T2D", 280.0, 305.0),
        ("Q2D", 0.005, 0.02),
        ("PSFC", 98000.0, 102000.0),
        ("SWDOWN", 0.0, 800.0),
        ("LWDOWN", 300.0, 450.0),
        ("LQFRAC", 0.0, 1.0),
        ("RAINRATE", 0.0, 5.0e-5),
    ]

    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)

        tv = ds.createVariable("time", "f4", ("time",))
        tv.units = "minutes since 1970-01-01 00:00:00"
        tv[:] = [time_minutes]

        for name, lo, hi in var_ranges:
            v = ds.createVariable(name, "f4", ("time", "south_north", "west_east"))
            v[:] = rng.uniform(lo, hi, shape)


# ---------------------------------------------------------------------------
# ESMFMESH file  (for regrid_forcings / ESMF.Mesh loading)
# ---------------------------------------------------------------------------


def make_esmfmesh_nc(path: Path) -> None:
    """Write a minimal ESMFMESH NetCDF file.

    Six nodes forming four triangular elements, covering the same Hawaii-like
    domain as the other fixtures.  Node indices in ``elementConn`` are
    1-based per the ESMFMESH specification.

    Node layout::

        4 --- 5 --- 6    lat=21.0
        |  \\ |  \\ |
        1 --- 2 --- 3    lat=20.5
      lon=-157.5  -156.5
    """
    node_lons = np.array([-157.5, -157.0, -156.5, -157.5, -157.0, -156.5], dtype="f8")
    node_lats = np.array([20.5, 20.5, 20.5, 21.0, 21.0, 21.0], dtype="f8")

    # 1-based connectivity for four triangles
    conn = np.array(
        [
            [1, 2, 5],
            [1, 5, 4],
            [2, 3, 6],
            [2, 6, 5],
        ],
        dtype="i4",
    )
    num_nodes_per_elem = np.array([3, 3, 3, 3], dtype="i1")

    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.gridType = "unstructured mesh"
        ds.version = "0.9"

        ds.createDimension("nodeCount", 6)
        ds.createDimension("elementCount", 4)
        ds.createDimension("maxNodePElement", 3)
        ds.createDimension("coordDim", 2)

        nc = ds.createVariable("nodeCoords", "f8", ("nodeCount", "coordDim"))
        nc.units = "degrees"
        nc[:, 0] = node_lons
        nc[:, 1] = node_lats

        ec = ds.createVariable("elementConn", "i4", ("elementCount", "maxNodePElement"))
        ec.start_index = np.int32(1)
        ec[:] = conn

        nec = ds.createVariable("numElementConn", "i1", ("elementCount",))
        nec[:] = num_nodes_per_elem
