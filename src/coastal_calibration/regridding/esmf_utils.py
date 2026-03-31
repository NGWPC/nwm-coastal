"""ESMF helper utilities inspired by xESMF's design patterns.

Provides clean abstractions for common ESMF operations:
- Grid and LocStream construction from numpy arrays
- Regridder classes that compute weights once and reuse them
- MPI-aware data gathering utilities

Design principles (from xESMF):
1. Build grids from plain numpy arrays, not raw ESMF API calls
2. Separate weight computation from weight application
3. Reuse regridding weights across timesteps where possible
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import esmpy as ESMF

# esmpy (>=8.4.0) requires Manager() before local_pet() returns the true MPI
# rank.  Call it here so all modules that import esmf_utils get correct ranks.
ESMF.Manager(debug=False)  # pyright: ignore[reportCallIssue]

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD


# ---------------------------------------------------------------------------
# Grid / LocStream construction
# ---------------------------------------------------------------------------


class GridBounds(NamedTuple):
    """Local MPI partition bounds for an ESMF Grid.

    These bounds define the slice of the global coordinate arrays
    that belong to this MPI rank.
    """

    y_lo: int
    y_hi: int
    x_lo: int
    x_hi: int


def build_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_corners: np.ndarray | None = None,
    lat_corners: np.ndarray | None = None,
) -> tuple[ESMF.Grid, GridBounds]:
    """Create an ESMF Grid from 2D lon/lat arrays.

    Inspired by ``xesmf.backend.Grid.from_xarray``. Handles the boilerplate
    of creating the Grid, extracting partition bounds, and populating
    coordinate arrays.

    Parameters
    ----------
    lon, lat
        Cell center coordinates, shape ``(Nlon, Nlat)``.
        Fortran-contiguous arrays are recommended for ESMF performance.
    lon_corners, lat_corners
        Cell corner coordinates, shape ``(Nlon+1, Nlat+1)``.
        Required for conservative regridding methods.

    Returns
    -------
    grid
        ESMF Grid with coordinates populated.
    bounds
        Local partition bounds for slicing data arrays.
    """
    if lon.ndim != 2 or lat.ndim != 2:
        raise ValueError("lon and lat must be 2D arrays")
    if lon.shape != lat.shape:
        raise ValueError(f"lon shape {lon.shape} != lat shape {lat.shape}")

    for name, arr in [("lon", lon), ("lat", lat)]:
        if not arr.flags["F_CONTIGUOUS"]:
            warnings.warn(
                f"{name} is not Fortran-contiguous; this may affect ESMF performance.",
                stacklevel=2,
            )

    nlon, nlat = lon.shape
    LON, LAT = 0, 1

    stagger: list[int] = [ESMF.StaggerLoc.CENTER]
    if lon_corners is not None:
        stagger.append(ESMF.StaggerLoc.CORNER)  # pyright: ignore[reportArgumentType]

    grid = ESMF.Grid(
        np.array([nlon, nlat]),
        staggerloc=stagger,
        coord_sys=ESMF.CoordSys.SPH_DEG,
    )

    # Extract local partition bounds
    x_lo = grid.lower_bounds[ESMF.StaggerLoc.CENTER][LON]  # pyright: ignore[reportOptionalSubscript]
    x_hi = grid.upper_bounds[ESMF.StaggerLoc.CENTER][LON]  # pyright: ignore[reportOptionalSubscript]
    y_lo = grid.lower_bounds[ESMF.StaggerLoc.CENTER][LAT]  # pyright: ignore[reportOptionalSubscript]
    y_hi = grid.upper_bounds[ESMF.StaggerLoc.CENTER][LAT]  # pyright: ignore[reportOptionalSubscript]

    # Populate center coordinates for this partition
    grid.get_coords(LON)[...] = lon[x_lo:x_hi, y_lo:y_hi]  # pyright: ignore[reportOptionalSubscript]
    grid.get_coords(LAT)[...] = lat[x_lo:x_hi, y_lo:y_hi]  # pyright: ignore[reportOptionalSubscript]

    # Populate corner coordinates if provided
    if lon_corners is not None and lat_corners is not None:
        xc_lo = grid.lower_bounds[ESMF.StaggerLoc.CORNER][LON]  # pyright: ignore[reportCallIssue, reportArgumentType]
        xc_hi = grid.upper_bounds[ESMF.StaggerLoc.CORNER][LON]  # pyright: ignore[reportCallIssue, reportArgumentType]
        yc_lo = grid.lower_bounds[ESMF.StaggerLoc.CORNER][LAT]  # pyright: ignore[reportCallIssue, reportArgumentType]
        yc_hi = grid.upper_bounds[ESMF.StaggerLoc.CORNER][LAT]  # pyright: ignore[reportCallIssue, reportArgumentType]
        grid.get_coords(LON, staggerloc=ESMF.StaggerLoc.CORNER)[...] = lon_corners[
            xc_lo:xc_hi, yc_lo:yc_hi
        ]
        grid.get_coords(LAT, staggerloc=ESMF.StaggerLoc.CORNER)[...] = lat_corners[
            xc_lo:xc_hi, yc_lo:yc_hi
        ]

    return grid, GridBounds(y_lo, y_hi, x_lo, x_hi)


def build_locstream(lon: np.ndarray, lat: np.ndarray) -> ESMF.LocStream:
    """Create an ESMF LocStream from 1D **global** coordinate arrays.

    The *global* array is partitioned across MPI ranks so that each rank
    owns a contiguous slice.  ``ESMF.LocStream(n)`` creates *n* points
    **locally** on the calling rank, so we must compute the local share
    and pass the corresponding coordinate slice.

    Parameters
    ----------
    lon, lat
        **Global** point coordinates (1D, same on every rank).

    Returns
    -------
    ESMF.LocStream
        LocStream with coordinates populated for the local partition.
        Use ``locstream.lower_bounds[0]`` / ``locstream.upper_bounds[0]``
        to determine the global index range owned by this rank.
    """
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("lon and lat must be 1D arrays")
    if len(lon) != len(lat):
        raise ValueError(f"lon length {len(lon)} != lat length {len(lat)}")

    n_global = len(lon)
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Simple contiguous partitioning: distribute n_global across ranks
    base, remainder = divmod(n_global, size)
    if rank < remainder:
        local_start = rank * (base + 1)
        local_count = base + 1
    else:
        local_start = rank * base + remainder
        local_count = base

    locstream = ESMF.LocStream(local_count, coord_sys=ESMF.CoordSys.SPH_DEG)
    locstream["ESMF:Lon"] = lon[local_start : local_start + local_count].astype(np.float64)
    locstream["ESMF:Lat"] = lat[local_start : local_start + local_count].astype(np.float64)

    # Store the global index range so callers can slice data arrays
    locstream._global_lower = local_start  # pyright: ignore[reportAttributeAccessIssue]
    locstream._global_upper = local_start + local_count  # pyright: ignore[reportAttributeAccessIssue]
    return locstream


# ---------------------------------------------------------------------------
# Regridder classes
# ---------------------------------------------------------------------------


class Regridder:
    """Reusable ESMF regridder that computes weights once.

    Inspired by xESMF's core pattern: compute weights on first call,
    then reuse the ESMF.Regrid handle for all subsequent regridding
    operations. This avoids the expensive weight computation per timestep.

    Parameters
    ----------
    src_field
        Source ESMF Field (defines the source grid geometry).
    dst_field
        Destination ESMF Field (defines the target grid geometry).
    method
        ESMF regridding method.
    unmapped_action
        How to handle unmapped destination cells.
    src_mask_values
        Source mask values to exclude from regridding.
    extrap_method
        Extrapolation method for unmapped destination cells.
    """

    def __init__(
        self,
        src_field: ESMF.Field,
        dst_field: ESMF.Field,
        method: ESMF.RegridMethod = ESMF.RegridMethod.BILINEAR,  # pyright: ignore[reportArgumentType]
        unmapped_action: ESMF.UnmappedAction = ESMF.UnmappedAction.IGNORE,  # pyright: ignore[reportArgumentType]
        src_mask_values: list[int] | None = None,
        extrap_method: ESMF.ExtrapMethod | None = None,
    ):
        kwargs: dict[str, Any] = {
            "srcfield": src_field,
            "dstfield": dst_field,
            "regrid_method": method,
            "unmapped_action": unmapped_action,
        }
        if src_mask_values is not None:
            kwargs["src_mask_values"] = src_mask_values
        if extrap_method is not None:
            kwargs["extrap_method"] = extrap_method

        self._handle = ESMF.Regrid(**kwargs)

    def destroy(self):
        """Release ESMF resources held by the regrid handle."""
        self._handle.destroy()

    def __call__(
        self,
        src_field: ESMF.Field,
        dst_field: ESMF.Field,
        zero_region: ESMF.Region | None = None,
    ) -> ESMF.Field:
        """Apply pre-computed regridding weights."""
        kwargs: dict[str, Any] = {"srcfield": src_field, "dstfield": dst_field}
        if zero_region is not None:
            kwargs["zero_region"] = zero_region
        return self._handle(**kwargs)


class MaskedRegridder:
    """Regridder for time-varying source masks.

    When the source data mask changes each timestep (e.g. ESTOFS data with
    varying wet/dry cells), the ESMF.Regrid weights must be recomputed.
    This class encapsulates the regridding parameters so only the fields
    need to be passed on each call.

    Parameters
    ----------
    method
        ESMF regridding method.
    unmapped_action
        How to handle unmapped destination cells.
    src_mask_values
        Source mask values to exclude from regridding.
    """

    def __init__(
        self,
        method: ESMF.RegridMethod = ESMF.RegridMethod.NEAREST_STOD,  # pyright: ignore[reportArgumentType]
        unmapped_action: ESMF.UnmappedAction = ESMF.UnmappedAction.IGNORE,  # pyright: ignore[reportArgumentType]
        src_mask_values: list[int] | None = None,
    ):
        self.method = method
        self.unmapped_action = unmapped_action
        self.src_mask_values = src_mask_values or []

    def __call__(
        self,
        src_field: ESMF.Field,
        dst_field: ESMF.Field,
    ) -> ESMF.Field:
        """Build a fresh regridder with the current mask state and apply it."""
        kwargs: dict[str, Any] = {
            "srcfield": src_field,
            "dstfield": dst_field,
            "regrid_method": self.method,
            "unmapped_action": self.unmapped_action,
        }
        if self.src_mask_values:
            kwargs["src_mask_values"] = self.src_mask_values

        handle = ESMF.Regrid(**kwargs)
        return handle(src_field, dst_field)


# ---------------------------------------------------------------------------
# MPI gather helpers
# ---------------------------------------------------------------------------


def gather_reduce(
    local_data: np.ndarray,
    global_shape: tuple[int, ...],
    root: int = 0,
) -> np.ndarray | None:
    """Sum-reduce distributed field data to a single rank.

    Each MPI rank contributes its local partition of the data. The root
    rank receives the element-wise sum across all ranks.

    Parameters
    ----------
    local_data
        This rank's contribution (same shape as global output).
    global_shape
        Shape of the full global array.
    root
        MPI rank that receives the result.

    Returns
    -------
    Summed array on root, ``None`` on other ranks.
    """
    result = np.zeros(global_shape) if comm.Get_rank() == root else None
    comm.Reduce(local_data, result, op=MPI.SUM, root=root)
    return result


def gatherv_1d(
    local_data: np.ndarray,
    local_count: int,
    root: int = 0,
) -> np.ndarray | None:
    """Gather variable-length 1D data from all ranks to root.

    Parameters
    ----------
    local_data
        This rank's data (1D).
    local_count
        Number of elements on this rank.
    root
        MPI rank that receives the result.

    Returns
    -------
    Concatenated array on root, ``None`` on other ranks.
    """
    count_arr = np.asarray([local_count], dtype="i")
    all_counts = np.empty(comm.Get_size(), dtype="i") if comm.Get_rank() == root else None
    comm.Gather(count_arr, all_counts, root=root)

    result = np.zeros(int(all_counts.sum())) if comm.Get_rank() == root else None  # pyright: ignore[reportOptionalMemberAccess]

    comm.Gatherv(sendbuf=local_data, recvbuf=(result, all_counts), root=root)
    return result


def allreduce_minmax(values: np.ndarray) -> tuple[float, float]:
    """Compute global min and max of an array across all MPI ranks.

    Parameters
    ----------
    values
        Local array.

    Returns
    -------
    (global_min, global_max)
    """
    g_min = np.empty(1, dtype=np.float32)
    g_max = np.empty(1, dtype=np.float32)
    comm.Allreduce(np.float32(values.min()), g_min, op=MPI.MIN)
    comm.Allreduce(np.float32(values.max()), g_max, op=MPI.MAX)
    return float(g_min[0]), float(g_max[0])
