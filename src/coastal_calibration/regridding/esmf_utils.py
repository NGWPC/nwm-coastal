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
from typing import TYPE_CHECKING, Any, NamedTuple

import esmpy as ESMF

# esmpy (>=8.4.0) requires Manager() before local_pet() returns the true MPI
# rank.  Call it here so all modules that import esmf_utils get correct ranks.
ESMF.Manager(debug=False)  # pyright: ignore[reportCallIssue]

import numpy as np
from mpi4py import MPI

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
    lon: NDArray[np.floating[Any]],
    lat: NDArray[np.floating[Any]],
    lon_corners: NDArray[np.floating[Any]] | None = None,
    lat_corners: NDArray[np.floating[Any]] | None = None,
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


def build_unstructured_mesh(
    lon: NDArray[np.floating[Any]],
    lat: NDArray[np.floating[Any]],
    elements: NDArray[np.integer[Any]],
    *,
    start_index: int = 0,
    bbox: tuple[float, float, float, float] | None = None,
    bbox_buffer_deg: float = 1.0,
    node_mask: NDArray[Any] | None = None,
) -> tuple[ESMF.Mesh, NDArray[np.integer[Any]]]:
    """Create an ESMF Mesh from an unstructured triangular grid.

    Used for BILINEAR-style regridding from unstructured sources where
    barycentric weights inside source triangles produce smooth
    interpolation between mesh nodes — the unstructured-mesh analog of
    fractional pixel sampling on rasters.

    A bounding box can be supplied to spatially subset very large
    sources (e.g. global STOFS, ~12M nodes / ~25M triangles) before
    handing them to ESMF.  Only nodes within ``bbox + bbox_buffer_deg``
    are kept, and only elements whose three vertices are all in the
    kept set survive.

    Parameters
    ----------
    lon, lat
        Global node coordinates, 1D arrays of length ``n_nodes``.
    elements
        Triangle connectivity, shape ``(n_elem, 3)``.
    start_index
        ``0`` if ``elements`` is 0-based, ``1`` if 1-based.
    bbox
        Optional ``(lon_min, lat_min, lon_max, lat_max)`` for spatial
        subsetting.  When provided only nodes within this bbox (plus
        ``bbox_buffer_deg`` margin) are loaded into the ESMF mesh.
    bbox_buffer_deg
        Margin added to ``bbox`` on each side to ensure all elements
        spanning the destination region remain intact.
    node_mask
        Optional per-node mask aligned with the global ``lon``/``lat``
        arrays.  Non-zero entries are excluded by ESMF when computing
        regridding weights via ``src_mask_values=[1]``.  The mask is
        subset to ``keep_idx`` internally.

    Returns
    -------
    ESMF.Mesh
        Mesh with the (possibly subset) nodes and triangular elements.
    keep_idx : numpy.ndarray
        1D array of node indices that were retained from the original
        ``lon``/``lat`` arrays.  Use this to subset per-node data
        arrays before assigning into ``field.data`` of an
        :class:`ESMF.Field` defined on the returned mesh.
    """
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("lon and lat must be 1D arrays")
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError(f"elements must be (N, 3); got {elements.shape}")

    n_global = len(lon)

    # Convert connectivity to 0-based
    if start_index == 1:
        elements = elements - 1
    elif start_index != 0:
        raise ValueError(f"start_index must be 0 or 1, got {start_index}")

    # Optional bbox filter to keep the mesh small
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        lat_in = (lat >= lat_min - bbox_buffer_deg) & (lat <= lat_max + bbox_buffer_deg)
        if lon_min <= lon_max:
            lon_in = (lon >= lon_min - bbox_buffer_deg) & (lon <= lon_max + bbox_buffer_deg)
        else:
            # bbox wraps the antimeridian (e.g. Aleutians): union of the
            # two longitude segments either side of +/-180.
            lon_in = (lon >= lon_min - bbox_buffer_deg) | (lon <= lon_max + bbox_buffer_deg)
        keep_idx = np.where(lon_in & lat_in)[0]
    else:
        keep_idx = np.arange(n_global, dtype=np.int64)

    # Filter elements: keep only triangles whose vertices are all in keep_idx
    keep_set = np.zeros(n_global, dtype=bool)
    keep_set[keep_idx] = True
    elem_kept_mask = keep_set[elements[:, 0]] & keep_set[elements[:, 1]] & keep_set[elements[:, 2]]
    kept_elements = elements[elem_kept_mask]

    # ESMF requires every node in the mesh to be referenced by at
    # least one element.  After bbox-filtering elements, some bbox
    # nodes near the boundary become orphans (their incident
    # triangles span outside the kept region and were dropped).
    # Restrict ``keep_idx`` to nodes that are actually referenced.
    keep_idx = np.unique(kept_elements)

    # Remap element vertex indices to the local 0..len(keep_idx) range
    remap = -np.ones(n_global, dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)
    local_elements = remap[kept_elements]

    n_local_nodes = len(keep_idx)
    n_local_elems = len(local_elements)

    # esmpy passes these arrays through ctypes to the ESMF C++ layer;
    # use the default Python ``int`` / ``float`` dtypes (matching the
    # official esmpy mesh-creation examples) to avoid undefined-
    # behavior crashes that occur with explicit int32 dtypes.
    node_ids = np.arange(1, n_local_nodes + 1)
    node_coords = np.empty(n_local_nodes * 2)
    node_coords[0::2] = lon[keep_idx]
    node_coords[1::2] = lat[keep_idx]
    node_owners = np.zeros(n_local_nodes)

    elem_ids = np.arange(1, n_local_elems + 1)
    elem_types = np.full(n_local_elems, ESMF.MeshElemType.TRI)
    elem_conn = local_elements.flatten()

    if node_mask is not None:
        local_mask = np.asarray(node_mask)[keep_idx].astype(int, copy=False)
    else:
        local_mask = None

    # Element centroids — esmpy is unstable when these are omitted on
    # large unstructured meshes; passing them explicitly avoids
    # segfaults inside the ESMF C++ layer.
    kept_lon = lon[keep_idx]
    kept_lat = lat[keep_idx]
    elem_centroid = np.empty(n_local_elems * 2)
    elem_centroid[0::2] = kept_lon[local_elements].mean(axis=1)
    elem_centroid[1::2] = kept_lat[local_elements].mean(axis=1)

    mesh = ESMF.Mesh(parametric_dim=2, spatial_dim=2, coord_sys=ESMF.CoordSys.SPH_DEG)
    mesh.add_nodes(
        n_local_nodes,
        node_ids,
        node_coords,
        node_owners,
        node_mask=local_mask,
    )
    mesh.add_elements(
        n_local_elems,
        elem_ids,
        elem_types,
        elem_conn,
        element_coords=elem_centroid,
    )

    return mesh, keep_idx


def build_locstream(
    lon: NDArray[np.floating[Any]], lat: NDArray[np.floating[Any]]
) -> ESMF.LocStream:
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
    local_data: NDArray[Any],
    global_shape: tuple[int, ...],
    root: int = 0,
) -> NDArray[Any] | None:
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
    local_data: NDArray[Any],
    local_count: int,
    root: int = 0,
) -> NDArray[Any] | None:
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

    # (buffer, counts) is valid mpi4py Gatherv recvbuf syntax that its stubs miss.
    comm.Gatherv(sendbuf=local_data, recvbuf=(result, all_counts), root=root)  # pyright: ignore[reportArgumentType]
    return result


def allreduce_minmax(values: NDArray[Any]) -> tuple[float, float]:
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
