"""User- and system-observation-point handling for water-level extraction.

This module turns an obs-points CSV (columns ``id, lon, lat`` in WGS84)
plus an optional set of NOAA-gauge points into a single time-series table
extracted from a model's 2-D water-level field. The intended caller is a
post-run plot stage that writes the result to an ``obs_water_level.parquet``
file alongside the model outputs.

The three responsibilities are:

- :func:`load_obs_points` parses the CSV and enforces its schema.
- :func:`validate_points_in_domain` builds an axis-aligned WGS84 box from
  the model mesh extent and checks that every point falls inside.
- :func:`extract_water_level_series` finds the nearest mesh cell (node /
  face / grid cell) for each point and pulls its time series, returning a
  :class:`pandas.DataFrame` indexed by time with one column per station.

All three dispatch on the canonical ``mesh_type`` attribute written by
:func:`coastal_calibration.schism.outputs.load_schism_elevation` and
:func:`coastal_calibration.sfincs.outputs.load_sfincs_water_level`, so the
same machinery works for SCHISM and SFINCS.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import shapely
    import xarray as xr
    from numpy.typing import NDArray

__all__ = [
    "combine_obs_points",
    "extract_water_level_series",
    "load_obs_points",
    "validate_points_in_domain",
]


#: Canonical column names required in the user-supplied CSV.
_REQUIRED_COLS: tuple[str, ...] = ("id", "lon", "lat")


def _is_wgs84(crs: str | None) -> bool:
    """Return True if *crs* is missing or names WGS84 (EPSG:4326).

    Handles case and whitespace variants (``"epsg:4326"``, ``"EPSG: 4326"``).
    A ``None`` CRS is treated as WGS84 — callers that can't identify the
    model CRS default to the lon/lat assumption used everywhere else in
    the package.
    """
    if crs is None:
        return True
    return crs.upper().replace(" ", "") == "EPSG:4326"


def combine_obs_points(user: pd.DataFrame | None, noaa: pd.DataFrame | None) -> pd.DataFrame:
    """Concat user- and NOAA-supplied obs-point frames; raise on id collision.

    Parameters
    ----------
    user, noaa : pandas.DataFrame or None
        Frames with at minimum ``id``, ``lon``, ``lat`` columns. Either
        can be ``None`` or empty.

    Returns
    -------
    pandas.DataFrame
        The row-wise concat, with the same three columns. When both
        inputs are empty / ``None``, returns an empty frame with the
        canonical schema.

    Raises
    ------
    ValueError
        If both frames contribute the same ``id``, so downstream
        time-series columns wouldn't be uniquely identifiable.
    """
    frames = [df for df in (user, noaa) if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=list(_REQUIRED_COLS))
    combined = pd.concat(frames, ignore_index=True)
    if combined["id"].duplicated().any():
        dupes = combined.loc[combined["id"].duplicated(keep=False), "id"].unique().tolist()
        msg = (
            "Duplicate obs-point ids detected between user CSV and NOAA gauges: "
            f"{sorted(dupes)}. Rename or remove the colliding user ids."
        )
        raise ValueError(msg)
    return combined


def load_obs_points(path: str | Path) -> pd.DataFrame:
    """Read a user-supplied observation-points CSV.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a CSV with columns ``id``, ``lon``, ``lat``.

    Returns
    -------
    pandas.DataFrame
        A frame with exactly those three columns. ``id`` is coerced to
        ``str``; ``lon`` / ``lat`` to ``float64``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If any required column is missing, if IDs are not unique, or if
        any coordinate is non-finite.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Observation points CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        msg = (
            f"{path.name} is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {_REQUIRED_COLS}."
        )
        raise ValueError(msg)

    df = df[list(_REQUIRED_COLS)].copy()
    df["id"] = df["id"].astype(str)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce").astype("float64")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce").astype("float64")

    if df["lon"].isna().any() or df["lat"].isna().any():
        bad = df[df["lon"].isna() | df["lat"].isna()]["id"].tolist()
        msg = f"Non-numeric lon/lat in {path.name} for id(s): {bad}"
        raise ValueError(msg)

    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(keep=False), "id"].unique().tolist()
        msg = f"Duplicate ids in {path.name}: {sorted(dupes)}"
        raise ValueError(msg)

    return df.reset_index(drop=True)


def _domain_bbox_wgs84(ds: xr.Dataset) -> shapely.Polygon:
    """Build an axis-aligned WGS84 bounding polygon from the mesh extent.

    Takes the min/max of ``node_x`` / ``node_y`` (or ``x`` / ``y`` for a
    regular grid), transforms to WGS84 if ``ds.attrs['crs']`` names a
    non-WGS84 CRS, and returns a shapely :class:`~shapely.Polygon`. A
    small ~1 cell padding (in degrees equivalent) is NOT added; points
    exactly on the boundary still pass ``contains``.
    """
    import shapely

    mesh_type = ds.attrs.get("mesh_type")
    if mesh_type == "regular":
        x = np.asarray(ds["x"].to_numpy(), dtype=np.float64)
        y = np.asarray(ds["y"].to_numpy(), dtype=np.float64)
    else:
        x = np.asarray(ds["node_x"].to_numpy(), dtype=np.float64)
        y = np.asarray(ds["node_y"].to_numpy(), dtype=np.float64)

    crs = ds.attrs.get("crs")
    if not _is_wgs84(crs):
        from pyproj import Transformer

        t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        # Project the four corners; that's enough for an axis-aligned box
        # because transverse-Mercator preserves the extreme corners along
        # each axis to within sub-degree at regional scale, and the caller
        # adds its own margin if it needs one.
        xx = np.array([x.min(), x.min(), x.max(), x.max()])
        yy = np.array([y.min(), y.max(), y.min(), y.max()])
        lon, lat = t.transform(xx, yy)
    else:
        lon = np.array([x.min(), x.max()])
        lat = np.array([y.min(), y.max()])

    return shapely.box(lon.min(), lat.min(), lon.max(), lat.max())


def validate_points_in_domain(points: pd.DataFrame, ds: xr.Dataset) -> shapely.Polygon:
    """Raise if any point in *points* falls outside the model WGS84 bbox.

    Parameters
    ----------
    points : pandas.DataFrame
        Must have ``lon`` and ``lat`` columns in WGS84.
    ds : xarray.Dataset
        Canonical dataset from a ``load_*`` reader.

    Returns
    -------
    shapely.Polygon
        The WGS84 bounding polygon used for the check. Callers typically
        discard this; it is returned for inspection / logging.

    Raises
    ------
    ValueError
        If one or more points are outside the domain. The error message
        lists offending IDs (up to 10).
    """
    import shapely

    bbox = _domain_bbox_wgs84(ds)
    pts = shapely.points(points["lon"].to_numpy(), points["lat"].to_numpy())
    # Use `covers` rather than `contains` so points exactly on the bbox
    # edge pass — `contains` excludes the boundary.
    inside = shapely.covers(bbox, pts)
    if not np.asarray(inside).all():
        bad_ids = points.loc[~np.asarray(inside), "id"].tolist()
        head = bad_ids[:10]
        more = "" if len(bad_ids) <= 10 else f" (and {len(bad_ids) - 10} more)"
        minx, miny, maxx, maxy = bbox.bounds
        msg = (
            f"{len(bad_ids)} observation point(s) fall outside the model "
            f"domain WGS84 bbox "
            f"[{minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}]: "
            f"{head}{more}"
        )
        raise ValueError(msg)
    return bbox


def _auto_variable(ds: xr.Dataset) -> str:
    """Pick the water-surface variable preferring ``zs`` then ``elevation``."""
    for candidate in ("zs", "elevation"):
        if candidate in ds.data_vars:
            return candidate
    msg = "Cannot auto-detect water-level variable (need 'zs' or 'elevation')."
    raise ValueError(msg)


def _query_points_in_model_crs(
    points: pd.DataFrame, ds: xr.Dataset
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (qx, qy) arrays in the model's native CRS.

    Points come in as WGS84 lon/lat. If the model's CRS is projected,
    transform; otherwise copy through.
    """
    lon = points["lon"].to_numpy(dtype=np.float64)
    lat = points["lat"].to_numpy(dtype=np.float64)
    crs = ds.attrs.get("crs")
    if not _is_wgs84(crs):
        from pyproj import Transformer

        t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        qx, qy = t.transform(lon, lat)
        return np.asarray(qx, dtype=np.float64), np.asarray(qy, dtype=np.float64)
    return lon, lat


def _face_centers(ds: xr.Dataset) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-face centroids from UGRID ``face_nodes`` + node coords."""
    fn = np.asarray(ds["face_nodes"].to_numpy(), dtype=np.int64)
    node_x = np.asarray(ds["node_x"].to_numpy(), dtype=np.float64)
    node_y = np.asarray(ds["node_y"].to_numpy(), dtype=np.float64)
    valid = fn >= 0
    # Replace fills with 0 (safe because they get zeroed out below).
    safe = np.where(valid, fn, 0)
    fx = np.where(valid, node_x[safe], 0.0).sum(axis=1) / valid.sum(axis=1)
    fy = np.where(valid, node_y[safe], 0.0).sum(axis=1) / valid.sum(axis=1)
    return fx, fy


def _argmin_per_query(
    cell_x: NDArray[np.float64],
    cell_y: NDArray[np.float64],
    qx: NDArray[np.float64],
    qy: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Return the index of the nearest cell for each query point.

    Uses :class:`scipy.spatial.KDTree` for an O(N log N) build and an
    O(M log N) query, replacing the previous O(M·N) Python loop.
    SciPy is already a transitive runtime dep (xarray, pandas) and is
    imported elsewhere in the package, so this adds no new constraint
    and scales to multi-million-node meshes against thousands of obs
    points without the per-query allocation overhead.
    """
    from scipy.spatial import KDTree

    tree = KDTree(np.column_stack([cell_x, cell_y]))
    _, idx = tree.query(np.column_stack([qx, qy]), k=1)
    return np.asarray(idx, dtype=np.int64)


def _nearest_cell_indices(ds: xr.Dataset, qx: NDArray[np.float64], qy: NDArray[np.float64]) -> Any:
    """Find nearest cell index per query point, dispatching on mesh type.

    Returns a dispatcher-specific payload the corresponding extractor
    knows how to handle: a flat index array for unstructured meshes,
    a ``(yi, xi)`` pair for regular grids.
    """
    mesh_type = ds.attrs.get("mesh_type")
    if mesh_type == "regular":
        x = np.asarray(ds["x"].to_numpy(), dtype=np.float64)
        y = np.asarray(ds["y"].to_numpy(), dtype=np.float64)
        xi = np.argmin(np.abs(x[np.newaxis, :] - qx[:, np.newaxis]), axis=1)
        yi = np.argmin(np.abs(y[np.newaxis, :] - qy[:, np.newaxis]), axis=1)
        return ("regular", yi.astype(np.int64), xi.astype(np.int64))

    if mesh_type == "ugrid-triangle-or-quad":
        nx = np.asarray(ds["node_x"].to_numpy(), dtype=np.float64)
        ny = np.asarray(ds["node_y"].to_numpy(), dtype=np.float64)
        idx = _argmin_per_query(nx, ny, qx, qy)
        return ("node", idx)

    if mesh_type == "ugrid-quadtree":
        fx, fy = _face_centers(ds)
        idx = _argmin_per_query(fx, fy, qx, qy)
        return ("face", idx)

    msg = f"Unsupported mesh_type for obs extraction: {mesh_type!r}"
    raise ValueError(msg)


def _pull_time_series(ds: xr.Dataset, variable: str, locator: Any) -> NDArray[np.float64]:
    """Pull the time series at each located cell; returns shape ``(n_time, n_pts)``."""
    import xarray as xr

    kind = locator[0]
    da = ds[variable]
    if kind == "regular":
        _, yi, xi = locator
        # Advanced indexing over (y, x) that preserves the ``time`` dim.
        out = da.isel(
            y=xr.DataArray(yi, dims="point"),
            x=xr.DataArray(xi, dims="point"),
        )
        return np.asarray(out.transpose("time", "point").to_numpy(), dtype=np.float64)
    if kind == "node":
        _, idx = locator
        return np.asarray(
            da.isel(node=xr.DataArray(idx, dims="point")).to_numpy(),
            dtype=np.float64,
        )
    if kind == "face":
        _, idx = locator
        return np.asarray(
            da.isel(face=xr.DataArray(idx, dims="point")).to_numpy(),
            dtype=np.float64,
        )
    msg = f"Unknown locator kind: {kind!r}"
    raise ValueError(msg)


def extract_water_level_series(
    ds: xr.Dataset,
    points: pd.DataFrame,
    *,
    variable: str | None = None,
) -> pd.DataFrame:
    """Extract per-point water-level time series by nearest-cell lookup.

    Parameters
    ----------
    ds : xarray.Dataset
        Canonical dataset from a ``load_*`` reader.
    points : pandas.DataFrame
        Must carry columns ``id``, ``lon``, ``lat`` in WGS84. Typically
        the concatenation of user-supplied points and NOAA gauges.
    variable : str, optional
        Data variable to extract. When *None* (default), auto-detects
        ``zs`` then ``elevation``.

    Returns
    -------
    pandas.DataFrame
        Indexed by the dataset's time axis, with one column per input
        point (named by its ``id``). Values are water-surface elevation
        (metres, MSL) interpolated by nearest-cell lookup in the model
        mesh.

    Notes
    -----
    The unstructured-mesh lookup is brute-force — one ``argmin`` scan
    of all mesh nodes/faces per observation point, implemented in
    Python. This is intentional (avoids a SciPy runtime dep and keeps
    peak memory bounded), but it scales as ``O(n_cells * n_points)``.
    Up to a few hundred obs points against a multi-million-node mesh
    stays well under a second; past that, pre-build a spatial index at
    the caller instead of calling this function inside a loop.
    """
    if variable is None:
        variable = _auto_variable(ds)
    if variable not in ds.data_vars:
        available = sorted(str(v) for v in ds.data_vars)
        msg = f"Variable {variable!r} not in dataset (have: {available})"
        raise KeyError(msg)
    if points.empty:
        return pd.DataFrame(
            index=pd.DatetimeIndex(ds["time"].to_numpy(), name="time"),
        )

    qx, qy = _query_points_in_model_crs(points, ds)
    locator = _nearest_cell_indices(ds, qx, qy)
    values = _pull_time_series(ds, variable, locator)

    time = pd.DatetimeIndex(ds["time"].to_numpy(), name="time")
    ids = points["id"].tolist()
    return pd.DataFrame(values, index=time, columns=ids)
