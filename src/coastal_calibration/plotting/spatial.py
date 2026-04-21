"""Shared spatial water-level renderer for SCHISM and SFINCS.

:func:`plot_water_level` accepts the canonical :class:`xarray.Dataset` layout
produced by :mod:`coastal_calibration.schism.outputs` (unstructured triangular
/ quad mesh) and :mod:`coastal_calibration.sfincs.outputs` (regular grid), and
dispatches on the ``mesh_type`` attribute to the right matplotlib primitive
(:func:`matplotlib.pyplot.tripcolor` vs :meth:`matplotlib.axes.Axes.pcolormesh`).

The single-frame renderer is reused by
:mod:`coastal_calibration.plotting.animate` to produce animations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.collections import Collection
    from matplotlib.tri import Triangulation

#: Valid ``shading`` argument for :meth:`matplotlib.axes.Axes.pcolormesh`.
PColorShading = Literal["flat", "nearest", "gouraud", "auto"]

__all__ = ["plot_water_level", "triangulate_faces", "triangulate_faces_indexed"]


#: Known ``mesh_type`` attribute values.
_REGULAR = "regular"
_UGRID_TRI_QUAD = "ugrid-triangle-or-quad"  # node-valued data (e.g. SCHISM)
_UGRID_QUADTREE = "ugrid-quadtree"  # face-valued data (e.g. SFINCS quadtree)


def _select_time(da: xr.DataArray, time: int | Any) -> xr.DataArray:
    """Return the slice at *time* (positional int or datetime-like label)."""
    if isinstance(time, (int, np.integer)):
        return da.isel(time=int(time))
    return da.sel(time=time, method="nearest")


def _auto_variable(ds: xr.Dataset) -> str:
    """Return the water-level variable present on *ds*.

    Prefers ``"zs"`` (SFINCS) then ``"elevation"`` (SCHISM); falls back to
    the first data variable with a ``time`` dim.
    """
    for candidate in ("zs", "elevation"):
        if candidate in ds.data_vars:
            return candidate
    for name, da in ds.data_vars.items():
        if "time" in da.dims:
            return str(name)
    msg = "Could not auto-detect a water-level variable on dataset"
    raise ValueError(msg)


def triangulate_faces_indexed(
    face_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split mixed triangle/quad connectivity into triangles, tracking source faces.

    Each quadrilateral face ``(a, b, c, d)`` becomes two triangles
    ``(a, b, c)`` and ``(a, c, d)``; existing triangles (``-1`` in the 4th
    column, or a 3-wide input) are kept as-is. Input is assumed 0-based.

    Parameters
    ----------
    face_nodes : ndarray of shape (n_face, k)
        Face-node connectivity with fill value ``-1``. ``k`` is 3 or 4.

    Returns
    -------
    triangles : ndarray of shape (n_tri, 3), int64
        Triangle-only connectivity for
        :class:`matplotlib.tri.Triangulation`.
    triangle_face : ndarray of shape (n_tri,), int64
        For each output triangle, the index of the original face in
        *face_nodes*. This lets callers broadcast face-valued data to
        triangle-valued arrays for ``tripcolor(shading="flat")``.
    """
    face_nodes = np.asarray(face_nodes, dtype=np.int64)
    if face_nodes.ndim != 2:
        msg = f"face_nodes must be 2-D, got shape {face_nodes.shape}"
        raise ValueError(msg)
    if face_nodes.shape[1] < 3:
        msg = f"face_nodes must have at least 3 columns, got {face_nodes.shape[1]}"
        raise ValueError(msg)

    n_face = face_nodes.shape[0]
    face_idx = np.arange(n_face, dtype=np.int64)

    if face_nodes.shape[1] == 3:
        return face_nodes.copy(), face_idx

    is_tri = face_nodes[:, 3] < 0
    tris = face_nodes[is_tri, :3]
    tri_source = face_idx[is_tri]

    quads = face_nodes[~is_tri]
    quad_source = face_idx[~is_tri]
    q1 = quads[:, [0, 1, 2]]
    q2 = quads[:, [0, 2, 3]]

    triangles = np.concatenate([tris, q1, q2], axis=0).astype(np.int64, copy=False)
    triangle_face = np.concatenate([tri_source, quad_source, quad_source], axis=0)
    return triangles, triangle_face


def triangulate_faces(face_nodes: np.ndarray) -> np.ndarray:
    """Flatten mixed triangle/quad connectivity into a pure triangle array.

    Thin wrapper around :func:`triangulate_faces_indexed` that discards the
    triangle → face mapping. See that function for details.
    """
    triangles, _ = triangulate_faces_indexed(face_nodes)
    return triangles


def _build_triangulation(ds: xr.Dataset) -> Triangulation:
    """Construct a :class:`matplotlib.tri.Triangulation` from *ds*."""
    from matplotlib.tri import Triangulation

    node_x = np.asarray(ds["node_x"].to_numpy(), dtype=np.float64)
    node_y = np.asarray(ds["node_y"].to_numpy(), dtype=np.float64)
    triangles = triangulate_faces(ds["face_nodes"].to_numpy())
    return Triangulation(node_x, node_y, triangles=triangles)


def _extended_cmap(name: str) -> Any:
    """Return a copy of *name* with ``set_under`` / ``set_over`` pinned.

    Matplotlib's default behaviour for out-of-range values depends on the
    colormap instance; by explicitly pinning the under/over colours we
    guarantee that data falling below *vmin* or above *vmax* renders with
    the nearest limit colour (rather than a transparent "bad" slot or a
    different colour chosen by the colormap's defaults).
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name).copy()
    cmap.set_under(cmap(0.0))
    cmap.set_over(cmap(1.0))
    return cmap


def _colorbar_extend(values: np.ndarray, vmin: float, vmax: float) -> str:
    """Return the ``extend`` argument for :func:`~matplotlib.figure.Figure.colorbar`.

    The arrow caps on the colorbar advertise that data exists below or above
    the visible range, signalling the caller that the limits have clamped
    the visual mapping.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "neither"
    below = bool((finite < vmin).any())
    above = bool((finite > vmax).any())
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def _plot_regular(
    ds: xr.Dataset,
    variable: str,
    time: int | Any,
    ax: Axes,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    shading: PColorShading,
) -> Collection:
    frame = _select_time(ds[variable], time)
    values = np.asarray(frame.to_numpy())
    x = np.asarray(ds["x"].to_numpy())
    y = np.asarray(ds["y"].to_numpy())
    return ax.pcolormesh(
        x,
        y,
        values,
        cmap=_extended_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        shading=shading,
    )


def _plot_unstructured(
    ds: xr.Dataset,
    variable: str,
    time: int | Any,
    ax: Axes,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> Collection:
    frame = _select_time(ds[variable], time)
    values = np.asarray(frame.to_numpy(), dtype=np.float64)
    triangulation = _build_triangulation(ds)
    # ``gouraud`` yields smooth interpolation across each triangle and is the
    # only shading that preserves the node-length value contract required by
    # in-place ``set_array`` updates in the animation writer.
    return ax.tripcolor(
        triangulation,
        values,
        cmap=_extended_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        shading="gouraud",
    )


def _plot_quadtree(
    ds: xr.Dataset,
    variable: str,
    time: int | Any,
    ax: Axes,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> Collection:
    """Render face-valued data on a UGRID quadtree mesh.

    Each quad is split into two triangles that share the face value via
    ``tripcolor(shading="flat")``. The returned collection's array is
    triangle-length; ``animate_water_level`` writers need to expand their
    face-length frames using the cached ``triangle_face`` mapping.
    """
    from matplotlib.tri import Triangulation

    frame = _select_time(ds[variable], time)
    face_values = np.asarray(frame.to_numpy(), dtype=np.float64)

    node_x = np.asarray(ds["node_x"].to_numpy(), dtype=np.float64)
    node_y = np.asarray(ds["node_y"].to_numpy(), dtype=np.float64)
    triangles, triangle_face = triangulate_faces_indexed(ds["face_nodes"].to_numpy())
    triangulation = Triangulation(node_x, node_y, triangles=triangles)

    tri_values = face_values[triangle_face]
    coll = ax.tripcolor(
        triangulation,
        facecolors=tri_values,
        cmap=_extended_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )
    # Stash the broadcast map on the collection so animation writers can
    # re-expand face-length frames to triangle-length without recomputing.
    coll.triangle_face_map = triangle_face  # type: ignore[attr-defined]
    return coll


def _percentile_limits(
    ds: xr.Dataset, variable: str, low: float = 1.0, high: float = 99.0
) -> tuple[float, float]:
    """Return (vmin, vmax) from the 1st/99th percentiles of the full time series."""
    vals = np.asarray(ds[variable].to_numpy())
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def _wet_mask(ds: xr.Dataset, dry_threshold: float) -> xr.DataArray | None:
    """Return a boolean DataArray that is True at wet cells.

    Prefers the model's authoritative classifier (``dryFlagNode``, written by
    SCHISM); falls back to ``h > dry_threshold`` (works for both SCHISM after
    F1 derivation and SFINCS after F2 derivation). Returns *None* if neither
    is present, signalling the caller to skip masking.
    """
    if "dryFlagNode" in ds.data_vars:
        return ds["dryFlagNode"] == 0
    if "h" in ds.data_vars:
        return ds["h"] > dry_threshold
    return None


def _apply_wet_mask(ds: xr.Dataset, variable: str, dry_threshold: float) -> xr.Dataset:
    """Return a shallow copy of *ds* with *variable* masked at dry cells.

    Dry cells become NaN, which matplotlib renders with the cmap's "bad"
    slot (transparent by default). The original dataset is not modified.
    """
    wet = _wet_mask(ds, dry_threshold)
    if wet is None:
        return ds
    out = ds.copy()
    out[variable] = ds[variable].where(wet)
    return out


def _add_basemap(
    ax: Axes,
    crs: str,
    source: Any | None,
    zoom: int | None,
) -> None:
    """Overlay a basemap on *ax* using contextily.

    *crs* is the data CRS (e.g. ``"EPSG:32614"``); contextily reprojects the
    tiles into that CRS so the data axes coordinates remain correct.
    *source* defaults to Esri WorldImagery (satellite). *zoom* defaults to
    contextily's automatic level when *None*.
    """
    try:
        import contextily as cx
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        msg = (
            "basemap=True requires the 'contextily' package. "
            "Install via the 'plot' optional dependency, e.g. "
            "`pixi add -e dev contextily`."
        )
        raise ImportError(msg) from exc

    if source is None:
        source = cx.providers.Esri.WorldImagery  # pyright: ignore[reportAttributeAccessIssue]
    kwargs: dict[str, Any] = {"crs": crs, "source": source}
    if zoom is not None:
        kwargs["zoom"] = zoom
    cx.add_basemap(ax, **kwargs)


def _default_title(ds: xr.Dataset, variable: str, time: int | Any) -> str:
    """Build a default title from the selected timestamp, when available."""
    selected = _select_time(ds[variable], time)
    if "time" in selected.coords:
        t_val = np.asarray(selected.coords["time"].to_numpy())
        if np.issubdtype(t_val.dtype, np.datetime64):
            t_str = np.datetime_as_string(t_val, unit="m")
            return f"{variable} @ {t_str}"
    return variable


def _resolve_limits(
    ds: xr.Dataset,
    variable: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    """Fill in unset vmin/vmax from the 1st/99th percentile of the full series."""
    if vmin is not None and vmax is not None:
        return vmin, vmax
    auto_lo, auto_hi = _percentile_limits(ds, variable)
    return (auto_lo if vmin is None else vmin, auto_hi if vmax is None else vmax)


def _dispatch_plot(
    ds: xr.Dataset,
    variable: str,
    time: int | Any,
    ax: Axes,
    *,
    mesh_type: str,
    cmap: str,
    vmin: float,
    vmax: float,
    shading_regular: PColorShading,
) -> Collection:
    """Route to the right matplotlib primitive based on mesh_type."""
    if mesh_type == _REGULAR:
        return _plot_regular(
            ds,
            variable,
            time,
            ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading=shading_regular,
        )
    if mesh_type == _UGRID_TRI_QUAD:
        return _plot_unstructured(
            ds,
            variable,
            time,
            ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    if mesh_type == _UGRID_QUADTREE:
        return _plot_quadtree(
            ds,
            variable,
            time,
            ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    msg = f"Unknown mesh_type={mesh_type!r}"
    raise ValueError(msg)


def plot_water_level(
    ds: xr.Dataset,
    time: int | Any = 0,
    *,
    variable: str | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 7),
    shading_regular: PColorShading = "auto",
    mask_dry: bool = True,
    dry_threshold: float = 0.05,
    basemap: bool = False,
    basemap_source: Any | None = None,
    basemap_zoom: int | None = None,
    crs: str | None = None,
) -> tuple[Axes, Collection]:
    """Render a single water-level (or water-depth) frame from *ds*.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by one of the ``load_*`` readers in
        :mod:`coastal_calibration.schism.outputs` /
        :mod:`coastal_calibration.sfincs.outputs`. Must carry a
        ``mesh_type`` attribute of ``"regular"``,
        ``"ugrid-triangle-or-quad"``, or ``"ugrid-quadtree"``.
    time : int or datetime-like, default ``0``
        Time selector. Integers are positional (``isel``); datetime-like
        values (``numpy.datetime64``, ``pandas.Timestamp``, ISO strings)
        are passed to ``sel(..., method="nearest")``.
    variable : str, optional
        Data variable to plot. Common choices:

        - ``"zs"`` / ``"elevation"`` — water-surface elevation (the default,
          auto-detected when omitted)
        - ``"h"`` — water depth
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure + axes are created when *None*.
    cmap : str, default ``"viridis"``
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Colormap limits. When *None*, both are computed from the 1st/99th
        percentiles of the full time series **after dry-cell masking** so
        outlier bed-elevation cells do not dominate the scale.
    colorbar : bool, default ``True``
        If *True*, attach a colorbar to the axes.
    title : str, optional
        Plot title. Defaults to ``"<variable> @ <time>"`` when the dataset
        has a datetime time axis.
    figsize : (float, float), default ``(10, 7)``
        Figure size when *ax* is *None*.
    shading_regular : str, default ``"auto"``
        ``shading`` passed through to ``pcolormesh`` on the regular-grid
        path. Ignored for unstructured meshes.
    mask_dry : bool, default ``True``
        Mask cells classified as dry. The mask is sourced (in priority order):

        1. ``dryFlagNode == 0`` if the SCHISM dry flag is in the dataset.
        2. ``h > dry_threshold`` otherwise (water depth above threshold).

        When neither variable is present the dataset is plotted unmasked.
    dry_threshold : float, default ``0.05``
        Water-depth threshold (m) used by the fallback mask. 5 cm is a
        common SFINCS/HydroMT convention for "wet enough to plot".
    basemap : bool, default ``False``
        If *True*, overlay a satellite basemap via :mod:`contextily`. The
        data CRS is taken from *crs*, falling back to ``ds.attrs["crs"]``.
        Raises if no CRS is available.
    basemap_source : optional
        Tile provider passed to :func:`contextily.add_basemap`. Defaults
        to ``cx.providers.Esri.WorldImagery`` (satellite imagery).
    basemap_zoom : int, optional
        Zoom level for the basemap tiles; *None* lets contextily pick.
    crs : str, optional
        Override the dataset CRS for basemap reprojection (e.g.
        ``"EPSG:32614"``). When *None*, ``ds.attrs["crs"]`` is used.

    Returns
    -------
    (ax, collection)
        The axes used for drawing and the primitive collection
        (``QuadMesh`` for regular grids, ``TriMesh`` / ``PolyCollection``
        for unstructured). Returning the collection allows
        :func:`coastal_calibration.plotting.animate.animate_water_level`
        to call ``collection.set_array(...)`` per frame.

    Raises
    ------
    KeyError
        If *ds* lacks the ``mesh_type`` attribute or *variable* is not a
        data variable.
    ValueError
        If no water-level variable can be auto-detected or the mesh type
        is unknown.
    """
    import matplotlib.pyplot as plt

    mesh_type = ds.attrs.get("mesh_type")
    if mesh_type is None:
        msg = (
            "Dataset is missing the 'mesh_type' attribute. "
            "Use coastal_calibration.{schism,sfincs}.outputs loaders to produce it."
        )
        raise KeyError(msg)

    if variable is None:
        variable = _auto_variable(ds)
    if variable not in ds.data_vars:
        available = sorted(str(name) for name in ds.data_vars)
        msg = f"Variable {variable!r} not in dataset (have: {available})"
        raise KeyError(msg)

    if mask_dry:
        ds = _apply_wet_mask(ds, variable, dry_threshold)

    vmin, vmax = _resolve_limits(ds, variable, vmin, vmax)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    coll = _dispatch_plot(
        ds,
        variable,
        time,
        ax,
        mesh_type=mesh_type,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading_regular=shading_regular,
    )

    # Equal aspect keeps lon/lat or projected meters looking correct.
    # ``adjustable="box"`` resizes the axes in figure-space instead of
    # clipping the data limits — important when a contextily basemap
    # comes in afterward, since contextily sets its own xlim/ylim and
    # ``adjustable="datalim"`` would otherwise emit a warning about
    # ignoring them.
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title if title is not None else _default_title(ds, variable, time))
    if mesh_type == _REGULAR:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")

    if colorbar:
        fig = ax.get_figure()
        if fig is not None:
            units = ds[variable].attrs.get("units", "m")
            extend = _colorbar_extend(np.asarray(ds[variable].to_numpy()), vmin, vmax)
            fig.colorbar(
                coll,
                ax=ax,
                label=f"{variable} ({units})",
                shrink=0.8,
                extend=extend,
            )

    if basemap:
        crs_label = crs if crs is not None else ds.attrs.get("crs")
        if not crs_label:
            msg = (
                "basemap=True but no CRS available. Pass `crs='EPSG:...'` or "
                "ensure the loader populated ds.attrs['crs']."
            )
            raise ValueError(msg)
        _add_basemap(ax, crs_label, basemap_source, basemap_zoom)

    return ax, coll
