"""Animate a time-dependent water-level field to an MP4 or GIF.

:func:`animate_water_level` builds one matplotlib frame with
:func:`coastal_calibration.plotting.plot_water_level`, then advances through
the time dimension by calling ``coll.set_array(...)`` on the primitive
collection returned by the frame builder. The animation is written with
matplotlib's :class:`~matplotlib.animation.FFMpegWriter` (for ``.mp4``) or
:class:`~matplotlib.animation.PillowWriter` (for ``.gif``).

The same renderer handles all three dispatch paths (regular grid SFINCS,
unstructured SCHISM, and UGRID-quadtree SFINCS) by picking the right
``set_array`` contract based on the collection type.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from coastal_calibration.plotting.spatial import (
    _apply_wet_mask,  # pyright: ignore[reportPrivateUsage]  -- shared intra-package helper
    _auto_variable,  # pyright: ignore[reportPrivateUsage]  -- shared intra-package helper
    _colorbar_extend,  # pyright: ignore[reportPrivateUsage]  -- shared intra-package helper
    _percentile_limits,  # pyright: ignore[reportPrivateUsage]  -- shared intra-package helper
    _resolve_limits,  # pyright: ignore[reportPrivateUsage]  -- shared intra-package helper
    plot_water_level,
)

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.animation import AbstractMovieWriter
    from numpy.typing import NDArray

_log = logging.getLogger(__name__)

__all__ = ["animate_water_level", "animate_water_level_comparison"]


#: Known output-file suffixes mapped to matplotlib writer names.
_WRITER_BY_SUFFIX: dict[str, Literal["ffmpeg", "pillow"]] = {
    ".mp4": "ffmpeg",
    ".mov": "ffmpeg",
    ".avi": "ffmpeg",
    ".gif": "pillow",
}


def _pick_writer(
    outfile: Path,
    fps: int,
    writer: Literal["auto", "ffmpeg", "pillow"],
) -> AbstractMovieWriter:
    """Return a matplotlib movie writer, falling back gracefully when needed."""
    from matplotlib.animation import FFMpegWriter, PillowWriter, writers

    if writer == "auto":
        choice = _WRITER_BY_SUFFIX.get(outfile.suffix.lower())
        if choice is None:
            msg = (
                f"Cannot infer animation writer from suffix {outfile.suffix!r}. "
                f"Pass writer='ffmpeg' or 'pillow' explicitly, or use "
                f"one of: {sorted(_WRITER_BY_SUFFIX)}."
            )
            raise ValueError(msg)
        writer = choice

    if writer == "ffmpeg":
        if not writers.is_available("ffmpeg"):
            # Gif writer is pure-Python and always available; warn and fall through.
            if outfile.suffix.lower() == ".gif":
                _log.warning("ffmpeg unavailable; using Pillow writer for %s", outfile)
                return PillowWriter(fps=fps)
            msg = (
                "ffmpeg is not available on PATH. Install via "
                "`pixi add -e dev ffmpeg` or `conda install ffmpeg -c conda-forge`, "
                "or pass writer='pillow' to write a GIF."
            )
            raise RuntimeError(msg)
        return FFMpegWriter(fps=fps, bitrate=2000)

    if writer == "pillow":
        return PillowWriter(fps=fps)

    msg = f"Unknown writer {writer!r}. Expected 'auto', 'ffmpeg', or 'pillow'."
    raise ValueError(msg)


def _make_updater(
    coll: Any,
    ds: xr.Dataset,
    variable: str,
    time_indices: NDArray[np.integer[Any]],
    *,
    mask_dry: bool,
    dry_threshold: float,
) -> Any:
    """Build the per-frame update function.

    Each frame:

    1. Pulls the time slice for *variable*.
    2. Optionally masks dry cells using ``dryFlagNode`` (preferred) or
       ``h > dry_threshold``. Both source variables are themselves
       time-varying, so the mask is recomputed per frame.
    3. Calls ``coll.set_array(...)`` with values in the shape the
       collection expects:

       - :class:`~matplotlib.collections.QuadMesh` → flat ``values.ravel()``
       - :class:`~matplotlib.tri.TriMesh` (``shading="gouraud"``) → node-length
       - ``tripcolor(facecolors=..., shading="flat")`` → triangle-length,
         expanded from face-length via the ``triangle_face_map`` stashed
         on the collection.
    """
    from matplotlib.collections import QuadMesh

    has_tri_map = hasattr(coll, "triangle_face_map")
    da = ds[variable]

    # Pre-pick the mask source (varies per time so we re-index each frame).
    mask_var: str | None = None
    use_dry_flag = False
    if mask_dry:
        if "dryFlagNode" in ds.data_vars:
            mask_var = "dryFlagNode"
            use_dry_flag = True
        elif "h" in ds.data_vars:
            mask_var = "h"

    def update(i: int) -> tuple[Any, ...]:
        t_idx = int(time_indices[i])
        frame = np.asarray(da.isel(time=t_idx).to_numpy(), dtype=np.float64)
        if mask_var is not None:
            mask_frame = np.asarray(ds[mask_var].isel(time=t_idx).to_numpy())
            wet = (mask_frame == 0) if use_dry_flag else (mask_frame > dry_threshold)
            frame = np.where(wet, frame, np.nan)
        if isinstance(coll, QuadMesh):
            coll.set_array(frame.ravel())
        elif has_tri_map:
            coll.set_array(frame[coll.triangle_face_map])
        else:
            coll.set_array(frame)
        return (coll,)

    return update


def _frame_title(ds: xr.Dataset, variable: str, frame_idx: int, prefix: str | None) -> str:
    """Build the per-frame title string."""
    t = np.asarray(ds["time"].to_numpy())[frame_idx]
    if np.issubdtype(np.asarray(t).dtype, np.datetime64):
        t_str = np.datetime_as_string(np.asarray(t), unit="m")
    else:
        t_str = str(t)
    base = f"{variable} @ {t_str}"
    return f"{prefix} — {base}" if prefix else base


def animate_water_level(
    ds: xr.Dataset,
    outfile: str | Path,
    *,
    variable: str | None = None,
    fps: int = 10,
    time_stride: int = 1,
    dpi: int = 150,
    writer: Literal["auto", "ffmpeg", "pillow"] = "auto",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (10, 7),
    title_prefix: str | None = None,
    mask_dry: bool = True,
    dry_threshold: float = 0.05,
) -> Path:
    """Render a time-animation of the water-level field to a movie file.

    Parameters
    ----------
    ds : xarray.Dataset
        Canonical dataset from a ``load_*`` reader in
        :mod:`coastal_calibration.schism.outputs` /
        :mod:`coastal_calibration.sfincs.outputs`. Must carry a
        ``mesh_type`` attribute that :func:`plot_water_level` recognizes.
    outfile : str or pathlib.Path
        Destination path. The suffix selects the writer:
        ``.mp4`` / ``.mov`` / ``.avi`` use ``FFMpegWriter`` (requires an
        ``ffmpeg`` binary on PATH); ``.gif`` uses the pure-Python
        ``PillowWriter``.
    variable : str, optional
        Variable to animate. Defaults to the result of :func:`_auto_variable`.
    fps : int, default 10
        Frames per second in the output.
    time_stride : int, default 1
        Keep every ``time_stride``-th frame from the dataset's time axis.
        Useful for previews or reducing file size on long runs.
    dpi : int, default 150
        Output resolution.
    writer : {"auto", "ffmpeg", "pillow"}, default ``"auto"``
        Writer selector; ``"auto"`` infers from the output suffix.
    cmap : str, default ``"viridis"``
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Colormap limits — shared across all frames. When *None*, each is
        filled in from the 1st/99th percentile of the full time series
        (see :func:`plot_water_level`).
    figsize : tuple[float, float], default ``(10, 7)``
        Figure size.
    title_prefix : str, optional
        Prefix prepended to the auto-generated per-frame title.
    mask_dry : bool, default ``True``
        Mask dry cells in every frame using ``dryFlagNode == 0`` (preferred)
        or ``h > dry_threshold`` as a fallback. See
        :func:`plot_water_level` for details.
    dry_threshold : float, default ``0.05``
        Water-depth threshold (m) for the fallback mask.

    Returns
    -------
    pathlib.Path
        The resolved output path.

    Raises
    ------
    RuntimeError
        If an ``.mp4``/``.mov``/``.avi`` is requested but ``ffmpeg`` is not
        on PATH.
    ValueError
        If the output suffix is unrecognized and no explicit *writer* was
        given.

    Notes
    -----
    The renderer is built from :func:`plot_water_level`, so any future
    additions to the frame layout (basemaps, projections, annotations)
    automatically flow through to animations.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    outfile = Path(outfile).expanduser().resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Validate the writer selection up front so a bad suffix raises before
    # we build the figure / animation.
    movie_writer = _pick_writer(outfile, fps=fps, writer=writer)

    if variable is None:
        variable = _auto_variable(ds)

    # Compute shared color limits on the *masked* dataset so dry-cell
    # outliers do not stretch the colormap. The original ds is untouched
    # by _apply_wet_mask (it returns a shallow copy).
    ds_for_limits = _apply_wet_mask(ds, variable, dry_threshold) if mask_dry else ds
    vmin_r, vmax_r = _resolve_limits(ds_for_limits, variable, vmin, vmax)

    stride = int(time_stride)
    if stride < 1:
        msg = f"time_stride must be >= 1 (got {time_stride})"
        raise ValueError(msg)
    time_indices = np.arange(0, ds.sizes["time"], stride, dtype=np.int64)
    if time_indices.size == 0:
        msg = "No frames to animate — time dimension is empty."
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=figsize)
    # Build the first frame; `plot_water_level` attaches a colorbar and
    # applies the same dry-cell mask we will reuse in every frame.
    _, coll = plot_water_level(
        ds,
        time=int(time_indices[0]),
        variable=variable,
        ax=ax,
        cmap=cmap,
        vmin=vmin_r,
        vmax=vmax_r,
        colorbar=True,
        title=_frame_title(ds, variable, int(time_indices[0]), title_prefix),
        mask_dry=mask_dry,
        dry_threshold=dry_threshold,
    )

    base_update = _make_updater(
        coll,
        ds,
        variable,
        time_indices,
        mask_dry=mask_dry,
        dry_threshold=dry_threshold,
    )

    def update(i: int) -> tuple[Any, ...]:
        artists = base_update(i)
        ax.set_title(_frame_title(ds, variable, int(time_indices[i]), title_prefix))
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(time_indices),
        interval=max(1, int(1000 / fps)),
        blit=False,
    )
    anim.save(str(outfile), writer=movie_writer, dpi=dpi)
    plt.close(fig)

    _log.info("wrote %s (%d frames, %d fps)", outfile, len(time_indices), fps)
    return outfile


def _trim_to_common_window(
    ds_left: xr.Dataset, ds_right: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    """Slice both datasets to their overlapping time window.

    The two model outputs are expected to span the same simulation period
    but at different output cadences. This helper enforces that the
    *animated* range starts and ends at coincident timestamps by slicing
    each dataset to the intersection of its time coverage.
    """
    t_left = np.asarray(ds_left["time"].to_numpy())
    t_right = np.asarray(ds_right["time"].to_numpy())
    if t_left.size == 0 or t_right.size == 0:
        msg = "One of the datasets has an empty time axis."
        raise ValueError(msg)

    start = max(t_left[0], t_right[0])
    end = min(t_left[-1], t_right[-1])
    if start > end:
        msg = (
            f"Datasets have no overlapping time window: "
            f"left=[{t_left[0]}..{t_left[-1]}], right=[{t_right[0]}..{t_right[-1]}]."
        )
        raise ValueError(msg)
    return ds_left.sel(time=slice(start, end)), ds_right.sel(time=slice(start, end))


def _nearest_index_array(
    target_times: NDArray[np.datetime64], lookup_times: NDArray[np.datetime64]
) -> NDArray[np.int64]:
    """For each *target_times* entry return the index of the nearest *lookup_times* entry.

    Both inputs must be sorted (xarray time coords always are after the
    upstream slice). Conversion to ``int64`` nanoseconds keeps the
    arithmetic numerical and lets ``np.searchsorted`` work uniformly across
    different datetime resolutions (e.g. ``ns``/``us``/``s``).
    """
    target = np.asarray(target_times).astype("datetime64[ns]").astype("int64")
    lookup = np.asarray(lookup_times).astype("datetime64[ns]").astype("int64")
    pos = np.searchsorted(lookup, target)
    pos = np.clip(pos, 0, lookup.size - 1)
    left = np.clip(pos - 1, 0, lookup.size - 1)
    use_left = np.abs(target - lookup[left]) < np.abs(target - lookup[pos])
    return np.where(use_left, left, pos).astype(np.int64)


def _resolve_shared_limits(
    ds_left: xr.Dataset,
    variable_left: str,
    ds_right: xr.Dataset,
    variable_right: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    """Span both datasets' 1st/99th percentiles when limits are not user-supplied."""
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    lo_l, hi_l = _percentile_limits(ds_left, variable_left)
    lo_r, hi_r = _percentile_limits(ds_right, variable_right)
    auto_lo = min(lo_l, lo_r)
    auto_hi = max(hi_l, hi_r)
    if auto_lo == auto_hi:
        auto_hi = auto_lo + 1.0
    return (
        auto_lo if vmin is None else float(vmin),
        auto_hi if vmax is None else float(vmax),
    )


def _format_time(t: Any) -> str:
    """Format a time scalar; minute precision for datetime64, ``str(t)`` otherwise."""
    arr = np.asarray(t)
    if np.issubdtype(arr.dtype, np.datetime64):
        return str(np.datetime_as_string(arr, unit="m"))
    return str(t)


def animate_water_level_comparison(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    outfile: str | Path,
    *,
    labels: tuple[str, str] = ("left", "right"),
    variable_left: str | None = None,
    variable_right: str | None = None,
    fps: int = 10,
    time_stride: int = 1,
    dpi: int = 150,
    writer: Literal["auto", "ffmpeg", "pillow"] = "auto",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (16, 7),
    title_prefix: str | None = None,
    mask_dry: bool = True,
    dry_threshold: float = 0.05,
) -> Path:
    """Render a side-by-side time-animation of two water-level fields.

    The left and right panels animate independently — each one is fed by its
    own dataset, can have a different mesh layout (e.g. SCHISM unstructured
    on the left, SFINCS quadtree or regular grid on the right), and may
    output at a different cadence. The two panels share a single colorbar
    placed to the right of both axes.

    The two datasets are expected to cover the same simulation period; the
    animation is run on the intersection of their time coverage and a
    nearest-neighbour lookup is used to pull each panel's frame at the
    master clock's timestamps.

    Parameters
    ----------
    ds_left, ds_right : xarray.Dataset
        Canonical datasets from a ``load_*`` reader in
        :mod:`coastal_calibration.schism.outputs` /
        :mod:`coastal_calibration.sfincs.outputs`. Each must carry a
        ``mesh_type`` attribute that :func:`plot_water_level` recognizes.
    outfile : str or pathlib.Path
        Destination path. Suffix selects the writer; see
        :func:`animate_water_level`.
    labels : (str, str), default ``("left", "right")``
        Per-panel titles drawn above each axes (e.g. model names).
    variable_left, variable_right : str, optional
        Variables to animate on each side. ``None`` auto-detects via
        :func:`_auto_variable` (``"zs"`` then ``"elevation"``).
    fps : int, default ``10``
        Frames per second in the output.
    time_stride : int, default ``1``
        Stride applied to the master clock's time axis (the dataset with
        fewer frames in the overlap window). Useful for previews.
    dpi : int, default ``150``
        Output resolution.
    writer : {"auto", "ffmpeg", "pillow"}, default ``"auto"``
        Writer selector; ``"auto"`` infers from the output suffix.
    cmap : str, default ``"viridis"``
        Matplotlib colormap name; shared by both panels.
    vmin, vmax : float, optional
        Shared colormap limits for both panels. ``None`` fills the unset
        side(s) by spanning the 1st/99th percentiles of *both* datasets so
        the colorbar is meaningful for either field.
    figsize : (float, float), default ``(16, 7)``
        Figure size; the default is wide enough for two square-ish panels
        plus the colorbar.
    title_prefix : str, optional
        Prefix prepended to the figure-level (suptitle) timestamp.
    mask_dry : bool, default ``True``
        Mask dry cells per frame on each side. See
        :func:`plot_water_level`.
    dry_threshold : float, default ``0.05``
        Water-depth threshold (m) for the fallback mask.

    Returns
    -------
    pathlib.Path
        The resolved output path.

    Raises
    ------
    ValueError
        If the two datasets share no overlapping time window, or if the
        master clock has no frames after striding.
    RuntimeError
        If an ``.mp4``/``.mov``/``.avi`` is requested but ``ffmpeg`` is not
        on PATH.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    outfile = Path(outfile).expanduser().resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    movie_writer = _pick_writer(outfile, fps=fps, writer=writer)

    var_l = variable_left if variable_left is not None else _auto_variable(ds_left)
    var_r = variable_right if variable_right is not None else _auto_variable(ds_right)

    ds_l, ds_r = _trim_to_common_window(ds_left, ds_right)

    stride = int(time_stride)
    if stride < 1:
        msg = f"time_stride must be >= 1 (got {time_stride})"
        raise ValueError(msg)

    # Master clock = side with fewer post-stride frames; the other side is
    # sampled by nearest-neighbour lookup so frame count stays predictable.
    n_l = (ds_l.sizes["time"] + stride - 1) // stride
    n_r = (ds_r.sizes["time"] + stride - 1) // stride
    if n_l <= n_r:
        master_ds = ds_l
        master_indices = np.arange(0, ds_l.sizes["time"], stride, dtype=np.int64)
    else:
        master_ds = ds_r
        master_indices = np.arange(0, ds_r.sizes["time"], stride, dtype=np.int64)
    if master_indices.size == 0:
        msg = "No frames to animate — master clock is empty after striding."
        raise ValueError(msg)
    master_times = np.asarray(master_ds["time"].to_numpy())[master_indices]

    left_indices = _nearest_index_array(master_times, ds_l["time"].to_numpy())
    right_indices = _nearest_index_array(master_times, ds_r["time"].to_numpy())

    ds_l_for_limits = _apply_wet_mask(ds_l, var_l, dry_threshold) if mask_dry else ds_l
    ds_r_for_limits = _apply_wet_mask(ds_r, var_r, dry_threshold) if mask_dry else ds_r
    vmin_r, vmax_r = _resolve_shared_limits(
        ds_l_for_limits, var_l, ds_r_for_limits, var_r, vmin, vmax
    )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    # Build first frame on each axes; suppress per-axes colorbars in favor
    # of the single shared one we attach below.
    _, coll_left = plot_water_level(
        ds_l,
        time=int(left_indices[0]),
        variable=var_l,
        ax=ax_left,
        cmap=cmap,
        vmin=vmin_r,
        vmax=vmax_r,
        colorbar=False,
        title=labels[0],
        mask_dry=mask_dry,
        dry_threshold=dry_threshold,
    )
    _, coll_right = plot_water_level(
        ds_r,
        time=int(right_indices[0]),
        variable=var_r,
        ax=ax_right,
        cmap=cmap,
        vmin=vmin_r,
        vmax=vmax_r,
        colorbar=False,
        title=labels[1],
        mask_dry=mask_dry,
        dry_threshold=dry_threshold,
    )

    units = ds_l[var_l].attrs.get("units") or ds_r[var_r].attrs.get("units", "m")
    combined = np.concatenate(
        [
            np.asarray(ds_l[var_l].to_numpy()).ravel(),
            np.asarray(ds_r[var_r].to_numpy()).ravel(),
        ]
    )
    extend = _colorbar_extend(combined, vmin_r, vmax_r)
    fig.colorbar(
        coll_left,
        ax=[ax_left, ax_right],
        label=f"water level ({units})",
        shrink=0.8,
        extend=extend,
    )

    update_left = _make_updater(
        coll_left,
        ds_l,
        var_l,
        left_indices,
        mask_dry=mask_dry,
        dry_threshold=dry_threshold,
    )
    update_right = _make_updater(
        coll_right,
        ds_r,
        var_r,
        right_indices,
        mask_dry=mask_dry,
        dry_threshold=dry_threshold,
    )

    def update(i: int) -> tuple[Any, ...]:
        artists_l = update_left(i)
        artists_r = update_right(i)
        t_str = _format_time(master_times[i])
        suptitle = f"{title_prefix} — {t_str}" if title_prefix else t_str
        fig.suptitle(suptitle)
        return artists_l + artists_r

    fig.suptitle(
        f"{title_prefix} — {_format_time(master_times[0])}"
        if title_prefix
        else _format_time(master_times[0])
    )

    anim = FuncAnimation(
        fig,
        update,
        frames=master_indices.size,
        interval=max(1, int(1000 / fps)),
        blit=False,
    )
    anim.save(str(outfile), writer=movie_writer, dpi=dpi)
    plt.close(fig)

    _log.info(
        "wrote %s (%d frames, %d fps, master=%s)",
        outfile,
        master_indices.size,
        fps,
        "left" if n_l <= n_r else "right",
    )
    return outfile
