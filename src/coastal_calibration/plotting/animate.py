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
    _resolve_limits,  # pyright: ignore[reportPrivateUsage]  -- shared intra-package helper
    plot_water_level,
)

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.animation import AbstractMovieWriter

_log = logging.getLogger(__name__)

__all__ = ["animate_water_level"]


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
    time_indices: np.ndarray,
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
        ``mesh_type`` attribute that :func:`plot_water_level` recognises.
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
        If the output suffix is unrecognised and no explicit *writer* was
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

    # Compute shared colour limits on the *masked* dataset so dry-cell
    # outliers do not stretch the colormap. The original ds is untouched
    # by _apply_wet_mask (it returns a shallow copy).
    ds_for_limits = _apply_wet_mask(ds, variable, dry_threshold) if mask_dry else ds
    vmin_r, vmax_r = _resolve_limits(ds_for_limits, variable, vmin, vmax)

    time_indices = np.arange(0, ds.sizes["time"], max(1, int(time_stride)), dtype=np.int64)
    if time_indices.size == 0:
        msg = "No frames to animate — time dimension is empty after applying time_stride."
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
