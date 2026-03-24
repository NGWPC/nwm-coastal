"""Flood depth map visualization.

Provides :func:`plot_floodmap` which reads a flood-depth COG (produced
by :func:`~coastal_calibration.utils.floodmap.create_flood_depth_map`),
prints summary metadata, and plots the depth on a satellite basemap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_floodmap"]


def plot_floodmap(
    floodmap_path: Path | str,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    basemap: bool = True,
    basemap_source: Any | None = None,
    basemap_zoom: int = 12,
    max_display_px: int = 2000,
    vmax_percentile: float = 98,
    figsize: tuple[float, float] = (11, 7),
) -> tuple[Figure, Axes]:
    """Plot a flood-depth COG with an optional satellite basemap.

    Reads at an overview level that keeps the longest axis under
    *max_display_px* pixels, masks dry / NaN pixels, and renders
    with a viridis color map.

    Parameters
    ----------
    floodmap_path:
        Path to the flood-depth GeoTIFF (e.g. ``floodmap_hmax.tif``).
    ax:
        Existing axes to plot into.  A new figure is created when *None*.
    title:
        Plot title.  Defaults to ``"Flood depth (hmax)"``.
    basemap:
        If *True* (default), overlay satellite imagery via *contextily*.
    basemap_source:
        Tile provider passed to ``contextily.add_basemap``.
    basemap_zoom:
        Zoom level for the basemap tiles.
    max_display_px:
        Target maximum dimension (in pixels) for the rendered raster.
        Controls which overview level is read.
    vmax_percentile:
        Upper percentile for the color-map range.
    figsize:
        Figure size when *ax* is *None*.

    Returns
    -------
    (Figure, Axes)
    """
    import matplotlib.pyplot as plt
    import rasterio

    floodmap_path = Path(floodmap_path)
    if not floodmap_path.exists():
        raise FileNotFoundError(
            f"Flood map not found: {floodmap_path} — "
            "ensure floodmap_dem is set and sfincs_map.nc contains zsmax."
        )

    # ── Read metadata at full resolution ─────────────────────────
    with rasterio.open(floodmap_path) as src:
        bounds = src.bounds
        raster_crs = src.crs

        overviews = src.overviews(1)
        ovr_idx = next(
            (
                i
                for i, f in enumerate(overviews)
                if max(src.height, src.width) / f <= max_display_px
            ),
            len(overviews) - 1,
        )

    # ── Read at overview level ───────────────────────────────────
    with rasterio.open(floodmap_path, overview_level=ovr_idx) as src:
        hmax = src.read(1)

    hmax_masked = np.where(np.isfinite(hmax) & (hmax > 0), hmax, np.nan)
    valid = np.isfinite(hmax_masked)
    if valid.any():
        pass

    # ── Plot ─────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
        if fig is None:  # pragma: no cover
            msg = "ax must be attached to a Figure"
            raise ValueError(msg)

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(alpha=0)
    hmax_plot = np.ma.masked_invalid(hmax_masked)

    im = ax.imshow(
        hmax_plot,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=0,
        vmax=np.nanpercentile(hmax_masked, vmax_percentile) if valid.any() else 1,
        interpolation="nearest",
        zorder=2,
    )
    fig.colorbar(im, ax=ax, label="Flood depth (m)", shrink=0.6, pad=0.02, extend="both")

    if title is None:
        title = "Flood depth (hmax)"
    ax.set_title(title)

    if basemap:
        import contextily as cx

        if basemap_source is None:
            basemap_source = cx.providers.Esri.WorldImagery  # ty: ignore[unresolved-attribute]
        cx.add_basemap(ax, crs=raster_crs, source=basemap_source, zoom=basemap_zoom)

    return fig, ax  # ty: ignore[invalid-return-type]
