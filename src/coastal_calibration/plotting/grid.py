"""SFINCS grid inspection and mesh plotting.

Provides :class:`SfincsGridInfo` for loading and summarising a SFINCS
grid (quadtree or regular) and :func:`plot_mesh` for visualising the
mesh with an optional satellite basemap.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pyproj import CRS

__all__ = ["LevelInfo", "SfincsGridInfo", "plot_mesh"]


class LevelInfo(NamedTuple):
    """Cell count and resolution for one refinement level."""

    count: int
    resolution: float


@dataclasses.dataclass
class SfincsGridInfo:
    """Summary of a SFINCS model grid.

    Use :meth:`from_model_root` to construct from a SFINCS model
    directory.  The instance carries enough pre-computed state to
    drive :func:`plot_mesh` without re-loading the model.

    Examples
    --------
    >>> info = SfincsGridInfo.from_model_root("run/sfincs_model")
    >>> print(info)
    SfincsGridInfo(quadtree, EPSG:32619)
      Faces:     293,850
      Edges:     596,123
      Level 1:    7,090 cells (512 m)
      ...
    """

    grid_type: str
    crs: CRS
    base_resolution: float
    levels: dict[int, LevelInfo]
    n_faces: int | None = None
    n_edges: int | None = None
    shape: tuple[int, int] | None = None

    # Internal arrays for plotting — not part of the public API.
    _verts: np.ndarray | None = dataclasses.field(default=None, repr=False)
    _level_per_face: np.ndarray | None = dataclasses.field(default=None, repr=False)
    _mask: np.ndarray | None = dataclasses.field(default=None, repr=False)
    _grid_extent: tuple[float, float, float, float] | None = dataclasses.field(
        default=None, repr=False
    )

    # ── factory ──────────────────────────────────────────────────

    @classmethod
    def from_model_root(
        cls,
        model_root: Path | str,
    ) -> SfincsGridInfo:
        """Load grid metadata from a SFINCS model directory.

        Parameters
        ----------
        model_root:
            Path to the SFINCS model directory (must contain ``sfincs.inp``
            and, for quadtree models, ``sfincs.nc``).
        """
        from coastal_calibration.stages._hydromt_compat import apply_all_patches
        from coastal_calibration.utils.logging import suppress_hydromt_output

        apply_all_patches()

        with suppress_hydromt_output():
            from hydromt_sfincs import SfincsModel

            sf = SfincsModel(root=str(model_root), mode="r+")
            sf.read()
        return cls._from_loaded_model(sf)

    @classmethod
    def _from_loaded_model(
        cls,
        model: Any,
    ) -> SfincsGridInfo:
        """Build from an already-loaded ``SfincsModel``."""
        crs = model.crs
        is_quadtree = model.grid_type == "quadtree"

        if is_quadtree:
            return cls._build_quadtree(model, crs)
        return cls._build_regular(model, crs)

    # ── quadtree builder ─────────────────────────────────────────

    @classmethod
    def _build_quadtree(
        cls,
        model: Any,
        crs: CRS,
    ) -> SfincsGridInfo:
        grid_ds = model.quadtree_grid.data
        grid = grid_ds.ugrid.grid
        fnc = grid.face_node_connectivity
        node_x, node_y = grid.node_x, grid.node_y

        # Mask fill values (-1) so they don't pollute min/max widths.
        fill_value = grid.fill_value
        valid = fnc != fill_value
        fnc_safe = np.where(valid, fnc, fnc[:, 0:1])

        cell_width = node_x[fnc_safe].max(axis=1) - node_x[fnc_safe].min(axis=1)

        # Derive base resolution from the coarsest cell so that levels
        # always start at 1 regardless of the grid configuration.
        base_resolution = float(cell_width.max())
        level_arr = np.round(np.log2(base_resolution / cell_width) + 1).astype(int)

        unique_levels, counts = np.unique(level_arr, return_counts=True)
        levels = {
            int(lv): LevelInfo(int(cnt), base_resolution / 2.0 ** (lv - 1))
            for lv, cnt in zip(unique_levels, counts, strict=True)
        }

        # Pre-compute vertex array for plotting.  For faces with fewer
        # nodes than *n_verts*, duplicate the first node so the polygon
        # closes properly without stretching to a distant fill-node.
        n_verts = fnc.shape[1]
        verts = np.zeros((grid.n_face, n_verts, 2))
        for j in range(n_verts):
            verts[:, j, 0] = node_x[fnc_safe[:, j]]
            verts[:, j, 1] = node_y[fnc_safe[:, j]]

        return cls(
            grid_type="quadtree",
            crs=crs,
            base_resolution=base_resolution,
            levels=levels,
            n_faces=int(grid.n_face),
            n_edges=int(grid.n_edge),
            _verts=verts,
            _level_per_face=level_arr,
        )

    # ── regular grid builder ─────────────────────────────────────

    @classmethod
    def _build_regular(
        cls,
        model: Any,
        crs: CRS,
    ) -> SfincsGridInfo:
        grid = model.grid
        mask = grid.mask.to_numpy()
        dx = float(grid.dx)
        dy = float(grid.dy)
        nmax = int(grid.nmax)
        mmax = int(grid.mmax)
        x0 = float(grid.x0)
        y0 = float(grid.y0)
        extent = (x0, x0 + mmax * dx, y0, y0 + nmax * dy)

        levels = {1: LevelInfo(int(mask.size), dx)}

        return cls(
            grid_type="regular",
            crs=crs,
            base_resolution=dx,
            levels=levels,
            shape=(nmax, mmax),
            _mask=mask,
            _grid_extent=extent,
        )

    # ── display ──────────────────────────────────────────────────

    def __str__(self) -> str:  # noqa: D105
        epsg = self.crs.to_epsg() if self.crs is not None else "?"
        lines = [f"SfincsGridInfo({self.grid_type}, EPSG:{epsg})"]

        if self.grid_type == "quadtree":
            lines.append(f"  Faces:     {self.n_faces:>10,}")
            lines.append(f"  Edges:     {self.n_edges:>10,}")
            for lv, info in sorted(self.levels.items()):
                lines.append(
                    f"  Level {lv}:   {info.count:>10,} cells ({info.resolution:.0f} m)"
                )
        else:
            assert self.shape is not None
            lines.append(f"  Shape:      {self.shape[0]} x {self.shape[1]}")
            res = next(iter(self.levels.values())).resolution
            lines.append(f"  Resolution: {res:.0f} m")

        return "\n".join(lines)

    def __repr__(self) -> str:  # noqa: D105
        return self.__str__()


# ── mesh plotting ────────────────────────────────────────────────────


_LEVEL_COLORS = ["#4575b4", "#91bfdb", "#fee090", "#d73027"]


def plot_mesh(
    info: SfincsGridInfo,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    basemap: bool = True,
    basemap_source: Any | None = None,
    basemap_zoom: int = 11,
    figsize: tuple[float, float] = (11, 7),
) -> tuple[Figure, Axes]:
    """Plot the SFINCS mesh coloured by refinement level.

    Parameters
    ----------
    info:
        Grid metadata from :meth:`SfincsGridInfo.from_model_root`.
    ax:
        Existing axes to plot into.  A new figure is created when *None*.
    title:
        Plot title.  Defaults to a description derived from *info*.
    basemap:
        If *True* (default), overlay satellite imagery via *contextily*.
    basemap_source:
        Tile provider passed to ``contextily.add_basemap``.  Defaults to
        ``cx.providers.Esri.WorldImagery``.
    basemap_zoom:
        Zoom level for the basemap tiles.
    figsize:
        Figure size when *ax* is *None*.

    Returns
    -------
    (Figure, Axes)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
        assert fig is not None

    if info.grid_type == "quadtree":
        _plot_quadtree(info, ax)
    else:
        _plot_regular(info, ax)

    if title is None:
        title = f"SFINCS {info.grid_type} mesh ({info.base_resolution:.0f} m)"
    ax.set_title(title)

    if basemap:
        _add_basemap(ax, info.crs, basemap_source, basemap_zoom)

    return fig, ax  # type: ignore[return-value]


def _plot_quadtree(info: SfincsGridInfo, ax: Axes) -> None:
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    assert info._verts is not None
    assert info._level_per_face is not None

    sorted_levels = sorted(info.levels)
    n_levels = len(sorted_levels)
    colors = _LEVEL_COLORS[:n_levels]
    cmap = ListedColormap(colors)
    bounds = [lv - 0.5 for lv in sorted_levels] + [sorted_levels[-1] + 0.5]
    norm = BoundaryNorm(bounds, ncolors=n_levels)

    pc = PolyCollection(list(info._verts), edgecolors="black", linewidths=0.1, alpha=0.4)
    pc.set_array(info._level_per_face.astype(float))
    pc.set_cmap(cmap)
    pc.set_norm(norm)
    ax.add_collection(pc)
    ax.autoscale_view()

    legend_handles = [
        Patch(
            facecolor=cmap(norm(lv)),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.5,
            label=f"Level {lv} ({info.levels[lv].resolution:.0f} m)",
        )
        for lv in sorted_levels
    ]
    ax.legend(handles=legend_handles, loc="lower right", title="Refinement level")


def _plot_regular(info: SfincsGridInfo, ax: Axes) -> None:
    assert info._mask is not None
    assert info._grid_extent is not None
    x0, x1, y0, y1 = info._grid_extent
    extent = (x0, x1, y0, y1)
    ax.imshow(
        np.where(info._mask > 0, info._mask, np.nan),
        extent=extent,
        origin="lower",
        alpha=0.4,
        cmap="Blues",
        interpolation="nearest",
    )


def _add_basemap(
    ax: Axes,
    crs: CRS,
    source: Any | None,
    zoom: int,
) -> None:
    import contextily as cx

    if source is None:
        source = cx.providers.Esri.WorldImagery  # type: ignore[attr-defined]
    cx.add_basemap(ax, crs=crs, source=source, zoom=zoom)
