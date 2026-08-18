"""Flood depth map generation from SFINCS output.

Downscales coarse-grid water surface elevations onto a high-resolution
DEM to produce a Cloud Optimized GeoTIFF (COG) of maximum flood depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from coastal_calibration.logging import logger as _log

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel
    from numpy.typing import NDArray

__all__ = ["FloodmapInputError", "create_flood_depth_map"]


class FloodmapInputError(Exception):
    """A flood map was asked for but its inputs are not there.

    Distinct from the errors hydromt and rasterio raise, so the workflow
    stage can skip on "this run has nothing to downscale" without also
    swallowing a genuine defect from deeper down.
    """


def _ensure_overviews(tif_path: Path, log: Any) -> None:
    """Build overviews on a GeoTIFF if it has none."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(tif_path) as src:
        if src.overviews(1):
            return

    log(f"GeoTIFF has no overviews, building them: {tif_path.name}")

    # Compute overview levels based on a 256-pixel target block size.
    # The upstream ``build_overviews`` auto-mode uses the file's own
    # blockxsize, which for untiled GeoTIFFs equals the full width
    # and yields 0 levels.
    with rasterio.open(tif_path, "r+") as src:
        min_dim: int = min(int(src.width), int(src.height))
        levels: list[int] = []
        factor = 2
        while min_dim // factor >= 256:
            levels.append(factor)
            factor *= 2
        if not levels:
            levels = [2]
        log(f"Building {len(levels)} overview levels: {levels}")
        src.build_overviews(levels, Resampling.average)
        src.update_tags(ns="rio_overview", resampling="average")


def _reduce_zsmax(zsmax: Any) -> tuple[Any, NDArray[np.float32]]:
    """Reduce zsmax to max over the time dimension and return a flat array.

    Returns ``(zsmax_reduced, zs_flat)`` where *zsmax_reduced* is the
    time-collapsed DataArray and *zs_flat* is a 1-D float32 numpy array
    suitable for index-based lookup.
    """
    import hydromt  # noqa: F401  # pyright: ignore[reportUnusedImport]  # registers .raster
    import xugrid as xu

    if isinstance(zsmax, xu.UgridDataArray):
        grid_dims: Any = zsmax.ugrid.grid.dims
        timedim: set[Any] = set(zsmax.dims) - set(grid_dims)  # pyright: ignore[reportArgumentType]
    else:
        timedim = set(zsmax.dims) - set(zsmax.raster.dims)
    if timedim:
        zsmax = zsmax.max(timedim)  # pyright: ignore[reportCallIssue, reportOptionalCall]

    if isinstance(zsmax, xu.UgridDataArray):
        zs_flat: NDArray[np.float32] = np.asarray(
            zsmax.to_numpy(),  # pyright: ignore[reportCallIssue, reportOptionalCall]
            dtype=np.float32,
        )
    else:
        # Regular grid: flatten in Fortran (column-major) order so that
        # indices from ``SfincsGrid.get_indices_at_points`` (which computes
        # ``col * nmax + row``) map to the correct values.
        zs_flat = np.asarray(zsmax.values, dtype=np.float32).flatten(order="F")
    zs_flat[~np.isfinite(zs_flat)] = np.nan
    return zsmax, zs_flat


def _assert_index_hits_the_model(index_path: Path, dem_path: Path) -> None:
    """Fail loudly when the index COG maps no DEM pixel to any model cell.

    This is the one failure in the whole flood-map path that is otherwise
    silent: every pixel lands on nodata, the depths all come out NaN, and an
    empty GeoTIFF is written as if nothing were wrong. The usual cause is a
    DEM whose CRS differs from the model's, which puts every sampled point
    outside the grid -- exactly the upstream ``make_index_cog`` bug that
    :func:`~coastal_calibration.sfincs._hydromt_compat.patch_make_index_cog`
    exists to fix. Checking here means that patch can be removed and the
    result re-tested without an empty map slipping through.
    """
    import rasterio

    with rasterio.open(index_path) as src:
        nodata = 2147483647 if src.nodata is None else int(src.nodata)
        for _, window in src.block_windows(1):
            if (src.read(1, window=window) != nodata).any():
                return

    msg = (
        f"The index in {index_path.name} maps no DEM pixel to a model cell, so the "
        f"flood map would come out empty. Check that {dem_path.name} overlaps the "
        "model domain and that its CRS is handled when sampling the grid."
    )
    raise ValueError(msg)


#: A cell counts as never wet only at (numerically) zero depth. Deliberately
#: not ``hmin``: that is the threshold for calling a *pixel* flooded, and a cell
#: holding less than it can still contain DEM pixels below the cell's recorded
#: minimum that carry more.
_DRY_CELL_TOL = 1e-6


def _blank_dry_cells(zsmax: Any, output: Any, log: Any) -> Any:
    """Blank out model cells the simulation never actually flooded.

    On a cell that stayed dry, SFINCS writes ``zsmax`` equal to that cell's
    bed level.  Downscaling subtracts the *high-resolution* DEM instead, so
    every DEM pixel lying below the cell's mean bed elevation comes out as
    flooded, by as much as the relief within one cell.  Setting those cells
    to NaN drops them from the map: the model itself reports zero depth
    there.  ``msk == 0`` cells are outside the active domain entirely.
    """
    if "zb" not in output:
        return zsmax

    zb = np.asarray(output["zb"].to_numpy(), dtype=np.float64)
    reduced, _ = _reduce_zsmax(zsmax)
    depth = np.asarray(reduced.to_numpy(), dtype=np.float64) - zb

    drop = ~np.isfinite(depth) | (depth <= _DRY_CELL_TOL)
    if "msk" in output:
        drop |= np.asarray(output["msk"].to_numpy()) == 0

    n_drop = int(drop.sum())
    if n_drop:
        log(f"Excluding {n_drop} of {drop.size} model cells that never flooded")
    return reduced.where(~drop)


def _baseline_water_surface(output: Any) -> Any:
    """Return the lowest water surface each cell reaches, or *None*.

    A pixel under water even at the model's driest moment is permanently
    wet: sea, estuary, lake. Deriving that from the model beats cutting on
    DEM elevation, which also discards land lying below the vertical datum
    but hydraulically dry, such as leveed polders or subsided urban ground.
    """
    if "zs" not in output or "zb" not in output:
        return None
    zs = output["zs"]
    timedim = [d for d in zs.dims if "time" in str(d).lower()]
    baseline = zs.min(timedim) if timedim else zs

    # On a dry cell SFINCS writes zs == zb, so a naive comparison against the
    # DEM would read every sub-cell pixel below the cell bed as already wet.
    # A cell holding no water at baseline has no permanently wet pixels at
    # all, so blank it and let those pixels be judged on depth alone.
    zb = output["zb"]
    return baseline.where((baseline - zb) > _DRY_CELL_TOL)


def _depth_from_index(
    idx_src: Any, zs_flat: Any, dep_block: Any, window: Any, idx_nodata: int
) -> Any:
    """Compute flood depth for a DEM block using an index COG."""
    idx_block = idx_src.read(1, window=window)
    nodata_mask = idx_block == idx_nodata
    safe_idx = idx_block.copy()
    oob_mask = (safe_idx < 0) | (safe_idx >= len(zs_flat))
    invalid_mask = nodata_mask | oob_mask
    safe_idx[invalid_mask] = 0
    h = zs_flat[safe_idx] - dep_block
    h[invalid_mask] = np.nan
    return h


def _depth_from_rasterize(
    zsmax: Any,
    dep_block: Any,
    dem_crs: Any,
    dem_transform: Any,
    bm0: int,
    bm1: int,
    bn0: int,
    bn1: int,
    reproj_method: str,
) -> Any:
    """Compute flood depth for a DEM block by rasterizing zsmax."""
    import hydromt  # noqa: F401  # pyright: ignore[reportUnusedImport]  # registers .raster
    import xarray as xr
    import xugrid as xu

    x_coords = dem_transform[2] + (np.arange(bm0, bm1) + 0.5) * dem_transform[0]
    y_coords = dem_transform[5] + (np.arange(bn0, bn1) + 0.5) * dem_transform[4]
    dep_da = xr.DataArray(
        dep_block,
        dims=("y", "x"),
        coords={"y": ("y", y_coords), "x": ("x", x_coords)},
    )
    dep_da.raster.set_crs(dem_crs.to_wkt())

    zs_block: Any
    if isinstance(zsmax, xu.UgridDataArray):
        zs_block = zsmax.ugrid.rasterize_like(dep_da)
    else:
        zs_block = zsmax.raster.reproject_like(dep_da, method=reproj_method)
    zs_block = zs_block.raster.mask_nodata()
    return np.asarray(zs_block.values - dep_block, dtype=np.float32)


def _write_floodmap_cog(
    zsmax: Any,
    dem_path: Path,
    index_path: Path | None,
    output_path: Path,
    hmin: float,
    reproj_method: str,
    nrmax: int,
    baseline: Any,
    dem_offset: float,
) -> None:
    """Write a flood-depth COG at the full DEM resolution.

    Reads the DEM (and optional index COG) at full resolution---avoiding
    the upstream ``overview_level=0`` bug---and writes block-by-block.

    When *baseline* is given (the per-cell minimum water surface), pixels
    already under water at the model's driest moment are dropped, so the
    result is inundation rather than inundation plus the permanently wet
    sea.

    *dem_offset* is added to the DEM as it is read. The model bed carries any
    vertical offset its elevation dataset was merged with, but the raster on
    disk does not, and subtracting an unshifted DEM from a shifted water
    surface biases every depth by exactly that offset.
    """
    import rasterio
    from rasterio.windows import Window

    zsmax, zs_flat = _reduce_zsmax(zsmax)
    base_flat = None if baseline is None else _reduce_zsmax(baseline)[1]

    with rasterio.open(dem_path) as src:
        dem_crs: Any = src.crs
        dem_transform: Any = src.transform
        n1: int
        m1: int
        n1, m1 = src.shape

        idx_src: Any = None
        idx_nodata: int = 2147483647
        if index_path is not None:
            idx_src = rasterio.open(index_path)
            idx_nodata = int(idx_src.nodata or 2147483647)

        try:
            profile: dict[str, Any] = {
                "driver": "GTiff",
                "width": m1,
                "height": n1,
                "count": 1,
                "dtype": "float32",
                "crs": dem_crs,
                "transform": dem_transform,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "compress": "deflate",
                "predictor": 2,
                "nodata": float("nan"),
                "BIGTIFF": "YES",
            }

            with rasterio.open(output_path, "w", **profile) as dst:
                nrcb = nrmax
                nrbn = int(np.ceil(n1 / nrcb))
                nrbm = int(np.ceil(m1 / nrcb))

                for jj in range(nrbn):
                    bn0 = jj * nrcb
                    bn1_val = min(bn0 + nrcb, n1)
                    for ii in range(nrbm):
                        bm0 = ii * nrcb
                        bm1_val = min(bm0 + nrcb, m1)

                        window = Window(bm0, bn0, bm1_val - bm0, bn1_val - bn0)  # pyright: ignore[reportCallIssue]
                        dep_block: Any = src.read(1, window=window).astype("float32")
                        if dem_offset:
                            dep_block += dem_offset

                        if np.all(np.isnan(dep_block)):
                            continue

                        if idx_src is not None:
                            h = _depth_from_index(idx_src, zs_flat, dep_block, window, idx_nodata)
                        else:
                            h = _depth_from_rasterize(
                                zsmax,
                                dep_block,
                                dem_crs,
                                dem_transform,
                                bm0,
                                bm1_val,
                                bn0,
                                bn1_val,
                                reproj_method,
                            )

                        h[~np.isfinite(h)] = np.nan
                        h[h <= hmin] = np.nan
                        if base_flat is not None and idx_src is not None:
                            # Permanently wet: under water even at the minimum.
                            wet = _depth_from_index(
                                idx_src, base_flat, dep_block, window, idx_nodata
                            )
                            h[np.isfinite(wet) & (wet > 0.0)] = np.nan
                        dst.write(h[np.newaxis, :, :], window=window)
        finally:
            if idx_src is not None:
                idx_src.close()


def create_flood_depth_map(
    model_root: Path | str,
    dem_path: Path | str,
    output_path: Path | str | None = None,
    *,
    index_path: Path | str | None = None,
    create_index: bool = True,
    hmin: float = 0.05,
    land_only: bool = True,
    dem_offset: float = 0.0,
    reproj_method: str = "nearest",
    nrmax: int = 2000,
    model: SfincsModel | None = None,
    log: Any = None,
) -> Path:
    """Create a downscaled flood depth map from SFINCS output.

    Reads the maximum water surface elevation (``zsmax``) from the SFINCS
    map output, optionally builds an index COG that maps DEM pixels to
    SFINCS grid cells, then downscales onto a high-resolution DEM to
    produce a Cloud Optimized GeoTIFF of maximum flood depth.

    Parameters
    ----------
    model_root : Path or str
        Path to the SFINCS model directory (must contain ``sfincs_map.nc``).
    dem_path : Path or str
        Path to a high-resolution DEM GeoTIFF covering the model domain.
    output_path : Path or str, optional
        Output flood depth COG path.  Defaults to
        ``<model_root>/floodmap_hmax.tif``.
    index_path : Path or str, optional
        Path for the index COG (DEM pixels -> SFINCS cell mapping).
        Defaults to ``<model_root>/floodmap_index.tif``.
    create_index : bool
        If True (default), (re)generate the index COG via
        :func:`make_index_cog`.  The index speeds up the downscaling
        significantly for large DEMs.
    hmin : float
        Minimum flood depth (m) to classify a pixel as flooded.
    land_only : bool
        Drop pixels the model shows as permanently wet, so the map is
        inundation rather than inundation plus the sea.
    dem_offset : float
        Vertical offset (m) added to *dem_path* to put it on the model's
        datum. Non-zero when the dataset this DEM came from was merged with
        an ``offset``; see
        :attr:`~coastal_calibration.config.create_schema.ElevationDataset.offset`.
    reproj_method : str
        Reprojection method (``"nearest"`` or ``"bilinear"``).
    nrmax : int
        Maximum cells per processing block (controls peak memory).
    model : SfincsModel, optional
        An already-loaded :class:`SfincsModel` instance.  When provided,
        ``model_root`` is still used to resolve default output paths but
        the model is **not** re-read from disk.
    log : callable, optional
        Logging callback accepting a single message; falls back to the module
        logger when *None*.

    Returns
    -------
    Path
        Path to the generated flood depth COG.

    Raises
    ------
    FileNotFoundError
        If the DEM or ``sfincs_map.nc`` cannot be found.
    FloodmapInputError
        If ``zsmax`` is not present in the SFINCS map output.
    """
    # registers the ``DataArray.raster`` accessor used below
    import hydromt  # noqa: F401  # pyright: ignore[reportUnusedImport]

    # -- Ensure patches are applied before any hydromt-sfincs call --
    from coastal_calibration.logging import suppress_hydromt_output
    from coastal_calibration.sfincs._hydromt_compat import apply_all_patches

    apply_all_patches()

    # Import *after* patches so local references pick up the fixed versions.
    with suppress_hydromt_output():
        from hydromt_sfincs.workflows.downscaling import make_index_cog

    model_root = Path(model_root)
    dem_path = Path(dem_path)

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    map_file = model_root / "sfincs_map.nc"
    if not map_file.exists():
        raise FileNotFoundError(f"SFINCS map output not found: {map_file}")

    output_path = model_root / "floodmap_hmax.tif" if output_path is None else Path(output_path)
    index_path = model_root / "floodmap_index.tif" if index_path is None else Path(index_path)

    def _info(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            _log.info(msg)

    # -- Load model and read output ---------------------------------
    if model is None:
        with suppress_hydromt_output():
            from hydromt_sfincs import SfincsModel as _Sfincs

            # Use "r+" (same as the pipeline) so all components are writable
            # and the quadtree grid loads correctly.
            model = _Sfincs(root=str(model_root), mode="r+")
            model.read()

    with suppress_hydromt_output():
        model.output.read()

    if "zsmax" not in model.output.data:
        msg = (
            "Variable 'zsmax' not found in SFINCS map output. "
            "Ensure SFINCS was configured to write zsmax (storzsmax = 1 in sfincs.inp)."
        )
        raise FloodmapInputError(msg)
    zsmax = model.output.data["zsmax"]
    _info(f"Loaded zsmax from {map_file}")

    # -- (Re)create index COG ---------------------------------------
    # Always regenerate so the index stays consistent with the
    # current grid and the patched ``get_indices_at_points``.
    if create_index:
        _info(f"Creating index COG: {index_path}")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with suppress_hydromt_output():
            make_index_cog(
                model=model,
                indices_fn=str(index_path),
                topobathy_fn=str(dem_path),
                nrmax=nrmax,
            )
        _assert_index_hits_the_model(index_path, dem_path)
        _ensure_overviews(index_path, _info)
        _info(f"Index COG created ({index_path.stat().st_size / 1e6:.1f} MB)")

    # -- Downscale --------------------------------------------------
    # Read the DEM and index at full resolution.  The upstream
    # ``downscale_floodmap`` defaults to ``overview_level=0`` when
    # reading rasters from disk, which silently halves the resolution
    # and can mismatch with the index.  By loading the rasters into
    # memory first and passing them as DataArrays we bypass that bug.
    _info("Downscaling flood depth map")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _write_floodmap_cog(
        zsmax=_blank_dry_cells(zsmax, model.output.data, _info),
        dem_path=dem_path,
        index_path=index_path if index_path.exists() else None,
        output_path=output_path,
        hmin=hmin,
        reproj_method=reproj_method,
        nrmax=nrmax,
        dem_offset=dem_offset,
        baseline=_baseline_water_surface(model.output.data) if land_only else None,
    )

    _ensure_overviews(output_path, _info)

    size_mb = output_path.stat().st_size / 1e6
    _info(f"Flood depth map written: {output_path} ({size_mb:.1f} MB)")

    return output_path
