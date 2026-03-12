"""Flood depth map generation from SFINCS output.

Downscales coarse-grid water surface elevations onto a high-resolution
DEM to produce a Cloud Optimized GeoTIFF (COG) of maximum flood depth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

__all__ = ["create_flood_depth_map"]

_log = logging.getLogger(__name__)


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
        min_dim = min(src.width, src.height)
        levels = []
        factor = 2
        while min_dim // factor >= 256:
            levels.append(factor)
            factor *= 2
        if not levels:
            levels = [2]
        log(f"Building {len(levels)} overview levels: {levels}")
        src.build_overviews(levels, Resampling.average)
        src.update_tags(ns="rio_overview", resampling="average")


def _reduce_zsmax(zsmax: Any) -> Any:
    """Reduce zsmax to max over the time dimension and return a flat array.

    Returns ``(zsmax_reduced, zs_flat)`` where *zsmax_reduced* is the
    time-collapsed DataArray and *zs_flat* is a 1-D float32 numpy array
    suitable for index-based lookup.
    """
    import numpy as np
    import xugrid as xu

    if isinstance(zsmax, xu.UgridDataArray):
        timedim = set(zsmax.dims) - set(zsmax.ugrid.grid.dims)
    else:
        timedim = set(zsmax.dims) - set(zsmax.raster.dims)
    if timedim:
        zsmax = zsmax.max(timedim)

    if isinstance(zsmax, xu.UgridDataArray):
        zs_flat = zsmax.to_numpy().astype("float32")
    else:
        zs_flat = zsmax.values.flatten().astype("float32")
    zs_flat[~np.isfinite(zs_flat)] = np.nan
    return zsmax, zs_flat


def _depth_from_index(
    idx_src: Any, zs_flat: Any, dep_block: Any, window: Any, idx_nodata: int
) -> Any:
    """Compute flood depth for a DEM block using an index COG."""
    import numpy as np

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
    import numpy as np
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

    if isinstance(zsmax, xu.UgridDataArray):
        zs_block = zsmax.ugrid.rasterize_like(dep_da)
    else:
        zs_block = zsmax.raster.reproject_like(dep_da, method=reproj_method)
    zs_block = zs_block.raster.mask_nodata()
    return (zs_block.values - dep_block).astype("float32")


def _write_floodmap_cog(
    zsmax: Any,
    dem_path: Path,
    index_path: Path | None,
    output_path: Path,
    hmin: float,
    reproj_method: str,
    nrmax: int,
    log_fn: Any,
) -> None:
    """Write a flood-depth COG at the full DEM resolution.

    Reads the DEM (and optional index COG) at full resolution—avoiding
    the upstream ``overview_level=0`` bug—and writes block-by-block.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    zsmax, zs_flat = _reduce_zsmax(zsmax)

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_transform = src.transform
        n1, m1 = src.shape

        idx_src = None
        idx_nodata: int = 2147483647
        if index_path is not None:
            idx_src = rasterio.open(index_path)
            idx_nodata = int(idx_src.nodata or 2147483647)

        try:
            profile = {
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
                    bn1 = min(bn0 + nrcb, n1)
                    for ii in range(nrbm):
                        bm0 = ii * nrcb
                        bm1 = min(bm0 + nrcb, m1)

                        window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)  # type: ignore[too-many-positional-arguments]
                        dep_block = src.read(1, window=window).astype("float32")

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
                                bm1,
                                bn0,
                                bn1,
                                reproj_method,
                            )

                        h[~np.isfinite(h)] = np.nan
                        h[h <= hmin] = np.nan
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
        Path for the index COG (DEM pixels → SFINCS cell mapping).
        Defaults to ``<model_root>/floodmap_index.tif``.
    create_index : bool
        If True (default), (re)generate the index COG via
        :func:`make_index_cog`.  The index speeds up the downscaling
        significantly for large DEMs.
    hmin : float
        Minimum flood depth (m) to classify a pixel as flooded.
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
    KeyError
        If ``zsmax`` is not present in the SFINCS map output.
    """
    # ── Ensure patches are applied before any hydromt-sfincs call ──
    from coastal_calibration.stages._hydromt_compat import apply_all_patches

    apply_all_patches()

    # Import *after* patches so local references pick up the fixed versions.
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

    # ── Load model and read output ──────────────────────────────
    if model is None:
        from hydromt_sfincs import SfincsModel as _Sfincs

        # Use "r+" (same as the pipeline) so all components are writable
        # and the quadtree grid loads correctly.
        model = _Sfincs(root=str(model_root), mode="r+")
        model.read()

    model.output.read()

    if "zsmax" not in model.output.data:
        raise KeyError(
            "Variable 'zsmax' not found in SFINCS map output. "
            "Ensure SFINCS was configured to write zsmax (storzsmax = 1 in sfincs.inp)."
        )
    zsmax = model.output.data["zsmax"]
    _info(f"Loaded zsmax from {map_file}")

    # ── (Re)create index COG ────────────────────────────────────
    # Always regenerate so the index stays consistent with the
    # current grid and the patched ``get_indices_at_points``.
    if create_index:
        _info(f"Creating index COG: {index_path}")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        make_index_cog(
            model=model,
            indices_fn=str(index_path),
            topobathy_fn=str(dem_path),
            nrmax=nrmax,
        )
        _ensure_overviews(index_path, _info)
        _info(f"Index COG created ({index_path.stat().st_size / 1e6:.1f} MB)")

    # ── Downscale ───────────────────────────────────────────────
    # Read the DEM and index at full resolution.  The upstream
    # ``downscale_floodmap`` defaults to ``overview_level=0`` when
    # reading rasters from disk, which silently halves the resolution
    # and can mismatch with the index.  By loading the rasters into
    # memory first and passing them as DataArrays we bypass that bug.
    _info("Downscaling flood depth map")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _write_floodmap_cog(
        zsmax=zsmax,
        dem_path=dem_path,
        index_path=index_path if index_path.exists() else None,
        output_path=output_path,
        hmin=hmin,
        reproj_method=reproj_method,
        nrmax=nrmax,
        log_fn=_info,
    )

    _ensure_overviews(output_path, _info)

    size_mb = output_path.stat().st_size / 1e6
    _info(f"Flood depth map written: {output_path} ({size_mb:.1f} MB)")

    return output_path
