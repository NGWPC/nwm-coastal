"""GDAL CLI helpers for VRT creation and clipping.

Thin wrappers around ``gdalbuildvrt`` and ``gdalwarp`` that all
data-fetcher modules share.  Requires ``libgdal-core`` (installed
automatically by the conda/pixi environment).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import xarray as xr

__all__ = ["build_vrt", "clip_to_aoi", "compute_aoi_coverage"]


def _path_str(path: Path | str) -> str:
    """Convert a path to a string suitable for GDAL commands.

    Handles both :class:`~pathlib.Path` objects and plain strings
    (e.g. GDAL ``/vsicurl/`` virtual filesystem paths).
    """
    if isinstance(path, str):
        return path
    return path.resolve().as_posix()


def build_vrt(vrt_path: Path, tiff_files: Sequence[Path | str]) -> None:
    """Create a VRT from a list of GeoTIFF tiles.

    Parameters
    ----------
    vrt_path
        Path to save the output VRT file.
    tiff_files
        List of GeoTIFF file paths to include in the VRT.

    Raises
    ------
    ImportError
        If ``gdalbuildvrt`` is not on ``$PATH``.
    RuntimeError
        If the ``gdalbuildvrt`` command fails.
    """
    if shutil.which("gdalbuildvrt") is None:
        raise ImportError(
            "GDAL (libgdal-core) is required for VRT creation. Install via conda/pixi."
        )

    command = [
        "gdalbuildvrt",
        "-overwrite",
        _path_str(vrt_path),
        *[_path_str(f) for f in tiff_files],
    ]
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        msg = f"gdalbuildvrt failed:\n{e.stderr.strip()}"
        raise RuntimeError(msg) from e


def clip_to_aoi(
    input_path: Path | str,
    cutline_path: Path,
    output_path: Path,
    *,
    nodata: str = "nan",
    output_type: str = "Float32",
    compress: str = "deflate",
    target_extent: tuple[float, float, float, float] | None = None,
    target_extent_srs: str | None = None,
) -> None:
    """Clip a raster to an AOI polygon using ``gdalwarp``.

    Parameters
    ----------
    input_path
        Input raster or VRT path.  GDAL virtual filesystem paths
        (``/vsicurl/…``, ``/vsis3/…``) are supported.
    cutline_path
        Path to AOI polygon file (GeoJSON, Shapefile, …) used as
        the ``-cutline`` argument.  Its CRS must match the raster.
    output_path
        Output GeoTIFF path.
    nodata
        Destination nodata value (default ``"nan"``).
    output_type
        Output pixel type (default ``"Float32"``).
    compress
        GeoTIFF compression (default ``"deflate"``).
    target_extent
        ``(xmin, ymin, xmax, ymax)`` passed to ``gdalwarp -te``.
        Limits which tiles are read — especially useful with remote
        VRTs so GDAL skips tiles outside the area of interest.
    target_extent_srs
        SRS for interpreting *target_extent* (e.g. ``"EPSG:4326"``).
        Required when *target_extent* is in a CRS different from
        the output CRS.

    Raises
    ------
    ImportError
        If ``gdalwarp`` is not on ``$PATH``.
    RuntimeError
        If the ``gdalwarp`` command fails.
    """
    if shutil.which("gdalwarp") is None:
        raise ImportError("GDAL (libgdal-core) is required for clipping. Install via conda/pixi.")

    input_str = _path_str(input_path)

    # HTTP and caching settings for remote /vsicurl/ sources.
    # Harmless for local-only operations; critical for VRTs whose
    # tiles reference remote URLs.
    #
    # CPL_VSIL_CURL_CHUNK_SIZE: bytes per HTTP range request.
    #   Default 16 KB → ~125 000 requests for 2 GB of tiles.
    #   10 MB → ~200 requests — 600x fewer round-trips.
    # VSI_CACHE_SIZE: in-memory cache for /vsicurl/ reads (100 MB).
    # GDAL_CACHEMAX: raster block cache in MB (512 MB).
    # GDAL_HTTP_MERGE_CONSECUTIVE_RANGES: merge adjacent byte-range
    #   requests into a single HTTP call.
    command = [
        "gdalwarp",
        "--config",
        "GDAL_HTTP_MAX_RETRY",
        "5",
        "--config",
        "GDAL_HTTP_RETRY_DELAY",
        "2",
        "--config",
        "GDAL_DISABLE_READDIR_ON_OPEN",
        "EMPTY_DIR",
        "--config",
        "VSI_CACHE",
        "TRUE",
        "--config",
        "CPL_VSIL_CURL_CHUNK_SIZE",
        "10485760",
        "--config",
        "VSI_CACHE_SIZE",
        "100000000",
        "--config",
        "GDAL_CACHEMAX",
        "512",
        "--config",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES",
        "YES",
    ]

    # Target extent so GDAL only reads overlapping tiles.
    if target_extent is not None:
        command += ["-te", *(str(v) for v in target_extent)]
        if target_extent_srs is not None:
            command += ["-te_srs", target_extent_srs]

    command += [
        "-cutline",
        _path_str(cutline_path),
        "-crop_to_cutline",
        "-dstnodata",
        nodata,
        "-ot",
        output_type,
        "-co",
        f"COMPRESS={compress.upper()}",
        "-multi",
        "-wo",
        "NUM_THREADS=ALL_CPUS",
        "-wm",
        "500",
        "-overwrite",
        input_str,
        _path_str(output_path),
    ]
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        msg = f"gdalwarp failed:\n{e.stderr.strip()}"
        raise RuntimeError(msg) from e


def compute_aoi_coverage(
    raster_path: Path,
    zone_path: Path | str,
) -> float:
    """Compute the percentage of valid (non-nodata) pixels inside zone polygons.

    If the zone CRS differs from the raster CRS, zones are reprojected
    automatically.

    Uses ``rasterio`` block-based reading with
    ``rasterio.features.geometry_mask`` so the full raster and mask
    never live in memory at once (each 4096x4096 block is processed
    independently).

    Parameters
    ----------
    raster_path
        Input GeoTIFF with a nodata value set.
    zone_path
        OGR-readable vector file (GeoJSON, Shapefile, …).

    Returns
    -------
    float
        Percentage of valid pixels inside the zone (0--100).
    """
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window

    block_size = 4096
    total_inside = 0
    valid_inside = 0

    with rasterio.open(raster_path) as src:
        zone_gdf = gpd.read_file(zone_path)
        if zone_gdf.crs is not None and not zone_gdf.crs.equals(src.crs):
            zone_gdf = zone_gdf.to_crs(src.crs)
        geometries = zone_gdf.geometry.tolist()
        nodata = src.nodata
        for row_off in range(0, src.height, block_size):
            for col_off in range(0, src.width, block_size):
                h = min(block_size, src.height - row_off)
                w = min(block_size, src.width - col_off)
                win = Window(col_off, row_off, w, h)  # pyright: ignore[reportCallIssue]
                win_transform = src.window_transform(win)

                inside = geometry_mask(
                    geometries,
                    out_shape=(h, w),
                    transform=win_transform,
                    invert=True,
                )
                n_inside = int(inside.sum())
                total_inside += n_inside

                if n_inside > 0:
                    data = src.read(1, window=win)
                    if nodata is not None and np.isnan(nodata):
                        valid = ~np.isnan(data) & inside
                    elif nodata is not None:
                        valid = (data != nodata) & inside
                    else:
                        valid = inside
                    valid_inside += int(valid.sum())

    if total_inside == 0:
        return 0.0
    return valid_inside / total_inside * 100


# ---------------------------------------------------------------------------
# Raster clipping and reprojection (was utils/raster.py)
# ---------------------------------------------------------------------------


def clip_and_reproject(
    data: xr.Dataset | xr.DataArray,
    dst_bounds: tuple[float, float, float, float],
    dst_crs: object,
    dst_res: float,
    *,
    fill_value: float = 0.0,
    buffer: float = 10_000.0,
    reproject_method: str = "nearest_index",
) -> xr.Dataset | xr.DataArray:
    """Clip a raster in its source CRS, then reproject to a constrained grid.

    Steps:

    1. Transform *dst_bounds* from *dst_crs* into the data's native CRS.
    2. Clip the data in its native CRS with a generous buffer.
    3. Reproject to *dst_crs* with an output grid exactly covering
       *dst_bounds* ± *buffer* at *dst_res* resolution.

    This avoids the grid-inflation problem where
    ``rasterio.warp.calculate_default_transform`` produces an output
    grid spanning the entire source extent (e.g. CONUS-scale for NWM
    LCC → UTM reprojections).

    Parameters
    ----------
    data
        Input :class:`xarray.Dataset` or :class:`xarray.DataArray`
        with a CRS set (via the ``raster`` accessor from hydromt or
        rioxarray).
    dst_bounds
        ``(xmin, ymin, xmax, ymax)`` target bounding box in *dst_crs*
        units (e.g. metres for a UTM zone).
    dst_crs
        Target CRS (anything accepted by ``rasterio``/``pyproj``).
    dst_res
        Target resolution in *dst_crs* units.
    fill_value
        Value for pixels with no source data after reprojection.
    buffer
        Extra margin in *dst_crs* units added around *dst_bounds*
        for the output grid.  Default 10 km.
    reproject_method
        Resampling method passed to ``raster.reproject()``.
        Default ``"nearest_index"`` (KDTree-based, good for meteo).

    Returns
    -------
    Same type as *data*
        Clipped and reprojected raster.  Spatial dimensions are
        **not** renamed (caller handles SFINCS ``x``/``y`` convention).
    """
    import rasterio.warp
    from affine import Affine

    if dst_res <= 0:
        msg = f"dst_res must be positive, got {dst_res}"
        raise ValueError(msg)

    xmin_dst, ymin_dst, xmax_dst, ymax_dst = dst_bounds
    if xmax_dst <= xmin_dst or ymax_dst <= ymin_dst:
        msg = f"Invalid dst_bounds (need xmax > xmin, ymax > ymin): {dst_bounds}"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # 1. Clip in source CRS
    # ------------------------------------------------------------------
    src_crs = data.raster.crs

    # Buffer the destination bounds first, then transform to source CRS.
    # This avoids mixing units between dst_crs and src_crs.
    buffered_dst = (
        xmin_dst - buffer,
        ymin_dst - buffer,
        xmax_dst + buffer,
        ymax_dst + buffer,
    )
    src_bounds = rasterio.warp.transform_bounds(dst_crs, src_crs, *buffered_dst)

    # Add extra padding in source-CRS units (30 cells) to ensure the
    # clip region fully covers the buffered destination after reprojection.
    src_res = abs(float(data.raster.res[0]))
    src_pad = src_res * 30
    src_clip = (
        src_bounds[0] - src_pad,
        src_bounds[1] - src_pad,
        src_bounds[2] + src_pad,
        src_bounds[3] + src_pad,
    )
    data = data.raster.clip_bbox(src_clip)

    # ------------------------------------------------------------------
    # 2. Reproject with a constrained output grid
    # ------------------------------------------------------------------
    xmin = buffered_dst[0]
    ymin = buffered_dst[1]
    xmax = buffered_dst[2]
    ymax = buffered_dst[3]

    dst_width = max(1, int(np.ceil((xmax - xmin) / dst_res)))
    dst_height = max(1, int(np.ceil((ymax - ymin) / dst_res)))
    dst_transform = Affine(dst_res, 0.0, xmin, 0.0, -dst_res, ymax)

    result = data.raster.reproject(
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        dst_width=dst_width,
        dst_height=dst_height,
        method=reproject_method,
    ).fillna(fill_value)

    # Ensure spatial coordinates are monotonic after reprojection.
    # The ``nearest_index`` method (KDTree-based) can produce slightly
    # non-monotonic coordinates from floating-point drift.
    y_dim, x_dim = result.raster.dims
    if result[x_dim].size > 1 and not result.indexes[x_dim].is_monotonic_increasing:
        result = result.sortby(x_dim)
    if result[y_dim].size > 1 and not (
        result.indexes[y_dim].is_monotonic_increasing
        or result.indexes[y_dim].is_monotonic_decreasing
    ):
        result = result.sortby(y_dim)

    return result
