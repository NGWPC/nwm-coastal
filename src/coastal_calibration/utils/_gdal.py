"""GDAL CLI helpers for VRT creation and clipping.

Thin wrappers around ``gdalbuildvrt`` and ``gdalwarp`` that all
data-fetcher modules share.  Requires ``libgdal-core`` (installed
automatically by the conda/pixi environment).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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
                win = Window(col_off, row_off, w, h)  # ty: ignore[too-many-positional-arguments]
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
