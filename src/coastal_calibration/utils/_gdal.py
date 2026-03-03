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


def build_vrt(vrt_path: Path, tiff_files: list[Path | str]) -> None:
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
    input_path: Path,
    cutline_path: Path,
    output_path: Path,
    *,
    nodata: str = "nan",
    output_type: str = "Float32",
    compress: str = "deflate",
) -> None:
    """Clip a raster to an AOI polygon using ``gdalwarp``.

    Parameters
    ----------
    input_path
        Input raster or VRT path.
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

    Raises
    ------
    ImportError
        If ``gdalwarp`` is not on ``$PATH``.
    RuntimeError
        If the ``gdalwarp`` command fails.
    """
    if shutil.which("gdalwarp") is None:
        raise ImportError("GDAL (libgdal-core) is required for clipping. Install via conda/pixi.")

    command = [
        "gdalwarp",
        "-cutline",
        _path_str(cutline_path),
        "-crop_to_cutline",
        "-dstnodata",
        nodata,
        "-ot",
        output_type,
        "-co",
        f"COMPRESS={compress.upper()}",
        "-overwrite",
        _path_str(input_path),
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

    The raster and zone polygons **must** share the same CRS.

    Uses ``rasterio`` block-based reading with
    ``rasterio.features.geometry_mask`` so the full raster and mask
    never live in memory at once (each 4096x4096 block is processed
    independently).

    Parameters
    ----------
    raster_path
        Input GeoTIFF with a nodata value set.
    zone_path
        OGR-readable vector file whose CRS matches *raster_path*.

    Returns
    -------
    float
        Percentage of valid pixels inside the zone (0--100).
    """
    import fiona
    import numpy as np
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window

    with fiona.open(zone_path) as src:
        geometries = [feature["geometry"] for feature in src]

    block_size = 4096
    total_inside = 0
    valid_inside = 0

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        for row_off in range(0, src.height, block_size):
            for col_off in range(0, src.width, block_size):
                h = min(block_size, src.height - row_off)
                w = min(block_size, src.width - col_off)
                win = Window(col_off, row_off, w, h)
                win_transform = src.window_transform(win)

                # geometry_mask returns True *outside* geometries.
                outside = geometry_mask(
                    geometries,
                    out_shape=(h, w),
                    transform=win_transform,
                )
                inside = ~outside
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
