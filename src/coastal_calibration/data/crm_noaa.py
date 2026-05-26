"""Fetch NOAA Coastal Relief Model (CRM) topobathy via NCEI ArcGIS REST.

The NCEI CRM mosaic is exposed as an ArcGIS ImageServer at
https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/CRM_mosaic
.
Calling ``exportImage`` with a bounding box returns a GeoTIFF clipped
to that bbox at the requested pixel resolution.  The service stitches
the underlying regional CRM volumes server-side, so callers do not
need to worry about volume seams or selection.

At runtime, :func:`fetch_crm` computes the pixel dimensions for the
buffered AOI bbox at a target ground resolution (default 90 m, the
native CRM grid).  Small AOIs are fetched in a single REST call.
Large AOIs are decomposed into a grid of tiles, fetched, and
mosaicked with ``gdalbuildvrt`` + ``gdalwarp`` clipped to the AOI
cutline.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from coastal_calibration.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_EXPORT_URL = (
    "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/CRM_mosaic/ImageServer/exportImage"
)

#: Per-tile pixel budget.  The service caps each ``exportImage`` request
#: at 20000x20000, so a value well below the per-dim limit keeps
#: individual responses small and lets a partial fetch resume on retry.
_PIXEL_MAX = 16_000_000  # 4000 x 4000 in the square case

#: Hard service-side per-dimension limit (per ``?f=pjson`` introspection).
_HARD_PIXEL_LIMIT = 20000


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    r_earth = 6_371_000.0
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * r_earth * math.asin(math.sqrt(a))


def _bbox_pixels(bbox: tuple[float, float, float, float], res_m: float) -> tuple[int, int]:
    """Pixel width/height covering *bbox* at *res_m* meters/pixel."""
    w, s, e, n = bbox
    x_dist = _haversine_m(s, w, s, e)
    y_dist = _haversine_m(s, w, n, w)
    return max(1, math.ceil(x_dist / res_m)), max(1, math.ceil(y_dist / res_m))


def _decompose_bbox(
    bbox: tuple[float, float, float, float],
    res_m: float,
    pixel_max: int,
) -> list[tuple[tuple[float, float, float, float], int, int]]:
    """Split *bbox* into tiles that stay under *pixel_max* and the service limit.

    Returns a list of ``(sub_bbox, sub_w, sub_h)`` triples.  ``sub_w`` and
    ``sub_h`` are the pixel dimensions to request for each sub-bbox.
    """
    width, height = _bbox_pixels(bbox, res_m)
    fits = (
        width * height <= pixel_max and width <= _HARD_PIXEL_LIMIT and height <= _HARD_PIXEL_LIMIT
    )
    if fits:
        return [(bbox, width, height)]

    w, s, e, n = bbox
    aspect = width / height
    n_boxes = math.ceil((width * height) / pixel_max)
    nx = max(1, math.ceil(math.sqrt(n_boxes * aspect)))
    ny = max(1, math.ceil(n_boxes / nx))
    # Tighten further when either dim would still blow past the service limit.
    nx = max(nx, math.ceil(width / _HARD_PIXEL_LIMIT))
    ny = max(ny, math.ceil(height / _HARD_PIXEL_LIMIT))
    dx = (e - w) / nx
    dy = (n - s) / ny
    sub_w = max(1, math.ceil(width / nx))
    sub_h = max(1, math.ceil(height / ny))
    tiles: list[tuple[tuple[float, float, float, float], int, int]] = []
    for i in range(nx):
        for j in range(ny):
            sub_bbox = (
                w + i * dx,
                s + j * dy,
                w + (i + 1) * dx,
                s + (j + 1) * dy,
            )
            tiles.append((sub_bbox, sub_w, sub_h))
    return tiles


# ------------------------------------------------------------------
# REST + catalog helpers
# ------------------------------------------------------------------


def _fetch_tile(
    sub_bbox: tuple[float, float, float, float],
    sub_w: int,
    sub_h: int,
    output_path: Path,
    log: Callable[[str], None],
) -> None:
    """Fetch one ``exportImage`` tile and write it to *output_path*."""
    import urllib.request

    params = {
        "bbox": ",".join(f"{c:.6f}" for c in sub_bbox),
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{sub_w},{sub_h}",
        "format": "tiff",
        "pixelType": "F32",
        # Without an explicit ``noData`` parameter the service returns
        # ``0.0`` (a valid sea-level elevation) for cells outside CRM
        # coverage and leaves the GeoTIFF's nodata header unset, which
        # would cause HydroMT-SFINCS to treat those cells as real land.
        # ``noData=NaN`` propagates as both the nodata value and the
        # GeoTIFF header, matching the other fetchers' convention.
        "noData": "NaN",
        "interpolation": "RSP_BilinearInterpolation",
        "compression": "LZ77",
        "renderingRule": '{"rasterFunction":"none"}',
        "f": "image",
    }
    url = f"{_EXPORT_URL}?{urlencode(params)}"
    log(f"  exportImage bbox={tuple(round(c, 3) for c in sub_bbox)}, size={sub_w}x{sub_h}")
    with urllib.request.urlopen(url, timeout=180) as r, output_path.open("wb") as f:
        f.write(r.read())


def _write_catalog(
    catalog_path: Path,
    tif_name: str,
    catalog_name: str,
) -> None:
    """Write a minimal HydroMT data-catalog YAML next to the GeoTIFF."""
    import yaml

    catalog: dict[str, Any] = {
        "meta": {
            "version": "v1.0.0",
            "name": catalog_name,
            "hydromt_version": ">1.0a,<2",
        },
        catalog_name: {
            "data_type": "RasterDataset",
            "uri": tif_name,
            "driver": {"name": "rasterio"},
            "metadata": {
                "category": "topography",
                "crs": 4326,
            },
            "data_adapter": {
                "rename": {"elevation": "elevtn"},
            },
        },
    }
    catalog_path.write_text(yaml.dump(catalog, default_flow_style=False, sort_keys=False))


# ------------------------------------------------------------------
# Output assembly and validation helpers
# ------------------------------------------------------------------


def _aoi_bbox(
    aoi: Path,
    buffer_deg: float,
) -> tuple[Any, tuple[float, float, float, float]]:
    """Read *aoi*, project to EPSG:4326, and return a buffered bbox tuple."""
    import geopandas as gpd

    aoi_gdf = gpd.read_file(aoi)
    if aoi_gdf.crs is None:
        raise ValueError(f"AOI file {aoi} has no CRS")
    aoi_4326 = aoi_gdf.to_crs(epsg=4326)
    bounds = aoi_4326.total_bounds
    bbox = (
        float(bounds[0] - buffer_deg),
        float(bounds[1] - buffer_deg),
        float(bounds[2] + buffer_deg),
        float(bounds[3] + buffer_deg),
    )
    return aoi_4326, bbox


def _assemble_output(
    tiles: list[tuple[tuple[float, float, float, float], int, int]],
    bbox: tuple[float, float, float, float],
    aoi_4326: Any,
    sub_dir: Path,
    geotiff_path: Path,
    log: Callable[[str], None],
) -> None:
    """Fetch every tile and merge them into *geotiff_path*."""
    import shutil

    from coastal_calibration.data.transformation import build_vrt, clip_to_aoi

    temp_tifs: list[Path] = []
    for i, (sub_bbox, sub_w, sub_h) in enumerate(tiles):
        temp_path = sub_dir / f"tile_{i:03d}.tif"
        _fetch_tile(sub_bbox, sub_w, sub_h, temp_path, log)
        temp_tifs.append(temp_path)

    if len(temp_tifs) == 1:
        shutil.move(str(temp_tifs[0]), geotiff_path)
        return

    vrt_path = sub_dir / "crm_mosaic.vrt"
    build_vrt(vrt_path, temp_tifs)
    cutline_path = sub_dir / "cutline.geojson"
    aoi_4326.to_file(cutline_path, driver="GeoJSON")
    log(f"Mosaicking {len(temp_tifs)} tiles and clipping to AOI")
    clip_to_aoi(
        vrt_path,
        cutline_path,
        geotiff_path,
        nodata="nan",
        output_type="Float32",
        target_extent=bbox,
        target_extent_srs="EPSG:4326",
    )


def _valid_data_fraction(geotiff_path: Path) -> float:
    """Block-windowed scan of *geotiff_path* returning the valid-cell fraction.

    Reads the raster one internal block at a time so peak memory stays
    O(block size) instead of O(full raster), which matters for
    continental-scale AOIs where the mosaic can easily exceed 1 GB.
    """
    import numpy as np
    import rasterio

    n_total = 0
    n_valid = 0
    with rasterio.open(geotiff_path) as src:
        nodata = src.nodata
        for _ji, window in src.block_windows(1):
            block = src.read(1, window=window)
            n_total += block.size
            valid_mask = np.isfinite(block)
            if nodata is not None and not np.isnan(nodata):
                valid_mask &= block != nodata
            n_valid += int(valid_mask.sum())
    return (n_valid / n_total) if n_total else 0.0


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------


def fetch_crm(
    aoi: Path | str,
    output_dir: Path | str,
    *,
    resolution_m: float = 90.0,
    buffer_deg: float = 0.1,
    catalog_name: str = "noaa_crm",
    log: Callable[[str], None] | None = None,
) -> tuple[Path, Path, str]:
    """Fetch NOAA CRM topobathy for *aoi* from the NCEI ImageServer.

    Computes the pixel size required to cover the buffered AOI at
    *resolution_m* meters/pixel, then either issues a single REST call
    (small AOIs) or decomposes the bbox into tiles, fetches each, and
    mosaics them with ``gdalbuildvrt`` + ``gdalwarp`` clipped to the
    AOI cutline.  Writes a HydroMT data-catalog YAML beside the GeoTIFF.

    Parameters
    ----------
    aoi
        Path to an AOI polygon (GeoJSON, Shapefile, etc.).
    output_dir
        Directory where the GeoTIFF and catalog YAML are written.
    resolution_m
        Target ground resolution in meters (default 90.0, matching the
        native CRM 3 arc-second grid).
    buffer_deg
        Bounding-box buffer in degrees added around the AOI extent
        (default 0.1, approximately 11 km).
    catalog_name
        Name used in the HydroMT catalog entry.
    log
        Optional logging callback.

    Returns
    -------
    tuple[Path, Path, str]
        ``(geotiff_path, catalog_path, catalog_name)``.

    Raises
    ------
    ValueError
        If the AOI file has no CRS, or the final raster has < 10% valid
        data in the AOI bbox (likely the AOI is outside CRM coverage).
    """
    from pathlib import Path

    _log = log if log is not None else logger.info

    aoi = Path(aoi)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi_4326, bbox = _aoi_bbox(aoi, buffer_deg)

    tiles = _decompose_bbox(bbox, resolution_m, _PIXEL_MAX)
    _log(f"CRM fetch: {len(tiles)} tile(s) at ~{resolution_m:.0f} m resolution")

    sub_dir = output_dir / "_crm_temp"
    sub_dir.mkdir(parents=True, exist_ok=True)

    tif_name = f"{catalog_name}.tif"
    geotiff_path = output_dir / tif_name

    try:
        _assemble_output(tiles, bbox, aoi_4326, sub_dir, geotiff_path, _log)
        _log(f"GeoTIFF written ({geotiff_path.stat().st_size / 1e6:.1f} MB)")
    finally:
        for p in sub_dir.iterdir():
            p.unlink()
        sub_dir.rmdir()

    valid_frac = _valid_data_fraction(geotiff_path)
    _log(f"Valid-data fraction in output: {valid_frac * 100:.1f}%")
    if valid_frac < 0.10:
        geotiff_path.unlink(missing_ok=True)
        raise ValueError(
            f"CRM mosaic has only {valid_frac * 100:.1f}% valid data in the AOI "
            f"bbox (minimum 10% required).  The AOI may be largely outside CRM "
            f"coverage; pair with a global source like gebco_15arcs."
        )

    catalog_path = output_dir / f"{catalog_name}_catalog.yml"
    _write_catalog(catalog_path, tif_name, catalog_name)
    _log(f"Catalog written: {catalog_path}")

    return geotiff_path, catalog_path, catalog_name
