"""Fetch Copernicus DEM 30m tiles from AWS S3.

The ``copernicus-dem-30m`` public S3 bucket hosts global 1-arc-second
(~30 m) elevation data as Cloud-Optimized GeoTIFFs organized in 1x1 degree
tiles.

At runtime, :func:`fetch_copdem30` computes which tiles overlap the
user's AOI, downloads them via ``tiny_retriever``, builds a virtual
mosaic with ``gdalbuildvrt``, and clips to the AOI with ``gdalwarp``.
A companion HydroMT data-catalog YAML is written alongside the output.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from coastal_calibration.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


# ------------------------------------------------------------------
# Tile helpers
# ------------------------------------------------------------------


def _tile_indices(
    bbox: tuple[float, float, float, float],
) -> list[tuple[int, int]]:
    """Return (lat, lon) SW-corner indices of 1x1 degree tiles covering *bbox*.

    Parameters
    ----------
    bbox
        ``(west, south, east, north)`` in EPSG:4326.
    """
    west, south, east, north = bbox
    lon_min = math.floor(west)
    lon_max = math.ceil(east) - 1
    lat_min = math.floor(south)
    lat_max = math.ceil(north) - 1
    return [
        (lat, lon) for lat in range(lat_min, lat_max + 1) for lon in range(lon_min, lon_max + 1)
    ]


def _tile_url(lat: int, lon: int) -> str:
    """Construct the HTTPS URL for a single CopDEM 30m COG tile."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    lat_str = f"{abs(lat):02d}"
    lon_str = f"{abs(lon):03d}"
    name = f"Copernicus_DSM_COG_10_{ns}{lat_str}_00_{ew}{lon_str}_00_DEM"
    return f"{_S3_BASE}/{name}/{name}.tif"


# ------------------------------------------------------------------
# Catalog writer
# ------------------------------------------------------------------


def _write_catalog(
    catalog_path: Path,
    tif_name: str,
    catalog_name: str,
    crs_epsg: int,
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
                "crs": crs_epsg,
            },
            "data_adapter": {
                "rename": {"elevation": "elevtn"},
            },
        },
    }
    catalog_path.write_text(yaml.dump(catalog, default_flow_style=False, sort_keys=False))


# ------------------------------------------------------------------
# Download + VRT + clip
# ------------------------------------------------------------------


def fetch_copdem30(
    aoi: Path | str,
    output_dir: Path | str,
    *,
    buffer_deg: float = 0.1,
    catalog_name: str = "copdem_30m",
    log: Callable[[str], None] | None = None,
) -> tuple[Path, Path, str]:
    """Discover, download, and mosaic CopDEM 30m tiles for *aoi*.

    Downloads tiles via ``tiny_retriever``, creates a virtual mosaic
    with ``gdalbuildvrt``, then clips to the AOI polygon with
    ``gdalwarp``.  A HydroMT data-catalog YAML is written alongside
    the output GeoTIFF.

    Parameters
    ----------
    aoi
        Path to an AOI polygon (GeoJSON, Shapefile, etc.).
    output_dir
        Directory where the GeoTIFF and catalog YAML are written.
    buffer_deg
        Bounding-box buffer in degrees added around the AOI extent.
    catalog_name
        Name used in the HydroMT catalog entry.
    log
        Optional logging callback.

    Returns
    -------
    tuple[Path, Path, str]
        ``(geotiff_path, catalog_path, catalog_name)``.
    """
    from pathlib import Path

    import geopandas as gpd
    from tiny_retriever import download

    from coastal_calibration.data.transformation import build_vrt, clip_to_aoi

    _log = log if log is not None else logger.info

    aoi = Path(aoi)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi_gdf = gpd.read_file(aoi)
    if aoi_gdf.crs is None:
        raise ValueError(f"AOI file {aoi} has no CRS")

    aoi_4326 = aoi_gdf.to_crs(epsg=4326)
    bounds = aoi_4326.total_bounds  # (w, s, e, n)
    buffered = (
        bounds[0] - buffer_deg,
        bounds[1] - buffer_deg,
        bounds[2] + buffer_deg,
        bounds[3] + buffer_deg,
    )

    tiles = _tile_indices(buffered)
    _log(f"CopDEM30: {len(tiles)} tile(s) needed for AOI")

    # Download tiles to a temporary directory.
    sub_dir = output_dir / "_copdem_temp"
    sub_dir.mkdir(parents=True, exist_ok=True)

    urls = [_tile_url(lat, lon) for lat, lon in tiles]
    tile_files = [sub_dir / f"copdem_{lat}_{lon}.tif" for lat, lon in tiles]

    download(urls, tile_files, timeout=300, raise_status=False)

    # Keep only successfully downloaded files.
    valid_files = [f for f in tile_files if f.exists() and f.stat().st_size > 0]
    if not valid_files:
        for f in tile_files:
            f.unlink(missing_ok=True)
        sub_dir.rmdir()
        raise ValueError(
            f"No CopDEM 30m tiles found for AOI bbox {buffered}. "
            f"Check that the AOI is within the CopDEM coverage (-90°S to 84°N)."
        )

    _log(f"Downloaded {len(valid_files)} of {len(tiles)} tile(s)")

    # Build VRT from downloaded tiles.
    vrt_path = sub_dir / "copdem_mosaic.vrt"
    build_vrt(vrt_path, valid_files)

    # Write a temporary cutline in EPSG:4326 for gdalwarp.
    cutline_path = sub_dir / "cutline.geojson"
    aoi_4326.to_file(cutline_path, driver="GeoJSON")

    # Clip to AOI polygon with gdalwarp.
    tif_name = f"{catalog_name}.tif"
    geotiff_path = output_dir / tif_name
    clip_to_aoi(
        vrt_path,
        cutline_path,
        geotiff_path,
        nodata="nan",
        output_type="Float32",
    )
    _log(f"GeoTIFF written ({geotiff_path.stat().st_size / 1e6:.1f} MB)")

    # Clean up temporary files.
    for f in [*tile_files, vrt_path, cutline_path]:
        f.unlink(missing_ok=True)
    sub_dir.rmdir()

    catalog_path = output_dir / f"{catalog_name}_catalog.yml"
    _write_catalog(catalog_path, tif_name, catalog_name, 4326)
    _log(f"Catalog written: {catalog_path}")

    return geotiff_path, catalog_path, catalog_name
