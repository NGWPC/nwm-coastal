"""Fetch ESA WorldCover 2020 tiles from AWS S3.

The ``esa-worldcover`` S3 bucket hosts 10 m land-use / land-cover
classification maps as Cloud-Optimized GeoTIFFs in 3x3 degree tiles.

At runtime, :func:`fetch_esa_worldcover` downloads the tiles that
overlap the user's AOI via ``tiny_retriever``, builds a virtual
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

_TILE_SIZE = 3  # degrees
_S3_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map"


# ------------------------------------------------------------------
# Tile helpers
# ------------------------------------------------------------------


def _snap_to_grid(value: float, grid: int, mode: str) -> int:
    """Snap *value* to the nearest *grid*-degree boundary.

    Parameters
    ----------
    value
        Coordinate value in degrees.
    grid
        Grid spacing in degrees.
    mode
        ``"floor"`` or ``"ceil"``.
    """
    if mode == "floor":
        return int(math.floor(value / grid) * grid)
    return int(math.ceil(value / grid) * grid)


def _tile_indices(
    bbox: tuple[float, float, float, float],
) -> list[tuple[int, int]]:
    """Return (lat, lon) SW-corner indices of 3x3 degree tiles covering *bbox*.

    Parameters
    ----------
    bbox
        ``(west, south, east, north)`` in EPSG:4326.
    """
    west, south, east, north = bbox
    lon_min = _snap_to_grid(west, _TILE_SIZE, "floor")
    lon_max = _snap_to_grid(east, _TILE_SIZE, "ceil") - _TILE_SIZE
    lat_min = _snap_to_grid(south, _TILE_SIZE, "floor")
    lat_max = _snap_to_grid(north, _TILE_SIZE, "ceil") - _TILE_SIZE
    return [
        (lat, lon)
        for lat in range(lat_min, lat_max + 1, _TILE_SIZE)
        for lon in range(lon_min, lon_max + 1, _TILE_SIZE)
    ]


def _tile_url(lat: int, lon: int) -> str:
    """Construct the HTTPS URL for a single ESA WorldCover tile."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    lat_str = f"{abs(lat):02d}"
    lon_str = f"{abs(lon):03d}"
    return f"{_S3_BASE}/ESA_WorldCover_10m_2020_v100_{ns}{lat_str}{ew}{lon_str}_Map.tif"


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
                "category": "landuse",
                "crs": crs_epsg,
            },
        },
    }
    catalog_path.write_text(yaml.dump(catalog, default_flow_style=False, sort_keys=False))


# ------------------------------------------------------------------
# Download + VRT + clip
# ------------------------------------------------------------------


def fetch_esa_worldcover_vrt(
    bbox: tuple[float, float, float, float],
    cache_dir: Path | str,
    *,
    buffer_deg: float = 0.05,
    log: Callable[[str], None] | None = None,
) -> tuple[Path, list[tuple[int, int]], list[tuple[int, int]]]:
    """Download the WorldCover tiles covering *bbox* and mosaic them into a VRT.

    Unlike :func:`fetch_esa_worldcover` this does not clip to a polygon, so it
    suits point sampling over a large mesh.  Tiles already in *cache_dir* are
    reused.  Tiles that 404 are open ocean, which ESA does not publish; they are
    returned in the missing list rather than raising.

    Parameters
    ----------
    bbox
        ``(west, south, east, north)`` in EPSG:4326.
    cache_dir
        Directory holding the downloaded tiles and the VRT.
    buffer_deg
        Bounding-box buffer in degrees.
    log
        Optional logging callback.

    Returns
    -------
    tuple[Path, list[tuple[int, int]], list[tuple[int, int]]]
        ``(vrt_path, tiles_found, tiles_missing)``.
    """
    from pathlib import Path

    from tiny_retriever import download

    from coastal_calibration.data.transformation import build_vrt

    _log = log if log is not None else logger.info

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = bbox
    buffered = (west - buffer_deg, south - buffer_deg, east + buffer_deg, north + buffer_deg)

    tiles = _tile_indices(buffered)
    paths = {(lat, lon): cache_dir / f"esa_wc_{lat}_{lon}.tif" for lat, lon in tiles}

    pending = [t for t in tiles if not (paths[t].exists() and paths[t].stat().st_size > 0)]
    if pending:
        _log(f"ESA WorldCover: downloading {len(pending)} of {len(tiles)} tile(s)")
        download(
            [_tile_url(lat, lon) for lat, lon in pending],
            [paths[t] for t in pending],
            timeout=300,
            raise_status=False,
        )
    else:
        _log(f"ESA WorldCover: all {len(tiles)} tile(s) already cached")

    found = [t for t in tiles if paths[t].exists() and paths[t].stat().st_size > 0]
    missing = [t for t in tiles if t not in found]
    for tile in missing:
        paths[tile].unlink(missing_ok=True)

    if not found:
        raise ValueError(
            f"No ESA WorldCover tiles found for bbox {buffered}. "
            f"Check that it is within ESA WorldCover coverage (60°S to 84°N)."
        )
    if missing:
        _log(f"{len(missing)} tile(s) not published (open ocean); those nodes use the fallback")

    vrt_path = cache_dir / "esa_wc_mosaic.vrt"
    build_vrt(vrt_path, [paths[t] for t in found])
    return vrt_path, found, missing


def fetch_esa_worldcover(
    aoi: Path | str,
    output_dir: Path | str,
    *,
    buffer_deg: float = 0.1,
    catalog_name: str = "esa_worldcover",
    log: Callable[[str], None] | None = None,
) -> tuple[Path, Path, str]:
    """Discover, download, and mosaic ESA WorldCover tiles for *aoi*.

    Downloads tiles via ``tiny_retriever``, creates a virtual mosaic
    with ``gdalbuildvrt``, then clips to the AOI polygon with
    ``gdalwarp``.  The output preserves the original ``uint8`` dtype
    (classification values 10-200).

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
    bounds = aoi_4326.total_bounds
    buffered = (
        bounds[0] - buffer_deg,
        bounds[1] - buffer_deg,
        bounds[2] + buffer_deg,
        bounds[3] + buffer_deg,
    )

    tiles = _tile_indices(buffered)
    _log(f"ESA WorldCover: {len(tiles)} tile(s) needed for AOI")

    # Download tiles to a temporary directory.
    sub_dir = output_dir / "_esa_wc_temp"
    sub_dir.mkdir(parents=True, exist_ok=True)

    urls = [_tile_url(lat, lon) for lat, lon in tiles]
    tile_files = [sub_dir / f"esa_wc_{lat}_{lon}.tif" for lat, lon in tiles]

    download(urls, tile_files, timeout=300, raise_status=False)

    # Keep only successfully downloaded files.
    valid_files = [f for f in tile_files if f.exists() and f.stat().st_size > 0]
    if not valid_files:
        for f in tile_files:
            f.unlink(missing_ok=True)
        sub_dir.rmdir()
        raise ValueError(
            f"No ESA WorldCover tiles found for AOI bbox {buffered}. "
            f"Check that the AOI is within ESA WorldCover coverage (60°S to 84°N)."
        )

    _log(f"Downloaded {len(valid_files)} of {len(tiles)} tile(s)")

    # Build VRT from downloaded tiles.
    vrt_path = sub_dir / "esa_wc_mosaic.vrt"
    build_vrt(vrt_path, valid_files)

    # Write a temporary cutline in EPSG:4326 for gdalwarp.
    cutline_path = sub_dir / "cutline.geojson"
    aoi_4326.to_file(cutline_path, driver="GeoJSON")

    # Clip to AOI polygon with gdalwarp (preserve uint8 dtype).
    tif_name = f"{catalog_name}.tif"
    geotiff_path = output_dir / tif_name
    clip_to_aoi(
        vrt_path,
        cutline_path,
        geotiff_path,
        nodata="0",
        output_type="Byte",
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
