"""Fetch GEBCO bathymetry from CEDA GeoTIFF tiles.

The GEBCO 2025 global grid is hosted at CEDA as eight 90x90 degree
GeoTIFF tiles.  At runtime, :func:`fetch_gebco` determines which
tile(s) overlap the user's AOI, builds a virtual mosaic with
``gdalbuildvrt`` using GDAL ``/vsicurl/`` for remote access, and
clips to the AOI polygon with ``gdalwarp``.  Only the pixels within
the AOI extent are transferred over the network.

A companion HydroMT data-catalog YAML is written alongside the output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coastal_calibration.utils.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_CEDA_BASE = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2025/ice_surface_elevation/geotiff"

# Eight tiles cover the globe in a 2 (lat) x 4 (lon) grid.
_TILE_LAT_EDGES = [-90.0, 0.0, 90.0]
_TILE_LON_EDGES = [-180.0, -90.0, 0.0, 90.0, 180.0]


# ------------------------------------------------------------------
# Tile helpers
# ------------------------------------------------------------------


def _tile_indices(
    bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    """Return ``(south, north, west, east)`` for each tile overlapping *bbox*.

    Parameters
    ----------
    bbox
        ``(west, south, east, north)`` in EPSG:4326.
    """
    west, south, east, north = bbox
    tiles = []
    for i in range(len(_TILE_LAT_EDGES) - 1):
        t_south = _TILE_LAT_EDGES[i]
        t_north = _TILE_LAT_EDGES[i + 1]
        if south >= t_north or north <= t_south:
            continue
        for j in range(len(_TILE_LON_EDGES) - 1):
            t_west = _TILE_LON_EDGES[j]
            t_east = _TILE_LON_EDGES[j + 1]
            if west >= t_east or east <= t_west:
                continue
            tiles.append((t_south, t_north, t_west, t_east))
    return tiles


def _tile_url(south: float, north: float, west: float, east: float) -> str:
    """Construct the ``/vsicurl/`` URL for a CEDA GEBCO tile."""
    name = f"gebco_2025_n{north:.1f}_s{south:.1f}_w{west:.1f}_e{east:.1f}.tif"
    return f"/vsicurl/{_CEDA_BASE}/{name}"


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
# VRT + clip (remote access via /vsicurl/)
# ------------------------------------------------------------------


def fetch_gebco(
    aoi: Path | str,
    output_dir: Path | str,
    *,
    buffer_deg: float = 0.1,
    catalog_name: str = "gebco",
    log: Callable[[str], None] | None = None,
) -> tuple[Path, Path, str]:
    """Fetch GEBCO bathymetry for *aoi* from CEDA GeoTIFF tiles.

    Determines which CEDA tile(s) overlap the AOI, builds a virtual
    mosaic with ``gdalbuildvrt`` using GDAL ``/vsicurl/`` for remote
    access, then clips to the AOI polygon with ``gdalwarp``.  Only
    the pixels within the buffered AOI extent are transferred.

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

    from coastal_calibration.utils._gdal import build_vrt, clip_to_aoi

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
        float(bounds[0] - buffer_deg),
        float(bounds[1] - buffer_deg),
        float(bounds[2] + buffer_deg),
        float(bounds[3] + buffer_deg),
    )

    tiles = _tile_indices(buffered)
    if not tiles:
        raise ValueError(
            f"No GEBCO tiles found for AOI bbox {buffered}. "
            f"Check that the AOI is within valid lat/lon ranges."
        )

    vsicurl_paths = [_tile_url(*t) for t in tiles]
    _log(f"GEBCO: {len(tiles)} CEDA tile(s) needed for AOI")

    # Build VRT from remote tiles via /vsicurl/.
    sub_dir = output_dir / "_gebco_temp"
    sub_dir.mkdir(parents=True, exist_ok=True)

    vrt_path = sub_dir / "gebco_mosaic.vrt"
    build_vrt(vrt_path, vsicurl_paths)

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
    for f in [vrt_path, cutline_path]:
        f.unlink(missing_ok=True)
    sub_dir.rmdir()

    catalog_path = output_dir / f"{catalog_name}_catalog.yml"
    _write_catalog(catalog_path, tif_name, catalog_name, 4326)
    _log(f"Catalog written: {catalog_path}")

    return geotiff_path, catalog_path, catalog_name
