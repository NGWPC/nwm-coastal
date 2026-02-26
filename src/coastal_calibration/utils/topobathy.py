"""Fetch NWS topobathy DEM from icechunk and write a HydroMT data-catalog entry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# S3 prefixes keyed by short domain name.
_PREFIXES: dict[str, str] = {
    "atlgulf": "surface/nws-topobathy/tbdem_conus_atlantic_gulf_30m",
    "hi": "surface/nws-topobathy/tbdem_hawaii_30m",
    "prvi": "surface/nws-topobathy/tbdem_pr_usvi_30m",
    "pacific": "surface/nws-topobathy/tbdem_conus_pacific_30m",
    "ak": "surface/nws-topobathy/tbdem_alaska_30m",
}

# Aliases so callers can use the CoastalDomain literals directly.
_DOMAIN_ALIASES: dict[str, str] = {
    "hawaii": "hi",
    "alaska": "ak",
}

_S3_BUCKET = "edfs-data"
_S3_REGION = "us-east-1"


def _resolve_domain(domain: str) -> str:
    """Normalise *domain* to a key in :data:`_PREFIXES`."""
    key = _DOMAIN_ALIASES.get(domain, domain)
    if key not in _PREFIXES:
        valid = sorted({*_PREFIXES, *_DOMAIN_ALIASES})
        msg = f"Unknown topobathy domain {domain!r}. Valid: {', '.join(valid)}"
        raise ValueError(msg)
    return key


def _write_catalog(
    catalog_path: Path,
    tif_name: str,
    crs_epsg: int,
) -> None:
    """Write a minimal HydroMT data-catalog YAML next to the GeoTIFF."""
    import yaml

    catalog: dict[str, Any] = {
        "meta": {
            "version": "v1.0.0",
            "name": "nws_topobathy",
            "hydromt_version": ">1.0a,<2",
        },
        "nws_topobathy": {
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
    catalog_path.write_text(
        yaml.dump(catalog, default_flow_style=False, sort_keys=False)
    )


def fetch_topobathy(
    domain: str,
    aoi: Path,
    output_dir: Path,
    *,
    buffer_deg: float = 0.1,
) -> tuple[Path, Path]:
    """Clip NWS topobathy DEM to *aoi* and write a HydroMT data-catalog entry.

    Parameters
    ----------
    domain
        Coastal domain identifier (``"atlgulf"``, ``"hi"``, ``"prvi"``,
        ``"pacific"``, ``"ak"``).  The aliases ``"hawaii"`` and
        ``"alaska"`` are also accepted.
    aoi
        Path to an AOI polygon (GeoJSON, Shapefile, etc.).
    output_dir
        Directory where the GeoTIFF and catalog YAML are written.
    buffer_deg
        Bounding-box buffer in degrees added around the AOI extent
        (default 0.1 ≈ 11 km).

    Returns
    -------
    tuple[Path, Path]
        ``(geotiff_path, catalog_path)`` — the local GeoTIFF and the
        HydroMT data-catalog YAML that references it.

    Raises
    ------
    ValueError
        If *domain* is not recognised.
    ImportError
        If ``icechunk``, ``rioxarray``, or ``geopandas`` are not installed.
    """
    import dotenv
    import geopandas as gpd
    import icechunk as ic
    import xarray as xr

    dotenv.load_dotenv()

    key = _resolve_domain(domain)
    prefix = _PREFIXES[key]

    aoi_gdf = gpd.read_file(aoi)
    if aoi_gdf.crs is None:
        raise ValueError(f"AOI file {aoi} has no CRS. Please ensure it is georeferenced.")
    bbox = aoi_gdf.buffer(buffer_deg).total_bounds

    logger.info("Opening and clipping icechunk store: s3://%s/%s", _S3_BUCKET, prefix)
    storage = ic.s3_storage(
        bucket=_S3_BUCKET, prefix=prefix, region=_S3_REGION, from_env=True
    )
    store = ic.Repository.open(storage).readonly_session("main").store
    clipped = (
        xr.open_zarr(store, consolidated=False, decode_coords="all")
        .squeeze("band", drop=True)
        .elevation
        .rio.clip_box(*bbox, crs=aoi_gdf.crs)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tif_name = f"nws_topobathy_{key}.tif"
    geotiff_path = output_dir / tif_name

    logger.info("Writing GeoTIFF: %s", geotiff_path)
    # SFINCS reads bathymetry as real*4; avoid writing unnecessary float64.
    if clipped.dtype != "float32":
        clipped = clipped.astype("float32")
    clipped.rio.to_raster(str(geotiff_path), driver="GTiff", compress="deflate")
    logger.info("GeoTIFF written (%.1f MB)", geotiff_path.stat().st_size / 1e6)

    catalog_path = output_dir / "nws_topobathy_catalog.yml"
    _write_catalog(catalog_path, tif_name, clipped.rio.crs)
    logger.info("Catalog written: %s", catalog_path)

    return geotiff_path, catalog_path
