"""Discover and fetch NOAA coastal DEMs from the NOS Coastal Lidar S3 bucket.

The ``noaa-nos-coastal-lidar-pds`` public S3 bucket hosts 300+ coastal
DEM datasets.  A static JSON index shipped with the package lists the
verified NCEI 1/9 arc-second topobathy datasets with bounding boxes,
resolution, year, and VRT filenames.

At runtime, :func:`fetch_noaa_dem` uses the index to select the best
DEM overlapping the user's AOI, clips it via ``rioxarray``, and writes
a local GeoTIFF plus a HydroMT data-catalog YAML.

.. note::

   The NCEI topobathy does **not** provide full coverage in every
   coastal region.  Always pair it with a global fallback dataset
   (e.g. ``gebco``) in the elevation dataset list so that HydroMT
   fills nodata gaps.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_S3_BASE = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem"


# ------------------------------------------------------------------
# Index helpers
# ------------------------------------------------------------------


def load_index() -> list[dict[str, Any]]:
    """Load the packaged NOAA DEM spatial index.

    Returns
    -------
    list of dict
        Each entry has keys ``dataset_name``, ``title``, ``bbox``,
        ``resolution_m``, ``year``, ``is_topobathy``, ``epsg``,
        and ``vrt_filename``.
    """
    ref = importlib.resources.files("coastal_calibration.data").joinpath(
        "noaa_dem_index.json"
    )
    return json.loads(ref.read_text(encoding="utf-8"))


def query_overlapping(
    index: list[dict[str, Any]],
    bbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    """Return index entries whose bbox intersects *bbox*.

    Parameters
    ----------
    index
        As returned by :func:`load_index`.
    bbox
        ``(west, south, east, north)`` in geographic coordinates.
    """
    w, s, e, n = bbox
    return [
        rec
        for rec in index
        if not (rec["bbox"][2] < w or rec["bbox"][0] > e or rec["bbox"][3] < s or rec["bbox"][1] > n)
    ]


def _overlap_fraction(
    rec_bbox: list[float],
    query_bbox: tuple[float, float, float, float],
) -> float:
    """Fraction of *query_bbox* area covered by *rec_bbox*."""
    w, s, e, n = query_bbox
    rw, rs, re, rn = rec_bbox
    iw = max(w, rw)
    ie = min(e, re)
    is_ = max(s, rs)
    in_ = min(n, rn)
    if iw >= ie or is_ >= in_:
        return 0.0
    query_area = (e - w) * (n - s)
    if query_area <= 0:
        return 0.0
    return ((ie - iw) * (in_ - is_)) / query_area


def select_best(
    candidates: list[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    *,
    prefer_topobathy: bool = True,
) -> dict[str, Any]:
    """Select the single best DEM from *candidates*.

    Selection criteria (in priority order):

    1. Topobathy datasets are preferred over topo-only.
    2. Higher overlap fraction with *bbox*.
    3. Finer resolution (smaller ``resolution_m``).
    4. Most recent (largest ``year``).

    Parameters
    ----------
    candidates
        Non-empty list as returned by :func:`query_overlapping`.
    bbox
        ``(west, south, east, north)`` query bounding box.
    prefer_topobathy
        When ``True`` (default), topobathy datasets rank higher.

    Returns
    -------
    dict
        The best matching index entry.

    Raises
    ------
    ValueError
        If *candidates* is empty.
    """
    if not candidates:
        raise ValueError("No candidate DEMs to select from")

    def _sort_key(rec: dict[str, Any]) -> tuple[int, float, float, int]:
        tb = int(rec.get("is_topobathy", False)) if prefer_topobathy else 0
        overlap = _overlap_fraction(rec["bbox"], bbox)
        # Negate resolution so smaller (finer) sorts higher.
        res = -rec.get("resolution_m", 999.0)
        year = rec.get("year", 0)
        return (tb, overlap, res, year)

    return max(candidates, key=_sort_key)


def get_vrt_url(record: dict[str, Any]) -> str:
    """Construct the HTTPS URL for a dataset's VRT file.

    Parameters
    ----------
    record
        A single index entry (must contain ``dataset_name`` and
        ``vrt_filename``).

    Returns
    -------
    str
        Full HTTPS URL to the VRT on S3.
    """
    return f"{_S3_BASE}/{record['dataset_name']}/{record['vrt_filename']}"


# ------------------------------------------------------------------
# Fetch + clip
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
    catalog_path.write_text(
        yaml.dump(catalog, default_flow_style=False, sort_keys=False)
    )


def fetch_noaa_dem(
    aoi: Path | str,
    output_dir: Path | str,
    *,
    dataset_name: str | None = None,
    buffer_deg: float = 0.1,
    catalog_name: str = "noaa_topobathy",
    log: Callable[[str], None] | None = None,
) -> tuple[Path, Path, str]:
    """Discover, clip, and save a NOAA DEM overlapping *aoi*.

    Uses ``rioxarray.open_rasterio`` to lazily read a remote VRT,
    clips to the buffered AOI bounding box, masks nodata values,
    and writes a local GeoTIFF + HydroMT data-catalog YAML.

    Parameters
    ----------
    aoi
        Path to an AOI polygon (GeoJSON, Shapefile, etc.).
    output_dir
        Directory where the GeoTIFF and catalog YAML are written.
    dataset_name
        Explicit NOAA dataset name.  When ``None``, the best
        overlapping dataset is auto-discovered from the index.
    buffer_deg
        Bounding-box buffer in degrees added around the AOI extent
        (default 0.1 ≈ 11 km).
    catalog_name
        Name used in the HydroMT catalog entry (default
        ``"noaa_topobathy"``).
    log
        Optional logging callback (e.g. ``stage._log``).  When
        provided, progress messages use this callback so they
        inherit the caller's indentation.  Falls back to the
        module logger.

    Returns
    -------
    tuple[Path, Path, str]
        ``(geotiff_path, catalog_path, catalog_name)``.

    Raises
    ------
    ValueError
        If no overlapping DEM is found, *dataset_name* is not in
        the index, or the VRT has no valid data for the AOI region.
    """
    from pathlib import Path as _Path

    import geopandas as gpd
    import numpy as np
    import rioxarray  # noqa: F401 (registers accessor)

    _log = log if log is not None else logger.info

    aoi = _Path(aoi)
    output_dir = _Path(output_dir)

    aoi_gdf = gpd.read_file(aoi)
    if aoi_gdf.crs is None:
        raise ValueError(f"AOI file {aoi} has no CRS")

    index = load_index()

    if dataset_name is not None:
        matches = [r for r in index if r["dataset_name"] == dataset_name]
        if not matches:
            raise ValueError(
                f"Dataset {dataset_name!r} not found in NOAA DEM index. "
                f"Available: {[r['dataset_name'] for r in index]}"
            )
        record = matches[0]
    else:
        # Reproject AOI to EPSG:4326 for bbox query (works for all datasets).
        aoi_4326 = aoi_gdf.to_crs(epsg=4326)
        bbox = tuple(aoi_4326.total_bounds)  # (w, s, e, n)
        candidates = query_overlapping(index, bbox)
        if not candidates:
            raise ValueError(
                f"No NOAA DEM overlaps AOI bbox {bbox}. "
                f"Run 'coastal-calibration update-dem-index' to refresh the index."
            )
        record = select_best(candidates, bbox)

    target_epsg = record["epsg"]

    _log(
        f"Selected DEM: {record['dataset_name']}"
        f" ({record['title']}, {record['resolution_m']:.0f} m, {record['year']})"
    )

    vrt_url = get_vrt_url(record)
    _log(f"Opening VRT: {vrt_url}")

    # Reproject AOI to the dataset's native CRS for clipping.
    aoi_native = aoi_gdf.to_crs(epsg=target_epsg)
    bounds = aoi_native.total_bounds  # (w, s, e, n)
    buffered = (
        bounds[0] - buffer_deg,
        bounds[1] - buffer_deg,
        bounds[2] + buffer_deg,
        bounds[3] + buffer_deg,
    )

    # Use rioxarray.open_rasterio for robust remote VRT access.
    da = rioxarray.open_rasterio(vrt_url, chunks={"x": 4096, "y": 4096})
    da = da.rio.clip_box(*buffered, crs=f"EPSG:{target_epsg}")  # type: ignore[union-attr]
    da = da.squeeze(drop=True)

    # Clip to AOI polygon and mask nodata values.
    da = da.rio.clip(aoi_native.geometry, aoi_native.crs, drop=True)
    nodata_val = da.rio.nodata
    if nodata_val is not None:
        da = da.where(da > nodata_val, drop=False)

    # Validate that the clipped raster has actual data.
    # Sample from the center of the array (coastal data is sparse near edges).
    if da.size < 1_000_000:
        sample = da.values
    else:
        ny, nx = da.shape[-2], da.shape[-1]
        cy, cx = ny // 2, nx // 2
        hw = min(250, nx // 2)
        hh = min(250, ny // 2)
        sample = da.isel(
            x=slice(cx - hw, cx + hw),
            y=slice(cy - hh, cy + hh),
        ).values
    valid_count = int(np.sum(~np.isnan(sample)))
    if valid_count == 0:
        raise ValueError(
            f"NOAA DEM '{record['dataset_name']}' has no valid pixels in the "
            f"clipped region. The VRT may lack data coverage for this area. "
            f"Try a different dataset or use 'prepare-topobathy' to fetch "
            f"from NWS icechunk instead."
        )

    # Finalize raster metadata before writing.
    da = da.rio.write_transform()
    da = da.rio.write_nodata(np.nan, encoded=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    tif_name = f"{catalog_name}.tif"
    geotiff_path = output_dir / tif_name

    _log(f"Writing GeoTIFF: {geotiff_path}")
    if da.dtype != "float32":
        da = da.astype("float32")
    da.rio.to_raster(str(geotiff_path), driver="GTiff", compress="deflate")
    _log(f"GeoTIFF written ({geotiff_path.stat().st_size / 1e6:.1f} MB)")

    catalog_path = output_dir / f"{catalog_name}_catalog.yml"
    _write_catalog(catalog_path, tif_name, catalog_name, target_epsg)
    _log(f"Catalog written: {catalog_path}")

    return geotiff_path, catalog_path, catalog_name


# ------------------------------------------------------------------
# Index rebuild from S3 (CLI / offline use)
# ------------------------------------------------------------------


def build_index_from_s3(
    *,
    max_datasets: int | None = None,
) -> list[dict[str, Any]]:
    """Scan the NOAA S3 bucket and build a DEM spatial index.

    Reads STAC collection JSON files to extract spatial extent,
    resolution, and year for each dataset.

    Parameters
    ----------
    max_datasets
        Limit the number of datasets scanned (for testing).

    Returns
    -------
    list of dict
        Index entries ready to be written to JSON.
    """
    import re

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket = "noaa-nos-coastal-lidar-pds"
    prefix = "dem/"

    # List top-level dataset directories.
    paginator = s3.get_paginator("list_objects_v2")
    dataset_names: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"].rstrip("/").split("/")[-1]
            if name and not name.startswith("."):
                dataset_names.append(name)

    if max_datasets is not None:
        dataset_names = dataset_names[:max_datasets]

    logger.info("Found %d dataset directories", len(dataset_names))

    entries: list[dict[str, Any]] = []
    for ds_name in dataset_names:
        stac_prefix = f"dem/{ds_name}/stac/"
        try:
            stac_resp = s3.list_objects_v2(Bucket=bucket, Prefix=stac_prefix)
            stac_files = [
                obj["Key"]
                for obj in stac_resp.get("Contents", [])
                if obj["Key"].endswith(".json") and "collection" in obj["Key"]
            ]
            if not stac_files:
                continue

            body = s3.get_object(Bucket=bucket, Key=stac_files[0])["Body"].read()
            collection = json.loads(body)
        except Exception:
            logger.debug("Skipping %s (no STAC collection)", ds_name)
            continue

        extent = collection.get("extent", {}).get("spatial", {}).get("bbox", [[]])
        if not extent or not extent[0]:
            continue
        raw_bbox = extent[0]

        title = collection.get("title", ds_name)
        description = collection.get("description", "")

        is_topobathy = any(
            kw in (title + description).lower()
            for kw in ("topobathy", "topo-bathy", "bathymetric topographic")
        )

        year_match = re.search(r"(\d{4})", ds_name)
        year = int(year_match.group(1)) if year_match else 0

        # Detect available VRTs.
        vrt_resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"dem/{ds_name}/", Delimiter="/")
        vrt_files = [
            obj["Key"].split("/")[-1]
            for obj in vrt_resp.get("Contents", [])
            if obj["Key"].endswith(".vrt")
        ]
        if not vrt_files:
            continue

        # Prefer geographic CRS VRTs (EPSG:4269 or 4326).
        epsg = 4269
        vrt_filename = ""
        for vf in vrt_files:
            m = re.search(r"EPSG-(\d+)", vf)
            if m:
                code = int(m.group(1))
                if code in (4269, 4326):
                    epsg = code
                    vrt_filename = vf

        if not vrt_filename:
            # Fall back to first available VRT.
            vrt_filename = vrt_files[0]
            m = re.search(r"EPSG-(\d+)", vrt_filename)
            if m:
                epsg = int(m.group(1))

        res_match = re.search(r"(\d+)[_-]?(?:arc)?[_-]?sec", title.lower())
        if res_match:
            arcsec = int(res_match.group(1))
            resolution_m = arcsec * 30.0 / 9  # approximate
        else:
            resolution_m = 1.0

        entries.append(
            {
                "dataset_name": ds_name,
                "title": title,
                "bbox": raw_bbox[:4],
                "resolution_m": round(resolution_m, 1),
                "year": year,
                "is_topobathy": is_topobathy,
                "epsg": epsg,
                "vrt_filename": vrt_filename,
            }
        )

    logger.info("Built index with %d entries", len(entries))
    return entries
