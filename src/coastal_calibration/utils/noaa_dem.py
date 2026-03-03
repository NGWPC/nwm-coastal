"""Discover and fetch NOAA coastal DEMs from the NOS Coastal Lidar S3 bucket.

The ``noaa-nos-coastal-lidar-pds`` public S3 bucket hosts 300+ coastal
DEM datasets.  A static JSON index shipped with the package lists the
verified NCEI 1/9 arc-second topobathy datasets with bounding boxes,
resolution, year, and VRT filenames.

At runtime, :func:`fetch_noaa_dem` uses the index to select the best
DEM overlapping the user's AOI, clips it via ``gdalwarp`` with GDAL
``/vsicurl/`` for remote access, and writes a local GeoTIFF plus a
HydroMT data-catalog YAML.  Only the pixels within the AOI extent are
transferred over the network.

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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
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
    ref = importlib.resources.files("coastal_calibration.data_catalog").joinpath(
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
        if not (
            rec["bbox"][2] < w or rec["bbox"][0] > e or rec["bbox"][3] < s or rec["bbox"][1] > n
        )
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
    catalog_path.write_text(yaml.dump(catalog, default_flow_style=False, sort_keys=False))


def _resolve_record(
    index: list[dict[str, Any]],
    dataset_name: str | None,
    aoi_gdf: Any,
) -> dict[str, Any]:
    """Find the best index record for the AOI."""
    if dataset_name is not None:
        matches = [r for r in index if r["dataset_name"] == dataset_name]
        if not matches:
            raise ValueError(
                f"Dataset {dataset_name!r} not found in NOAA DEM index. "
                f"Available: {[r['dataset_name'] for r in index]}"
            )
        return matches[0]

    aoi_4326 = aoi_gdf.to_crs(epsg=4326)
    bbox = tuple(aoi_4326.total_bounds)
    candidates = query_overlapping(index, bbox)
    if not candidates:
        raise ValueError(
            f"No NOAA DEM overlaps AOI bbox {bbox}. "
            f"Run 'coastal-calibration update-dem-index' to refresh the index."
        )
    return select_best(candidates, bbox)


def _localize_vrt(
    vrt_url: str,
    local_path: Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> int:
    """Download a remote VRT, rewrite tile refs, and optionally filter by bbox.

    GDAL must issue an HTTP HEAD request for *every* tile listed in a
    VRT to determine its internal block layout — even when ``-te``
    restricts the output extent.  For CONUS-wide VRTs with ~500 tiles
    this means hundreds of round-trips before any pixel data flows.

    This helper:

    1. Downloads the VRT XML in a single HTTP request.
    2. Rewrites each relative ``<SourceFilename>`` to an absolute
       ``/vsicurl/`` URL.
    3. When *bbox* is given, removes every tile whose geographic extent
       does not overlap, reducing the VRT from ~500 tiles to typically
       2-6.  ``gdalwarp`` then only issues HEAD requests for the tiles
       it will actually read.

    Parameters
    ----------
    vrt_url
        HTTPS URL of the remote VRT.
    local_path
        Where to save the rewritten VRT.
    bbox
        ``(west, south, east, north)`` in the VRT's native CRS.
        Tiles with no overlap are removed.  When ``None`` all tiles
        are kept.

    Returns
    -------
    int
        Number of tiles retained in the written VRT.
    """
    import urllib.request
    import xml.etree.ElementTree as ET

    base_url = vrt_url.rsplit("/", 1)[0]

    with urllib.request.urlopen(vrt_url) as resp:
        root = ET.fromstring(resp.read())  # noqa: S314

    # Rewrite relative tile paths to absolute /vsicurl/ URLs.
    for elem in root.iter("SourceFilename"):
        if elem.get("relativeToVRT", "0") == "1":
            elem.text = f"/vsicurl/{base_url}/{elem.text}"
            elem.set("relativeToVRT", "0")

    # Filter tiles to only those overlapping the bbox.
    kept = 0
    if bbox is not None:
        gt_text = root.findtext("GeoTransform")
        if gt_text is not None:
            gt = [float(v) for v in gt_text.split(",")]
            bw, bs, be, bn = bbox
            for band in root.findall("VRTRasterBand"):
                for src in list(band):
                    if src.tag not in ("SimpleSource", "ComplexSource"):
                        continue
                    dst_rect = src.find("DstRect")
                    if dst_rect is None:
                        continue
                    x_off = float(dst_rect.get("xOff", "0"))
                    y_off = float(dst_rect.get("yOff", "0"))
                    x_size = float(dst_rect.get("xSize", "0"))
                    y_size = float(dst_rect.get("ySize", "0"))

                    # Convert pixel coordinates to geographic coordinates
                    # using the VRT's GeoTransform.
                    tile_west = gt[0] + x_off * gt[1]
                    tile_east = gt[0] + (x_off + x_size) * gt[1]
                    tile_north = gt[3] + y_off * gt[5]
                    tile_south = gt[3] + (y_off + y_size) * gt[5]

                    # Remove tiles with no overlap.
                    if tile_east <= bw or tile_west >= be or tile_north <= bs or tile_south >= bn:
                        band.remove(src)
                    else:
                        kept += 1
        else:
            # No GeoTransform — keep all tiles.
            kept = sum(
                1
                for band in root.findall("VRTRasterBand")
                for src in band
                if src.tag in ("SimpleSource", "ComplexSource")
            )
    else:
        kept = sum(
            1
            for band in root.findall("VRTRasterBand")
            for src in band
            if src.tag in ("SimpleSource", "ComplexSource")
        )

    ET.ElementTree(root).write(local_path, xml_declaration=True, encoding="UTF-8")
    return kept


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

    Uses ``gdalwarp`` with GDAL ``/vsicurl/`` to remotely access the
    dataset's VRT, clips to the AOI polygon, and writes a local GeoTIFF
    plus a HydroMT data-catalog YAML.  Only the pixels within the AOI
    extent are transferred over the network.

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
    from pathlib import Path

    import geopandas as gpd

    from coastal_calibration.utils._gdal import clip_to_aoi, compute_aoi_coverage

    _log = log if log is not None else logger.info

    aoi = Path(aoi)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi_gdf = gpd.read_file(aoi)
    if aoi_gdf.crs is None:
        raise ValueError(f"AOI file {aoi} has no CRS")

    record = _resolve_record(load_index(), dataset_name, aoi_gdf)
    target_epsg = record["epsg"]

    _log(
        f"Selected DEM: {record['dataset_name']}"
        f" ({record['title']}, {record['resolution_m']:.0f} m, {record['year']})"
    )

    vrt_url = get_vrt_url(record)
    _log(f"VRT: {vrt_url}")

    sub_dir = output_dir / "_noaa_temp"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Write AOI as a GeoJSON cutline for gdalwarp.
    aoi_4326 = aoi_gdf.to_crs(epsg=4326)
    geo_bounds = aoi_4326.total_bounds  # (w, s, e, n)
    buffered_bbox = (
        geo_bounds[0] - buffer_deg,
        geo_bounds[1] - buffer_deg,
        geo_bounds[2] + buffer_deg,
        geo_bounds[3] + buffer_deg,
    )
    cutline_path = sub_dir / "cutline.geojson"
    aoi_4326.to_file(cutline_path, driver="GeoJSON")

    # Download the remote VRT XML once, rewrite tile paths to
    # /vsicurl/ URLs, and drop tiles outside the buffered bbox.
    # This avoids the per-tile HEAD requests that make large remote
    # VRTs (CONUS-wide, ~500 tiles) extremely slow: gdalwarp now
    # only issues HEAD + GET for the 2-6 tiles it actually reads.
    local_vrt = sub_dir / "noaa_dem.vrt"
    n_tiles = _localize_vrt(vrt_url, local_vrt, bbox=buffered_bbox)
    _log(f"VRT localized: {local_vrt.stat().st_size / 1e3:.0f} KB, {n_tiles} tiles")

    # Clip the local VRT to AOI polygon with gdalwarp.
    # Passing -te limits which remote tiles GDAL touches.
    tif_name = f"{catalog_name}.tif"
    geotiff_path = output_dir / tif_name
    _log(f"Clipping to AOI via gdalwarp: {geotiff_path}")
    clip_to_aoi(
        local_vrt,
        cutline_path,
        geotiff_path,
        nodata="nan",
        output_type="Float32",
        target_extent=buffered_bbox,
        target_extent_srs="EPSG:4326",
    )
    _log(f"GeoTIFF written ({geotiff_path.stat().st_size / 1e6:.1f} MB)")

    # Check data coverage within AOI.
    coverage_pct = compute_aoi_coverage(geotiff_path, cutline_path)

    # Clean up temporary files.
    for f in [cutline_path, local_vrt]:
        f.unlink(missing_ok=True)
    sub_dir.rmdir()

    _log(f"Data coverage within AOI: {coverage_pct:.1f}%")
    if coverage_pct < 10.0:
        geotiff_path.unlink(missing_ok=True)
        raise ValueError(
            f"NOAA DEM '{record['dataset_name']}' has only {coverage_pct:.1f}% "
            f"data coverage within the AOI (minimum 10% required). "
            f"The VRT may lack data for this area. Try a different dataset "
            f"or use 'prepare-topobathy' to fetch from NWS icechunk instead."
        )

    catalog_path = output_dir / f"{catalog_name}_catalog.yml"
    _write_catalog(catalog_path, tif_name, catalog_name, target_epsg)
    _log(f"Catalog written: {catalog_path}")

    return geotiff_path, catalog_path, catalog_name


# ------------------------------------------------------------------
# Index rebuild from S3 (CLI / offline use)
# ------------------------------------------------------------------


def _pick_vrt(vrt_files: list[str]) -> tuple[str, int]:
    """Select the best VRT filename and its EPSG code."""
    import re

    # Prefer geographic CRS VRTs (EPSG:4269 or 4326).
    for vf in vrt_files:
        m = re.search(r"EPSG-(\d+)", vf)
        if m and int(m.group(1)) in (4269, 4326):
            return vf, int(m.group(1))

    # Fall back to first available VRT.
    vrt_filename = vrt_files[0]
    m = re.search(r"EPSG-(\d+)", vrt_filename)
    return vrt_filename, int(m.group(1)) if m else 4269


def _parse_stac_collection(
    s3: Any,
    bucket: str,
    ds_name: str,
) -> dict[str, Any] | None:
    """Read a single dataset's STAC collection and VRT list into an index entry."""
    import re

    stac_prefix = f"dem/{ds_name}/stac/"
    try:
        stac_resp = s3.list_objects_v2(Bucket=bucket, Prefix=stac_prefix)
        stac_files = [
            obj["Key"]
            for obj in stac_resp.get("Contents", [])
            if obj["Key"].endswith(".json") and "collection" in obj["Key"]
        ]
        if not stac_files:
            return None

        body = s3.get_object(Bucket=bucket, Key=stac_files[0])["Body"].read()
        collection = json.loads(body)
    except Exception:
        logger.debug("Skipping %s (no STAC collection)", ds_name)
        return None

    extent = collection.get("extent", {}).get("spatial", {}).get("bbox", [[]])
    if not extent or not extent[0]:
        return None

    title = collection.get("title", ds_name)
    description = collection.get("description", "")

    is_topobathy = any(
        kw in (title + description).lower()
        for kw in ("topobathy", "topo-bathy", "bathymetric topographic")
    )

    year_match = re.search(r"(\d{4})", ds_name)
    year = int(year_match.group(1)) if year_match else 0

    vrt_resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"dem/{ds_name}/", Delimiter="/")
    vrt_files = [
        obj["Key"].split("/")[-1]
        for obj in vrt_resp.get("Contents", [])
        if obj["Key"].endswith(".vrt")
    ]
    if not vrt_files:
        return None

    vrt_filename, epsg = _pick_vrt(vrt_files)

    res_match = re.search(r"(\d+)[_-]?(?:arc)?[_-]?sec", title.lower())
    resolution_m = int(res_match.group(1)) * 30.0 / 9 if res_match else 1.0

    return {
        "dataset_name": ds_name,
        "title": title,
        "bbox": extent[0][:4],
        "resolution_m": round(resolution_m, 1),
        "year": year,
        "is_topobathy": is_topobathy,
        "epsg": epsg,
        "vrt_filename": vrt_filename,
    }


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
        entry = _parse_stac_collection(s3, bucket, ds_name)
        if entry is not None:
            entries.append(entry)

    logger.info("Built index with %d entries", len(entries))
    return entries
