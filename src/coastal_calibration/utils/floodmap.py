"""Flood depth map generation from SFINCS output.

Downscales coarse-grid water surface elevations onto a high-resolution
DEM to produce a Cloud Optimized GeoTIFF (COG) of maximum flood depth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

__all__ = ["create_flood_depth_map"]

_log = logging.getLogger(__name__)


def _ensure_overviews(tif_path: Path, log: Any) -> None:
    """Build overviews on a GeoTIFF if it has none.

    ``downscale_floodmap`` opens the DEM with ``overview_level=0``
    which requires at least one overview to exist.  This is a
    workaround for a bug in hydromt-sfincs where the default
    ``overview_level`` should be ``None`` (full resolution) rather
    than ``0`` (first overview).
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(tif_path) as src:
        if src.overviews(1):
            return

    log(f"GeoTIFF has no overviews, building them: {tif_path.name}")

    # Compute overview levels based on a 256-pixel target block size.
    # The upstream ``build_overviews`` auto-mode uses the file's own
    # blockxsize, which for untiled GeoTIFFs equals the full width
    # and yields 0 levels.
    with rasterio.open(tif_path, "r+") as src:
        min_dim = min(src.width, src.height)
        levels = []
        factor = 2
        while min_dim // factor >= 256:
            levels.append(factor)
            factor *= 2
        if not levels:
            levels = [2]
        log(f"Building {len(levels)} overview levels: {levels}")
        src.build_overviews(levels, Resampling.average)
        src.update_tags(ns="rio_overview", resampling="average")


def create_flood_depth_map(
    model_root: Path | str,
    dem_path: Path | str,
    output_path: Path | str | None = None,
    *,
    index_path: Path | str | None = None,
    create_index: bool = True,
    hmin: float = 0.05,
    reproj_method: str = "nearest",
    nrmax: int = 2000,
    model: SfincsModel | None = None,
    log: Any = None,
) -> Path:
    """Create a downscaled flood depth map from SFINCS output.

    Reads the maximum water surface elevation (``zsmax``) from the SFINCS
    map output, optionally builds an index COG that maps DEM pixels to
    SFINCS grid cells, then calls :func:`hydromt_sfincs.utils.downscale_floodmap`
    to produce a high-resolution flood depth GeoTIFF.

    Parameters
    ----------
    model_root : Path or str
        Path to the SFINCS model directory (must contain ``sfincs_map.nc``).
    dem_path : Path or str
        Path to a high-resolution DEM GeoTIFF covering the model domain.
    output_path : Path or str, optional
        Output flood depth COG path.  Defaults to
        ``<model_root>/floodmap_hmax.tif``.
    index_path : Path or str, optional
        Path for the index COG (DEM pixels → SFINCS cell mapping).
        Defaults to ``<model_root>/floodmap_index.tif``.
    create_index : bool
        If True (default) and *index_path* does not yet exist, generate
        the index COG via :func:`make_index_cog`.  The index speeds up
        the downscaling significantly for large DEMs.
    hmin : float
        Minimum flood depth (m) to classify a pixel as flooded.
    reproj_method : str
        Reprojection method (``"nearest"`` or ``"bilinear"``).
    nrmax : int
        Maximum cells per processing block (controls peak memory).
    model : SfincsModel, optional
        An already-loaded :class:`SfincsModel` instance.  When provided,
        ``model_root`` is still used to resolve default output paths but
        the model is **not** re-read from disk.
    log : callable, optional
        Logging callback ``(message, level)``; falls back to the module
        logger when *None*.

    Returns
    -------
    Path
        Path to the generated flood depth COG.

    Raises
    ------
    FileNotFoundError
        If the DEM or ``sfincs_map.nc`` cannot be found.
    KeyError
        If ``zsmax`` is not present in the SFINCS map output.
    """
    from hydromt_sfincs.utils import downscale_floodmap
    from hydromt_sfincs.workflows.downscaling import make_index_cog

    model_root = Path(model_root)
    dem_path = Path(dem_path)

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    map_file = model_root / "sfincs_map.nc"
    if not map_file.exists():
        raise FileNotFoundError(f"SFINCS map output not found: {map_file}")

    output_path = model_root / "floodmap_hmax.tif" if output_path is None else Path(output_path)
    index_path = model_root / "floodmap_index.tif" if index_path is None else Path(index_path)

    def _info(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            _log.info(msg)

    # ── Load model and read output ──────────────────────────────
    if model is None:
        from coastal_calibration.stages._hydromt_compat import apply_all_patches

        apply_all_patches()

        from hydromt_sfincs import SfincsModel as _Sfincs

        # Use "r+" (same as the pipeline) so all components are writable
        # and the quadtree grid loads correctly.
        model = _Sfincs(root=str(model_root), mode="r+")
        model.read()

    model.output.read()

    if "zsmax" not in model.output.data:
        raise KeyError(
            "Variable 'zsmax' not found in SFINCS map output. "
            "Ensure SFINCS was configured to write zsmax (storzsmax = 1 in sfincs.inp)."
        )
    zsmax = model.output.data["zsmax"]
    _info(f"Loaded zsmax from {map_file}")

    # ── Optionally create index COG ─────────────────────────────
    if create_index and not index_path.exists():
        _info(f"Creating index COG: {index_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        make_index_cog(
            model=model,
            indices_fn=str(index_path),
            topobathy_fn=str(dem_path),
            nrmax=nrmax,
        )
        _ensure_overviews(index_path, _info)
        _info(f"Index COG created ({index_path.stat().st_size / 1e6:.1f} MB)")

    indices_arg: str | None = str(index_path) if index_path.exists() else None

    # ── Ensure DEM has overviews ────────────────────────────────
    # ``downscale_floodmap`` opens the DEM with ``overview_level=0``
    # (first overview) by default.  If the file has no overviews
    # rasterio raises ``RasterioIOError``.  Build them once so the
    # downstream call succeeds.
    _ensure_overviews(dem_path, _info)

    # ── Downscale ───────────────────────────────────────────────
    _info("Downscaling flood depth map")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    downscale_floodmap(
        zsmax=zsmax,  # type: ignore[invalid-argument-type]
        dep=str(dem_path),
        indices=indices_arg,  # type: ignore[invalid-argument-type]
        hmin=hmin,
        floodmap_fn=str(output_path),
        reproj_method=reproj_method,
        nrmax=nrmax,
    )

    size_mb = output_path.stat().st_size / 1e6
    _info(f"Flood depth map written: {output_path} ({size_mb:.1f} MB)")

    return output_path
