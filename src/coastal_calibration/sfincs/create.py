"""SFINCS model creation stages using HydroMT-SFINCS Python API.

These stages build a new SFINCS quadtree model from an AOI polygon.
All stages subclass :class:`CreateStage` and accept a
:class:`~coastal_calibration.config.create_schema.SfincsCreateConfig`.

The HydroMT ``SfincsModel`` instance is shared between stages via a
module-level registry keyed by config ``id``, following the same pattern
as :mod:`coastal_calibration.sfincs.stages`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import shapely
from shapely import MultiPolygon, Point

from coastal_calibration.logging import (
    WorkflowMonitor,
    configure_logger,
    generate_log_path,
    silence_third_party_loggers,
    suppress_hydromt_output,
)
from coastal_calibration.runner import WorkflowResult
from coastal_calibration.sfincs._hydromt_compat import apply_all_patches
from coastal_calibration.utils import utc_now

apply_all_patches()

if TYPE_CHECKING:
    import geopandas as gpd
    from hydromt_sfincs import SfincsModel
    from numpy.typing import NDArray
    from shapely import Polygon

    from coastal_calibration.config.create_schema import SfincsCreateConfig


# ---------------------------------------------------------------------------
# Shared model instance management
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[int, SfincsModel] = {}


def _set_model(config: SfincsCreateConfig, model: SfincsModel) -> None:
    """Store the SfincsModel instance for the given config."""
    _MODEL_REGISTRY[id(config)] = model


def _get_model(config: SfincsCreateConfig) -> SfincsModel:
    """Retrieve the SfincsModel instance for the given config.

    Raises
    ------
    RuntimeError
        If no model has been registered (e.g. ``create_grid`` was skipped).
    """
    try:
        return _MODEL_REGISTRY[id(config)]
    except KeyError:
        raise RuntimeError(
            "No SfincsModel found in registry.  "
            "Ensure the 'create_grid' stage runs before other creation stages."
        ) from None


#: Config each stage consumes, as attribute paths on
#: :class:`SfincsCreateConfig`.  ``--start-from`` skips earlier stages and
#: reuses their outputs, so these are fingerprinted to catch a config edit
#: that would pair those stale outputs with new settings.  Entries name only
#: what changes a stage's result, down to the field where a whole section
#: would refuse a resume the user is entitled to: ``create_mask`` and
#: ``create_boundary`` split :class:`MaskConfig` between them so that tuning
#: ``boundary_zmax`` does not read as a mask change.
#:
#: Known limit: ``data_catalog`` covers which catalogs are listed, not what is
#: inside them, so editing a catalog file in place is invisible here.
_STAGE_CONFIG_DEPS: dict[str, tuple[str, ...]] = {
    "create_grid": ("aoi", "aoi_simplify_neck_m", "grid", "data_catalog"),
    # ``aoi_key`` is the identity the fetcher already caches under, so a
    # setting that changes the merge but not the download does not force a
    # re-fetch.  ``download_dir`` decides where the catalogs are looked for.
    "create_fetch_data": ("aoi_key", "download_dir"),
    "create_elevation": ("elevation", "data_catalog"),
    "create_mask": ("mask.zmin", "mask.keep_largest_only"),
    "create_boundary": ("mask.boundary_zmax", "mask.reset_bounds"),
    "create_discharge": ("river_discharge",),
    "create_obs": (
        "add_noaa_gages",
        "merge_observations",
        "observation_locations_file",
        "observation_points",
        "obs_snap_depth_threshold",
        "obs_snap_search_radius_m",
    ),
    # Merges the elevation datasets a second time, so it moves with them.
    "create_subgrid": ("subgrid", "elevation", "data_catalog"),
    "create_write": (),
}

#: The only stage that puts the model on disk.
_WRITE_STAGE = "create_write"


def _canonical(value: Any) -> Any:
    """Represent *value* so that equal configs hash equally.

    Files stand for their contents at any depth, not for where they sit, so
    editing a refinement polygon or a reclassification table in place counts
    as a change while relocating a project does not.  A path with no file
    behind it keeps a distinct shape of its own rather than borrowing the
    representation of a file that happens to exist.
    """
    if isinstance(value, Path):
        if value.is_file():
            return {"sha256": hashlib.sha256(value.read_bytes()).hexdigest()[:16]}
        return {"missing": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in cast("dict[Any, Any]", value).items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in cast("list[Any]", value)]
    return value


def _user_data_libs(cfg: SfincsCreateConfig) -> list[str]:
    """Return the configured catalogs, minus the ones the fetch stage adds.

    ``create_fetch_data`` appends a catalog per fetched raster to
    ``data_libs`` in memory, so the list differs between a fresh run and a
    resume that has not reached that stage.  Only the user's own entries are
    stable enough to compare, and they are the ones that decide where a
    dataset with no ``source`` resolves from.
    """
    derived = f"{cfg.effective_download_dir}/"
    return [lib for lib in cfg.data_catalog.data_libs if not lib.startswith(derived)]


def _stage_fingerprint(cfg: SfincsCreateConfig, stage: str) -> str:
    """Hash the config *stage* consumes.

    Lists keep their order, so reordering ``elevation.datasets`` registers as
    a change.
    """
    payload: dict[str, Any] = {}
    for dep in _STAGE_CONFIG_DEPS.get(stage, ()):
        if dep == "data_catalog":
            payload[dep] = _user_data_libs(cfg)
            continue
        value: Any = cfg
        for part in dep.split("."):
            value = getattr(value, part)
        payload[dep] = _canonical(value)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _elevation_list(cfg: SfincsCreateConfig) -> list[dict[str, Any]]:
    """Build the ``elevation_list`` hydromt-sfincs merges datasets from.

    ``offset`` is only emitted when non-zero, so configs that do not set it
    produce exactly the request they did before. hydromt-sfincs applies the
    offset before the ``zmin`` filter, which is why ``zmin`` is documented
    against the shifted elevations.

    Shared by the elevation and subgrid stages so the two cannot drift apart
    in which datasets they merge, in what order, or with what offset. They
    still differ in ways this does not control: the subgrid builder merges
    with ``buffer_cells=0`` and fills nodata by inverse-distance
    interpolation, so grid bed and subgrid bed can still disagree at dataset
    seams and holes.
    """
    entries: list[dict[str, Any]] = []
    for d in cfg.elevation.datasets:
        entry: dict[str, Any] = {"elevation": d.name, "zmin": d.zmin}
        if d.offset:
            entry["offset"] = d.offset
        entries.append(entry)
    return entries


def _clear_model(config: SfincsCreateConfig) -> None:
    """Remove the SfincsModel from the registry."""
    _MODEL_REGISTRY.pop(id(config), None)


def _load_existing_model(config: SfincsCreateConfig) -> None:
    """Load an existing SfincsModel from *config.output_dir* into the registry.

    Used when resuming from a later stage (``--start-from``) where
    ``create_grid`` is skipped but a model already exists on disk.
    """
    if id(config) in _MODEL_REGISTRY:
        return

    from hydromt_sfincs import SfincsModel

    sf = SfincsModel(
        root=str(config.output_dir),
        mode="r+",
        data_libs=config.data_catalog.data_libs,
    )
    sf.read()
    _set_model(config, sf)


# ---------------------------------------------------------------------------
# CreateStage ABC
# ---------------------------------------------------------------------------


class CreateStage(ABC):
    """Abstract base class for SFINCS model creation stages.

    Mirrors the :class:`~coastal_calibration.base.WorkflowStage`
    interface but is decoupled from
    :class:`~coastal_calibration.config.schema.CoastalCalibConfig`.
    """

    name: str = "create_base"
    description: str = "Base creation stage"

    def __init__(
        self,
        config: SfincsCreateConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        self.config = config
        self.monitor = monitor

    def _log(self, message: str, level: str = "info") -> None:
        """Log *message* via the monitor if available."""
        if self.monitor:
            getattr(self.monitor, level)(f"  {message}")

    def _update_substep(self, substep: str) -> None:
        """Update the current substep label in the monitor."""
        if self.monitor:
            self.monitor.update_substep(self.name, substep)

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the stage and return a result dict."""

    def validate(self) -> list[str]:
        """Validate stage prerequisites.  Return a list of errors."""
        return []


# ---------------------------------------------------------------------------
# Convenience base that exposes ``self.sfincs``
# ---------------------------------------------------------------------------


class _CreateStageBase(CreateStage):
    """Convenience base providing ``self.sfincs`` via the model registry."""

    @property
    def sfincs(self) -> SfincsModel:
        return _get_model(self.config)

    @abstractmethod
    def run(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Concrete stages
# ---------------------------------------------------------------------------


class CreateGridStage(CreateStage):
    """Initialize a new ``SfincsModel`` and generate the quadtree grid."""

    name = "create_grid"
    description = "Create SFINCS grid from AOI"

    def _simplify_aoi_thin_necks(self, aoi_path: Path, neck_m: float, output_dir: Path) -> Path:
        """Erode → keep largest → dilate to drop thin-neck-connected sub-regions.

        Returns the path to the cleaned-AOI GeoJSON written under
        ``output_dir``. Morphology runs in a metric CRS (UTM-projected
        when the input is geographic) so the *neck_m* threshold is
        meaningful in meters; the output is reprojected back to the
        original CRS for compatibility with downstream consumers.
        """
        import geopandas as gpd
        from pyproj import CRS

        gdf = gpd.read_file(str(aoi_path))
        original_crs = gdf.crs
        if original_crs is None or CRS.from_user_input(original_crs).is_geographic:
            centroid = gdf.geometry.union_all().centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            metric_epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone
            gdf_metric = gdf.to_crs(epsg=metric_epsg)
        else:
            gdf_metric = gdf

        poly = gdf_metric.union_all()
        eroded = poly.buffer(-neck_m)
        if eroded.is_empty:
            msg = (
                f"AOI vanished after erosion by {neck_m} m. Reduce "
                f"aoi_simplify_neck_m or set it to 0 to disable cleanup."
            )
            raise ValueError(msg)

        # A negative buffer can split the polygon, so MultiPolygon is reachable
        # even though shapely types buffer() as returning a Polygon.
        if eroded.geom_type == "MultiPolygon":  # pyright: ignore[reportUnnecessaryComparison]
            # ``Polygon``/``MultiPolygon`` are TYPE_CHECKING-only
            # imports; the annotation and the cast() are erased at
            # runtime so the symbol form here costs nothing and lets
            # vulture see the import as used.
            parts: list[Polygon] = list(cast(MultiPolygon, eroded).geoms)
            largest = max(parts, key=lambda p: float(p.area))
            n_dropped = len(parts) - 1
            dropped_area = sum(p.area for p in parts) - largest.area
            self._log(
                f"AOI cleanup: erosion(-{neck_m:.0f} m) split into "
                f"{len(parts)} parts; dropped {n_dropped} smaller "
                f"piece(s) totalling {dropped_area / 1e6:.2f} km²"
            )
        else:
            largest = eroded
            self._log(
                f"AOI cleanup: erosion(-{neck_m:.0f} m) produced a single "
                "piece (no thin necks detected); proceeding with the "
                "morphologically-closed polygon"
            )

        cleaned = largest.buffer(neck_m)
        cleaned_gdf = gpd.GeoDataFrame(geometry=[cleaned], crs=gdf_metric.crs)
        if original_crs is not None and CRS.from_user_input(original_crs) != gdf_metric.crs:
            cleaned_gdf = cleaned_gdf.to_crs(original_crs)

        out_path = output_dir / "aoi_cleaned.geojson"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()  # GeoJSON driver appends if file exists
        cleaned_gdf.to_file(out_path, driver="GeoJSON")
        return out_path

    def _resolve_grid_crs(self, grid_crs_str: str) -> Any:
        """Return the target ``pyproj.CRS`` for the grid, auto-deriving UTM."""
        from pyproj import CRS

        if grid_crs_str != "utm":
            return CRS.from_user_input(grid_crs_str)

        import geopandas as gpd

        aoi_gdf: gpd.GeoDataFrame = gpd.read_file(str(self.config.aoi))
        centroid = cast("Point", aoi_gdf.to_crs(4326).geometry.centroid.iloc[0])
        utm_zone = int((centroid.x + 180) / 6) + 1
        grid_epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone
        return CRS.from_epsg(grid_epsg)

    def _prepare_refinement_polygon(self, ref: Any, target_crs: Any) -> Any:
        """Return a (reprojected, buffered, exploded) GDF for one refinement entry.

        Returns ``None`` when the polygon collapses to empty after buffering
        — the caller should skip empty entries silently.
        """
        import geopandas as gpd

        gdf: gpd.GeoDataFrame = gpd.read_file(ref.polygon)
        if gdf.crs is not None and gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        if ref.buffer_m != 0.0:
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.buffer(ref.buffer_m)
            empty = gdf.geometry.is_empty
            if empty.any():
                n_dropped = int(empty.sum())
                self._log(
                    f"Refinement polygon '{ref.polygon.name}': "
                    f"{n_dropped} geometry(ies) collapsed to empty after "
                    f"buffer_m={ref.buffer_m} m; dropped",
                    "warning",
                )
                gdf = gdf[~empty]

        # Inward buffering of polygons with thin necks can split a
        # single Polygon into a MultiPolygon. hydromt-sfincs's
        # refinement-polygon code only handles single Polygons
        # (it calls ``.exterior`` directly), so explode here to
        # hand each component piece in as its own row.
        exploded = gdf.explode(index_parts=False, ignore_index=True)
        if len(exploded) > len(gdf):
            self._log(
                f"Refinement polygon '{ref.polygon.name}': split into "
                f"{len(exploded)} component(s) after buffer_m={ref.buffer_m} m"
            )
            gdf = exploded
        if gdf.empty:
            return None
        gdf["refinement_level"] = ref.level
        return gdf

    def _build_refinement_gdf(self) -> Any:
        """Build the combined refinement GeoDataFrame from config, or None."""
        cfg = self.config
        if not cfg.grid.refinement:
            return None

        import geopandas as gpd
        import pandas as pd

        target_crs = self._resolve_grid_crs(cfg.grid.crs)

        parts: list[gpd.GeoDataFrame] = []
        for ref in cfg.grid.refinement:
            gdf = self._prepare_refinement_polygon(ref, target_crs)
            if gdf is not None:
                parts.append(gdf)

        if not parts:
            self._log(
                "All refinement polygons collapsed after buffering; building uniform grid",
                "warning",
            )
            return None

        self._log(
            f"Refinement: {len(parts)} polygon(s), "
            f"max level {max(r.level for r in cfg.grid.refinement)}"
        )
        return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True))

    def run(self) -> dict[str, Any]:
        """Initialize SfincsModel and generate the quadtree grid."""
        from hydromt_sfincs import SfincsModel

        cfg = self.config

        self._update_substep("Initializing SfincsModel")
        sf = SfincsModel(
            root=str(cfg.output_dir),
            mode="w+",
            write_gis=True,
            data_libs=cfg.data_catalog.data_libs,
        )

        self._update_substep("Creating quadtree grid from AOI")
        self._log(f"AOI: {cfg.aoi}")
        self._log(f"Resolution: {cfg.grid.resolution} m, CRS: {cfg.grid.crs}")

        # Optional thin-neck cleanup: erode → keep largest → dilate.
        # Drops sub-regions that connect to the main domain only via
        # narrow necks; those regions are typically poorly resolved at
        # the SFINCS quadtree base resolution and produce visible
        # water-level artifacts. The cleaned polygon is written next
        # to the model output and reused by the discharge stage.
        aoi_for_region: Path | str = str(cfg.aoi)
        if cfg.aoi_simplify_neck_m > 0:
            cleaned_path = self._simplify_aoi_thin_necks(
                cfg.aoi, cfg.aoi_simplify_neck_m, cfg.output_dir
            )
            aoi_for_region = str(cleaned_path)
            self._log(f"Using cleaned AOI: {cleaned_path.name}")

        refinement_gdf = self._build_refinement_gdf()

        with suppress_hydromt_output():
            sf.quadtree_grid.create_from_region(
                region={"geom": aoi_for_region},
                res=cfg.grid.resolution,
                crs=cfg.grid.crs,
                rotated=cfg.grid.rotated,
                refinement_polygons=refinement_gdf,
            )

        _set_model(cfg, sf)

        self._log("Grid created successfully")
        return {"status": "completed"}


class CreateFetchDataStage(_CreateStageBase):
    """Fetch elevation and land-cover data for the AOI.

    Dispatches to source-specific fetchers based on the ``source``
    field of each :class:`ElevationDataset` and the ``lulc_source``
    field of :class:`SubgridConfig`.
    """

    name = "create_fetch_data"
    description = "Fetch elevation and land cover data for AOI"

    def _register_catalog(self, catalog_path: Path) -> None:
        """Add *catalog_path* to both config data_libs and the live model."""
        path_str = str(catalog_path.resolve())
        if path_str not in self.config.data_catalog.data_libs:
            self.config.data_catalog.data_libs.append(path_str)
        self.sfincs.data_catalog.from_yml(path_str)

    def _fetch_elevation(
        self, fetched: list[str], rasters: dict[str, str], offsets: dict[str, float]
    ) -> None:
        """Fetch elevation datasets that have a ``source`` set.

        Records each raster's path in *rasters*, in config order, so the run
        workflow can find the DEM to downscale onto without the user having
        to repeat a path only this workflow knows.
        """
        cfg = self.config
        dl_dir = cfg.effective_download_dir

        for ds in cfg.elevation.datasets:
            if ds.source is None:
                continue

            catalog_name = ds.name
            tif = dl_dir / f"{catalog_name}.tif"
            cat = dl_dir / f"{catalog_name}_catalog.yml"

            # Resumability: skip if already fetched.
            if tif.exists() and cat.exists():
                self._log(f"Reusing existing {tif.name}")
                self._register_catalog(cat)
                fetched.append(catalog_name)
                rasters[catalog_name] = str(tif)
                offsets[catalog_name] = ds.offset
                continue

            if ds.source == "nws_30m":
                from coastal_calibration.data.topobathy_nws import fetch_topobathy

                if ds.coastal_domain is None:
                    raise ValueError(
                        f"elevation.datasets[{ds.name}].coastal_domain is required "
                        f"when source is 'nws_30m'"
                    )
                self._update_substep(f"Fetching NWS topobathy for '{catalog_name}'")
                _, cat_path, _ = fetch_topobathy(
                    domain=ds.coastal_domain,
                    aoi=cfg.aoi,
                    output_dir=dl_dir,
                    buffer_deg=0.1,
                    catalog_name=catalog_name,
                    log=self._log,
                )
            elif ds.source == "noaa_3m":
                from coastal_calibration.data.topobathy_noaa import fetch_noaa_dem

                self._update_substep(f"Fetching NOAA DEM for '{catalog_name}'")
                _, cat_path, _ = fetch_noaa_dem(
                    aoi=cfg.aoi,
                    output_dir=dl_dir,
                    dataset_name=ds.noaa_dataset,
                    buffer_deg=0.1,
                    catalog_name=catalog_name,
                    log=self._log,
                )
            elif ds.source == "copdem_30m":
                from coastal_calibration.data.copdem import fetch_copdem30

                self._update_substep(f"Fetching Copernicus DEM 30m for '{catalog_name}'")
                _, cat_path, _ = fetch_copdem30(
                    aoi=cfg.aoi,
                    output_dir=dl_dir,
                    catalog_name=catalog_name,
                    log=self._log,
                )
            elif ds.source == "noaa_crm":
                from coastal_calibration.data.crm_noaa import fetch_crm

                self._update_substep(f"Fetching NOAA CRM topobathy for '{catalog_name}'")
                _, cat_path, _ = fetch_crm(
                    aoi=cfg.aoi,
                    output_dir=dl_dir,
                    catalog_name=catalog_name,
                    log=self._log,
                )
            elif ds.source == "gebco_15arcs":
                from coastal_calibration.data.gebco_wms import fetch_gebco

                self._update_substep(f"Fetching GEBCO 15-arcsec bathymetry for '{catalog_name}'")
                _, cat_path, _ = fetch_gebco(
                    aoi=cfg.aoi,
                    output_dir=dl_dir,
                    catalog_name=catalog_name,
                    log=self._log,
                )
            else:
                raise ValueError(f"Unknown elevation source: {ds.source!r}")

            self._register_catalog(cat_path)
            fetched.append(catalog_name)
            rasters[catalog_name] = str(tif)
            offsets[catalog_name] = ds.offset

    def _fetch_lulc(self, fetched: list[str]) -> None:
        """Fetch the LULC dataset if ``lulc_source`` is set."""
        cfg = self.config
        if cfg.subgrid.lulc_source is None:
            return

        dl_dir = cfg.effective_download_dir
        lulc_name = cfg.subgrid.lulc_dataset
        tif = dl_dir / f"{lulc_name}.tif"
        cat = dl_dir / f"{lulc_name}_catalog.yml"

        if tif.exists() and cat.exists():
            self._log(f"Reusing existing {tif.name}")
            self._register_catalog(cat)
            fetched.append(lulc_name)
            return

        if cfg.subgrid.lulc_source == "esa_worldcover":
            from coastal_calibration.data.esa_worldcover import fetch_esa_worldcover

            self._update_substep(f"Fetching ESA WorldCover for '{lulc_name}'")
            _, cat_path, _ = fetch_esa_worldcover(
                aoi=cfg.aoi,
                output_dir=dl_dir,
                catalog_name=lulc_name,
                log=self._log,
            )
        else:
            raise ValueError(f"Unknown LULC source: {cfg.subgrid.lulc_source!r}")

        self._register_catalog(cat_path)
        fetched.append(lulc_name)

    def run(self) -> dict[str, Any]:
        """Fetch elevation and LULC datasets for the AOI."""
        fetched: list[str] = []
        elevation_rasters: dict[str, str] = {}
        # The flood-map stage downscales onto one of these rasters, which is
        # the raw fetched file -- the model bed carries the offset, so the
        # stage has to re-apply it or every depth is wrong by that much.
        elevation_offsets: dict[str, float] = {}
        self._fetch_elevation(fetched, elevation_rasters, elevation_offsets)
        self._fetch_lulc(fetched)
        self._log(f"Fetched {len(fetched)} dataset(s): {fetched}")
        return {
            "status": "completed",
            "fetched": fetched,
            "elevation_rasters": elevation_rasters,
            "elevation_offsets": elevation_offsets,
        }


class CreateElevationStage(_CreateStageBase):
    """Add elevation / bathymetry data to the model grid."""

    name = "create_elevation"
    description = "Add elevation and bathymetry data"

    def run(self) -> dict[str, Any]:
        """Add elevation and bathymetry layers to the grid."""
        cfg = self.config

        self._update_substep("Creating elevation layers")
        elevation_list = _elevation_list(cfg)
        self._log(f"Elevation datasets: {[d.name for d in cfg.elevation.datasets]}")

        self.sfincs.quadtree_elevation.create(
            elevation_list=elevation_list,
            buffer_cells=cfg.elevation.buffer_cells,
        )

        self._log("Elevation created successfully")
        return {"status": "completed"}


class CreateMaskStage(_CreateStageBase):
    """Create the active-cell mask."""

    name = "create_mask"
    description = "Create active cell mask"

    def _drop_disconnected_components(self) -> int:
        """Set mask=0 for active cells that are not in the largest component.

        Builds a face-to-face adjacency from the SFINCS quadtree
        neighbour arrays (``mu*``, ``md*``, ``nu*``, ``nd*`` — 1-based,
        0 = no neighbour), runs ``scipy.sparse.csgraph.connected_components``
        on the active sub-graph, and returns the number of cells that
        were dropped.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        qt_data = self.sfincs.quadtree_mask.data
        mask = np.asarray(qt_data["mask"].values)
        n = len(mask)

        srcs: list[NDArray[np.integer[Any]]] = []
        dsts: list[NDArray[np.integer[Any]]] = []
        for nb in (
            "mu",
            "mu1",
            "mu2",
            "md",
            "md1",
            "md2",
            "nu",
            "nu1",
            "nu2",
            "nd",
            "nd1",
            "nd2",
        ):
            if nb not in qt_data:
                continue
            arr = np.asarray(qt_data[nb].values).astype(np.int64)
            valid = arr > 0
            srcs.append(np.arange(n)[valid])
            dsts.append(arr[valid] - 1)
        if not srcs:
            return 0
        src = np.concatenate(srcs)
        dst = np.concatenate(dsts)
        adj = csr_matrix(
            (np.ones(len(src), dtype=np.int8), (src, dst)),
            shape=(n, n),
        )

        active_idx = np.where(mask >= 1)[0]
        if active_idx.size == 0:
            return 0
        sub = adj[active_idx][:, active_idx]
        n_comp, labels = connected_components(sub, directed=False)
        if n_comp <= 1:
            return 0

        sizes = np.bincount(labels)
        largest_label = int(np.argmax(sizes))
        drop_idx = active_idx[labels != largest_label]
        if drop_idx.size == 0:
            return 0
        # In-place write so the change propagates back through the
        # UgridDataset wrapper (assigning a new DataArray would detach
        # it from the live model).
        qt_data["mask"].values[drop_idx] = 0
        return int(drop_idx.size)

    def run(self) -> dict[str, Any]:
        """Create active-cell mask based on elevation threshold."""
        cfg = self.config

        self._update_substep("Creating active mask")
        self._log(f"zmin={cfg.mask.zmin}")

        self.sfincs.quadtree_mask.create_active(
            zmin=cfg.mask.zmin,
        )

        if cfg.mask.keep_largest_only:
            n_dropped = self._drop_disconnected_components()
            if n_dropped:
                self._log(
                    f"Dropped {n_dropped} active cell(s) in disconnected "
                    f"components; kept the largest connected component only"
                )

        self._log("Active mask created successfully")
        return {"status": "completed"}


class CreateBoundaryStage(_CreateStageBase):
    """Create water-level boundary cells on the mask."""

    name = "create_boundary"
    description = "Create water level boundary cells"

    def run(self) -> dict[str, Any]:
        """Create water-level boundary cells on the mask."""
        cfg = self.config

        self._update_substep("Creating waterlevel boundary")
        self._log(f"boundary_zmax={cfg.mask.boundary_zmax}")

        self.sfincs.quadtree_mask.create_boundary(
            btype="waterlevel",
            zmax=cfg.mask.boundary_zmax,
            reset_bounds=cfg.mask.reset_bounds,
        )

        self._log("Boundary cells created successfully")
        return {"status": "completed"}


class CreateDischargeStage(_CreateStageBase):
    """Add river discharge source points derived from flowpath linestrings.

    Flowpath linestrings are read from a GeoJSON file (e.g., exported
    from the QGIS plugin), intersected with the AOI boundary, and the
    resulting points are registered as discharge source locations in the
    SFINCS model.
    """

    name = "create_discharge"
    description = "Add river discharge source points"

    def validate(self) -> list[str]:
        """Validate flowlines GeoJSON and NWM ID column."""
        cfg = self.config
        if cfg.river_discharge is None:
            return []

        nd = cfg.river_discharge
        errors: list[str] = []

        if not nd.flowlines.exists():
            errors.append(f"river_discharge.flowlines not found: {nd.flowlines}")
            return errors

        if nd.flowlines.suffix.lower() != ".geojson":
            errors.append(
                f"river_discharge.flowlines must be a .geojson file, got: {nd.flowlines.name}"
            )
            return errors

        # Validate that the file is readable and contains the ID column.
        import geopandas as gpd

        try:
            gdf: gpd.GeoDataFrame = gpd.read_file(nd.flowlines, force_2d=True)
        except Exception as exc:
            errors.append(f"Cannot read flowlines GeoJSON: {exc}")
            return errors

        if nd.nwm_id_column not in gdf.columns:
            errors.append(
                f"Column '{nd.nwm_id_column}' not found in "
                f"{nd.flowlines.name}. Available columns: {list(gdf.columns)}"
            )
            return errors

        # NWM feature IDs must be integers.
        try:
            gdf[nd.nwm_id_column] = gdf[nd.nwm_id_column].astype(int)
        except (ValueError, TypeError):
            errors.append(
                f"Column '{nd.nwm_id_column}' in {nd.flowlines.name} "
                f"cannot be converted to integer (NWM feature IDs must be integers)"
            )

        return errors

    @staticmethod
    def _crs_unit_to_meter(model_crs: Any) -> float:
        """Return the conversion factor from CRS linear units to meters.

        Raises ``ValueError`` for geographic (degree-based) CRS because
        Euclidean KDTree distances in degrees are not meaningful for
        metric comparisons.
        """
        from pyproj import CRS

        crs = CRS(model_crs)
        if crs.is_geographic:
            raise ValueError(
                f"Model CRS {crs.to_epsg() or crs} is geographic (degree-based). "
                "Discharge snapping requires a projected CRS so that distances "
                "are in linear units (meters or feet)."
            )
        # axis_info[0].unit_conversion_factor converts CRS units → meters
        # (e.g. 1.0 for meters, ~0.3048 for US survey feet).
        return float(crs.axis_info[0].unit_conversion_factor)

    def _snap_to_active_cells(
        self,
        points: list[tuple[float, float, str]],
        model: Any,
        max_snap_distance_m: float,
    ) -> list[tuple[float, float, str]]:
        """Snap discharge points to the nearest active grid cell.

        Each point is relocated to the face center of the nearest active
        cell.  The KDTree distance (in CRS units) is converted to meters
        before comparing with *max_snap_distance_m*.  Points whose
        nearest active cell is farther than the threshold are dropped
        with a warning.

        Returns the list of (x, y, name) tuples on active cells.
        """
        import numpy as np
        from scipy.spatial import KDTree

        grid_ds = model.quadtree_grid.data
        ugrid = grid_ds.ugrid.grid
        face_xy = np.column_stack([ugrid.face_x, ugrid.face_y])
        mask = grid_ds["mask"].to_numpy()
        unit_to_m = self._crs_unit_to_meter(ugrid.crs)

        active_idx = np.where(mask == 1)[0]
        if len(active_idx) == 0:
            self._log("No active cells in grid, cannot place discharge points", level="warning")
            return []
        tree_active = KDTree(face_xy[active_idx])

        snapped: list[tuple[float, float, str]] = []
        for x, y, name in points:
            dist_crs_raw, active_pos_raw = tree_active.query([x, y])
            dist_crs = float(dist_crs_raw)
            active_pos = int(active_pos_raw)
            dist_m = dist_crs * unit_to_m
            if dist_m > max_snap_distance_m:
                self._log(
                    f"  {name}: DROPPED, nearest active cell is {dist_m:.0f} m away "
                    f"(exceeds max_snap_distance_m={max_snap_distance_m:.0f})",
                    level="warning",
                )
                continue
            real_idx = active_idx[active_pos]
            cx, cy = float(face_xy[real_idx, 0]), float(face_xy[real_idx, 1])
            if dist_m > 0:
                self._log(f"  {name}: snapped to active cell ({dist_m:.0f} m away)")
            snapped.append((cx, cy, name))

        return snapped

    @staticmethod
    def _line_extreme_endpoints(geom: Any) -> tuple[Any, Any]:
        """Return the (first, last) shapely Points of a (Multi)LineString.

        For a ``MultiLineString`` we use the first coordinate of the
        first part and the last coordinate of the last part. NWM
        hydrofabric flowpaths come through a ``shapely.line_merge``
        upstream so most are plain ``LineString``; this helper
        gracefully handles the residual multipart cases.
        """
        from shapely import MultiLineString, Point

        if isinstance(geom, MultiLineString):
            first_coord = next(iter(geom.geoms)).coords[0][:2]
            last_coord = list(geom.geoms)[-1].coords[-1][:2]
        else:
            first_coord = geom.coords[0][:2]
            last_coord = geom.coords[-1][:2]
        return Point(first_coord), Point(last_coord)

    @staticmethod
    def _collapse_to_points(crossings: Any) -> list[Any]:
        """Flatten a shapely intersection result to a list of ``Point``s.

        Accepts ``Point``, ``MultiPoint``, ``LineString`` (tangential
        overlap — keeps the segment endpoints), and any ``GeometryCollection``
        containing those types.
        """
        from shapely import LineString, MultiPoint, Point

        if isinstance(crossings, Point):
            return [crossings]
        if isinstance(crossings, MultiPoint):
            return list(crossings.geoms)
        if isinstance(crossings, LineString):
            return [Point(c[:2]) for c in crossings.coords]

        points: list[Any] = []
        if hasattr(crossings, "geoms"):
            for sub in crossings.geoms:
                if isinstance(sub, Point):
                    points.append(sub)
                elif isinstance(sub, LineString):
                    points.extend(Point(c[:2]) for c in sub.coords)
        return points

    @classmethod
    def _pick_upstream_crossing(cls, geom: Any, aoi_polygon: Any, points: list[Any]) -> Any | None:
        """Pick the crossing closest to the line's upstream endpoint.

        Returns ``None`` when flow direction cannot be inferred from
        endpoint-in-polygon membership (e.g. the line passes through
        without terminating inside, or both endpoints land inside).
        """
        first_pt, last_pt = cls._line_extreme_endpoints(geom)
        first_in = aoi_polygon.contains(first_pt)
        last_in = aoi_polygon.contains(last_pt)
        if first_in and not last_in:
            upstream = last_pt
        elif last_in and not first_in:
            upstream = first_pt
        else:
            return None
        distances = [upstream.distance(p) for p in points]
        return points[min(range(len(points)), key=distances.__getitem__)]

    @classmethod
    def _inflow_intersection_point(cls, geom: Any, aoi_polygon: Any) -> tuple[float, float] | None:
        """Pick the discharge point on (or near) the AOI boundary for *geom*.

        Strategy:

        1. Compute the intersection of the flowline with the AOI's
           boundary. The result is a ``Point``, ``MultiPoint``, or — in
           degenerate cases — a ``GeometryCollection``; we collapse it
           to a flat list of points.
        2. If there is exactly one crossing, that's the discharge point.
        3. If there are multiple crossings, pick the one closest to the
           *upstream* end of the line. The upstream end is identified
           by membership: the line endpoint that lies *outside* the
           AOI is upstream; the one *inside* is downstream.
        4. If the line never crosses the boundary (entirely inside or
           entirely outside the AOI), fall back to whichever line
           endpoint sits closest to the AOI boundary. The downstream
           snap step then moves it onto the nearest active grid cell
           (or drops it when the snap distance is exceeded).

        TODO: When the line crosses the boundary multiple times **and**
        both endpoints fall outside the AOI (the line passes through
        without terminating inside), flow direction cannot be inferred
        from endpoint membership alone. A robust solution would
        consult NWM stream order or DEM-derived flow direction. For
        the walkthrough demo data this case does not arise.
        """
        if geom is None or geom.is_empty:
            return None

        boundary = aoi_polygon.boundary
        crossings = geom.intersection(boundary)
        points = cls._collapse_to_points(crossings) if not crossings.is_empty else []

        if len(points) == 1:
            p = points[0]
            return (float(p.x), float(p.y))

        if len(points) > 1:
            chosen = cls._pick_upstream_crossing(geom, aoi_polygon, points)
            if chosen is not None:
                return (float(chosen.x), float(chosen.y))

        # No usable boundary crossing — fall back to the line endpoint
        # closest to the boundary. The snap step downstream will move
        # the chosen point onto the nearest active grid cell, or drop
        # it if it is farther than ``max_snap_distance_m``.
        first_pt, last_pt = cls._line_extreme_endpoints(geom)
        closer: Point = (
            first_pt if boundary.distance(first_pt) <= boundary.distance(last_pt) else last_pt
        )
        return (float(closer.x), float(closer.y))

    def run(self) -> dict[str, Any]:
        """Extract flowpath outlets from GeoJSON and add as discharge points."""
        cfg = self.config
        if cfg.river_discharge is None:
            self._log("No river discharge configuration, skipping")
            return {"status": "skipped"}

        import geopandas as gpd

        nd = cfg.river_discharge
        model = self.sfincs

        self._update_substep("Reading flowpath geometries from GeoJSON")
        id_col = nd.nwm_id_column
        flowpaths_gdf: gpd.GeoDataFrame = gpd.read_file(nd.flowlines, force_2d=True)
        self._log(f"Read {len(flowpaths_gdf)} flowpath(s) from {nd.flowlines.name}")

        if flowpaths_gdf.empty:
            self._log(
                "No flowpaths found in GeoJSON, skipping discharge",
                level="warning",
            )
            return {"status": "skipped", "reason": "no flowpaths in file"}

        # Normalize: cast IDs to int and rename the user-supplied column to
        # ``"name"``. SFINCS's discharge_points GDF uses ``"name"`` as the
        # feature-ID column; the run stage's _assign_discharge_timeseries
        # reads gdf["name"] back as an int when looking up NWM CHRTOUT,
        # so writing it under the same name here keeps the contract
        # explicit instead of going through the .src round-trip.
        flowpaths_gdf[id_col] = flowpaths_gdf[id_col].astype(int)
        if id_col != "name":
            if "name" in flowpaths_gdf.columns:
                flowpaths_gdf = flowpaths_gdf.drop(columns=["name"])
            flowpaths_gdf = flowpaths_gdf.rename(columns={id_col: "name"})
        # Merge MultiLineStrings into LineStrings for clean endpoint selection.
        flowpaths_gdf["geometry"] = shapely.line_merge(flowpaths_gdf.geometry)

        # Determine model CRS from the SfincsModel grid
        grid_ds = model.quadtree_grid.data
        model_crs: Any = grid_ds.ugrid.grid.crs

        # Reproject flowpaths and AOI to model CRS
        if flowpaths_gdf.crs is not None and flowpaths_gdf.crs != model_crs:
            flowpaths_gdf = flowpaths_gdf.to_crs(model_crs)

        # Prefer the cleaned AOI when CreateGridStage produced one
        # (aoi_simplify_neck_m > 0 in the config). The cleaned polygon
        # matches the actual model domain, so flowpath-boundary
        # intersections line up with cells SFINCS will simulate.
        cleaned_aoi = cfg.output_dir / "aoi_cleaned.geojson"
        aoi_path_for_intersection = cleaned_aoi if cleaned_aoi.exists() else cfg.aoi
        aoi_gdf: gpd.GeoDataFrame = gpd.read_file(str(aoi_path_for_intersection))
        if aoi_gdf.crs is not None and aoi_gdf.crs != model_crs:
            aoi_gdf = aoi_gdf.to_crs(model_crs)
        aoi_polygon = aoi_gdf.union_all()

        # Identify the inflow point on the AOI boundary for each flowline.
        # Lines that never cross the boundary (entirely outside the
        # domain, or entirely inside with no entry/exit) are skipped
        # with a log message — there is no obvious discharge location
        # for them. Lines that cross multiple times resolve to the
        # most-upstream crossing.
        self._update_substep("Extracting flowpath outlet points")
        discharge_points: list[tuple[float, float, str]] = []
        for feature_id, geom in flowpaths_gdf[["name", "geometry"]].itertuples(
            name=None, index=False
        ):
            point = self._inflow_intersection_point(geom, aoi_polygon)
            if point is None:
                self._log(f"Flowpath {feature_id}: no usable AOI-boundary crossing, skipping")
                continue
            discharge_points.append((point[0], point[1], str(feature_id)))

        if not discharge_points:
            self._log("No discharge points extracted from flowpaths")
            return {"status": "completed", "points_added": 0}

        # Snap discharge points to nearest active (wet) grid cell.
        # Intersection points land on the AOI boundary which often
        # falls on inactive cells; snapping moves them inward.
        self._update_substep("Snapping discharge points to active cells")
        snapped = self._snap_to_active_cells(discharge_points, model, nd.max_snap_distance_m)
        if not snapped:
            self._log(
                "No discharge points could be placed on active cells",
                level="warning",
            )
            return {"status": "completed", "points_added": 0}

        # Write a standalone .src file with the snapped locations.
        # The points are NOT added to the model here — the run stage
        # reads this file, adds the points, and assigns real discharge
        # timeseries from the NWM CHRTOUT data catalog.
        self._update_substep("Writing discharge source locations")
        src_path = cfg.output_dir / f"sfincs_{nd.source}.src"
        with src_path.open("w") as f:
            for x, y, name in snapped:
                f.write(f'{x:.2f} {y:.2f} "{name}"\n')

        self._log(f"Wrote {len(snapped)} discharge source location(s) to {src_path.name}")
        return {
            "status": "completed",
            "points_added": len(snapped),
            "src_file": str(src_path),
        }


class CreateObservationPointsStage(_CreateStageBase):
    """Add observation points to the model during creation.

    Supports three sources of observation points:

    * A GeoJSON file (``observation_locations_file``).
    * An inline list (``observation_points``).
    * Automatic NOAA CO-OPS station discovery (``add_noaa_gages``).

    After adding points, all are snapped to the nearest wet cell so
    that they produce dynamic water-level output during the SFINCS run.
    A mapping file (``obs_station_map.json``) is written next to the
    model files for downstream stages (e.g. plotting) to use.
    """

    name = "create_obs"
    description = "Add observation points"

    # ------------------------------------------------------------------
    # NOAA CO-OPS helpers
    # ------------------------------------------------------------------

    def _add_noaa_gages(self, model: SfincsModel) -> int:
        """Query NOAA CO-OPS and add water-level stations as observation points.

        Returns the number of NOAA stations added.
        """
        from coastal_calibration.data.coops_api import COOPSAPIClient

        model_crs = model.crs
        if model_crs is None:
            self._log("Model CRS is undefined, cannot add NOAA CO-OPS stations")
            return 0

        region_4326 = model.region.to_crs(4326)
        domain_geom = region_4326.union_all()

        client = COOPSAPIClient()
        stations_gdf = client.stations_metadata
        selected = stations_gdf[stations_gdf.within(domain_geom)]

        if selected.empty:
            self._log("No NOAA CO-OPS stations found within model domain")
            return 0

        candidate_ids = selected["station_id"].tolist()
        valid_ids = client.filter_stations_by_datum(candidate_ids)
        dropped = set(candidate_ids) - valid_ids
        if dropped:
            self._log(
                f"Excluded {len(dropped)} station(s) without datum data: "
                f"{', '.join(sorted(dropped))}",
                "warning",
            )
        selected = selected[selected["station_id"].isin(sorted(valid_ids))]
        if selected.empty:
            self._log("No NOAA CO-OPS stations with valid datum data in domain")
            return 0

        dedup_distance_m = 100.0
        existing_points: list[tuple[float, float]] = []
        try:
            gdf = model.observation_points.data
            if gdf is not None and not gdf.empty:  # pyright: ignore[reportUnnecessaryComparison]
                existing_points = [
                    (cast("Point", geom).x, cast("Point", geom).y)
                    for geom in gdf.geometry
                    if geom is not None  # pyright: ignore[reportUnnecessaryComparison]
                ]
        except Exception:  # noqa: S110
            pass

        selected_projected = selected.to_crs(model_crs)

        added = 0
        for _, row in selected_projected.iterrows():
            cx, cy = row.geometry.x, row.geometry.y
            if any(math.hypot(cx - ex, cy - ey) < dedup_distance_m for ex, ey in existing_points):
                continue
            sid = row["station_id"]
            model.observation_points.add_point(x=cx, y=cy, name=f"noaa_{sid}")
            existing_points.append((cx, cy))
            added += 1

        return added

    # ------------------------------------------------------------------
    # Observation-point snapping
    # ------------------------------------------------------------------

    def _snap_obs_to_wet_cells(self, model: SfincsModel) -> int:
        """Snap every observation point to the center of the nearest wet cell.

        SFINCS's internal quadtree cell lookup maps (x, y) coordinates
        to cells via the (n, m) index structure.  When a point is *not*
        at the exact face center, the lookup can land on a neighboring
        (potentially dry or inactive) cell — especially after grid
        refinement changes the (n, m) layout.

        To guarantee correct placement we *always* relocate each point
        to the center of the nearest active wet face, regardless of
        whether the current cell appears wet.
        """
        from scipy.spatial import KDTree

        depth_threshold = self.config.obs_snap_depth_threshold
        search_radius = self.config.obs_snap_search_radius_m

        obs_gdf = model.observation_points.data
        if obs_gdf is None or obs_gdf.empty:  # pyright: ignore[reportUnnecessaryComparison]
            return 0

        grid_ds = model.quadtree_grid.data
        ugrid: Any = grid_ds.ugrid.grid
        fx = np.asarray(ugrid.face_x, dtype=np.float64)
        fy = np.asarray(ugrid.face_y, dtype=np.float64)
        z_elev = np.asarray(grid_ds["z"].to_numpy(), dtype=np.float64)  # pyright: ignore[reportIndexIssue]
        mask_arr = np.asarray(grid_ds["mask"].to_numpy(), dtype=np.float64)  # pyright: ignore[reportIndexIssue]

        # Build a KDTree of active wet face centers only.
        wet_active = (z_elev < depth_threshold) & (mask_arr > 0)
        wet_idx = np.where(wet_active)[0]
        if len(wet_idx) == 0:
            self._log("No active wet cells in grid; cannot snap observation points", "warning")
            return 0
        tree_wet = KDTree(np.column_stack([fx[wet_idx], fy[wet_idx]]))

        snapped = 0
        for i in range(len(obs_gdf)):
            geom = cast("Point", obs_gdf.geometry.iloc[i])
            if geom is None:  # pyright: ignore[reportUnnecessaryComparison]
                continue
            ox, oy = float(geom.x), float(geom.y)
            name = str(i)
            with contextlib.suppress(KeyError, IndexError):
                name = obs_gdf["name"].iloc[i]

            dist_raw, pos_raw = tree_wet.query([ox, oy])
            dist = float(dist_raw)
            pos = int(pos_raw)
            if dist > search_radius:
                self._log(
                    f"  {name}: no wet cell within {search_radius:.0f} m "
                    f"(nearest {dist:.0f} m away)",
                    "warning",
                )
                continue

            best = wet_idx[pos]
            new_x, new_y = float(fx[best]), float(fy[best])
            new_z = float(z_elev[best])
            obs_gdf.geometry.iloc[i] = Point(new_x, new_y)  # pyright: ignore[reportCallIssue, reportArgumentType]
            self._log(
                f"  {name}: placed at face center z={new_z:.3f} m ({dist:.0f} m from original)"
            )
            snapped += 1

        if snapped > 0:
            model.observation_points._data = obs_gdf  # pyright: ignore[reportPrivateUsage]
        return snapped

    # ------------------------------------------------------------------
    # Station mapping file
    # ------------------------------------------------------------------

    def _write_obs_station_map(self, model: SfincsModel) -> None:
        """Write ``obs_station_map.json`` mapping obs indices to station IDs."""
        obs_gdf = model.observation_points.data
        if obs_gdf is None or obs_gdf.empty:  # pyright: ignore[reportUnnecessaryComparison]
            return

        station_map: list[dict[str, Any]] = []
        for idx in range(len(obs_gdf)):
            name = ""
            with contextlib.suppress(KeyError, IndexError):
                name = str(obs_gdf["name"].iloc[idx])
            if not name:
                with contextlib.suppress(Exception):
                    name = str(obs_gdf.index[idx])

            station_id = ""
            if name.startswith("noaa_"):
                station_id = name[len("noaa_") :]

            geom = cast("Point", obs_gdf.geometry.iloc[idx])
            entry: dict[str, Any] = {
                "index": idx,
                "name": name,
                "x": float(geom.x) if geom else None,
                "y": float(geom.y) if geom else None,
            }
            if station_id:
                entry["station_id"] = station_id
            station_map.append(entry)

        map_path = self.config.output_dir / "obs_station_map.json"
        map_path.write_text(json.dumps(station_map, indent=2))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Add observation points from config, file, and/or NOAA gages."""
        cfg = self.config
        model = _get_model(cfg)

        has_file = cfg.observation_locations_file is not None
        has_points = bool(cfg.observation_points)
        has_noaa = cfg.add_noaa_gages

        if not has_file and not has_points and not has_noaa:
            self._log("No observation points configured, skipping")
            return {"status": "skipped"}

        self._update_substep("Adding observation points")

        if not cfg.merge_observations:
            try:
                existing = model.observation_points.nr_points
                if existing > 0:
                    model.observation_points.clear()
                    self._log(f"Cleared {existing} existing observation point(s)")
            except Exception:  # noqa: S110
                pass

        if has_file:
            model.observation_points.create(
                locations=str(cfg.observation_locations_file),
                merge=cfg.merge_observations,
            )
            self._log(f"Observation points added from {cfg.observation_locations_file}")
        elif has_points:
            for pt in cfg.observation_points:
                model.observation_points.add_point(
                    x=pt["x"],
                    y=pt["y"],
                    name=pt.get("name", f"obs_{cfg.observation_points.index(pt)}"),
                )
            self._log(f"Added {len(cfg.observation_points)} observation point(s)")

        noaa_count = 0
        if has_noaa:
            self._update_substep("Querying NOAA CO-OPS stations")
            noaa_count = self._add_noaa_gages(model)
            self._log(f"Added {noaa_count} NOAA CO-OPS observation point(s)")

        snapped = self._snap_obs_to_wet_cells(model)
        if snapped:
            self._log(f"Snapped {snapped} observation point(s) to nearest wet cell")

        self._write_obs_station_map(model)

        return {"status": "completed", "noaa_stations": noaa_count}


class CreateSubgridStage(_CreateStageBase):
    """Generate subgrid-derived lookup tables.

    Roughness is embedded in the quadtree subgrid tables, so there
    is no separate roughness stage.
    """

    name = "create_subgrid"
    description = "Create subgrid tables"

    def run(self) -> dict[str, Any]:
        """Generate subgrid lookup tables with embedded roughness."""
        cfg = self.config

        self._update_substep("Creating subgrid tables")
        self._log(f"nr_subgrid_pixels={cfg.subgrid.nr_subgrid_pixels}")

        elevation_list = _elevation_list(cfg)
        roughness_entry: dict[str, Any] = {"lulc": cfg.subgrid.lulc_dataset}
        if cfg.subgrid.reclass_table is not None:
            roughness_entry["reclass_table"] = str(cfg.subgrid.reclass_table)

        self.sfincs.quadtree_subgrid.create(
            elevation_list=elevation_list,
            roughness_list=[roughness_entry],
            nr_subgrid_pixels=cfg.subgrid.nr_subgrid_pixels,
            manning_land=cfg.subgrid.manning_land,
            manning_water=cfg.subgrid.manning_sea,
            quiet=True,
        )

        self._log("Subgrid tables created successfully")
        return {"status": "completed"}


class CreateWriteStage(_CreateStageBase):
    """Write the SFINCS model to disk and clean up the registry."""

    name = "create_write"
    description = "Write SFINCS model to disk"

    def run(self) -> dict[str, Any]:
        """Write the SFINCS model to disk and clean up the registry."""
        cfg = self.config

        self._update_substep("Writing model")
        self._log(f"Output directory: {cfg.output_dir}")

        self.sfincs.write()

        _clear_model(cfg)

        self._log("Model written successfully")
        return {"status": "completed", "output_dir": str(cfg.output_dir)}


# ---------------------------------------------------------------------------
# Stage construction helper
# ---------------------------------------------------------------------------

#: Mapping from stage name to its class.
STAGE_CLASSES: dict[str, type[CreateStage]] = {
    "create_grid": CreateGridStage,
    "create_fetch_data": CreateFetchDataStage,
    "create_elevation": CreateElevationStage,
    "create_mask": CreateMaskStage,
    "create_boundary": CreateBoundaryStage,
    "create_discharge": CreateDischargeStage,
    "create_subgrid": CreateSubgridStage,
    "create_obs": CreateObservationPointsStage,
    "create_write": CreateWriteStage,
}


def create_stages(
    config: SfincsCreateConfig,
    monitor: WorkflowMonitor | None = None,
) -> dict[str, CreateStage]:
    """Instantiate all creation stages for *config*.

    Only stages listed in ``config.stage_order`` are included.
    """
    return {
        name: STAGE_CLASSES[name](config, monitor)
        for name in config.stage_order
        if name in STAGE_CLASSES
    }


# ---------------------------------------------------------------------------
# SFINCS model creation runner (was creator.py)
# ---------------------------------------------------------------------------


class SfincsCreator:
    """Runner that orchestrates the SFINCS model creation pipeline.

    Mirrors :class:`~coastal_calibration.runner.CoastalCalibRunner` but
    operates on a :class:`SfincsCreateConfig` and delegates to
    :class:`~coastal_calibration.sfincs.create.CreateStage` instances.
    """

    _STATUS_FILENAME = ".create_status.json"

    def __init__(self, config: SfincsCreateConfig) -> None:
        self.config = config
        self._pending_status: dict[str, str] = {}

        # Ensure the output directory exists early so file logging can start.
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up file logging before creating the monitor.
        if not config.monitoring.log_file:
            log_path = generate_log_path(config.output_dir, prefix="sfincs-create")
            configure_logger(file=str(log_path), file_level="DEBUG")

        silence_third_party_loggers()

        self.monitor = WorkflowMonitor(config.monitoring)
        self._stages: dict[str, CreateStage] = {}
        self._results: dict[str, Any] = {}

    @property
    def stage_order(self) -> list[str]:
        """Active stage order based on config."""
        return self.config.stage_order

    def _init_stages(self) -> None:
        """Instantiate all creation stages."""
        self._stages = create_stages(self.config, self.monitor)

    # ------------------------------------------------------------------
    # Pipeline status tracking
    # ------------------------------------------------------------------

    @property
    def _status_path(self) -> Path:
        """Path to the pipeline status file."""
        return self.config.output_dir / self._STATUS_FILENAME

    def _load_status(self) -> dict[str, Any]:
        """Load pipeline status from disk (empty dict if missing)."""
        if self._status_path.exists():
            return json.loads(self._status_path.read_text())
        return {}

    def _record_stage(self, stage_name: str) -> None:
        """Buffer *stage_name* as completed until the model reaches disk."""
        self._pending_status[stage_name] = _stage_fingerprint(self.config, stage_name)

    def _commit_status(self) -> None:
        """Write the buffered completions, once the model is written.

        Every stage before ``create_write`` builds in memory, so a run that
        stops or fails ahead of it leaves nothing on disk to resume from.
        Recording each stage as it finished let such a run rewrite the
        fingerprints for work it then discarded: ``--stop-after`` in the
        middle of a resume made the *previous* model's files look like they
        matched the current config, and the next resume read that as an
        all-clear.  Buffering means a discarded run records nothing, so the
        status keeps describing the model that is actually there.
        """
        status = self._load_status()
        completed: list[str] = status.get("completed_stages", [])
        fingerprints: dict[str, str] = status.get("stage_fingerprints") or {}
        for stage_name, fingerprint in self._pending_status.items():
            if stage_name not in completed:
                completed.append(stage_name)
            fingerprints[stage_name] = fingerprint
        status["completed_stages"] = completed
        status["stage_fingerprints"] = fingerprints
        self._status_path.write_text(json.dumps(status, indent=2) + "\n")
        self._pending_status.clear()

    def _register_fetched_catalogs(self) -> None:
        """Re-add the catalogs ``create_fetch_data`` wrote, when it is skipped.

        That stage registers one catalog per dataset in memory, so a resume
        past it left ``data_libs`` empty and hydromt fell back to reading the
        dataset name as a path.  The files themselves persist under the
        download directory, which the fingerprint check has already confirmed
        was built from this config.
        """
        for catalog in sorted(self.config.effective_download_dir.glob("*_catalog.yml")):
            path_str = str(catalog.resolve())
            if path_str not in self.config.data_catalog.data_libs:
                self.config.data_catalog.data_libs.append(path_str)

    def _refresh_recorded_offsets(self, outputs: dict[str, Any]) -> None:
        """Bring the recorded elevation offsets back in line with the config.

        ``create_fetch_data`` snapshots each dataset's offset, and the
        flood-map stage reads that record to shift the DEM it downscales onto.
        Changing an offset does not change what is fetched, so the guard sends
        the user to ``create_elevation`` and the fetch stage never re-runs:
        the bed moved to the new datum while the record still described the
        old one, and every later flood map came out wrong by the difference.
        Which rasters were fetched is keyed by ``aoi_key``, so a resume that
        gets this far is reusing the right files under the wrong offsets.
        """
        fetched = outputs.get("create_fetch_data")
        if not isinstance(fetched, dict) or "elevation_offsets" not in fetched:
            return
        recorded = cast("dict[str, Any]", fetched)["elevation_offsets"]
        if not isinstance(recorded, dict):
            return
        current = {d.name: d.offset for d in self.config.elevation.datasets}
        for name in cast("dict[str, Any]", recorded):
            if name in current:
                cast("dict[str, Any]", recorded)[name] = current[name]

    def _save_result(self, result: WorkflowResult, start_from: str | None) -> None:
        """Write ``create_result.json``, folding in a resumed run's history.

        A resume only re-executes some stages, so overwriting would drop
        ``create_fetch_data`` and with it the elevation raster the flood-map
        stage resolves its DEM from.
        """
        result_file = self.config.output_dir / "create_result.json"
        if start_from and result_file.is_file():
            try:
                previous = json.loads(result_file.read_text()).get("outputs", {})
            except (OSError, ValueError):
                previous = {}
            result.outputs = {**previous, **result.outputs}
            self._refresh_recorded_offsets(result.outputs)
        result.save(result_file)

    def _check_prerequisites(self, start_from: str) -> list[str]:
        """Verify that all stages before *start_from* can be skipped safely.

        They must have completed, and their config must not have changed
        since they ran.
        """
        status = self._load_status()
        completed: set[str] = set(status.get("completed_stages", []))

        all_stages = self.stage_order
        if start_from not in all_stages:
            return [f"Unknown stage: {start_from}"]

        # A stage the config no longer asks for still left its artifacts in the
        # model, and dropping out of ``stage_order`` would take it out of the
        # comparison below entirely.
        dropped = [s for s in completed if s not in all_stages]
        if dropped:
            return [
                (
                    f"Cannot start from '{start_from}': stage(s) "
                    f"{', '.join(repr(s) for s in sorted(dropped))} completed for this "
                    f"model but are no longer part of the configured pipeline.  Their "
                    f"output is still in the model and a resume would write it back "
                    f"out.  Rebuild the model from scratch.  "
                    f"(Status file: {self._status_path})"
                )
            ]

        start_idx = all_stages.index(start_from)
        skipped = all_stages[:start_idx]
        missing = [s for s in skipped if s not in completed]
        if missing:
            return [
                (
                    f"Cannot start from '{start_from}': prerequisite stage(s) "
                    f"{', '.join(repr(s) for s in missing)} have not completed.  "
                    f"Run them first or start from an earlier stage.  "
                    f"(Status file: {self._status_path})"
                )
            ]

        # Models built before fingerprints were recorded have no entry, and
        # resume for them behaves as it always did.
        recorded: dict[str, str] = status.get("stage_fingerprints") or {}
        changed = [
            s
            for s in skipped
            if s in recorded and recorded[s] != _stage_fingerprint(self.config, s)
        ]
        if changed:
            return [
                (
                    f"Cannot start from '{start_from}': the configuration for "
                    f"already-completed stage(s) {', '.join(repr(s) for s in changed)} "
                    f"has changed since they ran.  Resuming would combine their "
                    f"existing outputs with the new settings.  Start from "
                    f"'{changed[0]}' instead, or rebuild the model from scratch.  "
                    f"(Status file: {self._status_path})"
                )
            ]
        return []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate configuration and stage prerequisites.

        Returns
        -------
        list of str
            Validation error messages (empty if valid).
        """
        errors: list[str] = []

        config_errors = self.config.validate()
        errors.extend(config_errors)

        self._init_stages()
        for name, stage in self._stages.items():
            stage_errors = stage.validate()
            errors.extend(f"[{name}] {error}" for error in stage_errors)

        return errors

    # ------------------------------------------------------------------
    # Stage selection
    # ------------------------------------------------------------------

    def _get_stages_to_run(
        self,
        start_from: str | None,
        stop_after: str | None,
    ) -> list[str]:
        """Determine which stages to run."""
        stages = self.stage_order.copy()

        if start_from:
            if start_from not in stages:
                raise ValueError(f"Unknown stage: {start_from}")
            start_idx = stages.index(start_from)
            stages = stages[start_idx:]

        if stop_after:
            if stop_after not in stages:
                raise ValueError(f"Unknown stage: {stop_after}")
            stop_idx = stages.index(stop_after)
            stages = stages[: stop_idx + 1]

        return stages

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(
        self,
        start_from: str | None = None,
        stop_after: str | None = None,
        dry_run: bool = False,
    ) -> WorkflowResult:
        """Execute the SFINCS model creation workflow.

        Parameters
        ----------
        start_from : str, optional
            Stage name to start from (skip earlier stages).
        stop_after : str, optional
            Stage name to stop after (skip later stages).
        dry_run : bool, default False
            If True, validate but don't execute.

        Returns
        -------
        WorkflowResult
            Result with execution details.
        """
        start_time = utc_now()
        stages_completed: list[str] = []
        stages_failed: list[str] = []
        outputs: dict[str, Any] = {}
        errors: list[str] = []

        validation_errors = self.validate()
        if not validation_errors and start_from:
            validation_errors = self._check_prerequisites(start_from)
        if validation_errors:
            return WorkflowResult(
                success=False,
                job_id=None,
                start_time=start_time,
                end_time=utc_now(),
                stages_completed=[],
                stages_failed=[],
                outputs={},
                errors=validation_errors,
            )

        if dry_run:
            self.monitor.info("Dry run mode - validation passed, no execution")
            return WorkflowResult(
                success=True,
                job_id=None,
                start_time=start_time,
                end_time=utc_now(),
                stages_completed=[],
                stages_failed=[],
                outputs={"dry_run": True},
                errors=[],
            )

        self.monitor.register_stages(self.stage_order)
        self.monitor.start_workflow()
        self.monitor.info("-" * 40)

        stages_to_run = self._get_stages_to_run(start_from, stop_after)

        # When resuming from a later stage, load the existing model so
        # that stages which reference ``self.sfincs`` can find it.
        if start_from and "create_grid" not in stages_to_run:
            if "create_fetch_data" not in stages_to_run:
                self._register_fetched_catalogs()
            _load_existing_model(self.config)

        current_stage = ""
        try:
            with suppress_hydromt_output():
                for current_stage in stages_to_run:
                    stage = self._stages[current_stage]

                    with self.monitor.stage_context(current_stage, stage.description):
                        result = stage.run()
                        self._results[current_stage] = result
                        outputs[current_stage] = result
                        stages_completed.append(current_stage)
                        self._record_stage(current_stage)
                        if current_stage == _WRITE_STAGE:
                            self._commit_status()

            self.monitor.end_workflow(success=True)
            success = True

        except Exception as e:
            self.monitor.error(f"Workflow failed: {e}")
            self.monitor.end_workflow(success=False)
            errors.append(str(e))
            stages_failed.append(current_stage)
            success = False

        result = WorkflowResult(
            success=success,
            job_id=None,
            start_time=start_time,
            end_time=utc_now(),
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            outputs=outputs,
            errors=errors,
        )

        self._save_result(result, start_from)
        self.monitor.save_progress(self.config.output_dir / "create_progress.json")

        return result
