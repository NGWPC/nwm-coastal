"""SFINCS model creation stages using HydroMT-SFINCS Python API.

These stages build a new SFINCS quadtree model from an AOI polygon.
All stages subclass :class:`CreateStage` and accept a
:class:`~coastal_calibration.config.create_schema.SfincsCreateConfig`.

The HydroMT ``SfincsModel`` instance is shared between stages via a
module-level registry keyed by config ``id``, following the same pattern
as :mod:`coastal_calibration.stages.sfincs_build`.
"""

from __future__ import annotations

import contextlib
import json
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import shapely
from shapely import Point

from coastal_calibration.stages._hydromt_compat import apply_all_patches
from coastal_calibration.utils.logging import suppress_hydromt_output

apply_all_patches()

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd
    from hydromt_sfincs import SfincsModel

    from coastal_calibration.config.create_schema import SfincsCreateConfig
    from coastal_calibration.utils.logging import WorkflowMonitor


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


def _clear_model(config: SfincsCreateConfig) -> None:
    """Remove the SfincsModel from the registry."""
    _MODEL_REGISTRY.pop(id(config), None)


def _load_existing_model(config: SfincsCreateConfig) -> None:  # pyright: ignore[reportUnusedFunction]
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

    Mirrors the :class:`~coastal_calibration.stages.base.WorkflowStage`
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

        region = {"geom": str(cfg.aoi)}

        # Build refinement GeoDataFrame from config (if any)
        refinement_gdf = None
        if cfg.grid.refinement:
            import geopandas as gpd
            import pandas as pd
            from pyproj import CRS

            # HydroMT's quadtree builder compares polygon coordinates
            # directly against the grid (no CRS reprojection).  Determine
            # the grid CRS first so we can reproject + buffer in meters.
            grid_crs_str = cfg.grid.crs
            if grid_crs_str == "utm":
                aoi_gdf: gpd.GeoDataFrame = gpd.read_file(str(cfg.aoi))
                centroid = cast("Point", aoi_gdf.to_crs(4326).geometry.centroid.iloc[0])
                utm_zone = int((centroid.x + 180) / 6) + 1
                grid_epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone
                target_crs = CRS.from_epsg(grid_epsg)
            else:
                target_crs: CRS = CRS.from_user_input(grid_crs_str)

            parts: list[gpd.GeoDataFrame] = []
            for ref in cfg.grid.refinement:
                gdf: gpd.GeoDataFrame = gpd.read_file(ref.polygon)
                # Reproject to the grid CRS first (buffer is in meters).
                if gdf.crs is not None and gdf.crs != target_crs:
                    gdf = gdf.to_crs(target_crs)
                # Apply inward buffer (negative = shrink) so that
                # cells near the grid boundary remain at a coarser
                # level — needed for valid quadtree transitions.
                if ref.buffer_m != 0.0:
                    gdf = gdf.copy()
                    gdf["geometry"] = gdf.geometry.buffer(ref.buffer_m)
                    # Drop geometries that collapsed to empty after buffering.
                    empty = gdf.geometry.is_empty
                    if empty.any():
                        n_dropped = int(empty.sum())
                        self._log(
                            f"Refinement polygon '{ref.polygon.name}': "
                            f"{n_dropped} geometry(ies) collapsed to empty after "
                            f"buffer_m={ref.buffer_m} m — dropped",
                            "warning",
                        )
                        gdf = gdf[~empty]
                if gdf.empty:
                    continue
                gdf["refinement_level"] = ref.level
                parts.append(gdf)
            if parts:
                refinement_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True))
                self._log(
                    f"Refinement: {len(parts)} polygon(s), "
                    f"max level {max(r.level for r in cfg.grid.refinement)}"
                )
            else:
                self._log(
                    "All refinement polygons collapsed after buffering — building uniform grid",
                    "warning",
                )

        with suppress_hydromt_output():
            sf.quadtree_grid.create_from_region(
                region=region,
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

    def _fetch_elevation(self, fetched: list[str]) -> None:
        """Fetch elevation datasets that have a ``source`` set."""
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
                continue

            if ds.source == "nws_30m":
                from coastal_calibration.utils.topobathy_nws import fetch_topobathy

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
                from coastal_calibration.utils.topobathy_noaa import fetch_noaa_dem

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
                from coastal_calibration.utils.copdem import fetch_copdem30

                self._update_substep(f"Fetching Copernicus DEM 30m for '{catalog_name}'")
                _, cat_path, _ = fetch_copdem30(
                    aoi=cfg.aoi,
                    output_dir=dl_dir,
                    catalog_name=catalog_name,
                    log=self._log,
                )
            elif ds.source == "gebco_15arcs":
                from coastal_calibration.utils.gebco_wms import fetch_gebco

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
            from coastal_calibration.utils.esa_worldcover import fetch_esa_worldcover

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
        self._fetch_elevation(fetched)
        self._fetch_lulc(fetched)
        self._log(f"Fetched {len(fetched)} dataset(s): {fetched}")
        return {"status": "completed", "fetched": fetched}


# Backward-compatible alias.
CreateFetchElevationStage = CreateFetchDataStage


class CreateElevationStage(_CreateStageBase):
    """Add elevation / bathymetry data to the model grid."""

    name = "create_elevation"
    description = "Add elevation and bathymetry data"

    def run(self) -> dict[str, Any]:
        """Add elevation and bathymetry layers to the grid."""
        cfg = self.config

        self._update_substep("Creating elevation layers")
        elevation_list = [{"elevation": d.name, "zmin": d.zmin} for d in cfg.elevation.datasets]
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

    def run(self) -> dict[str, Any]:
        """Create active-cell mask based on elevation threshold."""
        cfg = self.config

        self._update_substep("Creating active mask")
        self._log(f"zmin={cfg.mask.zmin}")

        self.sfincs.quadtree_mask.create_active(
            zmin=cfg.mask.zmin,
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
                    f"  {name}: DROPPED — nearest active cell is {dist_m:.0f} m away "
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
    def _downstream_endpoint(geom: Any, aoi_boundary: Any) -> tuple[float, float] | None:
        """Return the flowpath endpoint closest to the AOI boundary.

        NWM hydrofabric flowpath linestrings have no guaranteed
        direction, so we compare both the first and last coordinates
        and pick whichever is nearest to the AOI boundary.  In
        practice the outlet should sit on (or very near) the boundary.
        """
        from shapely import MultiLineString

        if geom is None or geom.is_empty:
            return None
        if isinstance(geom, MultiLineString):
            first_coord = next(iter(geom.geoms)).coords[0][:2]
            last_coord = list(geom.geoms)[-1].coords[-1][:2]
        else:
            first_coord = geom.coords[0][:2]
            last_coord = geom.coords[-1][:2]

        d_first = aoi_boundary.distance(shapely.Point(first_coord))
        d_last = aoi_boundary.distance(shapely.Point(last_coord))
        chosen = first_coord if d_first < d_last else last_coord
        return (float(chosen[0]), float(chosen[1]))

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

        # NWM feature IDs must be integers.
        flowpaths_gdf[id_col] = flowpaths_gdf[id_col].astype(int)
        # Merge MultiLineStrings into LineStrings for clean endpoint selection.
        flowpaths_gdf["geometry"] = shapely.line_merge(flowpaths_gdf.geometry)

        # Determine model CRS from the SfincsModel grid
        grid_ds = model.quadtree_grid.data
        model_crs: Any = grid_ds.ugrid.grid.crs

        # Reproject flowpaths and AOI to model CRS
        if flowpaths_gdf.crs is not None and flowpaths_gdf.crs != model_crs:
            flowpaths_gdf = flowpaths_gdf.to_crs(model_crs)

        aoi_gdf: gpd.GeoDataFrame = gpd.read_file(str(cfg.aoi))
        if aoi_gdf.crs is not None and aoi_gdf.crs != model_crs:
            aoi_gdf = aoi_gdf.to_crs(model_crs)
        aoi_boundary = aoi_gdf.union_all().boundary

        # Pick the flowpath endpoint closest to the AOI boundary
        self._update_substep("Extracting flowpath outlet points")
        discharge_points: list[tuple[float, float, str]] = []
        for feature_id, geom in flowpaths_gdf[[id_col, "geometry"]].itertuples(
            name=None, index=False
        ):
            endpoint = self._downstream_endpoint(geom, aoi_boundary)
            if endpoint is None:
                self._log(f"Flowpath {feature_id} has empty geometry, skipping")
                continue
            discharge_points.append((endpoint[0], endpoint[1], str(feature_id)))

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
        src_path = cfg.output_dir / "sfincs_nwm.src"
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

    #: Bed-elevation threshold (m): cells at or above this are "dry".
    _SNAP_DEPTH_THRESHOLD: float = -0.1
    #: Maximum search radius (m) when looking for a replacement wet cell.
    _SNAP_SEARCH_RADIUS_M: float = 1000.0

    # ------------------------------------------------------------------
    # NOAA CO-OPS helpers
    # ------------------------------------------------------------------

    def _add_noaa_gages(self, model: SfincsModel) -> int:
        """Query NOAA CO-OPS and add water-level stations as observation points.

        Returns the number of NOAA stations added.
        """
        from coastal_calibration.coops_api import COOPSAPIClient

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

        depth_threshold = self._SNAP_DEPTH_THRESHOLD
        search_radius = self._SNAP_SEARCH_RADIUS_M

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
            self._log("No active wet cells in grid — cannot snap observation points", "warning")
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

        elevation_list = [{"elevation": d.name, "zmin": d.zmin} for d in cfg.elevation.datasets]
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
    # Backward-compatible alias for saved status files.
    "create_fetch_elevation": CreateFetchDataStage,
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
