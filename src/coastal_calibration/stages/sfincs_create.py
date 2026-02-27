"""SFINCS model creation stages using HydroMT-SFINCS Python API.

These stages build a new SFINCS quadtree model from an AOI polygon.
All stages subclass :class:`CreateStage` and accept a
:class:`~coastal_calibration.config.create_schema.SfincsCreateConfig`.

The HydroMT ``SfincsModel`` instance is shared between stages via a
module-level registry keyed by config ``id``, following the same pattern
as :mod:`coastal_calibration.stages.sfincs_build`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

from coastal_calibration.stages._hydromt_compat import apply_all_patches

apply_all_patches()

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from hydromt_sfincs import SfincsModel

    from coastal_calibration.config.create_schema import SfincsCreateConfig
    from coastal_calibration.utils.logging import WorkflowMonitor


# ---------------------------------------------------------------------------
# stdout suppression
# ---------------------------------------------------------------------------


@contextmanager
def _suppress_stdout() -> Iterator[None]:
    """Redirect stdout to ``/dev/null``.

    HydroMT-SFINCS's quadtree builders use raw ``print()`` calls
    that cannot be silenced through Python's logging system.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


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
    """Initialise a new ``SfincsModel`` and generate the quadtree grid."""

    name = "create_grid"
    description = "Create SFINCS grid from AOI"

    def run(self) -> dict[str, Any]:
        """Initialise SfincsModel and generate the quadtree grid."""
        from hydromt_sfincs import SfincsModel

        cfg = self.config

        self._update_substep("Initialising SfincsModel")
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
            # the grid CRS first so we can reproject + buffer in metres.
            grid_crs_str = cfg.grid.crs
            if grid_crs_str == "utm":
                aoi_gdf = gpd.read_file(str(cfg.aoi))
                centroid = aoi_gdf.to_crs(4326).geometry.centroid.iloc[0]
                utm_zone = int((centroid.x + 180) / 6) + 1
                grid_epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone
                target_crs = CRS.from_epsg(grid_epsg)
            else:
                target_crs = CRS.from_user_input(grid_crs_str)

            parts: list[gpd.GeoDataFrame] = []
            for ref in cfg.grid.refinement:
                gdf = gpd.read_file(ref.polygon)
                # Reproject to the grid CRS first (buffer is in metres).
                if gdf.crs is not None and gdf.crs != target_crs:
                    gdf = gdf.to_crs(target_crs)
                # Apply inward buffer (negative = shrink) so that
                # cells near the grid boundary remain at a coarser
                # level — needed for valid quadtree transitions.
                if ref.buffer_m != 0.0:
                    gdf = gdf.copy()
                    gdf["geometry"] = gdf.geometry.buffer(ref.buffer_m)
                gdf["refinement_level"] = ref.level
                parts.append(gdf)
            refinement_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True))

            self._log(
                f"Refinement: {len(cfg.grid.refinement)} polygon(s), "
                f"max level {max(r.level for r in cfg.grid.refinement)}"
            )

        with _suppress_stdout():
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


class CreateFetchElevationStage(_CreateStageBase):
    """Fetch NOAA DEM(s) for the AOI before elevation creation."""

    name = "create_fetch_elevation"
    description = "Fetch NOAA topobathy DEM for AOI"

    def _register_catalog(self, catalog_path: Path) -> None:
        """Add *catalog_path* to both config data_libs and the live model."""
        path_str = str(catalog_path.resolve())
        if path_str not in self.config.data_catalog.data_libs:
            self.config.data_catalog.data_libs.append(path_str)
        self.sfincs.data_catalog.from_yml(path_str)

    def run(self) -> dict[str, Any]:
        """Discover and download NOAA DEM(s) overlapping the AOI."""
        from coastal_calibration.utils.noaa_dem import fetch_noaa_dem

        cfg = self.config
        dl_dir = cfg.effective_download_dir

        fetched: list[str] = []
        for ds in cfg.elevation.datasets:
            if ds.source != "noaa":
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

            self._update_substep(f"Fetching NOAA DEM for '{catalog_name}'")
            _, cat_path, _ = fetch_noaa_dem(
                aoi=cfg.aoi,
                output_dir=dl_dir,
                dataset_name=ds.noaa_dataset,
                buffer_deg=0.1,
                catalog_name=catalog_name,
                log=self._log,
            )
            self._register_catalog(cat_path)
            fetched.append(catalog_name)

        self._log(f"Fetched {len(fetched)} NOAA DEM(s): {fetched}")
        return {"status": "completed", "fetched": fetched}


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
    """Add NWM discharge source points derived from hydrofabric flowpaths.

    Flowpath linestrings are read from an NWM hydrofabric GeoPackage,
    intersected with the AOI boundary, and the resulting points are
    registered as discharge source locations in the SFINCS model.
    """

    name = "create_discharge"
    description = "Add NWM discharge source points"

    # Known-good sample dates per domain for NWM retro streamflow validation.
    _SAMPLE_DATES: ClassVar[dict[str, tuple[str, str]]] = {
        "conus": ("2020-01-01", "2020-01-01T01:00:00"),
        "atlgulf": ("2020-01-01", "2020-01-01T01:00:00"),
        "pacific": ("2020-01-01", "2020-01-01T01:00:00"),
        "hawaii": ("2010-01-01", "2010-01-01T01:00:00"),
        "prvi": ("2010-01-01", "2010-01-01T01:00:00"),
        "alaska": ("2018-01-01", "2018-01-01T01:00:00"),
    }

    def _validate_nwm_feature_ids(
        self,
        nd: Any,
    ) -> list[str]:
        """Download a sample NWM streamflow file and check feature IDs."""
        import tempfile
        from datetime import datetime

        import xarray as xr

        from coastal_calibration.downloader import (
            _build_nwm_retro_streamflow_urls,
            _execute_download,
        )

        errors: list[str] = []
        domain = nd.coastal_domain
        sample = self._SAMPLE_DATES.get(domain)
        if sample is None:
            errors.append(f"No sample date configured for domain '{domain}'")
            return errors

        start = datetime.fromisoformat(sample[0])
        end = datetime.fromisoformat(sample[1])

        with tempfile.TemporaryDirectory() as tmp_dir:
            from pathlib import Path

            tmp_path = Path(tmp_dir)
            urls, paths = _build_nwm_retro_streamflow_urls(start, end, tmp_path, domain)
            if not urls:
                errors.append("Could not build NWM streamflow URL for validation")
                return errors

            # Download only the first file
            result = _execute_download(
                urls[:1], paths[:1], "nwm_validation", timeout=120, raise_on_error=False
            )
            if result.errors or not paths[0].exists():
                self._log(
                    "Could not download NWM streamflow sample for ID validation; "
                    "skipping feature_id check",
                    level="warning",
                )
                return errors

            try:
                ds = xr.open_dataset(paths[0])
                try:
                    nwm_ids = set(ds["feature_id"].values.tolist())
                finally:
                    ds.close()
            except Exception as exc:
                self._log(
                    f"Could not read NWM streamflow sample: {exc}; skipping feature_id check",
                    level="warning",
                )
                return errors

            missing = [fid for fid in nd.flowpath_ids if fid not in nwm_ids]
            if missing:
                errors.append(
                    f"NWM feature_id(s) not found in streamflow data for "
                    f"domain '{domain}': {missing}"
                )

        return errors

    def validate(self) -> list[str]:
        """Validate hydrofabric GPKG structure and NWM feature IDs."""
        cfg = self.config
        if cfg.nwm_discharge is None:
            return []

        nd = cfg.nwm_discharge
        errors: list[str] = []

        if not nd.hydrofabric_gpkg.exists():
            errors.append(f"nwm_discharge.hydrofabric_gpkg not found: {nd.hydrofabric_gpkg}")
            return errors

        # --- pyogrio layer validation ---
        import pyogrio

        layers = pyogrio.list_layers(nd.hydrofabric_gpkg)
        layer_names = [name for name, _ in layers]
        if nd.flowpaths_layer not in layer_names:
            errors.append(
                f"Layer '{nd.flowpaths_layer}' not found in "
                f"{nd.hydrofabric_gpkg.name}. "
                f"Available layers: {layer_names}"
            )
            return errors

        # --- pyogrio column validation ---
        info = pyogrio.read_info(nd.hydrofabric_gpkg, layer=nd.flowpaths_layer)
        columns: list[str] = list(info["fields"])
        if nd.flowpath_id_column not in columns:
            errors.append(
                f"Column '{nd.flowpath_id_column}' not found in layer "
                f"'{nd.flowpaths_layer}'. Available columns: {columns}"
            )
            return errors

        # --- NWM streamflow feature_id validation ---
        nwm_errors = self._validate_nwm_feature_ids(nd)
        errors.extend(nwm_errors)

        return errors

    def _snap_to_active_cells(
        self,
        points: list[tuple[float, float, str]],
        model: Any,
    ) -> list[tuple[float, float, str]]:
        """Snap discharge points to the nearest active grid cell.

        For each point, if it already sits on an active cell (``mask == 1``)
        it is kept as-is.  Otherwise the nearest active cell is found via
        a KDTree search and the point is relocated to that cell's face
        centre.  Points with no active cell within the search radius are
        dropped with a warning.

        Returns the list of (x, y, name) tuples on active cells.
        """
        import numpy as np
        from scipy.spatial import KDTree

        grid_ds = model.quadtree_grid.data
        ugrid = grid_ds.ugrid.grid
        face_xy = np.column_stack([ugrid.face_x, ugrid.face_y])
        mask = grid_ds["mask"].to_numpy()

        # Build a KDTree of *all* face centres for initial lookup,
        # and a separate tree of *active-only* centres for snapping.
        tree_all = KDTree(face_xy)
        active_idx = np.where(mask == 1)[0]
        if len(active_idx) == 0:
            self._log("No active cells in grid, cannot place discharge points", level="warning")
            return []
        tree_active = KDTree(face_xy[active_idx])

        snapped: list[tuple[float, float, str]] = []
        for x, y, name in points:
            _, cell_idx = tree_all.query([x, y])
            if mask[cell_idx] == 1:
                # Already on an active cell — keep as-is
                snapped.append((x, y, name))
                continue

            # Snap to nearest active cell
            dist, active_pos = tree_active.query([x, y])
            real_idx = active_idx[active_pos]
            nx, ny = float(face_xy[real_idx, 0]), float(face_xy[real_idx, 1])
            snapped.append((nx, ny, name))
            self._log(f"  {name}: snapped to active cell ({dist:.0f} m away)")

        return snapped

    @staticmethod
    def _downstream_endpoint(geom: Any) -> tuple[float, float] | None:
        """Extract the downstream (last) coordinate of a flowpath geometry."""
        from shapely.geometry import MultiLineString

        if geom is None or geom.is_empty:
            return None
        if isinstance(geom, MultiLineString):
            last_line = list(geom.geoms)[-1]
            return last_line.coords[-1][:2]
        # LineString
        return geom.coords[-1][:2]

    def run(self) -> dict[str, Any]:
        """Extract flowpath outlets from hydrofabric and add as discharge points."""
        cfg = self.config
        if cfg.nwm_discharge is None:
            self._log("No NWM discharge configuration, skipping")
            return {"status": "skipped"}

        import geopandas as gpd

        nd = cfg.nwm_discharge
        model = self.sfincs

        self._update_substep("Reading flowpath geometries from hydrofabric")
        id_col = nd.flowpath_id_column
        ids_str = ", ".join(str(i) for i in nd.flowpath_ids)
        flowpaths_gdf = gpd.read_file(
            nd.hydrofabric_gpkg,
            layer=nd.flowpaths_layer,
            where=f'"{id_col}" IN ({ids_str})',
        )
        self._log(
            f"Read {len(flowpaths_gdf)} flowpath(s) from "
            f"{nd.hydrofabric_gpkg.name}:{nd.flowpaths_layer}"
        )

        if flowpaths_gdf.empty:
            self._log(
                "No flowpaths matched the specified IDs, skipping discharge",
                level="warning",
            )
            return {"status": "skipped", "reason": "no matching flowpaths"}

        # Determine model CRS from the SfincsModel grid
        grid_ds = model.quadtree_grid.data
        model_crs = grid_ds.ugrid.grid.crs

        # Reproject to model CRS
        if flowpaths_gdf.crs is not None and flowpaths_gdf.crs != model_crs:
            flowpaths_gdf = flowpaths_gdf.to_crs(model_crs)

        # Extract downstream endpoint of each flowpath as the discharge location
        self._update_substep("Extracting flowpath outlet points")
        discharge_points: list[tuple[float, float, str]] = []
        for _, row in flowpaths_gdf.iterrows():
            raw_id = row[id_col]
            feature_id = str(int(raw_id)) if isinstance(raw_id, float) else str(raw_id)
            endpoint = self._downstream_endpoint(row.geometry)
            if endpoint is None:
                self._log(f"Flowpath {feature_id} has empty geometry, skipping")
                continue
            discharge_points.append((endpoint[0], endpoint[1], feature_id))

        if not discharge_points:
            self._log("No discharge points extracted from flowpaths")
            return {"status": "completed", "points_added": 0}

        # Snap discharge points to nearest active (wet) grid cell.
        # Intersection points land on the AOI boundary which often
        # falls on inactive cells; snapping moves them inward.
        self._update_substep("Snapping discharge points to active cells")
        snapped = self._snap_to_active_cells(discharge_points, model)
        if not snapped:
            self._log(
                "No discharge points could be placed on active cells",
                level="warning",
            )
            return {"status": "completed", "points_added": 0}

        # Add snapped discharge points to the model
        self._update_substep("Adding discharge source points")
        for x, y, name in snapped:
            model.discharge_points.add_point(x=x, y=y, name=name)

        # Write a reference .src file
        src_path = cfg.output_dir / "sfincs_nwm.src"
        with src_path.open("w") as f:
            for x, y, name in snapped:
                f.write(f'{x:.2f} {y:.2f} "{name}"\n')

        self._log(f"Added {len(snapped)} discharge source point(s), wrote {src_path.name}")
        return {
            "status": "completed",
            "points_added": len(discharge_points),
            "src_file": str(src_path),
        }


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
    "create_fetch_elevation": CreateFetchElevationStage,
    "create_elevation": CreateElevationStage,
    "create_mask": CreateMaskStage,
    "create_boundary": CreateBoundaryStage,
    "create_discharge": CreateDischargeStage,
    "create_subgrid": CreateSubgridStage,
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
