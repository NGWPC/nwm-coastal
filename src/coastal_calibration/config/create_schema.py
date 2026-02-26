"""YAML configuration schema for SFINCS model creation workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from coastal_calibration.config.schema import LogLevel, MonitoringConfig


@dataclass
class RefinementLevel:
    """A single quadtree refinement polygon/level pair."""

    polygon: Path
    """Path to a polygon file (GeoJSON, Shapefile, etc.) defining the
    area to refine.  Cells overlapping this polygon are refined to
    *level*.  Use the AOI polygon itself for the innermost level."""

    level: int
    """Refinement level (1 = base resolution, 2 = base/2, 3 = base/4, …).
    Higher levels produce finer cells."""

    buffer_m: float = 0.0
    """Inward (negative) buffer in metres applied to the polygon before
    refinement.  A negative value shrinks the polygon so that cells
    near the grid boundary remain at a coarser level — required for
    valid quadtree transitions.  Set to ``0`` to disable buffering.
    When the refinement polygon coincides with the AOI, a buffer of
    at least ``-2 × base_resolution`` is recommended (e.g. ``-3072``
    for a 1024 m base resolution)."""

    def __post_init__(self) -> None:
        self.polygon = Path(self.polygon).expanduser().resolve()


@dataclass
class GridConfig:
    """Grid generation configuration."""

    resolution: float = 50.0
    """Grid cell resolution in metres (base resolution for quadtree grids)."""

    crs: str = "utm"
    """Coordinate reference system.  Use ``"utm"`` for automatic UTM zone
    detection from the AOI centroid, or an EPSG code string (e.g. ``"EPSG:32617"``)."""

    rotated: bool = True
    """Whether to allow grid rotation for tighter bounding-box fit."""

    refinement: list[RefinementLevel] = field(default_factory=list)
    """Quadtree refinement levels.  Each entry maps a polygon to a
    refinement level.  For example, to refine the whole AOI to level 4
    (base/8), supply::

        refinement:
          - polygon: ./texas_aoi.geojson
            level: 4

    Multiple entries with different polygons / levels enable spatially
    varying resolution (coarser offshore, finer near the coast)."""


@dataclass
class ElevationDataset:
    """A single elevation/bathymetry dataset entry."""

    name: str = "copdem30"
    """HydroMT data-catalog dataset name."""

    zmin: float = 0.001
    """Minimum elevation threshold for this dataset."""


@dataclass
class ElevationConfig:
    """Elevation and bathymetry configuration."""

    datasets: list[ElevationDataset] = field(
        default_factory=lambda: [
            ElevationDataset(name="copdem30", zmin=0.001),
            ElevationDataset(name="gebco", zmin=-20000),
        ]
    )
    """Ordered list of elevation datasets (later entries fill gaps)."""

    buffer_cells: int = 1
    """Number of buffer cells around the grid boundary."""


@dataclass
class MaskConfig:
    """Active-cell mask and boundary configuration."""

    zmin: float = -5.0
    """Minimum elevation for active cells."""

    boundary_zmax: float = -5.0
    """Maximum elevation for waterlevel boundary cells."""

    reset_bounds: bool = True
    """Reset existing boundary conditions before creating new ones."""


@dataclass
class SubgridConfig:
    """Subgrid table configuration.

    Roughness parameters are included here because for quadtree grids
    the Manning coefficients are embedded directly in the subgrid tables.
    """

    nr_subgrid_pixels: int = 5
    """Number of subgrid pixels per grid cell."""

    lulc_dataset: str = "esa_worldcover_2021"
    """Land-use / land-cover dataset for roughness classification."""

    reclass_table: Path | None = None
    """Optional CSV path for custom reclassification table.  When ``None``,
    the HydroMT-SFINCS built-in table for the chosen dataset is used."""

    manning_land: float = 0.04
    """Default Manning coefficient for land cells."""

    manning_sea: float = 0.02
    """Default Manning coefficient for sea cells."""

    def __post_init__(self) -> None:
        if self.reclass_table is not None:
            self.reclass_table = Path(self.reclass_table).expanduser().resolve()


@dataclass
class DataCatalogConfig:
    """HydroMT data catalog configuration."""

    data_libs: list[str] = field(default_factory=list)
    """Additional HydroMT data catalog YAML paths or predefined catalog names."""


#: Valid NWM domains for streamflow validation.
_VALID_NWM_DOMAINS = frozenset({"conus", "atlgulf", "pacific", "hawaii", "prvi", "alaska"})


@dataclass
class NWMDischargeConfig:
    """NWM discharge source point configuration.

    Derives discharge source points by intersecting NWM hydrofabric
    flowpaths with the model AOI boundary.  The intersection points
    are added as SFINCS discharge source locations.
    """

    hydrofabric_gpkg: Path
    """Path to an NWM hydrofabric GeoPackage file."""

    flowpaths_layer: str
    """Layer name inside the GeoPackage containing flowpath linestring
    geometries."""

    flowpath_id_column: str
    """Column in the flowpaths layer whose values identify each flowpath
    and correspond to NWM ``feature_id`` values in CHRTOUT files."""

    flowpath_ids: list[int] = field(default_factory=list)
    """List of NWM feature IDs to extract from the hydrofabric."""

    coastal_domain: str = "conus"
    """NWM coastal domain used for streamflow ID validation
    (``conus``, ``atlgulf``, ``pacific``, ``hawaii``, ``prvi``, or
    ``alaska``)."""

    def __post_init__(self) -> None:
        self.hydrofabric_gpkg = Path(self.hydrofabric_gpkg).expanduser().resolve()


@dataclass
class SfincsCreateConfig:
    """Root configuration for SFINCS model creation workflow.

    Loaded from YAML via :meth:`from_yaml`.  All paths are resolved to
    absolute paths during construction.
    """

    aoi: Path
    """Path to an AOI polygon file (GeoJSON, Shapefile, etc.)."""

    output_dir: Path
    """Directory where the SFINCS model will be written."""

    grid: GridConfig = field(default_factory=GridConfig)
    elevation: ElevationConfig = field(default_factory=ElevationConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    subgrid: SubgridConfig = field(default_factory=SubgridConfig)
    data_catalog: DataCatalogConfig = field(default_factory=DataCatalogConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    nwm_discharge: NWMDischargeConfig | None = None

    def __post_init__(self) -> None:
        self.aoi = Path(self.aoi).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()

    # ------------------------------------------------------------------
    # Stage ordering
    # ------------------------------------------------------------------

    @property
    def stage_order(self) -> list[str]:
        """Ordered list of creation stages to execute.

        Roughness is embedded in the quadtree subgrid tables, so there
        is no separate roughness stage.  The ``create_discharge`` stage
        is included only when :attr:`nwm_discharge` is configured.
        """
        stages = [
            "create_grid",
            "create_elevation",
            "create_mask",
            "create_boundary",
        ]
        if self.nwm_discharge is not None:
            stages.append("create_discharge")
        stages.extend(["create_subgrid", "create_write"])
        return stages

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SfincsCreateConfig:
        """Create config from a raw dictionary."""
        aoi = data.get("aoi")
        if aoi is None:
            raise ValueError("'aoi' is required (path to AOI polygon file)")
        output_dir = data.get("output_dir")
        if output_dir is None:
            raise ValueError("'output_dir' is required (model output directory)")

        grid_data = data.get("grid", {})
        refinement_raw = grid_data.pop("refinement", None)
        if refinement_raw is not None:
            grid_data["refinement"] = [RefinementLevel(**r) for r in refinement_raw]
        grid = GridConfig(**grid_data)

        elev_data = data.get("elevation", {})
        datasets_raw = elev_data.pop("datasets", None)
        if datasets_raw is not None:
            elev_data["datasets"] = [ElevationDataset(**d) for d in datasets_raw]
        elevation = ElevationConfig(**elev_data)

        mask_data = data.get("mask", {})
        mask = MaskConfig(**mask_data)

        subgrid_data = data.get("subgrid", {})
        if subgrid_data.get("reclass_table"):
            subgrid_data["reclass_table"] = Path(subgrid_data["reclass_table"])
        subgrid_cfg = SubgridConfig(**subgrid_data)

        catalog_data = data.get("data_catalog", {})
        data_catalog = DataCatalogConfig(**catalog_data)

        monitoring_data = data.get("monitoring", {})
        if monitoring_data.get("log_file"):
            monitoring_data["log_file"] = Path(monitoring_data["log_file"])
        monitoring = MonitoringConfig(**monitoring_data)

        nwm_discharge: NWMDischargeConfig | None = None
        nwm_data = data.get("nwm_discharge")
        if nwm_data is not None:
            nwm_data["hydrofabric_gpkg"] = Path(nwm_data["hydrofabric_gpkg"])
            nwm_discharge = NWMDischargeConfig(**nwm_data)

        return cls(
            aoi=Path(aoi),
            output_dir=Path(output_dir),
            grid=grid,
            elevation=elevation,
            mask=mask,
            subgrid=subgrid_cfg,
            data_catalog=data_catalog,
            monitoring=monitoring,
            nwm_discharge=nwm_discharge,
        )

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> SfincsCreateConfig:
        """Load configuration from a YAML file.

        Parameters
        ----------
        config_path : Path or str
            Path to YAML configuration file.

        Returns
        -------
        SfincsCreateConfig
            Loaded configuration.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        yaml.YAMLError
            If the YAML file is malformed.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            data = yaml.safe_load(config_path.read_text())
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}") from e

        if data is None:
            raise ValueError(f"Configuration file is empty: {config_path}")

        # Resolve relative paths against the YAML file's directory
        yaml_dir = config_path.parent
        for key in ("aoi", "output_dir"):
            val = data.get(key)
            if val and not Path(val).is_absolute():
                data[key] = str(yaml_dir / val)

        reclass = (data.get("subgrid") or {}).get("reclass_table")
        if reclass and not Path(reclass).is_absolute():
            data["subgrid"]["reclass_table"] = str(yaml_dir / reclass)

        # Resolve relative refinement polygon paths against YAML dir
        grid_data = data.get("grid") or {}
        for ref_entry in grid_data.get("refinement") or []:
            poly = ref_entry.get("polygon")
            if poly and not Path(poly).is_absolute():
                ref_entry["polygon"] = str(yaml_dir / poly)

        # Resolve relative hydrofabric_gpkg path against the YAML directory
        nwm_data = data.get("nwm_discharge") or {}
        gpkg = nwm_data.get("hydrofabric_gpkg")
        if gpkg and not Path(gpkg).is_absolute():
            nwm_data["hydrofabric_gpkg"] = str(yaml_dir / gpkg)
            data["nwm_discharge"] = nwm_data

        # Resolve relative data_libs paths against the YAML directory
        catalog_data = data.get("data_catalog") or {}
        libs = catalog_data.get("data_libs") or []
        resolved_libs: list[str] = []
        for lib in libs:
            lib_path = Path(lib)
            if not lib_path.is_absolute() and lib_path.suffix in (".yml", ".yaml"):
                resolved_libs.append(str((yaml_dir / lib_path).resolve()))
            else:
                resolved_libs.append(lib)
        if resolved_libs:
            catalog_data["data_libs"] = resolved_libs
            data["data_catalog"] = catalog_data

        return cls._from_dict(data)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate configuration and return a list of error messages.

        Returns
        -------
        list of str
            Validation errors (empty when the config is valid).
        """
        errors: list[str] = []

        if not self.aoi.exists():
            errors.append(f"AOI file not found: {self.aoi}")

        if self.grid.resolution <= 0:
            errors.append(f"grid.resolution must be positive, got {self.grid.resolution}")

        if self.elevation.buffer_cells < 0:
            errors.append("elevation.buffer_cells must be non-negative")

        if not self.elevation.datasets:
            errors.append("elevation.datasets must contain at least one entry")

        if self.subgrid.manning_land <= 0:
            errors.append("subgrid.manning_land must be positive")
        if self.subgrid.manning_sea <= 0:
            errors.append("subgrid.manning_sea must be positive")

        if self.subgrid.reclass_table is not None and not self.subgrid.reclass_table.exists():
            errors.append(f"subgrid.reclass_table not found: {self.subgrid.reclass_table}")

        if self.subgrid.nr_subgrid_pixels < 1:
            errors.append("subgrid.nr_subgrid_pixels must be >= 1")

        for ref in self.grid.refinement:
            if not ref.polygon.exists():
                errors.append(f"refinement polygon not found: {ref.polygon}")
            if ref.level < 1:
                errors.append(f"refinement level must be >= 1, got {ref.level}")

        if self.nwm_discharge is not None:
            nd = self.nwm_discharge
            if not nd.hydrofabric_gpkg.exists():
                errors.append(
                    f"nwm_discharge.hydrofabric_gpkg not found: {nd.hydrofabric_gpkg}"
                )
            if not nd.flowpath_ids:
                errors.append("nwm_discharge.flowpath_ids must contain at least one ID")
            if nd.coastal_domain not in _VALID_NWM_DOMAINS:
                errors.append(
                    f"nwm_discharge.coastal_domain must be one of "
                    f"{sorted(_VALID_NWM_DOMAINS)}, got '{nd.coastal_domain}'"
                )

        return errors

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a plain dictionary."""
        return {
            "aoi": str(self.aoi),
            "output_dir": str(self.output_dir),
            "grid": {
                "resolution": self.grid.resolution,
                "crs": self.grid.crs,
                "rotated": self.grid.rotated,
                "refinement": [
                    {"polygon": str(r.polygon), "level": r.level}
                    for r in self.grid.refinement
                ],
            },
            "elevation": {
                "datasets": [
                    {"name": d.name, "zmin": d.zmin} for d in self.elevation.datasets
                ],
                "buffer_cells": self.elevation.buffer_cells,
            },
            "mask": {
                "zmin": self.mask.zmin,
                "boundary_zmax": self.mask.boundary_zmax,
                "reset_bounds": self.mask.reset_bounds,
            },
            "subgrid": {
                "nr_subgrid_pixels": self.subgrid.nr_subgrid_pixels,
                "lulc_dataset": self.subgrid.lulc_dataset,
                "reclass_table": (
                    str(self.subgrid.reclass_table) if self.subgrid.reclass_table else None
                ),
                "manning_land": self.subgrid.manning_land,
                "manning_sea": self.subgrid.manning_sea,
            },
            "data_catalog": {
                "data_libs": self.data_catalog.data_libs,
            },
            "monitoring": {
                "log_level": self.monitoring.log_level,
                "log_file": (
                    str(self.monitoring.log_file) if self.monitoring.log_file else None
                ),
                "enable_progress_tracking": self.monitoring.enable_progress_tracking,
                "enable_timing": self.monitoring.enable_timing,
            },
            "nwm_discharge": (
                {
                    "hydrofabric_gpkg": str(self.nwm_discharge.hydrofabric_gpkg),
                    "flowpaths_layer": self.nwm_discharge.flowpaths_layer,
                    "flowpath_id_column": self.nwm_discharge.flowpath_id_column,
                    "flowpath_ids": self.nwm_discharge.flowpath_ids,
                    "coastal_domain": self.nwm_discharge.coastal_domain,
                }
                if self.nwm_discharge is not None
                else None
            ),
        }

    def to_yaml(self, path: Path | str) -> None:
        """Write configuration to a YAML file.

        Parameters
        ----------
        path : Path or str
            Path to YAML output file.  Parent directories are created
            automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False))
