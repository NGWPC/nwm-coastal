"""YAML configuration schema for SFINCS model creation workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from coastal_calibration.config.schema import MonitoringConfig


@dataclass
class RefinementLevel:
    """A single quadtree refinement polygon/level pair."""

    polygon: Path

    level: int

    #: Inward (negative) buffer in meters applied to the polygon before
    #: refinement.  A negative value shrinks the polygon so that cells
    #: near the grid boundary remain at a coarser level — required for
    #: valid quadtree transitions.  Set to ``0`` to disable buffering.
    #: When the refinement polygon coincides with the AOI, a buffer of
    #: at least ``-2 x base_resolution`` is recommended (e.g. ``-3072``
    #: for a 1024 m base resolution).
    buffer_m: float = 0.0

    def __post_init__(self) -> None:
        self.polygon = Path(self.polygon).expanduser().resolve()


@dataclass
class GridConfig:
    """Grid generation configuration."""

    #: Grid cell resolution in meters (base resolution for quadtree grids).
    resolution: float = 50.0

    #: Coordinate reference system.  Use ``"utm"`` for automatic UTM zone
    #: detection from the AOI centroid, or an EPSG code string (e.g. ``"EPSG:32617"``).
    crs: str = "utm"

    #: Whether to allow grid rotation for tighter bounding-box fit.
    rotated: bool = True

    #: Quadtree refinement levels.  Each entry maps a polygon to a
    #: refinement level.  For example, to refine the whole AOI to level 4
    #: (base/8), supply::
    #:
    #:     refinement:
    #:       - polygon: ./texas_aoi.geojson
    #:         level: 4
    #:
    #: Multiple entries with different polygons / levels enable spatially
    #: varying resolution (coarser offshore, finer near the coast).
    refinement: list[RefinementLevel] = field(default_factory=list)


@dataclass
class ElevationDataset:
    """A single elevation/bathymetry dataset entry."""

    #: HydroMT data-catalog dataset name.
    name: str = "copdem_30m"

    #: Minimum elevation threshold for this dataset.
    zmin: float = 0.001

    #: Data source for auto-fetching.  Supported values:
    #:
    #: - ``"nws_30m"`` — NWS 30 m topo-bathymetric DEM from icechunk (S3)
    #: - ``"noaa_3m"`` — NOAA ~3 m coastal topobathy DEM from S3
    #: - ``"copdem_30m"`` — Copernicus DEM 30 m from AWS S3
    #: - ``"gebco_15arcs"`` — GEBCO 15 arc-second bathymetry (~450 m) via CEDA
    #:
    #: When set, the ``create_fetch_data`` stage downloads the dataset
    #: for the AOI automatically.  When ``None``, the dataset must
    #: already exist in ``data_catalog.data_libs``.
    source: str | None = None

    #: Explicit NOAA dataset name (e.g. ``"TX_Coastal_DEM_2018_8899"``).
    #: Only used when ``source`` is ``"noaa_3m"``.  When ``None``, the best
    #: dataset is auto-discovered based on AOI overlap and resolution.
    noaa_dataset: str | None = None

    #: NWS topobathy coastal domain identifier (e.g. ``"atlgulf"``,
    #: ``"hi"``, ``"prvi"``, ``"pacific"``, ``"ak"``).
    #: Only used when ``source`` is ``"nws_30m"``.
    coastal_domain: str | None = None


@dataclass
class ElevationConfig:
    """Elevation and bathymetry configuration."""

    datasets: list[ElevationDataset] = field(
        default_factory=lambda: [
            ElevationDataset(name="copdem_30m", zmin=0.001, source="copdem_30m"),
            ElevationDataset(name="gebco_15arcs", zmin=-20000, source="gebco_15arcs"),
        ]
    )

    #: Number of buffer cells around the grid boundary.
    buffer_cells: int = 1


@dataclass
class MaskConfig:
    """Active-cell mask and boundary configuration."""

    #: Minimum elevation for active cells.
    zmin: float = -5.0

    #: Maximum elevation for waterlevel boundary cells.
    boundary_zmax: float = -5.0

    #: Reset existing boundary conditions before creating new ones.
    reset_bounds: bool = True


@dataclass
class SubgridConfig:
    """Subgrid table configuration.

    Roughness parameters are included here because for quadtree grids
    the Manning coefficients are embedded directly in the subgrid tables.
    """

    #: Number of subgrid pixels per grid cell.
    nr_subgrid_pixels: int = 5

    #: Land-use / land-cover dataset for roughness classification.
    lulc_dataset: str = "esa_worldcover"

    #: Data source for auto-fetching the LULC dataset.  Currently
    #: only ``"esa_worldcover"`` is supported (tiles from AWS S3).
    #: When ``None``, the dataset must already exist in
    #: ``data_catalog.data_libs``.
    lulc_source: str | None = "esa_worldcover"

    #: Optional CSV path for custom reclassification table.  When ``None``,
    #: the HydroMT-SFINCS built-in table for the chosen dataset is used.
    reclass_table: Path | None = None

    #: Default Manning coefficient for land cells.
    manning_land: float = 0.04

    #: Default Manning coefficient for sea cells.
    manning_sea: float = 0.02

    def __post_init__(self) -> None:
        if self.reclass_table is not None:
            self.reclass_table = Path(self.reclass_table).expanduser().resolve()
        # Backward compatibility: normalize legacy name.
        if self.lulc_dataset == "esa_worldcover_2021":
            self.lulc_dataset = "esa_worldcover"


@dataclass
class DataCatalogConfig:
    """HydroMT data catalog configuration."""

    #: Additional HydroMT data catalog YAML paths or predefined catalog names.
    data_libs: list[str] = field(default_factory=list)


#: Valid NWM domains for streamflow validation.
_VALID_NWM_DOMAINS = frozenset({"conus", "atlgulf", "pacific", "hawaii", "prvi", "alaska"})


@dataclass
class NWMDischargeConfig:
    """NWM discharge source point configuration.

    Derives discharge source points by intersecting NWM hydrofabric
    flowpaths with the model AOI boundary.  The intersection points
    are added as SFINCS discharge source locations.
    """

    #: Path to an NWM hydrofabric GeoPackage file.
    hydrofabric_gpkg: Path

    #: Layer name inside the GeoPackage containing flowpath linestring
    #: geometries.
    flowpaths_layer: str

    #: Column in the flowpaths layer whose values identify each flowpath
    #: and correspond to NWM ``feature_id`` values in CHRTOUT files.
    flowpath_id_column: str

    #: List of NWM feature IDs to extract from the hydrofabric.
    flowpath_ids: list[int] = field(default_factory=list)

    #: NWM coastal domain used for streamflow ID validation
    #: (``conus``, ``atlgulf``, ``pacific``, ``hawaii``, ``prvi``, or
    #: ``alaska``).
    coastal_domain: str = "conus"

    def __post_init__(self) -> None:
        self.hydrofabric_gpkg = Path(self.hydrofabric_gpkg).expanduser().resolve()


@dataclass
class SfincsCreateConfig:
    """Root configuration for SFINCS model creation workflow.

    Loaded from YAML via :meth:`from_yaml`.  All paths are resolved to
    absolute paths during construction.
    """

    #: Path to an AOI polygon file (GeoJSON, Shapefile, etc.).
    aoi: Path

    #: Directory where the SFINCS model will be written.
    output_dir: Path

    #: Directory for downloaded data (NOAA DEMs, etc.).  Defaults to
    #: ``output_dir / "downloads"`` when ``None``.
    download_dir: Path | None = None

    grid: GridConfig = field(default_factory=GridConfig)
    elevation: ElevationConfig = field(default_factory=ElevationConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    subgrid: SubgridConfig = field(default_factory=SubgridConfig)
    data_catalog: DataCatalogConfig = field(default_factory=DataCatalogConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    nwm_discharge: NWMDischargeConfig | None = None

    #: When True, automatically query NOAA CO-OPS for water level
    #: stations within the model domain and add them as observation
    #: points.  Requires the ``plot`` optional dependencies.
    add_noaa_gages: bool = False

    #: Observation point specifications as list of dicts with
    #: ``x``, ``y``, ``name`` keys (coordinates in model CRS).
    observation_points: list[dict[str, Any]] = field(default_factory=list)

    #: Path to a GeoJSON file with observation point locations.
    observation_locations_file: Path | None = None

    #: Whether to merge with pre-existing observation points.
    merge_observations: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    #: Valid ``ElevationDataset.source`` values.
    _VALID_ELEV_SOURCES = frozenset({"nws_30m", "noaa_3m", "copdem_30m", "gebco_15arcs"})

    #: Valid ``SubgridConfig.lulc_source`` values.
    _VALID_LULC_SOURCES = frozenset({"esa_worldcover"})

    def __post_init__(self) -> None:
        self.aoi = Path(self.aoi).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.observation_locations_file is not None:
            self.observation_locations_file = (
                Path(self.observation_locations_file).expanduser().resolve()
            )
        if self.download_dir is not None:
            self.download_dir = Path(self.download_dir).expanduser().resolve()

    @property
    def effective_download_dir(self) -> Path:
        """Effective download directory (fallback to output_dir/downloads)."""
        return self.download_dir or self.output_dir / "downloads"

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
        stages = ["create_grid"]
        has_elev_source = any(d.source is not None for d in self.elevation.datasets)
        has_lulc_source = self.subgrid.lulc_source is not None
        if has_elev_source or has_lulc_source:
            stages.append("create_fetch_data")
        stages.extend(
            [
                "create_elevation",
                "create_mask",
                "create_boundary",
            ]
        )
        if self.nwm_discharge is not None:
            stages.append("create_discharge")
        stages.append("create_subgrid")
        has_obs = (
            self.add_noaa_gages
            or bool(self.observation_points)
            or self.observation_locations_file is not None
        )
        if has_obs:
            stages.append("create_obs")
        stages.append("create_write")
        return stages

    # ------------------------------------------------------------------
    # Serialisation / deserialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SfincsCreateConfig:
        """Create config from a plain dictionary.

        Parameters
        ----------
        data : dict
            Configuration dictionary with the same structure as the YAML
            file (see :meth:`to_dict` for the expected keys).

        Returns
        -------
        SfincsCreateConfig
        """
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

        download_dir_raw = data.get("download_dir")
        download_dir = Path(download_dir_raw) if download_dir_raw else None

        add_noaa_gages = data.get("add_noaa_gages", False)
        observation_points = data.get("observation_points", [])
        obs_file_raw = data.get("observation_locations_file")
        observation_locations_file = Path(obs_file_raw) if obs_file_raw else None
        merge_observations = data.get("merge_observations", False)

        return cls(
            aoi=Path(aoi),
            output_dir=Path(output_dir),
            download_dir=download_dir,
            grid=grid,
            elevation=elevation,
            mask=mask,
            subgrid=subgrid_cfg,
            data_catalog=data_catalog,
            monitoring=monitoring,
            nwm_discharge=nwm_discharge,
            add_noaa_gages=add_noaa_gages,
            observation_points=observation_points,
            observation_locations_file=observation_locations_file,
            merge_observations=merge_observations,
        )

    @staticmethod
    def _resolve_relative_paths(data: dict[str, Any], yaml_dir: Path) -> None:
        """Resolve relative paths in *data* against *yaml_dir* in place."""
        for key in ("aoi", "output_dir", "download_dir", "observation_locations_file"):
            val = data.get(key)
            if val and not Path(val).is_absolute():
                data[key] = str(yaml_dir / val)

        reclass = (data.get("subgrid") or {}).get("reclass_table")
        if reclass and not Path(reclass).is_absolute():
            data["subgrid"]["reclass_table"] = str(yaml_dir / reclass)

        for ref_entry in (data.get("grid") or {}).get("refinement") or []:
            poly = ref_entry.get("polygon")
            if poly and not Path(poly).is_absolute():
                ref_entry["polygon"] = str(yaml_dir / poly)

        nwm_data = data.get("nwm_discharge") or {}
        gpkg = nwm_data.get("hydrofabric_gpkg")
        if gpkg and not Path(gpkg).is_absolute():
            nwm_data["hydrofabric_gpkg"] = str(yaml_dir / gpkg)
            data["nwm_discharge"] = nwm_data

        catalog_data = data.get("data_catalog") or {}
        libs = catalog_data.get("data_libs") or []
        resolved_libs = [
            str((yaml_dir / lib).resolve())
            if not Path(lib).is_absolute() and Path(lib).suffix in (".yml", ".yaml")
            else lib
            for lib in libs
        ]
        if resolved_libs:
            catalog_data["data_libs"] = resolved_libs
            data["data_catalog"] = catalog_data

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

        cls._resolve_relative_paths(data, config_path.parent)
        return cls.from_dict(data)

    def _validate_elevation(self) -> list[str]:
        """Validate elevation configuration."""
        errors: list[str] = []
        if self.elevation.buffer_cells < 0:
            errors.append("elevation.buffer_cells must be non-negative")
        if not self.elevation.datasets:
            errors.append("elevation.datasets must contain at least one entry")
        for ds in self.elevation.datasets:
            if ds.source is not None and ds.source not in self._VALID_ELEV_SOURCES:
                errors.append(
                    f"elevation.datasets[{ds.name}].source must be one of "
                    f"{sorted(self._VALID_ELEV_SOURCES)} or None, got '{ds.source}'"
                )
            if ds.noaa_dataset is not None and ds.source != "noaa_3m":
                errors.append(
                    f"elevation.datasets[{ds.name}].noaa_dataset is set but source is not 'noaa_3m'"
                )
            if ds.coastal_domain is not None and ds.source != "nws_30m":
                errors.append(
                    f"elevation.datasets[{ds.name}].coastal_domain is set but source is not 'nws_30m'"
                )
            if ds.source == "nws_30m" and ds.coastal_domain is None:
                errors.append(
                    f"elevation.datasets[{ds.name}].coastal_domain is required when source is 'nws_30m'"
                )
        return errors

    def _validate_subgrid(self) -> list[str]:
        """Validate subgrid configuration."""
        errors: list[str] = []
        if self.subgrid.manning_land <= 0:
            errors.append("subgrid.manning_land must be positive")
        if self.subgrid.manning_sea <= 0:
            errors.append("subgrid.manning_sea must be positive")
        if self.subgrid.reclass_table is not None and not self.subgrid.reclass_table.exists():
            errors.append(f"subgrid.reclass_table not found: {self.subgrid.reclass_table}")
        if self.subgrid.nr_subgrid_pixels < 1:
            errors.append("subgrid.nr_subgrid_pixels must be >= 1")
        if (
            self.subgrid.lulc_source is not None
            and self.subgrid.lulc_source not in self._VALID_LULC_SOURCES
        ):
            errors.append(
                f"subgrid.lulc_source must be one of "
                f"{sorted(self._VALID_LULC_SOURCES)} or None, "
                f"got '{self.subgrid.lulc_source}'"
            )
        return errors

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

        errors.extend(self._validate_elevation())
        errors.extend(self._validate_subgrid())

        for ref in self.grid.refinement:
            if not ref.polygon.exists():
                errors.append(f"refinement polygon not found: {ref.polygon}")
            if ref.level < 1:
                errors.append(f"refinement level must be >= 1, got {ref.level}")

        if self.nwm_discharge is not None:
            nd = self.nwm_discharge
            if not nd.hydrofabric_gpkg.exists():
                errors.append(f"nwm_discharge.hydrofabric_gpkg not found: {nd.hydrofabric_gpkg}")
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
            **({"download_dir": str(self.download_dir)} if self.download_dir else {}),
            "grid": {
                "resolution": self.grid.resolution,
                "crs": self.grid.crs,
                "rotated": self.grid.rotated,
                "refinement": [
                    {
                        "polygon": str(r.polygon),
                        "level": r.level,
                        **({"buffer_m": r.buffer_m} if r.buffer_m != 0.0 else {}),
                    }
                    for r in self.grid.refinement
                ],
            },
            "elevation": {
                "datasets": [
                    {
                        "name": d.name,
                        "zmin": d.zmin,
                        **({"source": d.source} if d.source else {}),
                        **({"noaa_dataset": d.noaa_dataset} if d.noaa_dataset else {}),
                        **({"coastal_domain": d.coastal_domain} if d.coastal_domain else {}),
                    }
                    for d in self.elevation.datasets
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
                **({"lulc_source": self.subgrid.lulc_source} if self.subgrid.lulc_source else {}),
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
                "log_file": (str(self.monitoring.log_file) if self.monitoring.log_file else None),
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
            "add_noaa_gages": self.add_noaa_gages,
            "observation_points": self.observation_points,
            "observation_locations_file": (
                str(self.observation_locations_file) if self.observation_locations_file else None
            ),
            "merge_observations": self.merge_observations,
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
