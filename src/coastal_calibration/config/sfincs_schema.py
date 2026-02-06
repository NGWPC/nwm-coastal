"""SFINCS-specific configuration schema and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from coastal_calibration.config.schema import (
    LogLevel,
    _deep_merge,  # pyright: ignore[reportPrivateUsage]
    _interpolate_config,  # pyright: ignore[reportPrivateUsage]
    _parse_datetime,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class SfincsGridConfig:
    """SFINCS computational grid configuration.

    Parameters
    ----------
    region_geojson : Path
        Path to GeoJSON file defining the model region.
    resolution : float
        Grid resolution in meters.
    crs : str
        Coordinate reference system. Use ``"utm"`` for automatic UTM zone detection.
    rotated : bool
        Whether the grid is rotated to align with the region.
    """

    region_geojson: Path
    resolution: float
    crs: str = "utm"
    rotated: bool = True

    def __post_init__(self) -> None:
        self.region_geojson = Path(self.region_geojson)


@dataclass
class ElevationEntry:
    """A single elevation dataset entry.

    Parameters
    ----------
    elevation : str
        Dataset name in the HydroMT data catalog.
    zmin : float, optional
        Minimum elevation threshold. Only elevations above this value are used.
    zmax : float, optional
        Maximum elevation threshold. Only elevations below this value are used.
    """

    elevation: str
    zmin: float | None = None
    zmax: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for HydroMT, excluding None fields."""
        result: dict[str, Any] = {"elevation": self.elevation}
        if self.zmin is not None:
            result["zmin"] = self.zmin
        if self.zmax is not None:
            result["zmax"] = self.zmax
        return result


@dataclass
class SfincsElevationConfig:
    """SFINCS elevation configuration.

    Parameters
    ----------
    datasets : list[ElevationEntry]
        Ordered list of elevation datasets. Later datasets fill gaps from earlier ones.
    buffer_cells : int
        Number of buffer cells around the domain for interpolation.
    """

    datasets: list[ElevationEntry]
    buffer_cells: int = 1


@dataclass
class SfincsMaskConfig:
    """SFINCS mask configuration for active, water level, and outflow cells.

    Parameters
    ----------
    active_zmin : float
        Minimum elevation for active cells (cells above this are active).
    active_fill_area : float, optional
        Fill inactive gaps smaller than this area in km2.
    active_drop_area : float, optional
        Drop active groups smaller than this area in km2.
    waterlevel_zmax : float
        Maximum elevation for water level boundary cells.
    outflow_polygon : Path, optional
        Path to GeoJSON defining outflow boundary locations.
    reset_bounds : bool
        Whether to reset existing boundary cells before adding new ones.
    """

    active_zmin: float = -5.0
    active_fill_area: float | None = None
    active_drop_area: float | None = None
    waterlevel_zmax: float = -5.0
    outflow_polygon: Path | None = None
    reset_bounds: bool = True

    def __post_init__(self) -> None:
        if self.outflow_polygon is not None:
            self.outflow_polygon = Path(self.outflow_polygon)


@dataclass
class RoughnessEntry:
    """A single roughness dataset entry.

    Parameters
    ----------
    lulc : str
        Land use / land cover dataset name in the HydroMT data catalog.
    reclass_table : Path, optional
        Path to CSV reclassification table mapping land use classes to Manning's n.
    """

    lulc: str
    reclass_table: Path | None = None

    def __post_init__(self) -> None:
        if self.reclass_table is not None:
            self.reclass_table = Path(self.reclass_table)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for HydroMT, excluding None fields."""
        result: dict[str, Any] = {"lulc": self.lulc}
        if self.reclass_table is not None:
            result["reclass_table"] = str(self.reclass_table)
        return result


@dataclass
class SfincsRoughnessConfig:
    """SFINCS roughness configuration.

    Parameters
    ----------
    datasets : list[RoughnessEntry]
        List of land use datasets for deriving roughness.
    manning_land : float
        Default Manning's n for land cells.
    manning_sea : float
        Default Manning's n for sea cells.
    rgh_lev_land : float
        Minimum elevation threshold separating land from sea for roughness.
    """

    datasets: list[RoughnessEntry] = field(default_factory=list)
    manning_land: float = 0.04
    manning_sea: float = 0.02
    rgh_lev_land: float = 0.0


@dataclass
class SfincsSubgridConfig:
    """SFINCS subgrid tables configuration.

    Parameters
    ----------
    enabled : bool
        Whether to generate subgrid-derived tables.
    nr_subgrid_pixels : int
        Number of subgrid pixels per grid cell (higher = finer subgrid resolution).
    write_dep_tif : bool
        Write merged topobathy as GeoTIFF.
    write_man_tif : bool
        Write merged Manning's n as GeoTIFF.
    """

    enabled: bool = True
    nr_subgrid_pixels: int = 20
    write_dep_tif: bool = True
    write_man_tif: bool = True


@dataclass
class SfincsTimingConfig:
    """SFINCS simulation time configuration.

    Parameters
    ----------
    tref : datetime
        Reference time for the simulation.
    tstart : datetime
        Simulation start time.
    tstop : datetime
        Simulation stop time.
    """

    tref: datetime
    tstart: datetime
    tstop: datetime


@dataclass
class SfincsForcingConfig:
    """SFINCS water level forcing configuration.

    Parameters
    ----------
    waterlevel_geodataset : str, optional
        Name of the geodataset containing water level time series for boundary
        forcing (e.g., ``"gtsmv3_eu_era5"``).
    """

    waterlevel_geodataset: str | None = None


@dataclass
class SfincsContainerConfig:
    """SFINCS Singularity container configuration.

    Parameters
    ----------
    docker_tag : str
        Tag for the ``deltares/sfincs-cpu`` Docker image.
    sif_path : Path, optional
        Path to a pre-pulled Singularity SIF file. When set, the runner uses
        this image directly instead of pulling from Docker Hub.
    """

    docker_tag: str = "latest"
    sif_path: Path | None = None

    def __post_init__(self) -> None:
        if self.sif_path is not None:
            self.sif_path = Path(self.sif_path)


@dataclass
class SfincsPathConfig:
    """SFINCS path configuration.

    Parameters
    ----------
    model_root : Path
        Directory where the SFINCS model is written to disk.
    download_dir : Path, optional
        Directory containing downloaded data (used for data catalog generation).
    data_catalog_path : Path, optional
        Path to a pre-existing HydroMT data catalog YAML file.
    """

    model_root: Path
    download_dir: Path | None = None
    data_catalog_path: Path | None = None

    def __post_init__(self) -> None:
        self.model_root = Path(self.model_root)
        if self.download_dir is not None:
            self.download_dir = Path(self.download_dir)
        if self.data_catalog_path is not None:
            self.data_catalog_path = Path(self.data_catalog_path)


@dataclass
class SfincsMonitoringConfig:
    """SFINCS monitoring configuration.

    Parameters
    ----------
    log_level : LogLevel
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : Path, optional
        Path to log file. If None, logs go to stdout only.
    """

    log_level: LogLevel = "INFO"
    log_file: Path | None = None

    def __post_init__(self) -> None:
        if self.log_file is not None:
            self.log_file = Path(self.log_file)


@dataclass
class SfincsConfig:
    """Complete SFINCS workflow configuration.

    Parameters
    ----------
    grid : SfincsGridConfig
        Grid configuration.
    elevation : SfincsElevationConfig
        Elevation datasets configuration.
    mask : SfincsMaskConfig
        Mask configuration for active and boundary cells.
    timing : SfincsTimingConfig
        Simulation time configuration.
    paths : SfincsPathConfig
        Path configuration.
    roughness : SfincsRoughnessConfig
        Roughness configuration.
    subgrid : SfincsSubgridConfig
        Subgrid tables configuration.
    forcing : SfincsForcingConfig
        Water level forcing configuration.
    container : SfincsContainerConfig
        Singularity container configuration.
    monitoring : SfincsMonitoringConfig
        Monitoring configuration.
    data_libs : list[str]
        HydroMT data library names (e.g., ``["artifact_data"]``).
    """

    grid: SfincsGridConfig
    elevation: SfincsElevationConfig
    mask: SfincsMaskConfig
    timing: SfincsTimingConfig
    paths: SfincsPathConfig
    roughness: SfincsRoughnessConfig = field(default_factory=SfincsRoughnessConfig)
    subgrid: SfincsSubgridConfig = field(default_factory=SfincsSubgridConfig)
    forcing: SfincsForcingConfig = field(default_factory=SfincsForcingConfig)
    container: SfincsContainerConfig = field(default_factory=SfincsContainerConfig)
    monitoring: SfincsMonitoringConfig = field(default_factory=SfincsMonitoringConfig)
    data_libs: list[str] = field(default_factory=list)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SfincsConfig:
        """Create config from dictionary."""
        grid_data = data.get("grid", {})
        grid = SfincsGridConfig(**grid_data)

        elev_data = data.get("elevation", {})
        elev_datasets = [
            ElevationEntry(**d) if isinstance(d, dict) else d for d in elev_data.pop("datasets", [])
        ]
        elevation = SfincsElevationConfig(datasets=elev_datasets, **elev_data)

        mask_data = data.get("mask", {})
        mask = SfincsMaskConfig(**mask_data)

        timing_data = data.get("timing", {})
        for key in ("tref", "tstart", "tstop"):
            if key in timing_data:
                timing_data[key] = _parse_datetime(timing_data[key])
        timing = SfincsTimingConfig(**timing_data)

        paths_data = data.get("paths", {})
        paths = SfincsPathConfig(**paths_data)

        rough_data = data.get("roughness", {})
        rough_datasets = [
            RoughnessEntry(**d) if isinstance(d, dict) else d
            for d in rough_data.pop("datasets", [])
        ]
        roughness = SfincsRoughnessConfig(datasets=rough_datasets, **rough_data)

        subgrid_data = data.get("subgrid", {})
        subgrid = SfincsSubgridConfig(**subgrid_data)

        forcing_data = data.get("forcing", {})
        forcing = SfincsForcingConfig(**forcing_data)

        container_data = data.get("container", {})
        container = SfincsContainerConfig(**container_data)

        monitoring_data = data.get("monitoring", {})
        monitoring = SfincsMonitoringConfig(**monitoring_data)

        data_libs = data.get("data_libs", [])

        return cls(
            grid=grid,
            elevation=elevation,
            mask=mask,
            timing=timing,
            paths=paths,
            roughness=roughness,
            subgrid=subgrid,
            forcing=forcing,
            container=container,
            monitoring=monitoring,
            data_libs=data_libs,
        )

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> SfincsConfig:
        """Load SFINCS configuration from YAML file with optional inheritance.

        Supports variable interpolation using ``${section.key}`` syntax and
        config inheritance via the ``_base`` field.

        Parameters
        ----------
        config_path : Path or str
            Path to YAML configuration file.

        Returns
        -------
        SfincsConfig
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

        if "_base" in data:
            base_path = Path(data.pop("_base"))
            if not base_path.is_absolute():
                base_path = config_path.parent / base_path
            base_data = yaml.safe_load(base_path.read_text())
            data = _deep_merge(base_data, data)

        data = _interpolate_config(data)
        return cls._from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "grid": {
                "region_geojson": str(self.grid.region_geojson),
                "resolution": self.grid.resolution,
                "crs": self.grid.crs,
                "rotated": self.grid.rotated,
            },
            "elevation": {
                "datasets": [d.to_dict() for d in self.elevation.datasets],
                "buffer_cells": self.elevation.buffer_cells,
            },
            "mask": {
                "active_zmin": self.mask.active_zmin,
                "active_fill_area": self.mask.active_fill_area,
                "active_drop_area": self.mask.active_drop_area,
                "waterlevel_zmax": self.mask.waterlevel_zmax,
                "outflow_polygon": str(self.mask.outflow_polygon)
                if self.mask.outflow_polygon
                else None,
                "reset_bounds": self.mask.reset_bounds,
            },
            "timing": {
                "tref": self.timing.tref.isoformat(),
                "tstart": self.timing.tstart.isoformat(),
                "tstop": self.timing.tstop.isoformat(),
            },
            "paths": {
                "model_root": str(self.paths.model_root),
                "download_dir": str(self.paths.download_dir) if self.paths.download_dir else None,
                "data_catalog_path": str(self.paths.data_catalog_path)
                if self.paths.data_catalog_path
                else None,
            },
            "roughness": {
                "datasets": [d.to_dict() for d in self.roughness.datasets],
                "manning_land": self.roughness.manning_land,
                "manning_sea": self.roughness.manning_sea,
                "rgh_lev_land": self.roughness.rgh_lev_land,
            },
            "subgrid": {
                "enabled": self.subgrid.enabled,
                "nr_subgrid_pixels": self.subgrid.nr_subgrid_pixels,
                "write_dep_tif": self.subgrid.write_dep_tif,
                "write_man_tif": self.subgrid.write_man_tif,
            },
            "forcing": {
                "waterlevel_geodataset": self.forcing.waterlevel_geodataset,
            },
            "container": {
                "docker_tag": self.container.docker_tag,
                "sif_path": str(self.container.sif_path) if self.container.sif_path else None,
            },
            "monitoring": {
                "log_level": self.monitoring.log_level,
                "log_file": str(self.monitoring.log_file) if self.monitoring.log_file else None,
            },
            "data_libs": self.data_libs,
        }

    def to_yaml(self, path: Path | str) -> None:
        """Write configuration to YAML file.

        Parameters
        ----------
        path : Path or str
            Path to YAML output file. Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False))

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns
        -------
        list of str
            Validation error messages (empty if valid).
        """
        errors: list[str] = []

        if not self.grid.region_geojson.exists():
            errors.append(f"Region GeoJSON not found: {self.grid.region_geojson}")

        if self.grid.resolution <= 0:
            errors.append("grid.resolution must be positive")

        if not self.elevation.datasets:
            errors.append("elevation.datasets must have at least one entry")

        if self.timing.tstart < self.timing.tref:
            errors.append("timing.tstart must be >= timing.tref")

        if self.timing.tstop <= self.timing.tstart:
            errors.append("timing.tstop must be after timing.tstart")

        if self.mask.outflow_polygon is not None and not self.mask.outflow_polygon.exists():
            errors.append(f"Outflow polygon not found: {self.mask.outflow_polygon}")

        if self.container.sif_path is not None and not self.container.sif_path.exists():
            errors.append(f"Singularity image not found: {self.container.sif_path}")

        if self.paths.data_catalog_path is not None and not self.paths.data_catalog_path.exists():
            errors.append(f"Data catalog not found: {self.paths.data_catalog_path}")

        return errors
