"""Configuration schema and validation."""

from __future__ import annotations

from coastal_calibration.config.create_schema import (
    DataCatalogConfig,
    ElevationConfig,
    ElevationDataset,
    GridConfig,
    MaskConfig,
    SfincsCreateConfig,
    SubgridConfig,
)
from coastal_calibration.config.schema import (
    BoundaryConfig,
    BoundarySource,
    CoastalCalibConfig,
    CoastalDomain,
    DownloadConfig,
    ModelConfig,
    ModelType,
    MonitoringConfig,
    PathConfig,
    SchismModelConfig,
    SfincsModelConfig,
    SimulationConfig,
)

__all__ = [
    "BoundaryConfig",
    "BoundarySource",
    "CoastalCalibConfig",
    "CoastalDomain",
    "DataCatalogConfig",
    "DownloadConfig",
    "ElevationConfig",
    "ElevationDataset",
    "GridConfig",
    "MaskConfig",
    "ModelConfig",
    "ModelType",
    "MonitoringConfig",
    "PathConfig",
    "SchismModelConfig",
    "SfincsCreateConfig",
    "SfincsModelConfig",
    "SimulationConfig",
    "SubgridConfig",
]
