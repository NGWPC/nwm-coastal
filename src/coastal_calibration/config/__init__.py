"""Configuration schema and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DataCatalogConfig": ("coastal_calibration.config.create_schema", "DataCatalogConfig"),
    "ElevationConfig": ("coastal_calibration.config.create_schema", "ElevationConfig"),
    "ElevationDataset": ("coastal_calibration.config.create_schema", "ElevationDataset"),
    "GridConfig": ("coastal_calibration.config.create_schema", "GridConfig"),
    "MaskConfig": ("coastal_calibration.config.create_schema", "MaskConfig"),
    "SfincsCreateConfig": ("coastal_calibration.config.create_schema", "SfincsCreateConfig"),
    "SubgridConfig": ("coastal_calibration.config.create_schema", "SubgridConfig"),
    "BoundaryConfig": ("coastal_calibration.config.schema", "BoundaryConfig"),
    "BoundarySource": ("coastal_calibration.config.schema", "BoundarySource"),
    "CoastalCalibConfig": ("coastal_calibration.config.schema", "CoastalCalibConfig"),
    "CoastalDomain": ("coastal_calibration.config.schema", "CoastalDomain"),
    "DownloadConfig": ("coastal_calibration.config.schema", "DownloadConfig"),
    "ModelConfig": ("coastal_calibration.config.schema", "ModelConfig"),
    "ModelType": ("coastal_calibration.config.schema", "ModelType"),
    "MonitoringConfig": ("coastal_calibration.config.schema", "MonitoringConfig"),
    "PathConfig": ("coastal_calibration.config.schema", "PathConfig"),
    "SchismModelConfig": ("coastal_calibration.config.schema", "SchismModelConfig"),
    "SfincsModelConfig": ("coastal_calibration.config.schema", "SfincsModelConfig"),
    "SimulationConfig": ("coastal_calibration.config.schema", "SimulationConfig"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


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
