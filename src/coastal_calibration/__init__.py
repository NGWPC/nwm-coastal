"""Coastal Calibration Workflow Python API.

This package provides a Python interface for running coastal model calibration
workflows (SCHISM, SFINCS) using Singularity containers on SLURM-managed HPC clusters.

Example usage:
    from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

    config = CoastalCalibConfig.from_yaml("config.yaml")
    runner = CoastalCalibRunner(config)
    result = runner.submit()

    if result.success:
        print(f"Workflow completed in {result.duration_seconds:.1f}s")
"""

from __future__ import annotations

from coastal_calibration.config.schema import (
    BoundaryConfig,
    BoundarySource,
    CoastalCalibConfig,
    CoastalDomain,
    DownloadConfig,
    MeteoSource,
    ModelConfig,
    ModelType,
    MonitoringConfig,
    PathConfig,
    SchismModelConfig,
    SfincsModelConfig,
    SimulationConfig,
    SlurmConfig,
)
from coastal_calibration.downloader import (
    DATA_SOURCE_DATE_RANGES,
    CoastalSource,
    DateRange,
    Domain,
    DownloadResult,
    DownloadResults,
    GLOFSModel,
    HydroSource,
    download_data,
    get_date_range,
    get_default_sources,
    get_overlapping_range,
)
from coastal_calibration.runner import (
    CoastalCalibRunner,
    WorkflowResult,
    run_workflow,
    submit_workflow,
)
from coastal_calibration.stages.sfincs import (
    CatalogEntry,
    CatalogMetadata,
    DataAdapter,
    DataCatalog,
    create_nc_symlinks,
    generate_data_catalog,
    remove_nc_symlinks,
)
from coastal_calibration.utils.workflow import (
    nwm_coastal_merge_source_sink,
    post_nwm_coastal,
    post_nwm_forcing_coastal,
    pre_nwm_forcing_coastal,
)

__version__ = "0.1.0"

__all__ = [
    "DATA_SOURCE_DATE_RANGES",
    "BoundaryConfig",
    "BoundarySource",
    "CatalogEntry",
    "CatalogMetadata",
    # Config classes
    "CoastalCalibConfig",
    # Runner
    "CoastalCalibRunner",
    "CoastalDomain",
    "CoastalSource",
    "DataAdapter",
    "DataCatalog",
    "DateRange",
    "Domain",
    "DownloadConfig",
    "DownloadResult",
    "DownloadResults",
    "GLOFSModel",
    "HydroSource",
    "MeteoSource",
    "ModelConfig",
    "ModelType",
    "MonitoringConfig",
    "PathConfig",
    "SchismModelConfig",
    "SfincsModelConfig",
    "SimulationConfig",
    "SlurmConfig",
    "WorkflowResult",
    # Data Catalog (SFINCS)
    "create_nc_symlinks",
    # Downloader
    "download_data",
    "generate_data_catalog",
    "get_date_range",
    "get_default_sources",
    "get_overlapping_range",
    # Workflow utilities (Python implementations of bash scripts)
    "nwm_coastal_merge_source_sink",
    "post_nwm_coastal",
    "post_nwm_forcing_coastal",
    "pre_nwm_forcing_coastal",
    "remove_nc_symlinks",
    "run_workflow",
    "submit_workflow",
]
