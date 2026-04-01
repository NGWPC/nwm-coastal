"""Coastal Calibration Workflow Python API."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coastal_calibration.config.create_schema import SfincsCreateConfig
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
    )
    from coastal_calibration.creator import SfincsCreator
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
    from coastal_calibration.plotting import (
        SfincsGridInfo,
        plot_floodmap,
        plot_mesh,
        plot_station_comparison,
        plotable_stations,
    )
    from coastal_calibration.runner import (
        CoastalCalibRunner,
        WorkflowResult,
        run_workflow,
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
    from coastal_calibration.utils.logging import configure_logger
    from coastal_calibration.utils.workflow import (
        nwm_coastal_merge_source_sink,
        post_nwm_coastal,
        post_nwm_forcing_coastal,
        pre_nwm_forcing_coastal,
    )

try:
    __version__ = version("coastal_calibration")
except PackageNotFoundError:
    __version__ = "999"

# ---------------------------------------------------------------------------
# Lazy public API: heavy imports (xarray, geopandas, matplotlib, …) are
# deferred until the caller actually accesses a name.  This keeps the CLI
# startup fast (``coastal-calibration --version`` no longer pulls the entire
# dependency tree).  The TYPE_CHECKING block above gives pyright/mypy full
# visibility into the public API without executing any imports at runtime.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Config classes
    "BoundaryConfig": ("coastal_calibration.config.schema", "BoundaryConfig"),
    "BoundarySource": ("coastal_calibration.config.schema", "BoundarySource"),
    "CoastalCalibConfig": ("coastal_calibration.config.schema", "CoastalCalibConfig"),
    "CoastalDomain": ("coastal_calibration.config.schema", "CoastalDomain"),
    "DownloadConfig": ("coastal_calibration.config.schema", "DownloadConfig"),
    "MeteoSource": ("coastal_calibration.config.schema", "MeteoSource"),
    "ModelConfig": ("coastal_calibration.config.schema", "ModelConfig"),
    "ModelType": ("coastal_calibration.config.schema", "ModelType"),
    "MonitoringConfig": ("coastal_calibration.config.schema", "MonitoringConfig"),
    "PathConfig": ("coastal_calibration.config.schema", "PathConfig"),
    "SchismModelConfig": ("coastal_calibration.config.schema", "SchismModelConfig"),
    "SfincsModelConfig": ("coastal_calibration.config.schema", "SfincsModelConfig"),
    "SimulationConfig": ("coastal_calibration.config.schema", "SimulationConfig"),
    # SFINCS creation
    "SfincsCreateConfig": ("coastal_calibration.config.create_schema", "SfincsCreateConfig"),
    "SfincsCreator": ("coastal_calibration.creator", "SfincsCreator"),
    # Downloader
    "DATA_SOURCE_DATE_RANGES": ("coastal_calibration.downloader", "DATA_SOURCE_DATE_RANGES"),
    "CoastalSource": ("coastal_calibration.downloader", "CoastalSource"),
    "DateRange": ("coastal_calibration.downloader", "DateRange"),
    "Domain": ("coastal_calibration.downloader", "Domain"),
    "DownloadResult": ("coastal_calibration.downloader", "DownloadResult"),
    "DownloadResults": ("coastal_calibration.downloader", "DownloadResults"),
    "GLOFSModel": ("coastal_calibration.downloader", "GLOFSModel"),
    "HydroSource": ("coastal_calibration.downloader", "HydroSource"),
    "download_data": ("coastal_calibration.downloader", "download_data"),
    "get_date_range": ("coastal_calibration.downloader", "get_date_range"),
    "get_default_sources": ("coastal_calibration.downloader", "get_default_sources"),
    "get_overlapping_range": ("coastal_calibration.downloader", "get_overlapping_range"),
    # Plotting
    "SfincsGridInfo": ("coastal_calibration.plotting", "SfincsGridInfo"),
    "plot_floodmap": ("coastal_calibration.plotting", "plot_floodmap"),
    "plot_mesh": ("coastal_calibration.plotting", "plot_mesh"),
    "plot_station_comparison": ("coastal_calibration.plotting", "plot_station_comparison"),
    "plotable_stations": ("coastal_calibration.plotting", "plotable_stations"),
    # Runner
    "CoastalCalibRunner": ("coastal_calibration.runner", "CoastalCalibRunner"),
    "WorkflowResult": ("coastal_calibration.runner", "WorkflowResult"),
    "run_workflow": ("coastal_calibration.runner", "run_workflow"),
    # Data Catalog (SFINCS)
    "CatalogEntry": ("coastal_calibration.stages.sfincs", "CatalogEntry"),
    "CatalogMetadata": ("coastal_calibration.stages.sfincs", "CatalogMetadata"),
    "DataAdapter": ("coastal_calibration.stages.sfincs", "DataAdapter"),
    "DataCatalog": ("coastal_calibration.stages.sfincs", "DataCatalog"),
    "create_nc_symlinks": ("coastal_calibration.stages.sfincs", "create_nc_symlinks"),
    "generate_data_catalog": ("coastal_calibration.stages.sfincs", "generate_data_catalog"),
    "remove_nc_symlinks": ("coastal_calibration.stages.sfincs", "remove_nc_symlinks"),
    # Logging
    "configure_logger": ("coastal_calibration.utils.logging", "configure_logger"),
    # Workflow utilities
    "nwm_coastal_merge_source_sink": (
        "coastal_calibration.utils.workflow",
        "nwm_coastal_merge_source_sink",
    ),
    "post_nwm_coastal": ("coastal_calibration.utils.workflow", "post_nwm_coastal"),
    "post_nwm_forcing_coastal": (
        "coastal_calibration.utils.workflow",
        "post_nwm_forcing_coastal",
    ),
    "pre_nwm_forcing_coastal": (
        "coastal_calibration.utils.workflow",
        "pre_nwm_forcing_coastal",
    ),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        # Cache on the module so __getattr__ is not called again
        globals()[name] = val
        return val
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "DATA_SOURCE_DATE_RANGES",
    "BoundaryConfig",
    "BoundarySource",
    "CatalogEntry",
    "CatalogMetadata",
    "CoastalCalibConfig",
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
    "SfincsCreateConfig",
    "SfincsCreator",
    "SfincsGridInfo",
    "SfincsModelConfig",
    "SimulationConfig",
    "WorkflowResult",
    "__version__",
    "configure_logger",
    "create_nc_symlinks",
    "download_data",
    "generate_data_catalog",
    "get_date_range",
    "get_default_sources",
    "get_overlapping_range",
    "nwm_coastal_merge_source_sink",
    "plot_floodmap",
    "plot_mesh",
    "plot_station_comparison",
    "plotable_stations",
    "post_nwm_coastal",
    "post_nwm_forcing_coastal",
    "pre_nwm_forcing_coastal",
    "remove_nc_symlinks",
    "run_workflow",
]
