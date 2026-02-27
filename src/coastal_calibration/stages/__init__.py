"""Workflow stage implementations."""

from __future__ import annotations

from coastal_calibration.stages.base import WorkflowStage
from coastal_calibration.stages.boundary import (
    BoundaryConditionStage,
    STOFSBoundaryStage,
    TPXOBoundaryStage,
    UpdateParamsStage,
)
from coastal_calibration.stages.download import DownloadStage
from coastal_calibration.stages.forcing import (
    NWMForcingStage,
    PostForcingStage,
    PreForcingStage,
)
from coastal_calibration.stages.schism import (
    PostSCHISMStage,
    PreSCHISMStage,
    SCHISMRunStage,
)
from coastal_calibration.stages.sfincs import (
    create_nc_symlinks,
    generate_data_catalog,
    remove_nc_symlinks,
)
from coastal_calibration.stages.sfincs_build import (
    SfincsDataCatalogStage,
    SfincsDischargeStage,
    SfincsForcingStage,
    SfincsInitStage,
    SfincsObservationPointsStage,
    SfincsPrecipitationStage,
    SfincsPressureStage,
    SfincsRunStage,
    SfincsSymlinksStage,
    SfincsTimingStage,
    SfincsWindStage,
    SfincsWriteStage,
)
from coastal_calibration.stages.sfincs_create import (
    CreateBoundaryStage,
    CreateElevationStage,
    CreateGridStage,
    CreateMaskStage,
    CreateStage,
    CreateSubgridStage,
    CreateWriteStage,
)

__all__ = [
    "BoundaryConditionStage",
    "CreateBoundaryStage",
    "CreateElevationStage",
    "CreateGridStage",
    "CreateMaskStage",
    "CreateStage",
    "CreateSubgridStage",
    "CreateWriteStage",
    "DownloadStage",
    "NWMForcingStage",
    "PostForcingStage",
    "PostSCHISMStage",
    "PreForcingStage",
    "PreSCHISMStage",
    "SCHISMRunStage",
    "STOFSBoundaryStage",
    "SfincsDataCatalogStage",
    "SfincsDischargeStage",
    "SfincsForcingStage",
    "SfincsInitStage",
    "SfincsObservationPointsStage",
    "SfincsPrecipitationStage",
    "SfincsPressureStage",
    "SfincsRunStage",
    "SfincsSymlinksStage",
    "SfincsTimingStage",
    "SfincsWindStage",
    "SfincsWriteStage",
    "TPXOBoundaryStage",
    "UpdateParamsStage",
    "WorkflowStage",
    "create_nc_symlinks",
    "generate_data_catalog",
    "remove_nc_symlinks",
]
