"""Boundary condition stages for SCHISM workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from coastal_calibration.base import WorkflowStage

if TYPE_CHECKING:
    from pathlib import Path

    from coastal_calibration.config.schema import CoastalCalibConfig, SchismModelConfig
    from coastal_calibration.logging import WorkflowMonitor


def _get_open_boundary_node_count(prebuilt_dir: Path) -> int | None:
    """Return total open boundary nodes from hgrid.gr3, or None on failure."""
    try:
        from coastal_calibration.schism import NWMSCHISMProject

        project = NWMSCHISMProject(prebuilt_dir, validate=False)
        return project.read_boundaries().total_open_nodes
    except Exception:
        return None


class UpdateParamsStage(WorkflowStage):
    """Update SCHISM parameter files."""

    name = "schism_params"
    description = "Create param.nml and symlink mesh files"

    def __init__(self, config: CoastalCalibConfig, monitor: WorkflowMonitor | None = None) -> None:
        super().__init__(config, monitor)
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

    def run(self) -> dict[str, Any]:
        """Create param.nml and symlink static mesh files."""
        from coastal_calibration.schism.prep import update_params

        self._update_substep("Building environment")
        self.build_environment()

        work_dir = self.config.paths.work_dir
        sim = self.config.simulation

        self._log("Creating param.nml and symlinking mesh files")
        self._update_substep("Running update_params")

        param_file = update_params(
            work_dir=work_dir,
            prebuilt_dir=self.model.prebuilt_dir,
            start_date=sim.start_date,
            duration_hours=sim.duration_hours,
            hot_start_file=self.config.paths.hot_start_file,
        )

        self._log(f"Parameter file created: {param_file}")
        return {
            "param_file": str(param_file),
            "status": "completed",
        }


class TPXOBoundaryStage(WorkflowStage):
    """Generate boundary conditions from TPXO tidal atlas."""

    name = "tpxo_boundary"
    description = "Create boundary forcing from TPXO"

    def __init__(self, config: CoastalCalibConfig, monitor: WorkflowMonitor | None = None) -> None:
        super().__init__(config, monitor)
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

    def run(self) -> dict[str, Any]:
        """Generate tidal boundary conditions from TPXO atlas."""
        from coastal_calibration.schism.prep import make_tpxo_boundary

        self._update_substep("Building environment")
        self.build_environment()

        sim = self.config.simulation
        prebuilt_dir = self.model.prebuilt_dir

        # Resolve optional elevation correction file and node count
        corr_path = prebuilt_dir / "elevation_correction.csv"
        correction_file = corr_path if corr_path.exists() else None
        n_open_boundary_nodes = _get_open_boundary_node_count(prebuilt_dir)

        self._log("Generating tidal boundary from TPXO atlas")
        self._update_substep("Running make_tpxo_ocean")

        elev_file = make_tpxo_boundary(
            work_dir=self.config.paths.work_dir,
            start_date=sim.start_date,
            duration_hours=sim.duration_hours,
            timestep_seconds=sim.timestep_seconds,
            prebuilt_dir=prebuilt_dir,
            otps_dir=self.config.paths.otps_dir,
            tpxo_data_dir=self.config.paths.tpxo_data_dir,
            correction_file=correction_file,
            n_open_boundary_nodes=n_open_boundary_nodes,
        )

        self._log(f"TPXO boundary file created: {elev_file}")
        return {
            "elev2d_file": str(elev_file),
            "status": "completed",
        }


class STOFSBoundaryStage(WorkflowStage):
    """Generate boundary conditions from STOFS data."""

    name = "stofs_boundary"
    description = "Regrid STOFS boundary data"

    def __init__(self, config: CoastalCalibConfig, monitor: WorkflowMonitor | None = None) -> None:
        super().__init__(config, monitor)
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

    def _resolve_stofs_file(self) -> Path:
        """Resolve STOFS file path from config or download directory."""
        if self.config.boundary.stofs_file:
            return self.config.boundary.stofs_file

        from coastal_calibration.data.downloader import get_stofs_path

        expected = get_stofs_path(
            self.config.simulation.start_date,
            self.config.paths.download_dir,
        )
        if expected.exists():
            self._log(f"Auto-resolved STOFS file: {expected}")
            return expected

        # Fallback: search for any STOFS file in the directory
        coastal_dir = self.config.paths.download_dir / "coastal" / "stofs"
        if coastal_dir.exists():
            stofs_files = sorted(coastal_dir.rglob("*.fields.cwl.nc"))
            if stofs_files:
                self._log(f"Auto-resolved STOFS file (fallback): {stofs_files[0]}")
                return stofs_files[0]

        msg = f"No STOFS file found. Set boundary.stofs_file or ensure data exists in {coastal_dir}"
        raise FileNotFoundError(msg)

    def run(self) -> dict[str, Any]:
        """Execute STOFS boundary condition regridding."""
        from coastal_calibration.schism.prep import make_stofs_boundary

        self._update_substep("Building environment")
        self.build_environment()

        stofs_file = self._resolve_stofs_file()
        self._log(f"Using STOFS file: {stofs_file}")

        sim = self.config.simulation
        prebuilt_dir = self.model.prebuilt_dir

        # Resolve optional elevation correction file and node count
        corr_path = prebuilt_dir / "elevation_correction.csv"
        correction_file = corr_path if corr_path.exists() else None
        n_open_boundary_nodes = _get_open_boundary_node_count(prebuilt_dir)

        self._update_substep("Running regrid_estofs.py with MPI")

        elev_file = make_stofs_boundary(
            work_dir=self.config.paths.work_dir,
            start_date=sim.start_date,
            duration_hours=sim.duration_hours,
            stofs_file=stofs_file,
            prebuilt_dir=prebuilt_dir,
            mpi_tasks=self.model.total_tasks,
            correction_file=correction_file,
            n_open_boundary_nodes=n_open_boundary_nodes,
        )

        self._log(f"STOFS boundary file created: {elev_file}")
        return {
            "elev2d_file": str(elev_file),
            "status": "completed",
        }

    def validate(self) -> list[str]:
        """Validate STOFS file exists (skipped when download is enabled)."""
        if self.config.download.enabled:
            return []
        errors = []
        if not self.config.boundary.stofs_file:
            errors.append("STOFS file must be specified for STOFS boundary source")
        elif not self.config.boundary.stofs_file.exists():
            errors.append(f"STOFS file not found: {self.config.boundary.stofs_file}")
        return errors


class BoundaryConditionStage(WorkflowStage):
    """Wrapper stage that selects TPXO or STOFS based on config."""

    name = "schism_boundary"
    description = "Generate boundary conditions"

    def run(self) -> dict[str, Any]:
        """Execute appropriate boundary condition stage."""
        if self.config.boundary.source == "tpxo":
            stage = TPXOBoundaryStage(self.config, self.monitor)
        else:
            stage = STOFSBoundaryStage(self.config, self.monitor)

        return stage.run()

    def validate(self) -> list[str]:
        """Validate based on boundary source."""
        if self.config.boundary.source == "stofs":
            stage = STOFSBoundaryStage(self.config, self.monitor)
            return stage.validate()
        return []
