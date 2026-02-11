"""Boundary condition stages for SCHISM workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coastal_calibration.stages.base import WorkflowStage

if TYPE_CHECKING:
    from pathlib import Path


class UpdateParamsStage(WorkflowStage):
    """Update SCHISM parameter files."""

    name = "update_params"
    description = "Create SCHISM param.nml"
    requires_container = True

    def run(self) -> dict[str, Any]:
        """Execute parameter file creation."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._log("Creating param.nml and symlinking mesh files")
        self._update_substep("Running update_params")
        script_path = self._get_scripts_dir() / "run_sing_coastal_workflow_update_params.bash"

        self.run_singularity_command(
            [str(script_path)],
            env=env,
        )

        param_file = self.config.paths.work_dir / "param.nml"
        self._log(f"Parameter file created: {param_file}")
        return {
            "param_file": str(param_file),
            "status": "completed",
        }


class TPXOBoundaryStage(WorkflowStage):
    """Generate boundary conditions from TPXO tidal atlas."""

    name = "tpxo_boundary"
    description = "Create boundary forcing from TPXO"
    requires_container = True

    def run(self) -> dict[str, Any]:
        """Execute TPXO boundary condition generation."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._log("Generating tidal boundary from TPXO atlas")
        self._update_substep("Running make_tpxo_ocean")
        script_path = self._get_scripts_dir() / "run_sing_coastal_workflow_make_tpxo_ocean.bash"

        self.run_singularity_command(
            [str(script_path)],
            env=env,
        )

        elev_file = self.config.paths.work_dir / "elev2D.th.nc"
        self._log(f"TPXO boundary file created: {elev_file}")
        return {
            "elev2d_file": str(elev_file),
            "status": "completed",
        }


class STOFSBoundaryStage(WorkflowStage):
    """Generate boundary conditions from STOFS data."""

    name = "stofs_boundary"
    description = "Regrid STOFS boundary data"
    requires_container = True

    def _resolve_stofs_file(self) -> Path:
        """Resolve STOFS file path from config or download directory."""
        if self.config.boundary.stofs_file:
            return self.config.boundary.stofs_file

        from coastal_calibration.downloader import get_stofs_path

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
        self._update_substep("Building environment")
        env = self.build_environment()

        stofs_file = self._resolve_stofs_file()
        self._log(f"Using STOFS file: {stofs_file}")

        self._update_substep("Pre-processing STOFS data")
        pre_script = self._get_scripts_dir() / "run_sing_coastal_workflow_pre_make_stofs_ocean.bash"

        result = self.run_singularity_command(
            [str(pre_script)],
            env=env,
        )

        length_hrs = (
            result.stdout.strip() if result.stdout else str(self.config.simulation.duration_hours)
        )

        self._update_substep("Running regrid_estofs.py with MPI")
        work_dir = self.config.paths.work_dir

        env["CYCLE_DATE"] = self.config.simulation.start_pdy
        env["CYCLE_TIME"] = f"{self.config.simulation.start_cyc}00"
        env["LENGTH_HRS"] = length_hrs
        env["ESTOFS_INPUT_FILE"] = str(stofs_file)
        env["SCHISM_OUTPUT_FILE"] = str(work_dir / "elev2D.th.nc")
        env["OPEN_BNDS_HGRID_FILE"] = str(work_dir / "open_bnds_hgrid.nc")

        conda_envs = env.get("CONDA_ENVS_PATH", "")
        conda_env = env.get("CONDA_ENV_NAME", "ngen_forcing_coastal")
        ush_dir = env.get("USHnwm", "")

        python_path = f"{conda_envs}/{conda_env}/bin/python"
        regrid_script = f"{ush_dir}/wrf_hydro_workflow_dev/coastal/regrid_estofs.py"

        self.run_singularity_command(
            [
                python_path,
                regrid_script,
                env["ESTOFS_INPUT_FILE"],
                env["OPEN_BNDS_HGRID_FILE"],
                env["SCHISM_OUTPUT_FILE"],
            ],
            env=env,
            use_mpi=True,
        )

        self._update_substep("Post-processing STOFS data")
        post_script = (
            self._get_scripts_dir() / "run_sing_coastal_workflow_post_make_stofs_ocean.bash"
        )

        self.run_singularity_command(
            [str(post_script)],
            env=env,
        )

        elev_file = work_dir / "elev2D.th.nc"
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

    name = "boundary_conditions"
    description = "Generate boundary conditions"
    requires_container = True

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
