"""SCHISM model execution stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coastal_calibration.config.schema import SchismModelConfig
from coastal_calibration.stages.base import WorkflowStage

if TYPE_CHECKING:
    from coastal_calibration.config.schema import CoastalCalibConfig
    from coastal_calibration.utils.logging import WorkflowMonitor


class PreSCHISMStage(WorkflowStage):
    """Prepare input files for SCHISM execution."""

    name = "pre_schism"
    description = "Prepare SCHISM inputs (discharge, partitioning)"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SchismModelConfig)  # noqa: S101
        self.model: SchismModelConfig = config.model_config

    def run(self) -> dict[str, Any]:
        """Execute SCHISM pre-processing."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._update_substep("Running pre_schism")
        script_path = self._get_scripts_dir() / "run_sing_coastal_workflow_pre_schism.bash"

        self.run_singularity_command(
            [str(script_path)],
            env=env,
        )

        work_dir = self.config.paths.work_dir

        # Verify critical files produced by earlier stages exist
        required_files = ["source.nc", "param.nml", "hgrid.gr3"]
        missing = [f for f in required_files if not (work_dir / f).exists()]
        if missing:
            raise RuntimeError(
                f"pre_schism: required files missing from {work_dir}: {', '.join(missing)}. "
                "Check logs from earlier stages (initial_discharge, combine_sink_source, "
                "merge_source_sink, update_params) for errors."
            )

        return {
            "partition_file": str(work_dir / "partition.prop"),
            "outputs_dir": str(work_dir / "outputs"),
            "status": "completed",
        }


class SCHISMRunStage(WorkflowStage):
    """Execute SCHISM model with MPI."""

    name = "schism_run"
    description = "Run SCHISM model (MPI)"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SchismModelConfig)  # noqa: S101
        self.model: SchismModelConfig = config.model_config

    def run(self) -> dict[str, Any]:
        """Execute SCHISM model run."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._update_substep("Setting up MPI environment")
        nfs_mount = str(self.config.paths.nfs_mount)
        env["PATH"] = f"{nfs_mount}/openmpi/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
        env["LD_LIBRARY_PATH"] = f"{nfs_mount}/openmpi/lib:{env.get('LD_LIBRARY_PATH', '')}"
        env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
        env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"

        nscribes = self.model.nscribes
        exec_dir = env.get("EXECnwm", "")
        schism_binary = f"{exec_dir}/{self.model.binary}"

        self._update_substep(f"Running pschism with {self.model.total_tasks} MPI tasks")

        self.run_singularity_command(
            ["/bin/bash", "-c", f"{schism_binary} {nscribes}"],
            env=env,
            pwd=self.config.paths.work_dir,
            use_mpi=True,
            mpi_tasks=self.model.total_tasks,
        )

        return {
            "outputs_dir": str(self.config.paths.work_dir / "outputs"),
            "status": "completed",
        }


class PostSCHISMStage(WorkflowStage):
    """Post-process SCHISM outputs."""

    name = "post_schism"
    description = "Post-process SCHISM outputs"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SchismModelConfig)  # noqa: S101
        self.model: SchismModelConfig = config.model_config

    def run(self) -> dict[str, Any]:
        """Execute SCHISM post-processing."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._update_substep("Checking for errors")
        fatal_error = self.config.paths.work_dir / "outputs" / "fatal.error"
        if fatal_error.exists() and fatal_error.stat().st_size > 0:
            error_content = fatal_error.read_text()[-2000:]
            raise RuntimeError(f"SCHISM run failed: {error_content}")

        self._update_substep("Running post_schism")
        script_path = self._get_scripts_dir() / "run_sing_coastal_workflow_post_schism.bash"

        self.run_singularity_command(
            [str(script_path)],
            env=env,
        )

        return {
            "outputs_dir": str(self.config.paths.work_dir / "outputs"),
            "status": "completed",
        }
