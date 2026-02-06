"""SCHISM model execution stages."""

from __future__ import annotations

from typing import Any

from coastal_calibration.stages.base import WorkflowStage


class PreSCHISMStage(WorkflowStage):
    """Prepare input files for SCHISM execution."""

    name = "pre_schism"
    description = "Prepare SCHISM inputs (discharge, partitioning)"

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
        return {
            "partition_file": str(work_dir / "partition.prop"),
            "outputs_dir": str(work_dir / "outputs"),
            "status": "completed",
        }


class SCHISMRunStage(WorkflowStage):
    """Execute SCHISM model with MPI."""

    name = "schism_run"
    description = "Run SCHISM model (MPI)"

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

        nscribes = self.config.mpi.nscribes
        exec_dir = env.get("EXECnwm", "")
        schism_binary = f"{exec_dir}/{self.config.mpi.schism_binary}"

        self._update_substep(f"Running pschism with {self.config.slurm.total_tasks} MPI tasks")

        self.run_singularity_command(
            ["/bin/bash", "-c", f"{schism_binary} {nscribes}"],
            env=env,
            pwd=self.config.paths.work_dir,
            use_mpi=True,
            mpi_tasks=self.config.slurm.total_tasks,
        )

        return {
            "outputs_dir": str(self.config.paths.work_dir / "outputs"),
            "status": "completed",
        }


class PostSCHISMStage(WorkflowStage):
    """Post-process SCHISM outputs."""

    name = "post_schism"
    description = "Post-process SCHISM outputs"

    def run(self) -> dict[str, Any]:
        """Execute SCHISM post-processing."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._update_substep("Checking for errors")
        fatal_error = self.config.paths.work_dir / "outputs" / "fatal.error"
        if fatal_error.exists() and fatal_error.stat().st_size > 0:
            error_content = fatal_error.read_text()[:500]
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
