"""Forcing preparation stages for SCHISM workflow."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from typing import Any

from coastal_calibration.stages.base import WorkflowStage


class PreForcingStage(WorkflowStage):
    """Prepare data for atmospheric forcing generation."""

    name = "pre_forcing"
    description = "Prepare NWM forcing data"
    requires_container = True

    def run(self) -> dict[str, Any]:
        """Execute pre-forcing preparation."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._update_substep("Creating output directories")
        work_dir = self.config.paths.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        forcing_output = work_dir / "coastal_forcing_output"
        if forcing_output.exists():
            shutil.rmtree(forcing_output)
        forcing_output.mkdir(exist_ok=True)

        self._update_substep("Running pre_nwm_forcing_coastal")
        self._log("Creating forcing symlinks and output directories")
        script_path = self._get_scripts_dir() / "run_sing_coastal_workflow_pre_forcing_coastal.bash"

        self.run_singularity_command(
            [str(script_path)],
            sif_path=self.config.model_config.singularity_image,
            env=env,
        )

        self._log(f"Pre-forcing complete — output dir: {forcing_output}")
        return {
            "forcing_output_dir": str(forcing_output),
            "status": "completed",
        }


class NWMForcingStage(WorkflowStage):
    """Generate atmospheric forcing using WRF-Hydro workflow driver."""

    name = "nwm_forcing"
    description = "Generate NWM atmospheric forcing (MPI)"
    requires_container = True

    def run(self) -> dict[str, Any]:
        """Execute NWM forcing generation with MPI."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._update_substep("Setting up forcing parameters")
        sim = self.config.simulation
        start_pdy = sim.start_pdy
        start_cyc = sim.start_cyc

        env["LENGTH_HRS"] = str(int(sim.duration_hours))

        forcing_begin = f"{start_pdy}{start_cyc}00"
        env["FORCING_BEGIN_DATE"] = forcing_begin

        start_dt = datetime.strptime(f"{start_pdy} {start_cyc}", "%Y%m%d %H").replace(tzinfo=UTC)
        end_dt = start_dt + timedelta(hours=sim.duration_hours)
        env["FORCING_END_DATE"] = end_dt.strftime("%Y%m%d%H00")

        nwm_forcing_output = str(self.config.paths.work_dir / "forcing_input")
        env["NWM_FORCING_OUTPUT_DIR"] = nwm_forcing_output
        env["COASTAL_FORCING_INPUT_DIR"] = f"{nwm_forcing_output}/{forcing_begin[:10]}"
        env["COASTAL_FORCING_OUTPUT_DIR"] = str(
            self.config.paths.work_dir / "coastal_forcing_output"
        )
        env["COASTAL_WORK_DIR"] = str(self.config.paths.work_dir)

        env["FORCING_START_YEAR"] = start_pdy[:4]
        env["FORCING_START_MONTH"] = start_pdy[4:6]
        env["FORCING_START_DAY"] = start_pdy[6:8]
        env["FORCING_START_HOUR"] = start_cyc

        env["FECPP_JOB_INDEX"] = "0"
        env["FECPP_JOB_COUNT"] = "1"

        conda_envs = env.get("CONDA_ENVS_PATH", "")
        conda_env = env.get("CONDA_ENV_NAME", "ngen_forcing_coastal")
        ush_dir = env.get("USHnwm", "")

        python_path = f"{conda_envs}/{conda_env}/bin/python"
        workflow_script = (
            f"{ush_dir}/wrf_hydro_workflow_dev/forcings/WrfHydroFECPP/workflow_driver.py"
        )

        self._log(f"Generating {sim.duration_hours}h forcing from {forcing_begin} via MPI")
        self._update_substep("Running workflow_driver.py with MPI")
        self.run_singularity_command(
            [python_path, workflow_script],
            sif_path=self.config.model_config.singularity_image,
            env=env,
            use_mpi=True,
            mpi_tasks=self.config.model_config.total_tasks,
        )

        self._log(f"NWM forcing generated in {nwm_forcing_output}")
        return {
            "forcing_output_dir": env["NWM_FORCING_OUTPUT_DIR"],
            "status": "completed",
        }


class PostForcingStage(WorkflowStage):
    """Clean up and post-process forcing generation."""

    name = "post_forcing"
    description = "Post-process forcing data"
    requires_container = True

    def run(self) -> dict[str, Any]:
        """Execute post-forcing cleanup."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._log("Running makeAtmo.py post-processing")
        self._update_substep("Running post_nwm_forcing_coastal")
        script_path = (
            self._get_scripts_dir() / "run_sing_coastal_workflow_post_forcing_coastal.bash"
        )

        self.run_singularity_command(
            [str(script_path)],
            sif_path=self.config.model_config.singularity_image,
            env=env,
        )

        # Verify sflux output was produced
        sflux_dir = self.config.paths.work_dir / "sflux"
        if not sflux_dir.exists() or not any(sflux_dir.iterdir()):
            raise RuntimeError(
                f"post_forcing: no sflux files produced in {sflux_dir}. "
                "Check makeAtmo.py log for errors."
            )

        n_sflux = sum(1 for _ in sflux_dir.iterdir())
        self._log(f"Post-processing complete — {n_sflux} sflux file(s) produced")
        return {"status": "completed"}
