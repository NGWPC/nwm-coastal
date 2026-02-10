"""Main workflow runner for coastal model calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coastal_calibration.config.schema import CoastalCalibConfig, SchismModelConfig
from coastal_calibration.utils.logging import (
    WorkflowMonitor,
    configure_logger,
    generate_log_path,
    silence_third_party_loggers,
)
from coastal_calibration.utils.slurm import JobState, SlurmManager

if TYPE_CHECKING:
    from coastal_calibration.stages.base import WorkflowStage


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    success: bool
    job_id: str | None
    start_time: datetime
    end_time: datetime | None
    stages_completed: list[str]
    stages_failed: list[str]
    outputs: dict[str, Any]
    errors: list[str]

    @property
    def duration_seconds(self) -> float | None:
        """Get workflow duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "job_id": self.job_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "stages_completed": self.stages_completed,
            "stages_failed": self.stages_failed,
            "outputs": self.outputs,
            "errors": self.errors,
        }

    def save(self, path: Path | str) -> None:
        """Save result to JSON file.

        Parameters
        ----------
        path : Path or str
            Path to output JSON file. Parent directories will be created
            if they don't exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


class CoastalCalibRunner:
    """Main workflow runner for coastal model calibration.

    This class orchestrates the entire calibration workflow, managing
    stage execution, SLURM job submission, and progress monitoring.

    Supports both SCHISM (``model="schism"``, default) and SFINCS
    (``model="sfincs"``) pipelines.  The model type is selected via
    ``config.model``.
    """

    def __init__(self, config: CoastalCalibConfig) -> None:
        """Initialize the workflow runner.

        Parameters
        ----------
        config : CoastalCalibConfig
            Coastal calibration configuration.
        """
        self.config = config

        # Ensure log directory exists early so file logging can start.
        config.paths.work_dir.mkdir(parents=True, exist_ok=True)

        # Set up file logging *before* creating the monitor so that
        # every message (including third-party) is captured on disk.
        if not config.monitoring.log_file:
            log_path = generate_log_path(config.paths.work_dir)
            configure_logger(file=str(log_path), file_level="DEBUG")

        # Silence noisy third-party loggers (HydroMT, xarray, ...)
        silence_third_party_loggers()

        self.monitor = WorkflowMonitor(config.monitoring)
        self._slurm: SlurmManager | None = None
        self._stages: dict[str, WorkflowStage] = {}
        self._results: dict[str, Any] = {}

    @property
    def STAGE_ORDER(self) -> list[str]:  # noqa: N802
        """Active stage order based on model config."""
        return self.config.model_config.stage_order

    @property
    def slurm(self) -> SlurmManager:
        """Lazily initialize SLURM manager (only needed for submit)."""
        if self._slurm is None:
            self._slurm = SlurmManager(self.config, self.monitor)
        return self._slurm

    def _init_stages(self) -> None:
        """Initialize all workflow stages via model config."""
        self._stages = self.config.model_config.create_stages(self.config, self.monitor)

    def validate(self) -> list[str]:
        """Validate configuration and prerequisites.

        Returns
        -------
        list of str
            List of validation error messages (empty if valid).
        """
        errors = []

        config_errors = self.config.validate()
        errors.extend(config_errors)

        self._init_stages()
        for name, stage in self._stages.items():
            stage_errors = stage.validate()
            errors.extend(f"[{name}] {error}" for error in stage_errors)

        return errors

    def _get_stages_to_run(
        self,
        start_from: str | None,
        stop_after: str | None,
    ) -> list[str]:
        """Determine which stages to run based on start/stop parameters."""
        stages = self.STAGE_ORDER.copy()

        # Skip download stage if disabled in config
        if not self.config.download.enabled and "download" in stages:
            stages.remove("download")

        if start_from:
            if start_from not in stages:
                raise ValueError(f"Unknown stage: {start_from}")
            start_idx = stages.index(start_from)
            stages = stages[start_idx:]

        if stop_after:
            if stop_after not in stages:
                raise ValueError(f"Unknown stage: {stop_after}")
            stop_idx = stages.index(stop_after)
            stages = stages[: stop_idx + 1]

        return stages

    def run(
        self,
        start_from: str | None = None,
        stop_after: str | None = None,
        dry_run: bool = False,
    ) -> WorkflowResult:
        """Execute the calibration workflow.

        Parameters
        ----------
        start_from : str, optional
            Stage name to start from (skip earlier stages).
        stop_after : str, optional
            Stage name to stop after (skip later stages).
        dry_run : bool, default False
            If True, validate but don't execute.

        Returns
        -------
        WorkflowResult
            Result with execution details.
        """
        start_time = datetime.now()
        stages_completed: list[str] = []
        stages_failed: list[str] = []
        outputs: dict[str, Any] = {}
        errors: list[str] = []

        validation_errors = self.validate()
        if validation_errors:
            return WorkflowResult(
                success=False,
                job_id=None,
                start_time=start_time,
                end_time=datetime.now(),
                stages_completed=[],
                stages_failed=[],
                outputs={},
                errors=validation_errors,
            )

        if dry_run:
            self.monitor.info("Dry run mode - validation passed, no execution")
            return WorkflowResult(
                success=True,
                job_id=None,
                start_time=start_time,
                end_time=datetime.now(),
                stages_completed=[],
                stages_failed=[],
                outputs={"dry_run": True},
                errors=[],
            )

        self.monitor.register_stages(self.STAGE_ORDER)
        self.monitor.start_workflow()

        stages_to_run = self._get_stages_to_run(start_from, stop_after)

        current_stage = ""
        try:
            for current_stage in stages_to_run:
                stage = self._stages[current_stage]

                with self.monitor.stage_context(current_stage, stage.description):
                    result = stage.run()
                    self._results[current_stage] = result
                    outputs[current_stage] = result
                    stages_completed.append(current_stage)

            self.monitor.end_workflow(success=True)
            success = True

        except Exception as e:
            self.monitor.error(f"Workflow failed: {e}")
            self.monitor.end_workflow(success=False)
            errors.append(str(e))
            stages_failed.append(current_stage)
            success = False

        result = WorkflowResult(
            success=success,
            job_id=None,
            start_time=start_time,
            end_time=datetime.now(),
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            outputs=outputs,
            errors=errors,
        )

        result_file = self.config.paths.work_dir / "workflow_result.json"
        result.save(result_file)
        self.monitor.save_progress(self.config.paths.work_dir / "workflow_progress.json")

        return result

    def _prepare_work_directory(self) -> None:
        """Prepare the work directory for execution."""
        work_dir = self.config.paths.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        config_file = work_dir / "config.yaml"
        self.config.to_yaml(config_file)
        self.monitor.info(f"Configuration saved to: {config_file}")

    def _generate_sfincs_runner_script(self) -> None:
        """Generate a SFINCS runner script for SLURM."""
        from coastal_calibration.config.schema import SfincsModelConfig
        from coastal_calibration.stages.sfincs_build import get_model_root, resolve_sif_path

        work_dir = self.config.paths.work_dir
        runner_script = work_dir / "sing_run_generated.bash"

        assert isinstance(self.config.model_config, SfincsModelConfig)  # noqa: S101
        model_root = get_model_root(self.config)
        sif_path = resolve_sif_path(self.config)

        script_lines = [
            "#!/usr/bin/env bash",
            "set -euox pipefail",
            "",
            "# Auto-generated SFINCS workflow runner script",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            f"export OMP_NUM_THREADS={self.config.model_config.omp_num_threads}",
            "",
            "# Run SFINCS in Singularity container",
            f"singularity run -B{model_root}:/data {sif_path}",
            "",
            'echo "=== SFINCS Workflow Complete ==="',
            "",
        ]

        script_content = "\n".join(script_lines)
        runner_script.write_text(script_content)
        runner_script.chmod(0o755)
        self.monitor.info(f"Generated SFINCS runner script: {runner_script}")

    def _generate_schism_runner_script(self) -> None:
        """Generate the SCHISM inner workflow runner script for SLURM job.

        This script mirrors the original sing_run.bash structure, calling
        the individual stage bash scripts directly rather than using the
        Python CLI (which may not be installed in the container).
        """
        work_dir = self.config.paths.work_dir
        runner_script = work_dir / "sing_run_generated.bash"

        sim = self.config.simulation
        paths = self.config.paths
        boundary = self.config.boundary

        assert isinstance(self.config.model_config, SchismModelConfig)  # noqa: S101
        model = self.config.model_config

        # Get scripts directory (where the bash stage scripts live)
        scripts_dir = Path(__file__).parent / "scripts"

        # Domain mappings (from sing_run.bash)
        domain_to_inland = {
            "hawaii": "domain_hawaii",
            "prvi": "domain_puertorico",
            "atlgulf": "domain",
            "pacific": "domain",
        }
        domain_to_geo_grid = {
            "hawaii": "geo_em_HI.nc",
            "prvi": "geo_em_PRVI.nc",
            "atlgulf": "geo_em_CONUS.nc",
            "pacific": "geo_em_CONUS.nc",
        }

        inland_domain = domain_to_inland.get(sim.coastal_domain, sim.coastal_domain)
        geo_grid = domain_to_geo_grid.get(sim.coastal_domain, "geo_em.d01.nc")

        # Compute derived values
        nprocs = model.total_tasks
        nscribes = min(model.nscribes, nprocs - 1) if nprocs > 1 else 0

        # Use STOFS or TPXO for boundary conditions
        use_tpxo = boundary.source == "tpxo"

        # Get SCHISM binary name from config
        schism_binary = model.binary

        # Get STOFS file path if using STOFS
        stofs_file = ""
        if not use_tpxo and boundary.stofs_file:
            stofs_file = str(boundary.stofs_file)
        elif not use_tpxo and paths.raw_download_dir:
            # Auto-resolve from download directory using date-aware path
            from coastal_calibration.downloader import get_stofs_path

            expected = get_stofs_path(sim.start_date, paths.raw_download_dir)
            if expected.exists():
                stofs_file = str(expected)
            else:
                # Fallback: search for any STOFS file
                stofs_dir = paths.raw_download_dir / "coastal" / "stofs"
                if stofs_dir.exists():
                    stofs_files = sorted(stofs_dir.rglob("*.fields.cwl.nc"))
                    if stofs_files:
                        stofs_file = str(stofs_files[0])

        script_lines = [
            "#!/usr/bin/env bash",
            "set -euox pipefail",
            "",
            "# Auto-generated SCHISM workflow runner script",
            f"# Generated: {datetime.now().isoformat()}",
            "# This script mirrors sing_run.bash but with configuration from Python",
            "",
            "# === Configuration ===",
            f"export NODES={model.nodes}",
            f"export NCORES={model.ntasks_per_node}",
            f"export NPROCS={nprocs}",
            f"export NSCRIBES={nscribes}",
            "",
            f"export STARTPDY={sim.start_pdy}",
            f"export STARTCYC={sim.start_cyc}",
            f"export FCST_LENGTH_HRS={int(sim.duration_hours)}",
            f"export FCST_TIMESTEP_LENGTH_SECS={sim.timestep_seconds}",
            "",
            f"export COASTAL_DOMAIN={sim.coastal_domain}",
            f"export METEO_SOURCE={sim.meteo_source.upper()}",
            f'export USE_TPXO="{"YES" if use_tpxo else "NO"}"',
            f"export STOFS_FILE='{stofs_file}'",
            f"export HOT_START_FILE='{paths.hot_start_file or ''}'",
            "",
            "# === Paths ===",
            f"export NFS_MOUNT={paths.nfs_mount}",
            f"export NGEN_APP_DIR={paths.ngen_app_dir}",
            f"export NGWPC_COASTAL_PARM_DIR={paths.parm_dir}",
            f"export COASTAL_WORK_DIR={work_dir}",
            f"export RAW_DOWNLOAD_DIR={paths.raw_download_dir or ''}",
            "export NWM_FORCING_DIR=$RAW_DOWNLOAD_DIR/meteo/${METEO_SOURCE,,}",
            "# Set NWM_CHROUT_DIR based on meteo source",
            'if [[ ${METEO_SOURCE} == "NWM_RETRO" ]]; then',
            "   export NWM_CHROUT_DIR=$RAW_DOWNLOAD_DIR/streamflow/nwm_retro",
            'elif [[ ${METEO_SOURCE} == "NWM_ANA" ]]; then',
            "   export NWM_CHROUT_DIR=$RAW_DOWNLOAD_DIR/hydro/nwm",
            "fi",
            "",
            f"export USHnwm={paths.ush_nwm}",
            f"export PARMnwm={paths.parm_nwm}",
            f"export EXECnwm={paths.exec_nwm}",
            "export DATAexec=$COASTAL_WORK_DIR",
            "export DATAlogs=$DATAexec",
            "",
            f"export OTPSDIR={paths.otps_dir}",
            f"export CONDA_ENVS_PATH={paths.conda_envs_path}",
            f"export CONDA_ENV_NAME={paths.conda_env_name}",
            "",
            f"export INLAND_DOMAIN={inland_domain}",
            f"export SCHISM_ESMFMESH=${{PARMnwm}}/coastal/{sim.coastal_domain}/hgrid.nc",
            f"export GEOGRID_FILE=${{PARMnwm}}/{inland_domain}/{geo_grid}",
            "",
            "# === Environment Setup ===",
            "export SAVE_ALL_TASKS=yes",
            f"export OMP_NUM_THREADS={model.omp_num_threads}",
            "export OMP_PLACES=cores",
            "export MPICH_OFI_STARTUP_CONNECT=1",
            "export MPICH_COLL_SYNC=MPI_Bcast",
            "export MPICH_REDUCE_NO_SMP=1",
            "export FI_OFI_RXM_SAR_LIMIT=3145728",
            "export FI_MR_CACHE_MAX_COUNT=0",
            "export FI_EFA_RECVWIN_SIZE=65536",
            "",
            "# Conda setup",
            "__conda_setup=\"$($NFS_MOUNT/ngen-app/conda/bin/conda 'shell.bash' 'hook' 2> /dev/null)\"",
            "if [ $? -eq 0 ]; then",
            '    eval "$__conda_setup"',
            "else",
            '    if [ -f "$NFS_MOUNT/ngen-app/conda/etc/profile.d/conda.sh" ]; then',
            '        . "$NFS_MOUNT/ngen-app/conda/etc/profile.d/conda.sh"',
            "    else",
            '        export PATH="$NFS_MOUNT/ngen-app/conda/bin:$PATH"',
            "    fi",
            "fi",
            "unset __conda_setup",
            "",
            "export PATH=$NFS_MOUNT/ngen-app/conda/bin:${PATH}",
            "export PATH=${CONDA_ENVS_PATH}/${CONDA_ENV_NAME}/bin:${PATH}",
            "conda activate ${CONDA_ENVS_PATH}/$CONDA_ENV_NAME",
            "export LD_LIBRARY_PATH=$NFS_MOUNT/ngen-app/conda/lib:${CONDA_ENVS_PATH}/lib:${LD_LIBRARY_PATH:-}",
            "",
            "# === Singularity Setup ===",
            f"export SIF_PATH={paths.singularity_image}",
            "if [[ ! -f $SIF_PATH ]]; then",
            '   echo "ERROR: Singularity image file not found at $SIF_PATH"',
            "   exit 1",
            "fi",
            "",
            "# Bind paths for Singularity (includes scripts directory)",
            f"export SCRIPTS_DIR={scripts_dir}",
            'export BINDINGS="$NFS_MOUNT,$CONDA_ENVS_PATH,$NGWPC_COASTAL_PARM_DIR,$SCRIPTS_DIR,/usr/bin/bc,/usr/bin/srun,'
            "/usr/lib64/libpmi2.so,/usr/lib64/libefa.so,/usr/lib64/libibmad.so,/usr/lib64/libibnetdisc.so,"
            "/usr/lib64/libibumad.so,/usr/lib64/libibverbs.so,/usr/lib64/libmana.so,/usr/lib64/libmlx4.so,"
            '/usr/lib64/libmlx5.so,/usr/lib64/librdmacm.so"',
            "",
            "work_dir=${NGEN_APP_DIR}/ngen-forcing/coastal/calib",
            'MPICOMMAND="mpiexec -n ${NPROCS}"',
            "",
            "run_in_container() {",
            '    singularity exec -B "$BINDINGS" --pwd "${work_dir}" "$SIF_PATH" "$@"',
            "}",
            "",
            "run_in_container_mpi() {",
            '    ${MPICOMMAND} singularity exec -B "$BINDINGS" --pwd "${work_dir}" "$SIF_PATH" "$@"',
            "}",
            "",
            "# === Derived Time Variables ===",
            'start_itime=$(date -u -d "${STARTPDY} ${STARTCYC}" +"%s")',
            "end_itime=$(( $start_itime + $FCST_LENGTH_HRS * 3600 + 3600 ))",
            'export start_dt=$(date -u -d "@${start_itime}" +"%Y-%m-%dT%H-%M-%SZ")',
            'export end_dt=$(date -u -d "@${end_itime}" +"%Y-%m-%dT%H-%M-%SZ")',
            f"export COASTAL_SOURCE={'tpxo' if use_tpxo else 'stofs'}",
            "",
            "# === Stage: pre_forcing ===",
            'echo "=== Stage: pre_forcing ==="',
            "run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_pre_forcing_coastal.bash",
            "",
            "# === Stage: nwm_forcing ===",
            'echo "=== Stage: nwm_forcing ==="',
            "export LENGTH_HRS=$FCST_LENGTH_HRS",
            "export FORCING_BEGIN_DATE=${STARTPDY}${STARTCYC}00",
            'start_timestamp=$(date -u -d "${STARTPDY} ${STARTCYC}" +"%s")',
            "itime=$(( 10#${LENGTH_HRS} * 3600 + $start_timestamp ))",
            'export FORCING_END_DATE=$(date -u -d "@${itime}" +"%Y%m%d%H00")',
            "",
            "export NWM_FORCING_OUTPUT_DIR=$DATAexec/forcing_input",
            "export COASTAL_FORCING_OUTPUT_DIR=$DATAexec/coastal_forcing_output",
            "export FECPP_JOB_INDEX=0",
            "export FECPP_JOB_COUNT=1",
            "",
            "run_in_container_mpi \\",
            "    $CONDA_ENVS_PATH/$CONDA_ENV_NAME/bin/python \\",
            "    $USHnwm/wrf_hydro_workflow_dev/forcings/WrfHydroFECPP/workflow_driver.py",
            "",
            "# === Stage: post_forcing ===",
            'echo "=== Stage: post_forcing ==="',
            "run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_post_forcing_coastal.bash",
            "",
            "# === Stage: update_params ===",
            'echo "=== Stage: update_params ==="',
            "run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_update_params.bash",
            "",
            "# === Stage: boundary_conditions ===",
            'echo "=== Stage: boundary_conditions ==="',
            'if [[ $USE_TPXO == "YES" ]]; then',
            "    run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_make_tpxo_ocean.bash",
            "else",
            "    export CYCLE_DATE=$STARTPDY",
            "    export CYCLE_TIME=${STARTCYC}00",
            "    export LENGTH_HRS=$(run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_pre_make_stofs_ocean.bash)",
            "",
            "    export ESTOFS_INPUT_FILE=$STOFS_FILE",
            "    export SCHISM_OUTPUT_FILE=$DATAexec/elev2D.th.nc",
            "    export OPEN_BNDS_HGRID_FILE=$DATAexec/open_bnds_hgrid.nc",
            "",
            "    run_in_container_mpi \\",
            "        $CONDA_ENVS_PATH/$CONDA_ENV_NAME/bin/python \\",
            "        $USHnwm/wrf_hydro_workflow_dev/coastal/regrid_estofs.py $ESTOFS_INPUT_FILE $OPEN_BNDS_HGRID_FILE $SCHISM_OUTPUT_FILE",
            "",
            "    run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_post_make_stofs_ocean.bash",
            "fi",
            "",
            "# === Stage: pre_schism ===",
            'echo "=== Stage: pre_schism ==="',
            "run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_pre_schism.bash",
            "",
            "# === Stage: schism_run ===",
            'echo "=== Stage: schism_run ==="',
            "# Switch to NFS OpenMPI for SCHISM",
            "export PATH=$NFS_MOUNT/openmpi/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin",
            "export LD_LIBRARY_PATH=$NFS_MOUNT/openmpi/lib:$LD_LIBRARY_PATH",
            "export OMPI_ALLOW_RUN_AS_ROOT=1",
            "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
            "",
            '${MPICOMMAND} singularity exec -B "$BINDINGS" --pwd "$COASTAL_WORK_DIR" "$SIF_PATH" \\',
            f'    /bin/bash -c "${{EXECnwm}}/{schism_binary} $NSCRIBES"',
            "",
            "# === Stage: post_schism ===",
            'echo "=== Stage: post_schism ==="',
            "run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_post_schism.bash",
            "",
            'echo "=== Workflow Complete ==="',
            "",
        ]

        script_content = "\n".join(script_lines)

        runner_script.write_text(script_content)
        runner_script.chmod(0o755)
        self.monitor.info(f"Generated runner script: {runner_script}")

    def _generate_runner_script(self) -> None:
        """Generate the inner workflow runner script for SLURM job.

        For SCHISM this mirrors the original sing_run.bash structure.
        For SFINCS this generates a simpler OpenMP-based script.
        """
        model_config = self.config.model_config
        if isinstance(model_config, SchismModelConfig):
            self._generate_schism_runner_script()
        else:
            self._generate_sfincs_runner_script()

    def submit(self, wait: bool = False, log_file: Path | None = None) -> WorkflowResult:
        """Submit workflow as a SLURM job.

        The download stage (if enabled) runs on the login node before
        submitting, so that expensive compute nodes are not wasted on
        sequential file downloads.

        Parameters
        ----------
        wait : bool, default False
            If True, wait for job completion (interactive mode).
            If False, return immediately after job submission.
        log_file : Path, optional
            Custom path for SLURM output log. If not provided, logs are
            written to <work_dir>/slurm-<job_id>.out.

        Returns
        -------
        WorkflowResult
            Result with job submission details.
        """
        start_time = datetime.now()

        validation_errors = self.validate()
        if validation_errors:
            return WorkflowResult(
                success=False,
                job_id=None,
                start_time=start_time,
                end_time=datetime.now(),
                stages_completed=[],
                stages_failed=[],
                outputs={},
                errors=validation_errors,
            )

        self._prepare_work_directory()

        # Run download on the login node before submitting the SLURM job
        if self.config.download.enabled:
            self.monitor.info("Running download stage on login node...")
            self._init_stages()
            download_stage = self._stages["download"]
            try:
                download_stage.run()
                self.monitor.info("Download stage completed")
            except Exception as e:
                return WorkflowResult(
                    success=False,
                    job_id=None,
                    start_time=start_time,
                    end_time=datetime.now(),
                    stages_completed=[],
                    stages_failed=["download"],
                    outputs={},
                    errors=[f"Download stage failed: {e}"],
                )

        job_script = self.config.paths.work_dir / "submit_job.sh"
        self.slurm.generate_job_script(job_script, log_file=log_file)

        self._generate_runner_script()

        job_id = self.slurm.submit_job(job_script)

        if not wait:
            # Return immediately without waiting for job completion
            return WorkflowResult(
                success=True,
                job_id=job_id,
                start_time=start_time,
                end_time=datetime.now(),
                stages_completed=[],
                stages_failed=[],
                outputs={"slurm_status": "SUBMITTED"},
                errors=[],
            )

        final_status = self.slurm.wait_for_job(job_id)

        success = final_status.state == JobState.COMPLETED
        errors = []
        if not success:
            errors.append(f"Job ended with state: {final_status.state.value}")
            # Filter out empty reasons and SLURM's literal "None" string
            if final_status.reason and final_status.reason.lower() != "none":
                errors.append(f"Reason: {final_status.reason}")

        return WorkflowResult(
            success=success,
            job_id=job_id,
            start_time=start_time,
            end_time=datetime.now(),
            stages_completed=self.STAGE_ORDER if success else [],
            stages_failed=[] if success else ["unknown"],
            outputs={"slurm_status": final_status.state.value},
            errors=errors,
        )


def run_workflow(
    config_path: Path | str,
    start_from: str | None = None,
    stop_after: str | None = None,
    dry_run: bool = False,
) -> WorkflowResult:
    """Run workflow from config file.

    Parameters
    ----------
    config_path : Path or str
        Path to YAML configuration file.
    start_from : str, optional
        Stage name to start from.
    stop_after : str, optional
        Stage name to stop after.
    dry_run : bool, default False
        If True, validate but don't execute.

    Returns
    -------
    WorkflowResult
        Result with execution details.
    """
    config = CoastalCalibConfig.from_yaml(Path(config_path))
    runner = CoastalCalibRunner(config)
    return runner.run(
        start_from=start_from,
        stop_after=stop_after,
        dry_run=dry_run,
    )


def submit_workflow(config_path: Path | str, wait: bool = True) -> WorkflowResult:
    """Submit workflow as SLURM job.

    Parameters
    ----------
    config_path : Path or str
        Path to YAML configuration file.
    wait : bool, default True
        If True, wait for job completion. If False, return immediately
        after job submission.

    Returns
    -------
    WorkflowResult
        Result with job submission details.
    """
    config = CoastalCalibConfig.from_yaml(Path(config_path))
    runner = CoastalCalibRunner(config)
    return runner.submit(wait=wait)
