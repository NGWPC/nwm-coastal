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

    # Name of the lightweight JSON file that tracks completed stages.
    _STATUS_FILENAME = ".pipeline_status.json"

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

    # ------------------------------------------------------------------
    # Pipeline status tracking
    # ------------------------------------------------------------------

    @property
    def _status_path(self) -> Path:
        """Path to the pipeline status file in the work directory."""
        return self.config.paths.work_dir / self._STATUS_FILENAME

    def _load_status(self) -> dict[str, Any]:
        """Load pipeline status from disk (empty dict if missing)."""
        if self._status_path.exists():
            return json.loads(self._status_path.read_text())  # type: ignore[no-any-return]
        return {}

    def _save_stage_status(self, stage_name: str) -> None:
        """Mark *stage_name* as completed in the pipeline status file."""
        status = self._load_status()
        completed: list[str] = status.get("completed_stages", [])
        if stage_name not in completed:
            completed.append(stage_name)
        status["completed_stages"] = completed
        self._status_path.write_text(json.dumps(status, indent=2) + "\n")

    def _check_prerequisites(self, start_from: str) -> list[str]:
        """Verify that all stages before *start_from* have completed.

        Returns a list of error messages (empty if all prerequisites met).
        """
        status = self._load_status()
        completed: set[str] = set(status.get("completed_stages", []))

        all_stages = self.STAGE_ORDER
        if start_from not in all_stages:
            return [f"Unknown stage: {start_from}"]

        start_idx = all_stages.index(start_from)
        missing = [s for s in all_stages[:start_idx] if s not in completed]
        if missing:
            return [
                f"Cannot start from '{start_from}': prerequisite stage(s) "
                f"{', '.join(repr(s) for s in missing)} have not completed.  "
                f"Run them first or start from an earlier stage.  "
                f"(Status file: {self._status_path})"
            ]
        return []

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
        # When resuming mid-pipeline, verify that earlier stages completed.
        if not validation_errors and start_from:
            validation_errors = self._check_prerequisites(start_from)
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
        self.monitor.info("-" * 40)

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
                    self._save_stage_status(current_stage)

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

    def _split_stages_for_submit(
        self,
        stages_to_run: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Partition stages into pre-job, job, and post-job groups.

        For ``submit``, Python-only stages run on the login node while
        container stages are bundled into a SLURM bash script.

        Returns
        -------
        pre_job : list[str]
            Python-only stages to run before SLURM submission.
            Includes stages naturally before the first container stage
            **and** any Python-only stages sandwiched between container
            stages (promoted to run early).
        job : list[str]
            Container stages for the SLURM bash script.
        post_job : list[str]
            Python-only stages to run after the SLURM job completes.
        """
        # Find first and last container stage indices
        first_container: int | None = None
        last_container: int | None = None
        for i, name in enumerate(stages_to_run):
            if self._stages[name].requires_container:
                if first_container is None:
                    first_container = i
                last_container = i

        if first_container is None:
            # No container stages at all — everything runs on login node
            return list(stages_to_run), [], []

        assert last_container is not None  # noqa: S101

        pre_job = list(stages_to_run[:first_container])
        post_job = list(stages_to_run[last_container + 1 :])
        job: list[str] = []

        # Walk from first_container to last_container (inclusive)
        for name in stages_to_run[first_container : last_container + 1]:
            if self._stages[name].requires_container:
                job.append(name)
            else:
                # Promote sandwiched Python-only stages to pre_job
                pre_job.append(name)

        return pre_job, job, post_job

    def _prepare_promoted_stage_deps(self, stage_name: str) -> None:
        """Pre-create dependencies for a Python-only stage promoted to pre-job.

        When a stage that normally runs *after* a container stage is
        promoted to run *before* the SLURM job, its prerequisites may
        not yet exist.  This method creates them.
        """
        if stage_name == "schism_obs":
            # schism_obs needs hgrid.gr3 in work_dir.  Normally the
            # update_params container stage symlinks it, but that hasn't
            # run yet.  Create the symlink from parm_nwm (the parameter
            # files directory, i.e. parm_dir / "parm").
            work_dir = self.config.paths.work_dir
            hgrid_src = (
                self.config.paths.parm_nwm
                / "coastal"
                / self.config.simulation.coastal_domain
                / "hgrid.gr3"
            )
            hgrid_dst = work_dir / "hgrid.gr3"
            if not hgrid_dst.exists() and hgrid_src.exists():
                hgrid_dst.symlink_to(hgrid_src)
                self.monitor.info(f"Symlinked hgrid.gr3 -> {hgrid_src}")

    def _run_stages_on_login_node(self, stage_names: list[str]) -> list[str]:
        """Run Python-only stages on the login node with monitoring.

        Parameters
        ----------
        stage_names : list[str]
            Stage names to execute.

        Returns
        -------
        list[str]
            Stages that completed successfully.

        Raises
        ------
        RuntimeError
            If any stage fails.
        """
        completed: list[str] = []
        for name in stage_names:
            stage = self._stages[name]
            self._prepare_promoted_stage_deps(name)

            with self.monitor.stage_context(name, stage.description):
                result = stage.run()
                self._results[name] = result
                completed.append(name)
                self._save_stage_status(name)

        return completed

    def _prepare_work_directory(self) -> None:
        """Prepare the work directory for execution."""
        work_dir = self.config.paths.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        config_file = work_dir / "config.yaml"
        self.config.to_yaml(config_file)
        self.monitor.info(f"Configuration saved to: {config_file}")

    @staticmethod
    def _logging_helpers() -> list[str]:
        """Return bash helper functions that mirror WorkflowMonitor output.

        The generated ``log_stage_start``, ``log_stage_end``, and
        ``log_workflow_*`` functions emit the same separator / timing
        format that :class:`WorkflowMonitor` produces during ``run``.
        """
        return [
            "# --- Logging helpers (mirrors WorkflowMonitor output) ---",
            "WORKFLOW_START_TS=$(date +%s)",
            "",
            "log_ts() { date '+[%Y/%m/%d %H:%M:%S]'; }",
            "",
            "log_workflow_start() {",
            '    echo "$(log_ts) INFO     ============================================================"',
            '    echo "$(log_ts) INFO     Coastal Calibration Workflow Started"',
            '    echo "$(log_ts) INFO     ============================================================"',
            "}",
            "",
            "log_stage_start() {",
            "    local name=$1",
            '    local desc=${2:-""}',
            '    echo "$(log_ts) INFO     Stage: ${name}"',
            '    if [[ -n "$desc" ]]; then',
            '        echo "$(log_ts) INFO       ${desc}"',
            "    fi",
            "    STAGE_START_TS=$(date +%s)",
            "}",
            "",
            "log_stage_end() {",
            "    local name=$1",
            "    local elapsed=$(( $(date +%s) - STAGE_START_TS ))",
            '    local fmt=""',
            "    if (( elapsed >= 3600 )); then",
            '        fmt="$(( elapsed/3600 ))h $(( (elapsed%3600)/60 ))m $(( elapsed%60 ))s"',
            "    elif (( elapsed >= 60 )); then",
            '        fmt="$(( elapsed/60 ))m $(( elapsed%60 ))s"',
            "    else",
            '        fmt="${elapsed}s"',
            "    fi",
            '    echo "$(log_ts) INFO       [✓] COMPLETED (${fmt})"',
            '    echo "$(log_ts) INFO     ----------------------------------------"',
            "}",
            "",
            "log_info() {",
            '    echo "$(log_ts) INFO     $1"',
            "}",
            "",
            "log_workflow_end() {",
            "    local elapsed=$(( $(date +%s) - WORKFLOW_START_TS ))",
            "    local hh=$(( elapsed/3600 ))",
            "    local mm=$(( (elapsed%3600)/60 ))",
            "    local ss=$(( elapsed%60 ))",
            '    local dur=$(printf "%d:%02d:%02d" $hh $mm $ss)',
            '    echo "$(log_ts) INFO     ============================================================"',
            '    echo "$(log_ts) INFO     Workflow COMPLETED | Total Duration: ${dur}"',
            '    echo "$(log_ts) INFO     ============================================================"',
            "}",
            "",
        ]

    @staticmethod
    def _stage_tracking_helpers() -> list[str]:
        """Return bash functions for per-stage completion tracking.

        These helpers allow the generated runner script to mark stages as
        completed in ``.pipeline_status.json`` and to skip stages that
        have already finished — enabling mid-pipeline restarts after a
        SLURM job failure without re-running expensive stages.

        Must be placed **after** the environment section that exports
        ``COASTAL_WORK_DIR``, since the status-file path depends on it.
        """
        return [
            "# --- Stage completion tracking ---",
            'PIPELINE_STATUS_FILE="${COASTAL_WORK_DIR}/.pipeline_status.json"',
            "",
            "mark_stage_complete() {",
            "    local stage=$1",
            '    python3 -c "',
            "import json, pathlib",
            "p = pathlib.Path('${PIPELINE_STATUS_FILE}')",
            "d = json.loads(p.read_text()) if p.exists() else {}",
            "c = d.get('completed_stages', [])",
            "if '${stage}' not in c:",
            "    c.append('${stage}')",
            "d['completed_stages'] = c",
            "p.write_text(json.dumps(d, indent=2) + '\\\\n')",
            '"',
            "}",
            "",
            "is_stage_complete() {",
            "    local stage=$1",
            '    python3 -c "',
            "import json, pathlib, sys",
            "p = pathlib.Path('${PIPELINE_STATUS_FILE}')",
            "if not p.exists(): sys.exit(1)",
            "d = json.loads(p.read_text())",
            "sys.exit(0 if '${stage}' in d.get('completed_stages', []) else 1)",
            '"',
            "}",
            "",
            "log_stage_skip() {",
            "    local name=$1",
            '    echo "$(log_ts) INFO     Stage: ${name}"',
            '    echo "$(log_ts) INFO       [⏭] SKIPPED (already completed)"',
            '    echo "$(log_ts) INFO     ----------------------------------------"',
            "}",
            "",
        ]

    def _generate_sfincs_runner_script(self, container_stages: list[str]) -> None:
        """Generate a SFINCS runner script for SLURM.

        Parameters
        ----------
        container_stages : list[str]
            Container stage names to include in the script.
        """
        from coastal_calibration.config.schema import SfincsModelConfig
        from coastal_calibration.stages.sfincs_build import (
            SFINCS_DOCKER_IMAGE,
            get_model_root,
            resolve_sif_path,
        )

        work_dir = self.config.paths.work_dir
        runner_script = work_dir / "sing_run_generated.bash"

        assert isinstance(self.config.model_config, SfincsModelConfig)  # noqa: S101
        model_root = get_model_root(self.config)
        sif_path = resolve_sif_path(self.config)
        docker_uri = f"docker://{SFINCS_DOCKER_IMAGE}:{self.config.model_config.container_tag}"

        script_lines = [
            "#!/usr/bin/env bash",
            "set -euox pipefail",
            "",
            "# Auto-generated SFINCS workflow runner script",
            f"# Generated: {datetime.now().isoformat()}",
            "",
        ]
        script_lines.extend(self._logging_helpers())
        script_lines.extend(
            [
                f"export OMP_NUM_THREADS={self.config.model_config.omp_num_threads}",
                # SFINCS doesn't export COASTAL_WORK_DIR — set it so
                # _stage_tracking_helpers() can derive the status file path.
                f'export COASTAL_WORK_DIR="{work_dir}"',
                "",
            ]
        )
        script_lines.extend(self._stage_tracking_helpers())
        script_lines.extend(
            [
                "log_workflow_start",
                "",
            ]
        )

        if "sfincs_run" in container_stages:
            sfincs_exe = self.config.model_config.sfincs_exe
            if sfincs_exe is not None:
                # Native executable — no Singularity required
                script_lines.extend(
                    [
                        "# === Stage: sfincs_run ===",
                        'if is_stage_complete "sfincs_run"; then',
                        '    log_stage_skip "sfincs_run"',
                        "else",
                        '    log_stage_start "sfincs_run" "Run SFINCS model (native)"',
                        f'    cd "{model_root}"',
                        f'    "{sfincs_exe}"',
                        '    log_stage_end "sfincs_run"',
                        '    mark_stage_complete "sfincs_run"',
                        "fi",
                        "",
                    ]
                )
            else:
                pull_lines = [
                    "# Pull Singularity image if not already present",
                    f'if [ ! -f "{sif_path}" ]; then',
                    f'    echo "Pulling Singularity image: {docker_uri}"',
                    f'    mkdir -p "{sif_path.parent}"',
                    f'    singularity pull "{sif_path}" "{docker_uri}"',
                    "fi",
                    "",
                ]
                script_lines.extend(pull_lines)
                script_lines.extend(
                    [
                        "# === Stage: sfincs_run ===",
                        'if is_stage_complete "sfincs_run"; then',
                        '    log_stage_skip "sfincs_run"',
                        "else",
                        '    log_stage_start "sfincs_run" "Run SFINCS model (Singularity)"',
                        f'    cd "{model_root}"',
                        f"    singularity run -B{model_root}:/data {sif_path}",
                        '    log_stage_end "sfincs_run"',
                        '    mark_stage_complete "sfincs_run"',
                        "fi",
                        "",
                    ]
                )

        script_lines.extend(
            [
                "log_workflow_end",
                "",
            ]
        )

        script_content = "\n".join(script_lines)
        runner_script.write_text(script_content)
        runner_script.chmod(0o755)
        self.monitor.info(f"Generated SFINCS runner script: {runner_script}")

    @staticmethod
    def _resolve_stofs_file(
        use_tpxo: bool,
        boundary: Any,
        paths: Any,
        sim: Any,
    ) -> str:
        """Resolve the STOFS file path for boundary conditions."""
        if use_tpxo:
            return ""
        if boundary.stofs_file:
            return str(boundary.stofs_file)
        if paths.raw_download_dir:
            from coastal_calibration.downloader import get_stofs_path

            expected = get_stofs_path(sim.start_date, paths.raw_download_dir)
            if expected.exists():
                return str(expected)
            stofs_dir = paths.raw_download_dir / "coastal" / "stofs"
            if stofs_dir.exists():
                stofs_files = sorted(stofs_dir.rglob("*.fields.cwl.nc"))
                if stofs_files:
                    return str(stofs_files[0])
        return ""

    @staticmethod
    def _schism_env_lines(
        sim: Any,
        paths: Any,
        model: SchismModelConfig,
        scripts_dir: Path,
        use_tpxo: bool,
        stofs_file: str,
    ) -> list[str]:
        """Return bash lines for SCHISM environment and configuration."""
        nprocs = model.total_tasks
        nscribes = min(model.nscribes, nprocs - 1) if nprocs > 1 else 0

        # Use the canonical mappings from SimulationConfig to avoid
        # maintaining duplicate domain lookup tables.
        inland_domain = sim.inland_domain
        geo_grid = sim.geo_grid
        work_dir = paths.work_dir

        return [
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
            f"export NWM_DIR={paths.nwm_dir}",
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
            "# Disable HDF5 file locking — unreliable on NFS",
            "export HDF5_USE_FILE_LOCKING=FALSE",
            "",
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
            f"export SIF_PATH={model.singularity_image}",
            "if [[ ! -f $SIF_PATH ]]; then",
            '   echo "ERROR: Singularity image file not found at $SIF_PATH"',
            "   exit 1",
            "fi",
            "",
            "# Bind paths for Singularity (includes scripts directory)",
            f"export SCRIPTS_DIR={scripts_dir}",
            "export WRF_HYDRO_DIR=$SCRIPTS_DIR/wrf_hydro_workflow_dev",
            "export COASTAL_SCRIPTS_DIR=$WRF_HYDRO_DIR/coastal",
            "export TPXO_SCRIPTS_DIR=$SCRIPTS_DIR/tpxo_to_open_bnds_hgrid",
            "export FORCINGS_SCRIPTS_DIR=$WRF_HYDRO_DIR/forcings",
            'export BINDINGS="$NFS_MOUNT,$CONDA_ENVS_PATH,$NGWPC_COASTAL_PARM_DIR,$SCRIPTS_DIR,/usr/bin/bc,/usr/bin/srun,'
            "/usr/lib64/libpmi2.so,/usr/lib64/libefa.so,/usr/lib64/libibmad.so,/usr/lib64/libibnetdisc.so,"
            "/usr/lib64/libibumad.so,/usr/lib64/libibverbs.so,/usr/lib64/libmana.so,/usr/lib64/libmlx4.so,"
            '/usr/lib64/libmlx5.so,/usr/lib64/librdmacm.so"',
            "",
            "work_dir=/ngen-app/ngen-forcing/coastal/calib",
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
            "# Date components used by makeAtmo.py, makeDischarge.py, etc.",
            "export PDY=$STARTPDY",
            "export cyc=$STARTCYC",
            "export FORCING_START_YEAR=${STARTPDY:0:4}",
            "export FORCING_START_MONTH=${STARTPDY:4:2}",
            "export FORCING_START_DAY=${STARTPDY:6:2}",
            "export FORCING_START_HOUR=$STARTCYC",
            "export FORCING_BEGIN_DATE=${STARTPDY}${STARTCYC}00",
            "forcing_itime=$(( 10#${FCST_LENGTH_HRS} * 3600 + $start_itime ))",
            'export FORCING_END_DATE=$(date -u -d "@${forcing_itime}" +"%Y%m%d%H00")',
            'export END_DATETIME=$(date -u -d "@${forcing_itime}" +"%Y%m%d%H")',
            "",
            "# SCHISM date variables used by update_param.bash",
            "if [[ $FCST_LENGTH_HRS -le 0 ]]; then",
            "    schism_begin_itime=$(( $start_itime + $FCST_LENGTH_HRS * 3600 ))",
            '    export SCHISM_BEGIN_DATE=$(date -u -d "@${schism_begin_itime}" +"%Y%m%d%H00")',
            "    export SCHISM_END_DATE=${STARTPDY}${STARTCYC}00",
            "else",
            "    export SCHISM_BEGIN_DATE=${STARTPDY}${STARTCYC}00",
            '    export SCHISM_END_DATE=$(date -u -d "@${forcing_itime}" +"%Y%m%d%H00")',
            "fi",
            "",
            "log_workflow_start",
            "",
        ]

    @staticmethod
    def _schism_stage_lines(
        container_stages: list[str],
        model: SchismModelConfig,
        work_dir: Path,
    ) -> list[str]:
        """Return bash lines for SCHISM container stage blocks.

        Each stage is wrapped with skip-if-complete / mark-complete
        logic so that a resubmitted SLURM job resumes from the first
        incomplete stage rather than re-running everything.
        """
        lines: list[str] = []
        schism_binary = model.binary

        if "pre_forcing" in container_stages:
            lines.extend(
                [
                    "# === Stage: pre_forcing ===",
                    'if is_stage_complete "pre_forcing"; then',
                    '    log_stage_skip "pre_forcing"',
                    "else",
                    '    log_stage_start "pre_forcing" "Prepare NWM forcing data"',
                    "    run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_pre_forcing_coastal.bash",
                    '    log_stage_end "pre_forcing"',
                    '    mark_stage_complete "pre_forcing"',
                    "fi",
                    "",
                ]
            )

        if "nwm_forcing" in container_stages:
            lines.extend(
                [
                    "# === Stage: nwm_forcing ===",
                    'if is_stage_complete "nwm_forcing"; then',
                    '    log_stage_skip "nwm_forcing"',
                    "else",
                    '    log_stage_start "nwm_forcing" "Generate NWM atmospheric forcing (MPI)"',
                    "    export LENGTH_HRS=$FCST_LENGTH_HRS",
                    "    export NWM_FORCING_OUTPUT_DIR=$DATAexec/forcing_input",
                    "    export COASTAL_FORCING_OUTPUT_DIR=$DATAexec/coastal_forcing_output",
                    "    export FECPP_JOB_INDEX=0",
                    "    export FECPP_JOB_COUNT=1",
                    "",
                    "    run_in_container_mpi \\",
                    "        $CONDA_ENVS_PATH/$CONDA_ENV_NAME/bin/python \\",
                    "        $USHnwm/wrf_hydro_workflow_dev/forcings/WrfHydroFECPP/workflow_driver.py \\",
                    "        > /dev/null",
                    '    log_stage_end "nwm_forcing"',
                    '    mark_stage_complete "nwm_forcing"',
                    "fi",
                    "",
                ]
            )

        if "post_forcing" in container_stages:
            lines.extend(
                [
                    "# === Stage: post_forcing ===",
                    'if is_stage_complete "post_forcing"; then',
                    '    log_stage_skip "post_forcing"',
                    "else",
                    '    log_stage_start "post_forcing" "Post-process forcing data"',
                    "    run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_post_forcing_coastal.bash",
                    '    log_stage_end "post_forcing"',
                    '    mark_stage_complete "post_forcing"',
                    "fi",
                    "",
                ]
            )

        if "update_params" in container_stages:
            lines.extend(
                [
                    "# === Stage: update_params ===",
                    'if is_stage_complete "update_params"; then',
                    '    log_stage_skip "update_params"',
                    "else",
                    '    log_stage_start "update_params" "Create SCHISM param.nml"',
                    "    run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_update_params.bash",
                ]
            )
            # Patch param.nml to enable station output when station.in was
            # created by the schism_obs stage on the login node.
            # nspool_sta must divide nhot_write or SCHISM aborts with
            # "mod(nhot_write,nspool_sta)/=0".  18 matches nspool and
            # divides every nhot_write value update_param.bash produces.
            if model.include_noaa_gages:
                lines.extend(
                    [
                        f'    if [[ -f "{work_dir}/station.in" ]]; then',
                        '        log_info "Patching param.nml: iout_sta = 1, nspool_sta = 18"',
                        f'        sed -i "s/^\\(\\s*\\)iout_sta\\s*=.*/\\1iout_sta = 1/" "{work_dir}/param.nml"',
                        f'        sed -i "s/^\\(\\s*\\)nspool_sta\\s*=.*/\\1nspool_sta = 18/" "{work_dir}/param.nml"',
                        f'        if ! grep -q "nspool_sta" "{work_dir}/param.nml"; then',
                        f'            sed -i "/^\\s*iout_sta/a\\  nspool_sta = 18" "{work_dir}/param.nml"',
                        "        fi",
                        "    fi",
                    ]
                )
            lines.extend(
                [
                    '    log_stage_end "update_params"',
                    '    mark_stage_complete "update_params"',
                    "fi",
                    "",
                ]
            )

        if "boundary_conditions" in container_stages:
            lines.extend(
                [
                    "# === Stage: boundary_conditions ===",
                    'if is_stage_complete "boundary_conditions"; then',
                    '    log_stage_skip "boundary_conditions"',
                    "else",
                    '    log_stage_start "boundary_conditions" "Create boundary forcing"',
                    '    if [[ $USE_TPXO == "YES" ]]; then',
                    '        log_info "Using TPXO tidal atlas"',
                    "        run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_make_tpxo_ocean.bash",
                    "    else",
                    '        log_info "Using STOFS data"',
                    "        export CYCLE_DATE=$STARTPDY",
                    "        export CYCLE_TIME=${STARTCYC}00",
                    "        export LENGTH_HRS=$(run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_pre_make_stofs_ocean.bash)",
                    "",
                    "        export ESTOFS_INPUT_FILE=$STOFS_FILE",
                    "        export SCHISM_OUTPUT_FILE=$DATAexec/elev2D.th.nc",
                    "        export OPEN_BNDS_HGRID_FILE=$DATAexec/open_bnds_hgrid.nc",
                    "",
                    "        run_in_container_mpi \\",
                    "            $CONDA_ENVS_PATH/$CONDA_ENV_NAME/bin/python \\",
                    "            $USHnwm/wrf_hydro_workflow_dev/coastal/regrid_estofs.py $ESTOFS_INPUT_FILE $OPEN_BNDS_HGRID_FILE $SCHISM_OUTPUT_FILE \\",
                    "            > /dev/null",
                    "",
                    "        run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_post_make_stofs_ocean.bash",
                    "    fi",
                    '    log_stage_end "boundary_conditions"',
                    '    mark_stage_complete "boundary_conditions"',
                    "fi",
                    "",
                ]
            )

        if "pre_schism" in container_stages:
            lines.extend(
                [
                    "# === Stage: pre_schism ===",
                    'if is_stage_complete "pre_schism"; then',
                    '    log_stage_skip "pre_schism"',
                    "else",
                    '    log_stage_start "pre_schism" "Prepare SCHISM run directory"',
                    "    run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_pre_schism.bash",
                    '    log_stage_end "pre_schism"',
                    '    mark_stage_complete "pre_schism"',
                    "fi",
                    "",
                ]
            )

        if "schism_run" in container_stages:
            lines.extend(
                [
                    "# === Stage: schism_run ===",
                    'if is_stage_complete "schism_run"; then',
                    '    log_stage_skip "schism_run"',
                    "else",
                    '    log_stage_start "schism_run" "Run SCHISM model (MPI)"',
                    "    # Switch to NFS OpenMPI for SCHISM",
                    "    export PATH=$NFS_MOUNT/openmpi/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin",
                    "    export LD_LIBRARY_PATH=$NFS_MOUNT/openmpi/lib:$LD_LIBRARY_PATH",
                    "    export OMPI_ALLOW_RUN_AS_ROOT=1",
                    "    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
                    "",
                    '    ${MPICOMMAND} singularity exec -B "$BINDINGS" --pwd "$COASTAL_WORK_DIR" "$SIF_PATH" \\',
                    f'        /bin/bash -c "${{EXECnwm}}/{schism_binary} $NSCRIBES"',
                    '    log_stage_end "schism_run"',
                    '    mark_stage_complete "schism_run"',
                    "fi",
                    "",
                ]
            )

        if "post_schism" in container_stages:
            lines.extend(
                [
                    "# === Stage: post_schism ===",
                    'if is_stage_complete "post_schism"; then',
                    '    log_stage_skip "post_schism"',
                    "else",
                    '    log_stage_start "post_schism" "Post-process SCHISM outputs"',
                    "    run_in_container $SCRIPTS_DIR/run_sing_coastal_workflow_post_schism.bash",
                    '    log_stage_end "post_schism"',
                    '    mark_stage_complete "post_schism"',
                    "fi",
                    "",
                ]
            )

        return lines

    def _generate_schism_runner_script(self, container_stages: list[str]) -> None:
        """Generate the SCHISM inner workflow runner script for SLURM job.

        This script mirrors the original sing_run.bash structure, calling
        the individual stage bash scripts directly rather than using the
        Python CLI (which may not be installed in the container).

        Parameters
        ----------
        container_stages : list[str]
            Container stage names to include in the script.
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

        # Compute derived values
        use_tpxo = boundary.source == "tpxo"
        stofs_file = self._resolve_stofs_file(use_tpxo, boundary, paths, sim)

        script_lines = [
            "#!/usr/bin/env bash",
            "set -euox pipefail",
            "",
            "# Auto-generated SCHISM workflow runner script",
            f"# Generated: {datetime.now().isoformat()}",
            "# This script mirrors sing_run.bash but with configuration from Python",
            "",
        ]
        script_lines.extend(self._logging_helpers())
        script_lines.extend(
            self._schism_env_lines(sim, paths, model, scripts_dir, use_tpxo, stofs_file)
        )
        script_lines.extend(self._stage_tracking_helpers())

        script_lines.extend(self._schism_stage_lines(container_stages, model, work_dir))

        script_lines.extend(
            [
                "log_workflow_end",
                "",
            ]
        )

        script_content = "\n".join(script_lines)

        runner_script.write_text(script_content)
        runner_script.chmod(0o755)
        self.monitor.info(f"Generated runner script: {runner_script}")

    def _generate_runner_script(self, container_stages: list[str]) -> None:
        """Generate the inner workflow runner script for SLURM job.

        For SCHISM this mirrors the original sing_run.bash structure.
        For SFINCS this generates a simpler OpenMP-based script.

        Parameters
        ----------
        container_stages : list[str]
            Container stage names to include in the bash script.
        """
        model_config = self.config.model_config
        if isinstance(model_config, SchismModelConfig):
            self._generate_schism_runner_script(container_stages)
        else:
            self._generate_sfincs_runner_script(container_stages)

    def _wait_for_slurm_job(
        self,
        *,
        job_id: str,
        job: list[str],
        post_job: list[str],
        start_time: datetime,
        stages_completed: list[str],
        errors: list[str],
    ) -> WorkflowResult:
        """Wait for a SLURM job and collect results."""
        final_status = self.slurm.wait_for_job(job_id)

        success = final_status.state == JobState.COMPLETED

        # The generated bash script marks each stage in the status
        # file as it completes.  Read it back so we know exactly which
        # stages finished — both for the success and failure paths.
        status = self._load_status()
        completed_set = set(status.get("completed_stages", []))
        completed_in_job = [s for s in job if s in completed_set]
        failed_in_job = [s for s in job if s not in completed_set]

        if not success:
            errors.append(f"Job ended with state: {final_status.state.value}")
            if final_status.reason and final_status.reason.lower() != "none":
                errors.append(f"Reason: {final_status.reason}")
            stages_completed.extend(completed_in_job)
            for stage_name in completed_in_job:
                self.monitor.mark_stage_completed(stage_name)
            self.monitor.end_workflow(success=False)
            return WorkflowResult(
                success=False,
                job_id=job_id,
                start_time=start_time,
                end_time=datetime.now(),
                stages_completed=stages_completed,
                stages_failed=failed_in_job,
                outputs={"slurm_status": final_status.state.value},
                errors=errors,
            )

        stages_completed.extend(completed_in_job)

        # Mark container stages as completed in the monitor so the
        # timing summary shows checkmarks instead of "?" for stages
        # that ran inside the SLURM job.
        for stage_name in completed_in_job:
            self.monitor.mark_stage_completed(stage_name)

        # --- Run post-job stages on login node ---
        if post_job:
            self.monitor.info("-" * 40)
        try:
            post_completed = self._run_stages_on_login_node(post_job)
            stages_completed.extend(post_completed)
        except Exception as e:
            self.monitor.warning(f"Post-job stage failed: {e}")
            # Post-job failures are non-fatal (job succeeded)

        self.monitor.end_workflow(success=True)
        return WorkflowResult(
            success=True,
            job_id=job_id,
            start_time=start_time,
            end_time=datetime.now(),
            stages_completed=stages_completed,
            stages_failed=[],
            outputs={"slurm_status": final_status.state.value},
            errors=errors,
        )

    def submit(
        self,
        wait: bool = False,
        log_file: Path | None = None,
        start_from: str | None = None,
        stop_after: str | None = None,
    ) -> WorkflowResult:
        """Submit workflow as a SLURM job.

        Executes the same stage pipeline as :meth:`run`, but Python-only
        stages run on the login node while container stages are bundled
        into a SLURM bash script and submitted as a job.

        Parameters
        ----------
        wait : bool, default False
            If True, wait for job completion (interactive mode).
            If False, return immediately after job submission.
        log_file : Path, optional
            Custom path for SLURM output log. If not provided, logs are
            written to <work_dir>/slurm-<job_id>.out.
        start_from : str, optional
            Stage name to start from (skip earlier stages).
        stop_after : str, optional
            Stage name to stop after (skip later stages).

        Returns
        -------
        WorkflowResult
            Result with job submission details.
        """
        start_time = datetime.now()
        stages_completed: list[str] = []
        errors: list[str] = []

        validation_errors = self.validate()
        # When resuming mid-pipeline, verify that earlier stages completed.
        if not validation_errors and start_from:
            validation_errors = self._check_prerequisites(start_from)
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

        stages_to_run = self._get_stages_to_run(start_from, stop_after)
        pre_job, job, post_job = self._split_stages_for_submit(stages_to_run)

        # Register all stages for monitoring
        self.monitor.register_stages(stages_to_run)
        self.monitor.start_workflow()

        self._prepare_work_directory()
        self.monitor.info("-" * 40)

        # --- Run pre-job stages on login node ---
        try:
            pre_completed = self._run_stages_on_login_node(pre_job)
            stages_completed.extend(pre_completed)
        except Exception as e:
            self.monitor.error(f"Pre-job stage failed: {e}")
            self.monitor.end_workflow(success=False)
            errors.append(str(e))
            return WorkflowResult(
                success=False,
                job_id=None,
                start_time=start_time,
                end_time=datetime.now(),
                stages_completed=stages_completed,
                stages_failed=[
                    pre_job[len(stages_completed)]
                    if len(stages_completed) < len(pre_job)
                    else "unknown"
                ],
                outputs={},
                errors=errors,
            )

        # --- Submit container stages as SLURM job ---
        if not job:
            # No container stages — everything ran on login node
            self.monitor.end_workflow(success=True)
            return WorkflowResult(
                success=True,
                job_id=None,
                start_time=start_time,
                end_time=datetime.now(),
                stages_completed=stages_completed,
                stages_failed=[],
                outputs={"slurm_status": "NO_JOB_NEEDED"},
                errors=[],
            )

        self.monitor.info(f"Submitting SLURM job with {len(job)} stage(s):")
        for stage_name in job:
            self.monitor.info(f"  \u2022 {stage_name}")

        self._generate_runner_script(job)

        job_script = self.config.paths.work_dir / "submit_job.sh"
        self.slurm.generate_job_script(job_script, log_file=log_file)

        job_id = self.slurm.submit_job(job_script)

        if not wait:
            # Return immediately without waiting for job completion
            self.monitor.end_workflow(success=True)
            return WorkflowResult(
                success=True,
                job_id=job_id,
                start_time=start_time,
                end_time=datetime.now(),
                stages_completed=stages_completed,
                stages_failed=[],
                outputs={"slurm_status": "SUBMITTED"},
                errors=[],
            )

        return self._wait_for_slurm_job(
            job_id=job_id,
            job=job,
            post_job=post_job,
            start_time=start_time,
            stages_completed=stages_completed,
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


def submit_workflow(
    config_path: Path | str,
    wait: bool = True,
    start_from: str | None = None,
    stop_after: str | None = None,
) -> WorkflowResult:
    """Submit workflow as SLURM job.

    Parameters
    ----------
    config_path : Path or str
        Path to YAML configuration file.
    wait : bool, default True
        If True, wait for job completion. If False, return immediately
        after job submission.
    start_from : str, optional
        Stage name to start from (skip earlier stages).
    stop_after : str, optional
        Stage name to stop after (skip later stages).

    Returns
    -------
    WorkflowResult
        Result with job submission details.
    """
    config = CoastalCalibConfig.from_yaml(Path(config_path))
    runner = CoastalCalibRunner(config)
    return runner.submit(
        wait=wait,
        start_from=start_from,
        stop_after=stop_after,
    )
