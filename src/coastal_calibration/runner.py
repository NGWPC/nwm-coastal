"""Main workflow runner for coastal model calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coastal_calibration.config.schema import CoastalCalibConfig
from coastal_calibration.utils.logging import (
    WorkflowMonitor,
    configure_logger,
    generate_log_path,
    silence_third_party_loggers,
)

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
    stage execution and progress monitoring.

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
        self._stages: dict[str, WorkflowStage] = {}
        self._results: dict[str, Any] = {}

    @property
    def STAGE_ORDER(self) -> list[str]:  # noqa: N802
        """Active stage order based on model config."""
        return self.config.model_config.stage_order

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
            return json.loads(self._status_path.read_text())
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
