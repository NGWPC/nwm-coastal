"""Runner for the SFINCS model creation workflow."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coastal_calibration.config.create_schema import SfincsCreateConfig
from coastal_calibration.runner import WorkflowResult
from coastal_calibration.stages.sfincs_create import (
    CreateStage,
    _load_existing_model,
    create_stages,
)
from coastal_calibration.utils.logging import (
    WorkflowMonitor,
    configure_logger,
    generate_log_path,
    silence_third_party_loggers,
)

if TYPE_CHECKING:
    pass


class SfincsCreator:
    """Runner that orchestrates the SFINCS model creation pipeline.

    Mirrors :class:`~coastal_calibration.runner.CoastalCalibRunner` but
    operates on a :class:`SfincsCreateConfig` and delegates to
    :class:`~coastal_calibration.stages.sfincs_create.CreateStage` instances.
    """

    _STATUS_FILENAME = ".create_status.json"

    def __init__(self, config: SfincsCreateConfig) -> None:
        self.config = config

        # Ensure the output directory exists early so file logging can start.
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up file logging before creating the monitor.
        if not config.monitoring.log_file:
            log_path = generate_log_path(config.output_dir, prefix="sfincs-create")
            configure_logger(file=str(log_path), file_level="DEBUG")

        silence_third_party_loggers()

        self.monitor = WorkflowMonitor(config.monitoring)
        self._stages: dict[str, CreateStage] = {}
        self._results: dict[str, Any] = {}

    @property
    def stage_order(self) -> list[str]:
        """Active stage order based on config."""
        return self.config.stage_order

    def _init_stages(self) -> None:
        """Instantiate all creation stages."""
        self._stages = create_stages(self.config, self.monitor)

    # ------------------------------------------------------------------
    # Pipeline status tracking
    # ------------------------------------------------------------------

    @property
    def _status_path(self) -> Path:
        """Path to the pipeline status file."""
        return self.config.output_dir / self._STATUS_FILENAME

    def _load_status(self) -> dict[str, Any]:
        """Load pipeline status from disk (empty dict if missing)."""
        if self._status_path.exists():
            return json.loads(self._status_path.read_text())
        return {}

    def _save_stage_status(self, stage_name: str) -> None:
        """Mark *stage_name* as completed in the status file."""
        status = self._load_status()
        completed: list[str] = status.get("completed_stages", [])
        if stage_name not in completed:
            completed.append(stage_name)
        status["completed_stages"] = completed
        self._status_path.write_text(json.dumps(status, indent=2) + "\n")

    def _check_prerequisites(self, start_from: str) -> list[str]:
        """Verify that all stages before *start_from* have completed."""
        status = self._load_status()
        completed: set[str] = set(status.get("completed_stages", []))

        all_stages = self.stage_order
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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate configuration and stage prerequisites.

        Returns
        -------
        list of str
            Validation error messages (empty if valid).
        """
        errors: list[str] = []

        config_errors = self.config.validate()
        errors.extend(config_errors)

        self._init_stages()
        for name, stage in self._stages.items():
            stage_errors = stage.validate()
            errors.extend(f"[{name}] {error}" for error in stage_errors)

        return errors

    # ------------------------------------------------------------------
    # Stage selection
    # ------------------------------------------------------------------

    def _get_stages_to_run(
        self,
        start_from: str | None,
        stop_after: str | None,
    ) -> list[str]:
        """Determine which stages to run."""
        stages = self.stage_order.copy()

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

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(
        self,
        start_from: str | None = None,
        stop_after: str | None = None,
        dry_run: bool = False,
    ) -> WorkflowResult:
        """Execute the SFINCS model creation workflow.

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

        self.monitor.register_stages(self.stage_order)
        self.monitor.start_workflow()
        self.monitor.info("-" * 40)

        stages_to_run = self._get_stages_to_run(start_from, stop_after)

        # When resuming from a later stage, load the existing model so
        # that stages which reference ``self.sfincs`` can find it.
        if start_from and "create_grid" not in stages_to_run:
            _load_existing_model(self.config)

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

        result_file = self.config.output_dir / "create_result.json"
        result.save(result_file)
        self.monitor.save_progress(self.config.output_dir / "create_progress.json")

        return result
