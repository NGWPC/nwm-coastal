"""Standalone SFINCS model build and run workflow runner."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from coastal_calibration.config.schema import MonitoringConfig
from coastal_calibration.config.sfincs_schema import SfincsConfig
from coastal_calibration.runner import WorkflowResult
from coastal_calibration.stages.sfincs_build import (
    SfincsElevationStage,
    SfincsForcingStage,
    SfincsGridStage,
    SfincsInitStage,
    SfincsMaskStage,
    SfincsRoughnessStage,
    SfincsStageBase,
    SfincsSubgridStage,
    SfincsTimingStage,
    SfincsWriteStage,
)
from coastal_calibration.stages.sfincs_run import SfincsRunStage
from coastal_calibration.utils.logging import WorkflowMonitor

if TYPE_CHECKING:
    from pathlib import Path


class SfincsRunner:
    """Standalone runner for SFINCS model build and run workflow.

    This runner orchestrates the complete SFINCS workflow: building a model
    from scratch using the HydroMT-SFINCS Python API and then running it
    via a Singularity container.

    Parameters
    ----------
    config : SfincsConfig
        SFINCS workflow configuration.

    Examples
    --------
    >>> from coastal_calibration.config.sfincs_schema import SfincsConfig
    >>> config = SfincsConfig.from_yaml("sfincs_config.yaml")  # doctest: +SKIP
    >>> runner = SfincsRunner(config)  # doctest: +SKIP
    >>> result = runner.run()  # doctest: +SKIP
    """

    STAGE_ORDER: ClassVar[list[str]] = [
        "init",
        "grid",
        "elevation",
        "mask",
        "roughness",
        "subgrid",
        "timing",
        "forcing",
        "write",
        "sfincs_run",
    ]

    BUILD_STAGES: ClassVar[list[str]] = [
        "init",
        "grid",
        "elevation",
        "mask",
        "roughness",
        "subgrid",
        "timing",
        "forcing",
        "write",
    ]

    def __init__(self, config: SfincsConfig) -> None:
        self.config = config
        self.monitor = WorkflowMonitor(
            MonitoringConfig(
                log_level=config.monitoring.log_level,
                log_file=config.monitoring.log_file,
            )
        )
        self._stages: dict[str, SfincsStageBase] = {}

    def _init_stages(self) -> None:
        """Initialize all workflow stages."""
        self._stages = {
            "init": SfincsInitStage(self.config, self.monitor),
            "grid": SfincsGridStage(self.config, self.monitor),
            "elevation": SfincsElevationStage(self.config, self.monitor),
            "mask": SfincsMaskStage(self.config, self.monitor),
            "roughness": SfincsRoughnessStage(self.config, self.monitor),
            "subgrid": SfincsSubgridStage(self.config, self.monitor),
            "timing": SfincsTimingStage(self.config, self.monitor),
            "forcing": SfincsForcingStage(self.config, self.monitor),
            "write": SfincsWriteStage(self.config, self.monitor),
            "sfincs_run": SfincsRunStage(self.config, self.monitor),
        }

    def validate(self, stages: list[str] | None = None) -> list[str]:
        """Validate configuration and stage prerequisites.

        Parameters
        ----------
        stages : list of str, optional
            Specific stages to validate. If None, validates all stages
            in ``STAGE_ORDER``.

        Returns
        -------
        list of str
            List of validation error messages (empty if valid).
        """
        errors = self.config.validate()

        self._init_stages()
        stage_names = stages or self.STAGE_ORDER
        for name in stage_names:
            if name in self._stages:
                stage_errors = self._stages[name].validate()
                errors.extend(f"[{name}] {error}" for error in stage_errors)

        return errors

    def _get_stages_to_run(
        self,
        stage_list: list[str],
        start_from: str | None,
        stop_after: str | None,
    ) -> list[str]:
        """Determine which stages to run based on start/stop parameters."""
        stages = stage_list.copy()

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

    def _execute_stages(
        self,
        stage_list: list[str],
        start_from: str | None = None,
        stop_after: str | None = None,
        dry_run: bool = False,
    ) -> WorkflowResult:
        """Execute a sequence of stages.

        Parameters
        ----------
        stage_list : list of str
            Ordered list of stage names to execute.
        start_from : str, optional
            Stage name to start from (skip earlier stages).
        stop_after : str, optional
            Stage name to stop after (skip later stages).
        dry_run : bool
            If True, validate but don't execute.

        Returns
        -------
        WorkflowResult
            Result with execution details.
        """
        start_time = datetime.now()
        stages_completed: list[str] = []
        stages_failed: list[str] = []
        errors: list[str] = []

        validation_errors = self.validate(stage_list)
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

        stages_to_run = self._get_stages_to_run(stage_list, start_from, stop_after)

        self.monitor.register_stages(stages_to_run)
        self.monitor.start_workflow()

        current_stage = ""
        try:
            for current_stage in stages_to_run:
                stage = self._stages[current_stage]

                with self.monitor.stage_context(current_stage, stage.description):
                    stage.run()
                    # Propagate the model instance to subsequent stages
                    if stage.has_model:
                        for s in self._stages.values():
                            s.model = stage.model
                    stages_completed.append(current_stage)

            self.monitor.end_workflow(success=True)
            success = True

        except Exception as e:
            self.monitor.error(f"Workflow failed at stage '{current_stage}': {e}")
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
            outputs={"model_root": str(self.config.paths.model_root)},
            errors=errors,
        )

        result_file = self.config.paths.model_root / "workflow_result.json"
        result.save(result_file)

        return result

    def build(
        self,
        start_from: str | None = None,
        stop_after: str | None = None,
        dry_run: bool = False,
    ) -> WorkflowResult:
        """Build the SFINCS model without running it.

        Executes all build stages (init through write) to produce a
        complete SFINCS model on disk.

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
        return self._execute_stages(
            self.BUILD_STAGES,
            start_from=start_from,
            stop_after=stop_after,
            dry_run=dry_run,
        )

    def run(
        self,
        start_from: str | None = None,
        stop_after: str | None = None,
        dry_run: bool = False,
    ) -> WorkflowResult:
        """Build and run the complete SFINCS workflow.

        Executes all stages including model build and Singularity execution.

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
        return self._execute_stages(
            self.STAGE_ORDER,
            start_from=start_from,
            stop_after=stop_after,
            dry_run=dry_run,
        )

    def run_only(self) -> WorkflowResult:
        """Run a pre-built SFINCS model.

        Executes only the Singularity run stage on an existing model
        directory. The model must have been previously built (``sfincs.inp``
        must exist in ``model_root``).

        Returns
        -------
        WorkflowResult
            Result with execution details.
        """
        return self._execute_stages(["sfincs_run"])


def build_sfincs(
    config_path: Path | str,
    start_from: str | None = None,
    stop_after: str | None = None,
    dry_run: bool = False,
) -> WorkflowResult:
    """Build a SFINCS model from a configuration file.

    Parameters
    ----------
    config_path : Path or str
        Path to SFINCS YAML configuration file.
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
    config = SfincsConfig.from_yaml(config_path)
    runner = SfincsRunner(config)
    return runner.build(start_from=start_from, stop_after=stop_after, dry_run=dry_run)


def run_sfincs_workflow(
    config_path: Path | str,
    start_from: str | None = None,
    stop_after: str | None = None,
    dry_run: bool = False,
) -> WorkflowResult:
    """Build and run a SFINCS model from a configuration file.

    Parameters
    ----------
    config_path : Path or str
        Path to SFINCS YAML configuration file.
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
    config = SfincsConfig.from_yaml(config_path)
    runner = SfincsRunner(config)
    return runner.run(start_from=start_from, stop_after=stop_after, dry_run=dry_run)
