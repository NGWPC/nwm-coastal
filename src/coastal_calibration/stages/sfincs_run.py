"""SFINCS model execution stage using Singularity container."""

from __future__ import annotations

from typing import Any

from coastal_calibration.stages.sfincs_build import SfincsStageBase


class SfincsRunStage(SfincsStageBase):
    """Run the SFINCS model using a Singularity container.

    Pulls the ``deltares/sfincs-cpu`` Docker image as a Singularity SIF
    file (if not already present) and executes SFINCS inside the container
    using :meth:`~SfincsStageBase.run_singularity_command`.
    """

    name = "sfincs_run"
    description = "Run SFINCS model (Singularity)"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute SFINCS via Singularity.

        Parameters
        ----------
        context : dict[str, Any]
            Shared context dictionary (model instance not required for run).

        Returns
        -------
        dict[str, Any]
            Updated context with run status.
        """
        self._update_substep("Pulling Singularity image")
        sif_path = self.pull_singularity_image()

        self._update_substep("Running SFINCS model")
        self.run_singularity_command(sif_path)

        self._log("SFINCS run completed")
        context["run_status"] = "completed"
        return context

    def validate(self) -> list[str]:
        """Validate that model inputs exist for running."""
        errors = super().validate()

        sfincs_inp = self.config.paths.model_root / "sfincs.inp"
        if not sfincs_inp.exists():
            errors.append(
                f"SFINCS input file not found: {sfincs_inp}. Build the model first before running."
            )

        return errors
