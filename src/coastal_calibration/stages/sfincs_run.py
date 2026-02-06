"""SFINCS model execution stage using Singularity container."""

from __future__ import annotations

from typing import Any

from coastal_calibration.stages.sfincs_build import SfincsStageBase


class SfincsRunStage(SfincsStageBase):
    """Run the SFINCS model using a Singularity container.

    Uses ``hydromt_sfincs.run.run_sfincs`` to execute SFINCS inside a
    Singularity container based on the ``deltares/sfincs-cpu`` Docker image.
    If a pre-pulled SIF file is configured, it is used as the executable
    directly.
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
        from hydromt_sfincs.run import run_sfincs  # pyright: ignore[reportMissingImports]

        self._update_substep("Running SFINCS model")

        model_root = self.config.paths.model_root
        container = self.config.container

        if container.sif_path is not None:
            self._log(f"Running SFINCS with SIF: {container.sif_path}")
            run_sfincs(
                model_root=model_root,
                sfincs_exe=str(container.sif_path),
                verbose=True,
            )
        else:
            self._log(f"Running SFINCS via Singularity (tag: {container.docker_tag})")
            run_sfincs(
                model_root=model_root,
                vm="singularity",
                docker_tag=container.docker_tag,
                verbose=True,
            )

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

        container = self.config.container
        if container.sif_path is not None and not container.sif_path.exists():
            errors.append(f"Singularity image not found: {container.sif_path}")

        return errors
