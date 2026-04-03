"""Forcing preparation stages for SCHISM workflow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from coastal_calibration.base import WorkflowStage

if TYPE_CHECKING:
    from pathlib import Path

    from coastal_calibration.config.schema import CoastalCalibConfig, SchismModelConfig
    from coastal_calibration.logging import WorkflowMonitor


class PreForcingStage(WorkflowStage):
    """Prepare data for atmospheric forcing generation."""

    name = "schism_forcing_prep"
    description = "Prepare LDASIN forcing data"

    def __init__(self, config: CoastalCalibConfig, monitor: WorkflowMonitor | None = None) -> None:
        super().__init__(config, monitor)
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

    def run(self) -> dict[str, Any]:
        """Stage LDASIN forcing files and create output directories."""
        from coastal_calibration.schism.prep import stage_ldasin_files

        self._update_substep("Building environment")
        self.build_environment()

        self._update_substep("Creating output directories")
        work_dir = self.config.paths.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        self._update_substep("Staging LDASIN files")
        self._log("Creating forcing symlinks and output directories")

        sim = self.config.simulation
        nwm_forcing_dir = self.config.paths.meteo_dir(sim.meteo_source)

        _forcing_input_dir, coastal_forcing_output = stage_ldasin_files(
            work_dir=work_dir,
            start_date=sim.start_date,
            duration_hours=sim.duration_hours,
            nwm_forcing_dir=nwm_forcing_dir,
        )

        self._log(f"Pre-forcing complete — output dir: {coastal_forcing_output}")
        return {
            "forcing_output_dir": str(coastal_forcing_output),
            "status": "completed",
        }


class NWMForcingStage(WorkflowStage):
    """Generate atmospheric forcing using WRF-Hydro workflow driver."""

    name = "schism_forcing"
    description = "Regrid atmospheric forcing (MPI)"

    def __init__(self, config: CoastalCalibConfig, monitor: WorkflowMonitor | None = None) -> None:
        super().__init__(config, monitor)
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

    def _build_mpi_command(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        geogrid_file: Path,
        schism_mesh: Path,
        length_hrs: int,
        forcing_begin_date: str,
        forcing_end_date: str,
        job_index: int = 0,
        job_count: int = 1,
    ) -> list[str]:
        """Build the MPI command for regrid_forcings.

        Extracted as a testable seam.
        """
        import sys

        from coastal_calibration.utils import build_mpi_cmd

        cmd = build_mpi_cmd(self.model.total_tasks, oversubscribe=self.model.oversubscribe)
        cmd.extend(
            [
                sys.executable,
                "-m",
                "coastal_calibration.regridding.regrid_forcings",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--geogrid-file",
                str(geogrid_file),
                "--schism-mesh",
                str(schism_mesh),
                "--length-hrs",
                str(length_hrs),
                "--forcing-begin-date",
                forcing_begin_date,
                "--forcing-end-date",
                forcing_end_date,
                "--job-index",
                str(job_index),
                "--job-count",
                str(job_count),
            ]
        )
        return cmd

    def run(self) -> dict[str, Any]:
        """Execute NWM forcing generation with MPI."""
        import subprocess

        self._update_substep("Building environment")
        env = self.build_environment()

        self._update_substep("Setting up forcing parameters")
        sim = self.config.simulation
        start_pdy = sim.start_pdy
        start_cyc = sim.start_cyc
        length_hrs = int(sim.duration_hours)

        forcing_begin = f"{start_pdy}{start_cyc}00"
        start_dt = datetime.strptime(f"{start_pdy} {start_cyc}", "%Y%m%d %H").replace(tzinfo=UTC)
        end_dt = start_dt + timedelta(hours=sim.duration_hours)
        forcing_end = end_dt.strftime("%Y%m%d%H00")

        nwm_forcing_output = self.config.paths.work_dir / "forcing_input"
        coastal_forcing_output = self.config.paths.work_dir / "coastal_forcing_output"
        geogrid_file = self.model.geogrid_file
        schism_mesh = self.model.schism_mesh

        cmd = self._build_mpi_command(
            input_dir=nwm_forcing_output,
            output_dir=coastal_forcing_output,
            geogrid_file=geogrid_file,
            schism_mesh=schism_mesh,
            length_hrs=length_hrs,
            forcing_begin_date=forcing_begin,
            forcing_end_date=forcing_end,
            job_index=0,
            job_count=1,
        )

        self._log(f"Generating {length_hrs}h forcing from {forcing_begin} via MPI")
        self._update_substep("Running regrid_forcings with MPI")

        result = subprocess.run(
            cmd,
            env=env,
            cwd=self.config.paths.work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"regrid_forcings failed (exit {result.returncode}): {result.stderr[-2000:]}"
            )

        self._log(f"NWM forcing generated in {nwm_forcing_output}")
        return {
            "forcing_output_dir": str(nwm_forcing_output),
            "status": "completed",
        }


class PostForcingStage(WorkflowStage):
    """Generate sflux atmospheric forcing from LDASIN data."""

    name = "schism_sflux"
    description = "Generate sflux atmospheric files"

    def __init__(self, config: CoastalCalibConfig, monitor: WorkflowMonitor | None = None) -> None:
        super().__init__(config, monitor)
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

    def run(self) -> dict[str, Any]:
        """Generate sflux files from LDASIN forcing data."""
        from coastal_calibration.schism.prep import make_sflux

        self._update_substep("Building environment")
        self.build_environment()

        work_dir = self.config.paths.work_dir
        sim = self.config.simulation

        # Symlink precip_source.nc from coastal_forcing_output
        precip_src = work_dir / "coastal_forcing_output" / "precip_source.nc"
        precip_dst = work_dir / "precip_source.nc"
        if precip_src.exists() and not precip_dst.exists():
            precip_dst.symlink_to(precip_src)

        self._log("Generating sflux from LDASIN files")
        self._update_substep("Running make_atmo_sflux")

        make_sflux(
            work_dir=work_dir,
            forcing_input_dir=work_dir / "forcing_input",
            start_date=sim.start_date,
            duration_hours=sim.duration_hours,
            geogrid_file=self.model.geogrid_file,
        )

        # Verify sflux output was produced
        sflux_dir = work_dir / "sflux"
        if not sflux_dir.exists() or not any(sflux_dir.iterdir()):
            raise RuntimeError(f"post_forcing: no sflux files produced in {sflux_dir}")

        n_sflux = sum(1 for _ in sflux_dir.iterdir())
        self._log(f"Post-processing complete — {n_sflux} sflux file(s) produced")
        return {"status": "completed"}
