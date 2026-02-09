"""SFINCS model build stages using HydroMT-SFINCS Python API."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from hydromt_sfincs import SfincsModel  # pyright: ignore[reportMissingImports]

    from coastal_calibration.config.sfincs_schema import SfincsConfig
    from coastal_calibration.utils.logging import WorkflowMonitor

SFINCS_DOCKER_IMAGE = "deltares/sfincs-cpu"


class SfincsStageBase(ABC):
    """Abstract base class for SFINCS workflow stages.

    Parameters
    ----------
    config : SfincsConfig
        SFINCS workflow configuration.
    monitor : WorkflowMonitor, optional
        Workflow monitor for logging and progress tracking.
    """

    name: str = "base"
    description: str = "Base SFINCS stage"

    def __init__(
        self,
        config: SfincsConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        self.config = config
        self.monitor = monitor
        self._model: SfincsModel | None = None

    @property
    def model(self) -> SfincsModel:
        """The ``hydromt_sfincs.SfincsModel`` instance shared across build stages.

        Set by :class:`SfincsInitStage` and propagated to other stages by
        the runner.

        Raises
        ------
        RuntimeError
            If accessed before the model has been initialised.
        """
        if self._model is None:
            raise RuntimeError(
                f"Stage '{self.name}' requires a SfincsModel instance but none has been set. "
                "Make sure the 'init' stage runs first."
            )
        return self._model

    @model.setter
    def model(self, value: SfincsModel) -> None:
        self._model = value

    @property
    def has_model(self) -> bool:
        """Whether a :class:`SfincsModel` instance has been set."""
        return self._model is not None

    def _log(self, message: str, level: str = "info") -> None:
        """Log message if monitor is available."""
        if self.monitor:
            getattr(self.monitor, level)(message)

    def _update_substep(self, substep: str) -> None:
        """Update current substep."""
        if self.monitor:
            self.monitor.update_substep(self.name, substep)

    def _resolve_sif_path(self) -> Path:
        """Resolve the Singularity SIF path from configuration.

        If ``container.sif_path`` is set, returns it directly.
        Otherwise returns a default path under ``model_root``.
        """
        container = self.config.container
        if container.sif_path is not None:
            return container.sif_path
        return self.config.paths.model_root / f"sfincs-cpu_{container.docker_tag}.sif"

    def pull_singularity_image(self, sif_path: Path | None = None) -> Path:
        """Pull the SFINCS Docker image as a Singularity SIF file.

        Equivalent to::

            singularity pull sfincs-cpu.sif docker://deltares/sfincs-cpu:<tag>

        If the SIF file already exists the pull is skipped.

        Parameters
        ----------
        sif_path : Path, optional
            Destination for the SIF file. When *None*, resolved via
            :meth:`_resolve_sif_path`.

        Returns
        -------
        Path
            Path to the (existing or newly pulled) SIF file.
        """
        if sif_path is None:
            sif_path = self._resolve_sif_path()

        if sif_path.exists():
            self._log(f"SIF already exists: {sif_path}")
            return sif_path

        sif_path.parent.mkdir(parents=True, exist_ok=True)
        docker_tag = self.config.container.docker_tag
        docker_uri = f"docker://{SFINCS_DOCKER_IMAGE}:{docker_tag}"
        cmd = ["singularity", "pull", str(sif_path), docker_uri]
        self._log(f"Pulling Singularity image: {docker_uri} -> {sif_path}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            self._log(f"singularity pull failed: {result.stderr[-2000:]}", "error")
            raise RuntimeError(f"singularity pull failed: {result.stderr}")

        return sif_path

    def run_singularity_command(
        self,
        sif_path: Path,
        model_root: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run SFINCS inside a Singularity container.

        Equivalent to::

            singularity run -B<model_root>:/data <sif_path>

        Stdout and stderr are streamed to the console and written to
        ``sfincs_log.txt`` inside *model_root*.

        Parameters
        ----------
        sif_path : Path
            Path to the Singularity SIF image.
        model_root : Path, optional
            Model root directory (must contain ``sfincs.inp``).
            Defaults to ``config.paths.model_root``.

        Returns
        -------
        subprocess.CompletedProcess
            Completed process result.

        Raises
        ------
        RuntimeError
            If ``singularity`` is not found or SFINCS exits with a
            non-zero return code.
        """
        if model_root is None:
            model_root = self.config.paths.model_root

        cmd = ["singularity", "run", f"-B{model_root}:/data", str(sif_path)]
        self._log(f"Running: {' '.join(cmd)}")

        log_path = model_root / "sfincs_log.txt"
        with subprocess.Popen(
            cmd,
            cwd=model_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as proc:
            with log_path.open("w") as f:
                assert proc.stdout is not None  # noqa: S101
                assert proc.stderr is not None  # noqa: S101
                for line in proc.stdout:
                    f.write(line)
                for line in proc.stderr:
                    f.write(line)
            proc.wait()

        if proc.returncode == 127:
            raise RuntimeError("singularity not found. Make sure it is installed and on PATH.")
        if proc.returncode != 0:
            raise RuntimeError(f"SFINCS run failed with return code {proc.returncode}")

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout="",
            stderr="",
        )

    @abstractmethod
    def run(self) -> None:
        """Execute the stage."""

    def validate(self) -> list[str]:
        """Validate stage prerequisites.

        Returns
        -------
        list of str
            List of validation error messages (empty if valid).
        """
        return []


class SfincsInitStage(SfincsStageBase):
    """Initialize a new SfincsModel instance."""

    name = "init"
    description = "Initialize SFINCS model"

    def run(self) -> None:
        """Create a SfincsModel with the configured data libraries and root."""
        from hydromt_sfincs import SfincsModel  # pyright: ignore[reportMissingImports]

        self._update_substep("Initializing SfincsModel")

        data_libs = list(self.config.data_libs)
        if self.config.paths.data_catalog_path is not None:
            data_libs.append(str(self.config.paths.data_catalog_path))

        model_root = str(self.config.paths.model_root)

        self.model = SfincsModel(
            data_libs=data_libs,
            root=model_root,
            mode="w+",
            write_gis=True,
        )

        self._log(f"SfincsModel initialized at {model_root}")


class SfincsGridStage(SfincsStageBase):
    """Create the SFINCS computational grid from a region geometry."""

    name = "grid"
    description = "Create computational grid"

    def run(self) -> None:
        """Generate grid from region GeoJSON."""
        cfg = self.config.grid

        self._update_substep("Creating grid from region")

        self.model.grid.create_from_region(
            region={"geom": str(cfg.region_geojson)},
            res=cfg.resolution,
            rotated=cfg.rotated,
            crs=cfg.crs,
        )

        self._log(f"Grid created: res={cfg.resolution}, crs={cfg.crs}, rotated={cfg.rotated}")

    def validate(self) -> list[str]:
        """Validate that the region GeoJSON exists."""
        errors = super().validate()
        if not self.config.grid.region_geojson.exists():
            errors.append(f"Region GeoJSON not found: {self.config.grid.region_geojson}")
        return errors


class SfincsElevationStage(SfincsStageBase):
    """Add elevation data to the SFINCS model."""

    name = "elevation"
    description = "Add elevation data"

    def run(self) -> None:
        """Load and merge elevation datasets onto the model grid."""
        cfg = self.config.elevation

        self._update_substep("Adding elevation data")

        elevation_list = [entry.to_dict() for entry in cfg.datasets]

        self.model.elevation.create(
            elevation_list=elevation_list,
            buffer_cells=cfg.buffer_cells,
        )

        self._log(f"Elevation created from {len(cfg.datasets)} dataset(s)")


class SfincsMaskStage(SfincsStageBase):
    """Create the mask of active, water level, and outflow boundary cells."""

    name = "mask"
    description = "Create cell mask and boundaries"

    def run(self) -> None:
        """Generate active cell mask and boundary conditions."""
        cfg = self.config.mask

        self._update_substep("Creating active cell mask")

        mask_kwargs: dict[str, Any] = {"zmin": cfg.active_zmin}
        if cfg.active_fill_area is not None:
            mask_kwargs["fill_area"] = cfg.active_fill_area
        if cfg.active_drop_area is not None:
            mask_kwargs["drop_area"] = cfg.active_drop_area

        self.model.mask.create_active(**mask_kwargs)

        self._update_substep("Adding water level boundary")
        self.model.mask.create_boundary(
            btype="waterlevel",
            zmax=cfg.waterlevel_zmax,
            reset_bounds=cfg.reset_bounds,
        )

        if cfg.outflow_polygon is not None:
            self._update_substep("Adding outflow boundary")
            gdf_include = self.model.data_catalog.get_geodataframe(str(cfg.outflow_polygon))
            if gdf_include is None:
                raise RuntimeError(f"Outflow polygon geodataset not found: {cfg.outflow_polygon}")
            self.model.mask.create_boundary(
                btype="outflow",
                include_polygon=gdf_include,
                reset_bounds=cfg.reset_bounds,
            )

        self._log("Mask and boundaries created")


class SfincsRoughnessStage(SfincsStageBase):
    """Add spatially varying roughness data."""

    name = "roughness"
    description = "Add roughness data"

    def run(self) -> None:
        """Derive Manning's n roughness from land use data."""
        cfg = self.config.roughness

        if not cfg.datasets:
            self._log("No roughness datasets configured, skipping")
            return

        self._update_substep("Adding roughness data")

        roughness_list = [entry.to_dict() for entry in cfg.datasets]

        self.model.roughness.create(
            roughness_list=roughness_list,
            manning_land=cfg.manning_land,
            manning_sea=cfg.manning_sea,
            rgh_lev_land=cfg.rgh_lev_land,  # pyright: ignore[reportArgumentType]
        )

        self._log(f"Roughness created from {len(cfg.datasets)} dataset(s)")


class SfincsSubgridStage(SfincsStageBase):
    """Create subgrid-derived tables for improved accuracy."""

    name = "subgrid"
    description = "Create subgrid tables"

    def run(self) -> None:
        """Generate subgrid tables from elevation and roughness data."""
        cfg = self.config.subgrid

        if not cfg.enabled:
            self._log("Subgrid tables disabled, skipping")
            return

        self._update_substep("Creating subgrid tables")

        elevation_list = [entry.to_dict() for entry in self.config.elevation.datasets]
        roughness_list = [entry.to_dict() for entry in self.config.roughness.datasets]

        self.model.subgrid.create(
            elevation_list=elevation_list,
            roughness_list=roughness_list,
            nr_subgrid_pixels=cfg.nr_subgrid_pixels,
            write_dep_tif=cfg.write_dep_tif,
            write_man_tif=cfg.write_man_tif,
        )

        self._log(f"Subgrid tables created with {cfg.nr_subgrid_pixels} pixels/cell")


class SfincsTimingStage(SfincsStageBase):
    """Set simulation timing parameters."""

    name = "timing"
    description = "Set simulation timing"

    def run(self) -> None:
        """Configure tref, tstart, and tstop."""
        cfg = self.config.timing

        self._update_substep("Setting simulation timing")

        self.model.config.update(
            {
                "tref": cfg.tref,
                "tstart": cfg.tstart,
                "tstop": cfg.tstop,
            }
        )

        self._log(f"Timing set: {cfg.tstart} to {cfg.tstop}")


class SfincsForcingStage(SfincsStageBase):
    """Add water level forcing to the model."""

    name = "forcing"
    description = "Add water level forcing"

    def run(self) -> None:
        """Add water level boundary forcing from a geodataset."""
        cfg = self.config.forcing

        if cfg.waterlevel_geodataset is None:
            self._log("No water level geodataset configured, skipping")
            return

        self._update_substep("Adding water level forcing")

        self.model.water_level.create(geodataset=cfg.waterlevel_geodataset)

        self._log(f"Water level forcing added from {cfg.waterlevel_geodataset}")


class SfincsWriteStage(SfincsStageBase):
    """Write the SFINCS model to disk."""

    name = "write"
    description = "Write model to disk"

    def run(self) -> None:
        """Write all model files to the model root directory."""
        self._update_substep("Writing model to disk")

        self.model.write()

        model_root = str(self.config.paths.model_root)
        self._log(f"Model written to {model_root}")
