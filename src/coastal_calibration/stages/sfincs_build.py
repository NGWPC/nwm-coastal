"""SFINCS model build stages using HydroMT-SFINCS Python API.

All stages subclass :class:`~coastal_calibration.stages.base.WorkflowStage`
and accept a :class:`~coastal_calibration.config.schema.CoastalCalibConfig`.
SFINCS-specific settings are read from ``config.model_config``
(:class:`~coastal_calibration.config.schema.SfincsModelConfig`).

The HydroMT ``SfincsModel`` instance is shared between stages via a
module-level registry keyed by config ``id``.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from coastal_calibration.config.schema import SfincsModelConfig
from coastal_calibration.stages.base import WorkflowStage
from coastal_calibration.stages.sfincs import create_nc_symlinks, generate_data_catalog

if TYPE_CHECKING:
    from pathlib import Path

    from hydromt_sfincs import SfincsModel  # pyright: ignore[reportMissingImports]

    from coastal_calibration.config.schema import CoastalCalibConfig
    from coastal_calibration.utils.logging import WorkflowMonitor

SFINCS_DOCKER_IMAGE = "deltares/sfincs-cpu"

# ---------------------------------------------------------------------------
# Shared model instance management
# ---------------------------------------------------------------------------
# The SFINCS build stages share a HydroMT ``SfincsModel`` between them.
# We use a module-level dictionary keyed by the ``CoastalCalibConfig`` id to
# store the model instance across stages within a single runner invocation.
_MODEL_REGISTRY: dict[int, SfincsModel] = {}


def _set_model(config: CoastalCalibConfig, model: SfincsModel) -> None:
    """Store the SfincsModel instance for the given config."""
    _MODEL_REGISTRY[id(config)] = model


def _get_model(config: CoastalCalibConfig) -> SfincsModel:
    """Retrieve the SfincsModel instance for the given config."""
    try:
        return _MODEL_REGISTRY[id(config)]
    except KeyError:
        raise RuntimeError(
            "SFINCS model not initialised. "
            "Ensure the 'sfincs_init' stage runs before other SFINCS stages."
        ) from None


def _clear_model(config: CoastalCalibConfig) -> None:
    """Remove the SfincsModel instance for the given config."""
    _MODEL_REGISTRY.pop(id(config), None)


# ---------------------------------------------------------------------------
# Helper — resolve paths from CoastalCalibConfig
# ---------------------------------------------------------------------------


def _sfincs_cfg(config: CoastalCalibConfig) -> SfincsModelConfig:
    """Return the SFINCS model config, raising if not the active model."""
    if not isinstance(config.model_config, SfincsModelConfig):
        raise TypeError("model_config must be SfincsModelConfig when model='sfincs'")
    return config.model_config


def get_model_root(config: CoastalCalibConfig) -> Path:
    """Effective model output directory."""
    sfincs = _sfincs_cfg(config)
    return sfincs.model_root or (config.paths.work_dir / "sfincs_model")


def _data_catalog_path(config: CoastalCalibConfig) -> Path | None:
    """Return catalog path if it exists on disk, else None."""
    candidate = config.paths.work_dir / "data_catalog.yml"
    return candidate if candidate.exists() else None


def _waterlevel_geodataset(config: CoastalCalibConfig) -> str | None:
    """Return the geodataset name for water-level forcing, or None."""
    catalog_path = _data_catalog_path(config)
    if catalog_path is None:
        return None
    coastal_source = config.boundary.source
    return f"{coastal_source}_waterlevel" if coastal_source != "tpxo" else "tpxo_tidal"


# ---------------------------------------------------------------------------
# Singularity helpers (used by the run stage)
# ---------------------------------------------------------------------------


def resolve_sif_path(config: CoastalCalibConfig) -> Path:
    """Resolve the Singularity SIF path from configuration."""
    sfincs = _sfincs_cfg(config)
    if sfincs.container_image is not None:
        return sfincs.container_image
    return get_model_root(config) / f"sfincs-cpu_{sfincs.container_tag}.sif"


def _pull_singularity_image(
    config: CoastalCalibConfig,
    sif_path: Path | None = None,
    *,
    _log: Any = None,
) -> Path:
    """Pull the SFINCS Docker image as a Singularity SIF file."""
    if sif_path is None:
        sif_path = resolve_sif_path(config)

    if sif_path.exists():
        return sif_path

    sif_path.parent.mkdir(parents=True, exist_ok=True)
    sfincs = _sfincs_cfg(config)
    docker_uri = f"docker://{SFINCS_DOCKER_IMAGE}:{sfincs.container_tag}"
    cmd = ["singularity", "pull", str(sif_path), docker_uri]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"singularity pull failed: {result.stderr}")

    return sif_path


def _run_singularity(
    config: CoastalCalibConfig,
    sif_path: Path,
    model_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run SFINCS inside a Singularity container."""
    if model_root is None:
        model_root = get_model_root(config)

    cmd = ["singularity", "run", f"-B{model_root}:/data", str(sif_path)]

    log_path = model_root / "sfincs_log.txt"
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
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
                stdout_lines.append(line)
                f.write(line)
            for line in proc.stderr:
                stderr_lines.append(line)
                f.write(line)
        proc.wait()

    if proc.returncode == 127:
        raise RuntimeError("singularity not found. Make sure it is installed and on PATH.")
    if proc.returncode != 0:
        # Include the last portion of stdout/stderr so the caller can
        # see what went wrong without having to open the log file.
        tail_stdout = "".join(stdout_lines[-20:]).rstrip()
        tail_stderr = "".join(stderr_lines[-20:]).rstrip()
        detail = ""
        if tail_stderr:
            detail += f"\n--- stderr (last 20 lines) ---\n{tail_stderr}"
        if tail_stdout:
            detail += f"\n--- stdout (last 20 lines) ---\n{tail_stdout}"
        if not detail:
            detail = f"\n(no output captured -- check {log_path})"
        raise RuntimeError(f"SFINCS run failed with return code {proc.returncode}{detail}")

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
    )


# ---------------------------------------------------------------------------
# SFINCS workflow stages  (all subclass WorkflowStage)
# ---------------------------------------------------------------------------


class SfincsInitStage(WorkflowStage):
    """Initialise the SFINCS model (pre-built mode).

    Copies pre-built model files to the output directory, opens the model
    in ``r+`` mode, reads it, and clears missing file references.
    """

    name = "sfincs_init"
    description = "Initialise SFINCS model (pre-built)"

    #: Config attributes that reference external files.  When a pre-built
    #: ``sfincs.inp`` lists placeholder names for files that haven't been
    #: generated yet, we clear them so HydroMT won't fail on read/write.
    _FILE_REF_ATTRS: tuple[str, ...] = (
        # Meteorological ASCII forcing files
        "amufile",
        "amvfile",
        "ampfile",
        "amprfile",
        # Other optional file references
        "sbgfile",
        "srcfile",
        "disfile",
        "bzsfile",
        "bzifile",
        "precipfile",
        "prcfile",
        "wndfile",
        "spwfile",
        "inifile",
        "rstfile",
        "ncinifile",
        "weirfile",
        "thdfile",
        "drnfile",
        "scsfile",
        "netspwfile",
        "netsrcdisfile",
        "netbndbzsbzifile",
    )

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SfincsModelConfig)  # noqa: S101
        self.sfincs: SfincsModelConfig = config.model_config

    def _clear_missing_file_refs(self, model: SfincsModel) -> None:
        """Clear config references to files that do not exist on disk."""
        model_root = model.root.path
        cfg_data = model.config.data
        cleared: list[str] = []
        for attr in self._FILE_REF_ATTRS:
            val = getattr(cfg_data, attr, None)
            if val and not (model_root / val).exists():
                setattr(cfg_data, attr, None)
                cleared.append(f"{attr}={val}")
        if cleared:
            self._log(f"Cleared missing file references: {', '.join(cleared)}")

    def run(self) -> dict[str, Any]:
        """Load a pre-built SFINCS model and register it for subsequent stages."""
        from hydromt_sfincs import SfincsModel  # pyright: ignore[reportMissingImports]

        root = get_model_root(self.config)

        data_libs: list[str] = []
        catalog_path = _data_catalog_path(self.config)
        if catalog_path is not None:
            data_libs.append(str(catalog_path))

        self._update_substep("Loading pre-built SFINCS model")

        # Copy pre-built files to model_root if source is different
        source_dir = self.sfincs.prebuilt_dir
        if source_dir.resolve() != root.resolve():
            root.mkdir(parents=True, exist_ok=True)
            for src_file in source_dir.iterdir():
                if src_file.is_file():
                    dst_file = root / src_file.name
                    if not dst_file.exists():
                        shutil.copy2(src_file, dst_file)
            self._log(f"Copied pre-built model from {source_dir} to {root}")

        model = SfincsModel(
            data_libs=data_libs,
            root=str(root),
            mode="r+",
            write_gis=True,
        )

        # Read existing model to detect grid type and load components
        model.read()

        # Clear config references to missing files
        self._clear_missing_file_refs(model)

        _set_model(self.config, model)

        self._log(f"SFINCS model initialised (grid_type={model.grid_type}) at {root}")

        return {
            "model_root": str(root),
            "grid_type": model.grid_type,
            "status": "completed",
        }


class SfincsTimingStage(WorkflowStage):
    """Set simulation timing on the SFINCS model."""

    name = "sfincs_timing"
    description = "Set SFINCS timing"

    def run(self) -> dict[str, Any]:
        """Configure tref, tstart, and tstop on the model."""
        model = _get_model(self.config)
        sim = self.config.simulation
        start = sim.start_date
        stop = start + timedelta(hours=sim.duration_hours)

        self._update_substep("Setting simulation timing")

        model.config.update(
            {
                "tref": start,
                "tstart": start,
                "tstop": stop,
            }
        )

        self._log(f"Timing set: {start} to {stop}")

        return {"status": "completed"}


class SfincsForcingStage(WorkflowStage):
    """Add water level boundary forcing."""

    name = "sfincs_forcing"
    description = "Add water level forcing"

    def run(self) -> dict[str, Any]:
        """Add water level boundary forcing from a geodataset."""
        wl_geodataset = _waterlevel_geodataset(self.config)

        if wl_geodataset is None:
            self._log("No water level geodataset configured, skipping")
            return {"status": "skipped"}

        model = _get_model(self.config)

        self._update_substep("Adding water level forcing")
        model.water_level.create(geodataset=wl_geodataset)
        self._log(f"Water level forcing added from {wl_geodataset}")

        return {"status": "completed"}


class SfincsObservationPointsStage(WorkflowStage):
    """Add observation points to the model."""

    name = "sfincs_obs"
    description = "Add observation points"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SfincsModelConfig)  # noqa: S101
        self.sfincs: SfincsModelConfig = config.model_config

    def run(self) -> dict[str, Any]:
        """Add observation points from config or file."""
        model = _get_model(self.config)

        if self.sfincs.observation_locations_file is None and not self.sfincs.observation_points:
            self._log("No observation points configured, skipping")
            return {"status": "skipped"}

        self._update_substep("Adding observation points")

        # When merge=False, clear existing observation points first
        if not self.sfincs.merge_observations:
            try:
                existing = model.observation_points.nr_points
                if existing > 0:
                    model.observation_points.clear()
                    self._log(f"Cleared {existing} existing observation point(s)")
            except Exception:  # noqa: S110
                pass  # No existing points to clear

        if self.sfincs.observation_locations_file is not None:
            model.observation_points.create(
                locations=str(self.sfincs.observation_locations_file),
                merge=self.sfincs.merge_observations,
            )
            self._log(f"Observation points added from {self.sfincs.observation_locations_file}")
        elif self.sfincs.observation_points:
            for pt in self.sfincs.observation_points:
                model.observation_points.add_point(
                    x=pt["x"],
                    y=pt["y"],
                    name=pt.get("name", f"obs_{self.sfincs.observation_points.index(pt)}"),
                )
            self._log(f"Added {len(self.sfincs.observation_points)} observation point(s)")

        return {"status": "completed"}


class SfincsDischargeStage(WorkflowStage):
    """Add discharge source points to the model."""

    name = "sfincs_discharge"
    description = "Add discharge sources"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SfincsModelConfig)  # noqa: S101
        self.sfincs: SfincsModelConfig = config.model_config

    @staticmethod
    def _parse_src_file(path: Path) -> list[tuple[float, float, str]]:
        """Parse a SFINCS ``.src`` file into (x, y, name) tuples."""
        from pathlib import Path as _Path

        points: list[tuple[float, float, str]] = []
        for raw_line in _Path(path).read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split('"')
            coords = parts[0].strip().split()
            name = parts[1].strip() if len(parts) > 1 else f"src_{len(points)}"
            x, y = float(coords[0]), float(coords[1])
            points.append((x, y, name))
        return points

    def run(self) -> dict[str, Any]:
        """Add discharge source points from a ``.src`` or GeoJSON file."""
        model = _get_model(self.config)

        if self.sfincs.discharge_locations_file is None:
            self._log("No discharge configuration, skipping")
            return {"status": "skipped"}

        self._update_substep("Adding discharge source points")

        # When merge=False, clear existing discharge points first
        if not self.sfincs.merge_discharge:
            try:
                existing = model.discharge_points.nr_points
                if existing > 0:
                    model.discharge_points.clear()
                    self._log(f"Cleared {existing} existing discharge point(s)")
            except Exception:  # noqa: S110
                pass  # No existing points to clear

        src_path = self.sfincs.discharge_locations_file
        suffix = src_path.suffix.lower()

        if suffix == ".src":
            parsed = self._parse_src_file(src_path)
            for x, y, name in parsed:
                model.discharge_points.add_point(x=x, y=y, name=name)
            self._log(f"Added {len(parsed)} discharge source point(s) from {src_path}")
        else:
            model.discharge_points.create(
                locations=str(src_path),
                merge=self.sfincs.merge_discharge,
            )
            self._log(f"Discharge source points added from {src_path}")

        return {"status": "completed"}


class SfincsPrecipitationStage(WorkflowStage):
    """Add precipitation forcing."""

    name = "sfincs_precip"
    description = "Add precipitation forcing"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SfincsModelConfig)  # noqa: S101
        self.sfincs: SfincsModelConfig = config.model_config

    def run(self) -> dict[str, Any]:
        """Add precipitation from a dataset in the data catalog."""
        if self.sfincs.precip_dataset is None:
            self._log("No precipitation configured, skipping")
            return {"status": "skipped"}

        model = _get_model(self.config)

        self._update_substep("Adding precipitation forcing")
        model.precipitation.create(precip=self.sfincs.precip_dataset)
        self._log(f"Precipitation forcing added from {self.sfincs.precip_dataset}")

        return {"status": "completed"}


class SfincsWriteStage(WorkflowStage):
    """Write the SFINCS model files to disk."""

    name = "sfincs_write"
    description = "Write SFINCS model"

    def run(self) -> dict[str, Any]:
        """Write all model files to the model root directory."""
        model = _get_model(self.config)

        self._update_substep("Writing model to disk")
        model.write()

        root = get_model_root(self.config)
        self._log(f"SFINCS model written to {root}")

        return {
            "model_root": str(root),
            "status": "completed",
        }


class SfincsRunStage(WorkflowStage):
    """Run the SFINCS model via Singularity container."""

    name = "sfincs_run"
    description = "Run SFINCS model (Singularity)"

    def run(self) -> dict[str, Any]:
        """Execute SFINCS via Singularity."""
        self._update_substep("Pulling Singularity image")
        sif_path = _pull_singularity_image(self.config, _log=self._log)

        self._update_substep("Running SFINCS model")
        _run_singularity(self.config, sif_path)

        self._log("SFINCS run completed")

        # Clean up model registry
        _clear_model(self.config)

        return {"status": "completed"}


# ---------------------------------------------------------------------------
# Infrastructure stages (symlinks, data catalog)
# ---------------------------------------------------------------------------


class SfincsSymlinksStage(WorkflowStage):
    """Create ``.nc`` symlinks for NWM files in the download directory.

    NWM LDASIN and CHRTOUT files lack a ``.nc`` extension, which confuses
    HydroMT's dataset readers.  This stage creates symlinks with the
    ``.nc`` suffix so the data catalog entries resolve correctly.

    If the download directory does not exist, this stage is a no-op.
    """

    name = "sfincs_symlinks"
    description = "Create .nc symlinks for NWM data"

    def run(self) -> dict[str, Any]:
        """Create ``.nc`` symlinks for NWM LDASIN and CHRTOUT files."""
        download_dir = self.config.paths.download_dir
        if not download_dir.exists():
            self._log(f"Download dir {download_dir} does not exist — skipping symlinks")
            return {"meteo_symlinks": 0, "streamflow_symlinks": 0, "status": "skipped"}

        self._update_substep("Creating .nc symlinks")
        meteo_source = self.config.simulation.meteo_source

        created = create_nc_symlinks(
            download_dir,
            meteo_source=meteo_source,
            include_meteo=True,
            include_streamflow=True,
        )

        n_meteo = len(created["meteo"])
        n_stream = len(created["streamflow"])
        self._log(f"Created {n_meteo} meteo + {n_stream} streamflow symlinks in {download_dir}")

        return {
            "meteo_symlinks": n_meteo,
            "streamflow_symlinks": n_stream,
            "status": "completed",
        }


class SfincsDataCatalogStage(WorkflowStage):
    """Generate HydroMT data catalog for the SFINCS pipeline.

    Delegates to :func:`generate_data_catalog` which already accepts a
    :class:`CoastalCalibConfig`.

    If the download directory does not exist (e.g. download is disabled),
    the catalog is skipped — there are no data files to reference.
    """

    name = "sfincs_data_catalog"
    description = "Generate HydroMT data catalog for SFINCS"

    def run(self) -> dict[str, Any]:
        """Generate a HydroMT data catalog YAML for downloaded data."""
        download_dir = self.config.paths.download_dir
        if not download_dir.exists():
            self._log(f"Download dir {download_dir} does not exist — skipping catalog generation")
            return {"catalog_path": None, "entries": [], "status": "skipped"}

        self._update_substep("Generating data catalog")

        catalog_path = self.config.paths.work_dir / "data_catalog.yml"

        catalog = generate_data_catalog(
            self.config,
            output_path=catalog_path,
            catalog_name=f"sfincs_{self.config.simulation.coastal_domain}",
        )

        self._log(f"Data catalog written to {catalog_path}")

        return {
            "catalog_path": str(catalog_path),
            "entries": [e.name for e in catalog.entries],
            "status": "completed",
        }
