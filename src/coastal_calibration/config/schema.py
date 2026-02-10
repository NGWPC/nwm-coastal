"""YAML configuration schema and validation for coastal calibration workflow."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml

from coastal_calibration.utils.time import parse_datetime as _parse_datetime

MeteoSource = Literal["nwm_retro", "nwm_ana"]
CoastalDomain = Literal["prvi", "hawaii", "atlgulf", "pacific"]
BoundarySource = Literal["tpxo", "stofs"]
ModelType = Literal["schism", "sfincs"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_SING_IMAGE_PATH = Path("/ngencerf-app/singularity/ngen-coastal.sif")
DEFAULT_PARM_DIR = Path("/ngen-test/coastal/ngwpc-coastal")
DEFAULT_NGEN_APP_DIR = Path("/ngen-app")
DEFAULT_NFS_MOUNT = Path("/ngen-test")
DEFAULT_CONDA_ENV_NAME = "ngen_forcing_coastal"
DEFAULT_NWM_VERSION = "v3.0.6"
DEFAULT_SLURM_PARTITION = "c5n-18xlarge"
# TPXO binary directory inside the Singularity container (not user-configurable)
DEFAULT_OTPS_DIR = Path("/ngen-app/OTPSnc")

# Default SCHISM binary name (used as default for SchismModelConfig.binary)
_DEFAULT_SCHISM_BINARY = "pschism_wcoss2_NO_PARMETIS_TVD-VL.openmpi"

# Default path templates using interpolation syntax.
# ${model} resolves to "schism" or "sfincs" based on the top-level ``model`` key.
DEFAULT_WORK_DIR_TEMPLATE = (
    "/ngen-test/coastal/${slurm.user}/"
    "${model}_${simulation.coastal_domain}_${boundary.source}_${simulation.meteo_source}/"
    "${model}_${simulation.start_date}"
)
DEFAULT_RAW_DOWNLOAD_DIR_TEMPLATE = (
    "/ngen-test/coastal/${slurm.user}/"
    "${model}_${simulation.coastal_domain}_${boundary.source}_${simulation.meteo_source}/"
    "raw_data"
)


@dataclass
class SlurmConfig:
    """SLURM job scheduling configuration.

    Contains only parameters related to job scheduling (partition, account,
    time limits).  Compute resources (nodes, tasks) are model-specific and
    live on the concrete :class:`ModelConfig` subclasses.
    """

    job_name: str = "coastal_calibration"
    partition: str = DEFAULT_SLURM_PARTITION
    time_limit: str | None = None
    account: str | None = None
    qos: str | None = None
    user: str | None = None


@dataclass
class SimulationConfig:
    """Simulation time and domain configuration."""

    start_date: datetime
    duration_hours: int
    coastal_domain: CoastalDomain
    meteo_source: MeteoSource
    timestep_seconds: int = 3600

    _INLAND_DOMAIN: ClassVar[dict[str, str]] = {
        "prvi": "domain_puertorico",
        "hawaii": "domain_hawaii",
        "atlgulf": "domain",
        "pacific": "domain",
    }
    _NWM_DOMAIN: ClassVar[dict[str, str]] = {
        "prvi": "prvi",
        "hawaii": "hawaii",
        "atlgulf": "conus",
        "pacific": "conus",
    }
    _GEO_GRID: ClassVar[dict[str, str]] = {
        "prvi": "geo_em_PRVI.nc",
        "hawaii": "geo_em_HI.nc",
        "atlgulf": "geo_em_CONUS.nc",
        "pacific": "geo_em_CONUS.nc",
    }

    @property
    def start_pdy(self) -> str:
        """Return start date as YYYYMMDD string."""
        return self.start_date.strftime("%Y%m%d")

    @property
    def start_cyc(self) -> str:
        """Return start cycle (hour) as HH string."""
        return self.start_date.strftime("%H")

    @property
    def inland_domain(self) -> str:
        """Inland domain directory name for this coastal domain."""
        return self._INLAND_DOMAIN[self.coastal_domain]

    @property
    def nwm_domain(self) -> str:
        """NWM domain identifier for this coastal domain."""
        return self._NWM_DOMAIN[self.coastal_domain]

    @property
    def geo_grid(self) -> str:
        """Geogrid filename for this coastal domain."""
        return self._GEO_GRID[self.coastal_domain]


@dataclass
class BoundaryConfig:
    """Boundary condition configuration."""

    source: BoundarySource = "tpxo"
    stofs_file: Path | None = None

    def __post_init__(self) -> None:
        if self.stofs_file is not None:
            self.stofs_file = Path(self.stofs_file).resolve()


@dataclass
class PathConfig:
    """Path configuration for data and executables."""

    METEO_SUBDIR: ClassVar[str] = "meteo"
    STREAMFLOW_SUBDIR: ClassVar[str] = "streamflow"
    HYDRO_SUBDIR: ClassVar[str] = "hydro"
    COASTAL_SUBDIR: ClassVar[str] = "coastal"

    work_dir: Path
    raw_download_dir: Path | None = None
    nfs_mount: Path = field(default_factory=lambda: DEFAULT_NFS_MOUNT)
    singularity_image: Path = field(default_factory=lambda: DEFAULT_SING_IMAGE_PATH)
    ngen_app_dir: Path = field(default_factory=lambda: DEFAULT_NGEN_APP_DIR)
    hot_start_file: Path | None = None
    conda_env_name: str = DEFAULT_CONDA_ENV_NAME
    parm_dir: Path = field(default_factory=lambda: DEFAULT_PARM_DIR)

    def __post_init__(self) -> None:
        # Resolve all path fields to absolute so that downstream paths
        # (model_root, sif_path, Singularity bind mounts, etc.) never
        # contain relative components that can double up when cwd changes.
        self.work_dir = Path(self.work_dir).resolve()
        self.parm_dir = Path(self.parm_dir).resolve()
        self.nfs_mount = Path(self.nfs_mount).resolve()
        self.singularity_image = Path(self.singularity_image).resolve()
        self.ngen_app_dir = Path(self.ngen_app_dir).resolve()
        if self.raw_download_dir:
            self.raw_download_dir = Path(self.raw_download_dir).resolve()
        if self.hot_start_file:
            self.hot_start_file = Path(self.hot_start_file).resolve()

    @property
    def otps_dir(self) -> Path:
        """TPXO binary directory (inside Singularity container, not configurable)."""
        return DEFAULT_OTPS_DIR

    @property
    def tpxo_data_dir(self) -> Path:
        """TPXO tidal atlas data directory."""
        return self.parm_dir / "TPXO10_atlas_v2_nc"

    @property
    def nwm_version_dir(self) -> Path:
        """NWM version directory (ush, exec live here)."""
        return self.ngen_app_dir / f"nwm.{DEFAULT_NWM_VERSION}"

    @property
    def ush_nwm(self) -> Path:
        """USH scripts directory."""
        return self.nwm_version_dir / "ush"

    @property
    def exec_nwm(self) -> Path:
        """Executables directory."""
        return self.nwm_version_dir / "exec"

    @property
    def parm_nwm(self) -> Path:
        """Parameter files directory."""
        return self.parm_dir / "parm"

    @property
    def conda_envs_path(self) -> Path:
        """Conda environments directory."""
        return self.nfs_mount / "ngen-app" / "conda" / "envs"

    @property
    def download_dir(self) -> Path:
        """Effective download directory (fallback to work_dir/downloads)."""
        return self.raw_download_dir or self.work_dir / "downloads"

    def meteo_dir(self, meteo_source: str) -> Path:
        """Directory for meteorological data."""
        return self.download_dir / self.METEO_SUBDIR / meteo_source

    def streamflow_dir(self, meteo_source: str) -> Path:
        """Directory for streamflow/hydro data."""
        if meteo_source == "nwm_retro":
            return self.download_dir / self.STREAMFLOW_SUBDIR / "nwm_retro"
        return self.download_dir / self.HYDRO_SUBDIR / "nwm"

    def coastal_dir(self, coastal_source: str) -> Path:
        """Directory for coastal boundary data."""
        return self.download_dir / self.COASTAL_SUBDIR / coastal_source

    def geogrid_file(self, sim: SimulationConfig) -> Path:
        """Geogrid file path for the given domain."""
        return self.parm_nwm / sim.inland_domain / sim.geo_grid


@dataclass
class MonitoringConfig:
    """Workflow monitoring configuration."""

    log_level: LogLevel = "INFO"
    log_file: Path | None = None
    enable_progress_tracking: bool = True
    enable_timing: bool = True

    def __post_init__(self) -> None:
        if self.log_file is not None:
            self.log_file = Path(self.log_file).resolve()


@dataclass
class DownloadConfig:
    """Data download configuration."""

    enabled: bool = True
    skip_existing: bool = True
    timeout: int = 600
    raise_on_error: bool = True
    limit_per_host: int = 4


# ---------------------------------------------------------------------------
# ModelConfig ABC and concrete implementations
# ---------------------------------------------------------------------------


class ModelConfig(ABC):
    """Abstract base class for model-specific configuration.

    Each concrete subclass owns its compute parameters, environment variable
    construction, stage ordering, validation, and SLURM script generation.
    This keeps model-specific concerns out of the shared configuration and
    makes adding new models straightforward: create a new subclass,
    implement the abstract methods, and register it in :data:`MODEL_REGISTRY`.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string (e.g. ``'schism'``, ``'sfincs'``)."""

    @abstractmethod
    def build_environment(self, env: dict[str, str], config: CoastalCalibConfig) -> dict[str, str]:
        """Add model-specific environment variables to *env* (mutating).

        Called by :meth:`WorkflowStage.build_environment` after shared
        variables have been populated.
        """

    @abstractmethod
    def validate(self, config: CoastalCalibConfig) -> list[str]:
        """Return model-specific validation errors."""

    @property
    @abstractmethod
    def stage_order(self) -> list[str]:
        """Ordered list of stage names for this model's pipeline."""

    @abstractmethod
    def create_stages(self, config: CoastalCalibConfig, monitor: Any) -> dict[str, Any]:
        """Construct and return the ``{name: stage}`` dictionary."""

    @abstractmethod
    def generate_job_script_lines(self, config: CoastalCalibConfig) -> list[str]:
        """Return ``#SBATCH`` directives specific to this model's compute needs."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize model-specific fields to a dictionary."""


@dataclass
class SchismModelConfig(ModelConfig):
    """SCHISM model configuration.

    Contains compute parameters (MPI layout, SCHISM binary) that were
    previously split across ``MPIConfig`` and ``SlurmConfig``.

    Parameters
    ----------
    nodes : int
        Number of SLURM nodes.
    ntasks_per_node : int
        MPI tasks per node.
    exclusive : bool
        Request exclusive node access.
    nscribes : int
        Number of SCHISM scribe processes.
    omp_num_threads : int
        OpenMP threads per MPI rank.
    oversubscribe : bool
        Allow MPI oversubscription.
    binary : str
        SCHISM executable name.
    """

    nodes: int = 2
    ntasks_per_node: int = 18
    exclusive: bool = True
    nscribes: int = 2
    omp_num_threads: int = 2
    oversubscribe: bool = False
    binary: str = _DEFAULT_SCHISM_BINARY
    include_noaa_gages: bool = False

    @property
    def model_name(self) -> str:  # noqa: D102
        return "schism"

    @property
    def total_tasks(self) -> int:
        """Total number of MPI tasks (nodes * ntasks_per_node)."""
        return self.nodes * self.ntasks_per_node

    def schism_mesh(self, sim: SimulationConfig, paths: PathConfig) -> Path:
        """SCHISM ESMF mesh file path for the given domain."""
        return paths.parm_nwm / "coastal" / sim.coastal_domain / "hgrid.nc"

    @property
    def stage_order(self) -> list[str]:  # noqa: D102
        return [
            "download",
            "pre_forcing",
            "nwm_forcing",
            "post_forcing",
            "update_params",
            "boundary_conditions",
            "pre_schism",
            "schism_run",
            "post_schism",
        ]

    def build_environment(  # noqa: D102
        self, env: dict[str, str], config: CoastalCalibConfig
    ) -> dict[str, str]:
        env["NODES"] = str(self.nodes)
        env["NCORES"] = str(self.ntasks_per_node)
        env["NPROCS"] = str(self.total_tasks)
        env["NSCRIBES"] = str(self.nscribes)
        env["OMP_NUM_THREADS"] = str(self.omp_num_threads)
        env["SCHISM_ESMFMESH"] = str(self.schism_mesh(config.simulation, config.paths))

        # SCHISM date variables
        sim = config.simulation
        start_dt = datetime.strptime(f"{sim.start_pdy} {sim.start_cyc}", "%Y%m%d %H").replace(
            tzinfo=UTC
        )
        length_hrs = int(sim.duration_hours)
        pdycyc = f"{sim.start_pdy}{sim.start_cyc}"

        if length_hrs <= 0:
            schism_begin_dt = start_dt + timedelta(hours=length_hrs)
            env["SCHISM_BEGIN_DATE"] = schism_begin_dt.strftime("%Y%m%d%H00")
            env["SCHISM_END_DATE"] = f"{pdycyc}00"
        else:
            env["SCHISM_BEGIN_DATE"] = f"{pdycyc}00"
            schism_end_dt = start_dt + timedelta(hours=length_hrs)
            env["SCHISM_END_DATE"] = schism_end_dt.strftime("%Y%m%d%H00")

        return env

    def validate(self, config: CoastalCalibConfig) -> list[str]:  # noqa: D102
        errors: list[str] = []

        if not config.slurm.user:
            errors.append("slurm.user is required")

        if self.nodes < 1:
            errors.append("model_config.nodes must be at least 1")

        if self.ntasks_per_node < 1:
            errors.append("model_config.ntasks_per_node must be at least 1")

        if self.nscribes >= self.total_tasks:
            errors.append("model_config.nscribes must be less than total MPI tasks")

        if config.paths.hot_start_file and not config.paths.hot_start_file.exists():
            errors.append(f"Hot start file not found: {config.paths.hot_start_file}")

        if not config.paths.raw_download_dir:
            errors.append("paths.raw_download_dir is required")

        if not config.paths.singularity_image.exists():
            errors.append(f"Singularity image not found: {config.paths.singularity_image}")

        return errors

    def create_stages(  # noqa: D102
        self, config: CoastalCalibConfig, monitor: Any
    ) -> dict[str, Any]:
        from coastal_calibration.stages.boundary import (
            BoundaryConditionStage,
            UpdateParamsStage,
        )
        from coastal_calibration.stages.download import DownloadStage
        from coastal_calibration.stages.forcing import (
            NWMForcingStage,
            PostForcingStage,
            PreForcingStage,
        )
        from coastal_calibration.stages.schism import (
            PostSCHISMStage,
            PreSCHISMStage,
            SCHISMRunStage,
        )

        return {
            "download": DownloadStage(config, monitor),
            "pre_forcing": PreForcingStage(config, monitor),
            "nwm_forcing": NWMForcingStage(config, monitor),
            "post_forcing": PostForcingStage(config, monitor),
            "update_params": UpdateParamsStage(config, monitor),
            "boundary_conditions": BoundaryConditionStage(config, monitor),
            "pre_schism": PreSCHISMStage(config, monitor),
            "schism_run": SCHISMRunStage(config, monitor),
            "post_schism": PostSCHISMStage(config, monitor),
        }

    def generate_job_script_lines(  # noqa: D102
        self, config: CoastalCalibConfig
    ) -> list[str]:
        lines = [
            f"#SBATCH -N {self.nodes}",
            f"#SBATCH --ntasks-per-node={self.ntasks_per_node}",
        ]
        if self.exclusive:
            lines.append("#SBATCH --exclusive")
        return lines

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            "nodes": self.nodes,
            "ntasks_per_node": self.ntasks_per_node,
            "exclusive": self.exclusive,
            "nscribes": self.nscribes,
            "omp_num_threads": self.omp_num_threads,
            "oversubscribe": self.oversubscribe,
            "binary": self.binary,
            "include_noaa_gages": self.include_noaa_gages,
        }


@dataclass
class SfincsModelConfig(ModelConfig):
    """SFINCS model configuration.

    SFINCS runs on a single node using OpenMP (all available cores).
    There is no MPI or multi-node support.

    Parameters
    ----------
    prebuilt_dir : Path
        Path to the directory containing the pre-built model files
        (``sfincs.inp``, ``sfincs.nc``, ``region.geojson``, etc.).
    model_root : Path, optional
        Output directory for the built model.  Defaults to
        ``{work_dir}/sfincs_model``.
    include_noaa_gages : bool
        When True, automatically query NOAA CO-OPS for water level
        stations within the model domain and add them as observation
        points.  Requires the ``plot`` optional dependencies.
    observation_points : list, optional
        Observation point specifications as list of dicts with
        ``x``, ``y``, ``name`` keys (coordinates in model CRS).
    observation_locations_file : Path, optional
        Path to a GeoJSON file with observation point locations.
    merge_observations : bool
        Whether to merge with pre-existing observation points.
    discharge_locations_file : Path, optional
        Path to a SFINCS ``.src`` or GeoJSON with discharge source point
        locations.
    merge_discharge : bool
        Whether to merge with pre-existing discharge source points.
    include_precip : bool
        When True, add precipitation forcing from the meteorological
        data catalog entry (derived from ``simulation.meteo_source``).
    include_wind : bool
        When True, add spatially-varying wind forcing (``wind10_u``,
        ``wind10_v``) from the meteorological data catalog entry.
    include_pressure : bool
        When True, add spatially-varying atmospheric pressure forcing
        (``press_msl``) and enable barometric correction (``baro=1``).
    container_tag : str
        Tag for the ``deltares/sfincs-cpu`` Docker/Singularity image.
    container_image : Path, optional
        Path to a pre-pulled Singularity SIF file.
    omp_num_threads : int
        Number of OpenMP threads.  Defaults to the number of physical CPU
        cores on the current machine (see :func:`~coastal_calibration.utils.system.get_cpu_count`).
        On HPC nodes this auto-detects correctly; on a local laptop it
        avoids over-subscribing the system.
    """

    prebuilt_dir: Path
    model_root: Path | None = None
    include_noaa_gages: bool = False
    observation_points: list[dict[str, Any]] = field(default_factory=list)
    observation_locations_file: Path | None = None
    merge_observations: bool = False
    discharge_locations_file: Path | None = None
    merge_discharge: bool = False
    include_precip: bool = False
    include_wind: bool = False
    include_pressure: bool = False
    container_tag: str = "latest"
    container_image: Path | None = None
    omp_num_threads: int = field(default=0)

    def __post_init__(self) -> None:
        self.prebuilt_dir = Path(self.prebuilt_dir).resolve()
        if self.model_root is not None:
            self.model_root = Path(self.model_root).resolve()
        if self.observation_locations_file is not None:
            self.observation_locations_file = Path(self.observation_locations_file).resolve()
        if self.discharge_locations_file is not None:
            self.discharge_locations_file = Path(self.discharge_locations_file).resolve()
        if self.container_image is not None:
            self.container_image = Path(self.container_image).resolve()
        if self.omp_num_threads <= 0:
            from coastal_calibration.utils.system import get_cpu_count

            self.omp_num_threads = get_cpu_count()

    @property
    def model_name(self) -> str:  # noqa: D102
        return "sfincs"

    @property
    def stage_order(self) -> list[str]:  # noqa: D102
        return [
            "download",
            "sfincs_symlinks",
            "sfincs_data_catalog",
            "sfincs_init",
            "sfincs_timing",
            "sfincs_forcing",
            "sfincs_obs",
            "sfincs_discharge",
            "sfincs_precip",
            "sfincs_wind",
            "sfincs_pressure",
            "sfincs_write",
            "sfincs_run",
            "sfincs_plot",
        ]

    def build_environment(  # noqa: D102
        self, env: dict[str, str], config: CoastalCalibConfig
    ) -> dict[str, str]:
        env["OMP_NUM_THREADS"] = str(self.omp_num_threads)
        return env

    def validate(self, config: CoastalCalibConfig) -> list[str]:  # noqa: D102
        errors: list[str] = []

        if not self.prebuilt_dir.exists():
            errors.append(f"model_config.prebuilt_dir not found: {self.prebuilt_dir}")
        else:
            required = ["sfincs.inp"]
            errors.extend(
                f"Required file missing in model_config.prebuilt_dir: {fname}"
                for fname in required
                if not (self.prebuilt_dir / fname).exists()
            )

        if self.observation_locations_file and not self.observation_locations_file.exists():
            errors.append(
                "model_config.observation_locations_file not found: "
                f"{self.observation_locations_file}"
            )

        if self.discharge_locations_file and not self.discharge_locations_file.exists():
            errors.append(
                f"model_config.discharge_locations_file not found: {self.discharge_locations_file}"
            )

        if self.container_image and not self.container_image.exists():
            errors.append(f"model_config.container_image not found: {self.container_image}")

        return errors

    def create_stages(  # noqa: D102
        self, config: CoastalCalibConfig, monitor: Any
    ) -> dict[str, Any]:
        from coastal_calibration.stages.download import DownloadStage
        from coastal_calibration.stages.sfincs_build import (
            SfincsDataCatalogStage,
            SfincsDischargeStage,
            SfincsForcingStage,
            SfincsInitStage,
            SfincsObservationPointsStage,
            SfincsPlotStage,
            SfincsPrecipitationStage,
            SfincsPressureStage,
            SfincsRunStage,
            SfincsSymlinksStage,
            SfincsTimingStage,
            SfincsWindStage,
            SfincsWriteStage,
        )

        return {
            "download": DownloadStage(config, monitor),
            "sfincs_symlinks": SfincsSymlinksStage(config, monitor),
            "sfincs_data_catalog": SfincsDataCatalogStage(config, monitor),
            "sfincs_init": SfincsInitStage(config, monitor),
            "sfincs_timing": SfincsTimingStage(config, monitor),
            "sfincs_forcing": SfincsForcingStage(config, monitor),
            "sfincs_obs": SfincsObservationPointsStage(config, monitor),
            "sfincs_discharge": SfincsDischargeStage(config, monitor),
            "sfincs_precip": SfincsPrecipitationStage(config, monitor),
            "sfincs_wind": SfincsWindStage(config, monitor),
            "sfincs_pressure": SfincsPressureStage(config, monitor),
            "sfincs_write": SfincsWriteStage(config, monitor),
            "sfincs_run": SfincsRunStage(config, monitor),
            "sfincs_plot": SfincsPlotStage(config, monitor),
        }

    def generate_job_script_lines(  # noqa: D102
        self, config: CoastalCalibConfig
    ) -> list[str]:
        return [
            "#SBATCH -N 1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={self.omp_num_threads}",
            "#SBATCH --exclusive",
        ]

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            "prebuilt_dir": str(self.prebuilt_dir),
            "model_root": str(self.model_root) if self.model_root else None,
            "include_noaa_gages": self.include_noaa_gages,
            "observation_points": self.observation_points,
            "observation_locations_file": (
                str(self.observation_locations_file) if self.observation_locations_file else None
            ),
            "merge_observations": self.merge_observations,
            "discharge_locations_file": (
                str(self.discharge_locations_file) if self.discharge_locations_file else None
            ),
            "merge_discharge": self.merge_discharge,
            "include_precip": self.include_precip,
            "include_wind": self.include_wind,
            "include_pressure": self.include_pressure,
            "container_tag": self.container_tag,
            "container_image": (str(self.container_image) if self.container_image else None),
            "omp_num_threads": self.omp_num_threads,
        }


MODEL_REGISTRY: dict[str, type[ModelConfig]] = {
    "schism": SchismModelConfig,
    "sfincs": SfincsModelConfig,
}


# ---------------------------------------------------------------------------
# Interpolation utilities
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _interpolate_value(value: Any, context: dict[str, Any]) -> Any:
    """Interpolate ${section.key} variables in a string value.

    Parameters
    ----------
    value : Any
        The value to interpolate. If not a string, returns unchanged.
    context : dict
        Flat dictionary of available variables (e.g., {"slurm.user": "john"}).

    Returns
    -------
    Any
        The interpolated value.

    Examples
    --------
    >>> ctx = {"slurm.user": "john", "simulation.coastal_domain": "hawaii"}
    >>> _interpolate_value("/data/${slurm.user}/${simulation.coastal_domain}", ctx)
    '/data/john/hawaii'
    """
    if not isinstance(value, str):
        return value

    import re

    pattern = re.compile(r"\$\{([^}]+)\}")

    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in context:
            return str(context[key])
        return match.group(0)  # Leave unresolved variables as-is

    return pattern.sub(replacer, value)


def _build_interpolation_context(data: dict[str, Any]) -> dict[str, Any]:
    """Build a flat context dictionary for variable interpolation.

    Parameters
    ----------
    data : dict
        The raw configuration dictionary.

    Returns
    -------
    dict
        Flat dictionary with keys like "slurm.user", "simulation.coastal_domain".
    """
    context: dict[str, Any] = {}
    for section, values in data.items():
        if isinstance(values, dict):
            for key, val in values.items():
                if val is not None and not isinstance(val, dict):
                    context[f"{section}.{key}"] = val
    # Top-level scalar keys (e.g., "model") are available without a section prefix.
    if "model" in data:
        context["model"] = data["model"]
    return context


def _interpolate_config(data: dict[str, Any]) -> dict[str, Any]:
    """Interpolate all ${section.key} variables in the configuration.

    Parameters
    ----------
    data : dict
        The raw configuration dictionary.

    Returns
    -------
    dict
        Configuration with all variables interpolated.
    """
    context = _build_interpolation_context(data)
    result: dict[str, Any] = {}

    for section, values in data.items():
        if isinstance(values, dict):
            result[section] = {}
            for key, val in values.items():
                result[section][key] = _interpolate_value(val, context)
        else:
            result[section] = _interpolate_value(values, context)

    return result


# ---------------------------------------------------------------------------
# Backward compatibility migration helpers
# ---------------------------------------------------------------------------

# Maps old SfincsModelConfig field names to new names.
_SFINCS_FIELD_MIGRATION: dict[str, str] = {
    "model_dir": "prebuilt_dir",
    "docker_tag": "container_tag",
    "sif_path": "container_image",
    "obs_points": "observation_points",
    "obs_locations": "observation_locations_file",
    "obs_merge": "merge_observations",
    "src_locations": "discharge_locations_file",
    "src_merge": "merge_discharge",
}

# Maps old SchismModelConfig field names to new names.
_SCHISM_FIELD_MIGRATION: dict[str, str] = {
    "schism_binary": "binary",
}


def _migrate_model_config_data(
    model_type: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Migrate old-style config keys into the new ``model_config`` dict.

    Handles two legacy patterns:

    1. **SCHISM** — fields were split across top-level ``mpi`` and ``slurm``
       sections.  We pull compute fields (``nodes``, ``ntasks_per_node``,
       ``exclusive``) from ``slurm`` and all fields from ``mpi`` into the
       model config dict.
    2. **SFINCS** — the model config lived under a top-level ``sfincs`` key
       and used different field names.

    In both cases the ``model_config`` key in *data* (if already present)
    takes precedence over migrated values.
    """
    model_config_data: dict[str, Any] = {}

    if model_type == "schism":
        # Migrate old mpi section
        mpi_data = data.pop("mpi", None)
        if mpi_data is not None:
            for old_key, new_key in _SCHISM_FIELD_MIGRATION.items():
                if old_key in mpi_data:
                    model_config_data[new_key] = mpi_data.pop(old_key)
            # Remaining mpi fields map directly (nscribes, omp_num_threads, etc.)
            model_config_data.update(mpi_data)

        # Migrate compute fields from old slurm section
        slurm_data = data.get("slurm", {})
        for compute_key in ("nodes", "ntasks_per_node", "exclusive"):
            if compute_key in slurm_data:
                model_config_data.setdefault(compute_key, slurm_data.pop(compute_key))

    elif model_type == "sfincs":
        # Migrate old top-level sfincs section
        sfincs_data = data.pop("sfincs", None)
        if sfincs_data is not None:
            for old_key, new_key in _SFINCS_FIELD_MIGRATION.items():
                if old_key in sfincs_data:
                    model_config_data[new_key] = sfincs_data.pop(old_key)
            model_config_data.update(sfincs_data)

    # Explicit model_config in YAML takes precedence
    explicit = data.pop("model_config", {}) or {}
    model_config_data.update(explicit)

    return model_config_data


# ---------------------------------------------------------------------------
# Main configuration class
# ---------------------------------------------------------------------------


@dataclass
class CoastalCalibConfig:
    """Complete coastal calibration workflow configuration.

    Supports both SCHISM and SFINCS models via the polymorphic
    :attr:`model_config` field.  The concrete type is selected by the
    ``model`` key in the YAML file and resolved through
    :data:`MODEL_REGISTRY`.
    """

    slurm: SlurmConfig
    simulation: SimulationConfig
    boundary: BoundaryConfig
    paths: PathConfig
    model_config: ModelConfig
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    _base_config: Path | None = field(default=None, repr=False)

    @property
    def model(self) -> str:
        """Model identifier string (convenience accessor)."""
        return self.model_config.model_name

    @classmethod
    def _from_dict(
        cls, data: dict[str, Any], base_config_path: Path | None = None
    ) -> CoastalCalibConfig:
        """Create config from dictionary."""
        if "model" not in data:
            raise ValueError("'model' is required (e.g., model: schism or model: sfincs)")
        model_type: str = data["model"]

        # Migrate legacy keys into model_config
        model_config_data = _migrate_model_config_data(model_type, data)

        slurm_data = data.get("slurm", {})
        slurm = SlurmConfig(**slurm_data)

        sim_data = data.get("simulation", {})
        if "start_date" in sim_data:
            sim_data["start_date"] = _parse_datetime(sim_data["start_date"])
        simulation = SimulationConfig(**sim_data)

        boundary_data = data.get("boundary", {})
        if boundary_data.get("stofs_file"):
            boundary_data["stofs_file"] = Path(boundary_data["stofs_file"])
        boundary = BoundaryConfig(**boundary_data)

        paths_data = data.get("paths", {})
        # Remove deprecated otps_dir - it's now hardcoded inside the container
        paths_data.pop("otps_dir", None)
        paths = PathConfig(**paths_data)

        monitoring_data = data.get("monitoring", {})
        if monitoring_data.get("log_file"):
            monitoring_data["log_file"] = Path(monitoring_data["log_file"])
        monitoring = MonitoringConfig(**monitoring_data)

        download_data = data.get("download", {})
        download = DownloadConfig(**download_data)

        if model_type not in MODEL_REGISTRY:
            msg = (
                f"Unknown model type: {model_type!r}. Supported models: {', '.join(MODEL_REGISTRY)}"
            )
            raise ValueError(msg)

        model_cls = MODEL_REGISTRY[model_type]
        model_config = model_cls(**model_config_data)

        return cls(
            slurm=slurm,
            simulation=simulation,
            boundary=boundary,
            paths=paths,
            model_config=model_config,
            monitoring=monitoring,
            download=download,
            _base_config=base_config_path,
        )

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> CoastalCalibConfig:
        """Load configuration from YAML file with optional inheritance.

        Supports variable interpolation using ${section.key} syntax.
        Variables are resolved from other config values, e.g.:

        - ``${slurm.user}`` -> value of ``slurm.user``
        - ``${simulation.coastal_domain}`` -> value of ``simulation.coastal_domain``
        - ``${model}`` -> the model type string (``"schism"`` or ``"sfincs"``)

        Parameters
        ----------
        config_path : Path or str
            Path to YAML configuration file.

        Returns
        -------
        CoastalCalibConfig
            Loaded configuration.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        yaml.YAMLError
            If the YAML file is malformed.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            data = yaml.safe_load(config_path.read_text())
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}") from e

        if data is None:
            raise ValueError(f"Configuration file is empty: {config_path}")

        base_config = None
        if "_base" in data:
            base_path = Path(data.pop("_base"))
            if not base_path.is_absolute():
                base_path = config_path.parent / base_path
            base_config = cls.from_yaml(base_path)
            data = _deep_merge(base_config.to_dict(), data)

        # Ensure model key has a default before interpolation
        data.setdefault("model", "schism")

        # Inject default path templates if not provided (before interpolation)
        if "paths" not in data:
            data["paths"] = {}
        if "work_dir" not in data["paths"]:
            data["paths"]["work_dir"] = DEFAULT_WORK_DIR_TEMPLATE
        if "raw_download_dir" not in data["paths"]:
            data["paths"]["raw_download_dir"] = DEFAULT_RAW_DOWNLOAD_DIR_TEMPLATE

        # Interpolate variables after merging
        data = _interpolate_config(data)

        return cls._from_dict(data, base_config_path=config_path if base_config else None)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model,
            "slurm": {
                "job_name": self.slurm.job_name,
                "partition": self.slurm.partition,
                "time_limit": self.slurm.time_limit,
                "account": self.slurm.account,
                "qos": self.slurm.qos,
                "user": self.slurm.user,
            },
            "simulation": {
                "start_date": self.simulation.start_date.isoformat(),
                "duration_hours": self.simulation.duration_hours,
                "coastal_domain": self.simulation.coastal_domain,
                "meteo_source": self.simulation.meteo_source,
                "timestep_seconds": self.simulation.timestep_seconds,
            },
            "boundary": {
                "source": self.boundary.source,
                "stofs_file": (str(self.boundary.stofs_file) if self.boundary.stofs_file else None),
            },
            "paths": {
                "work_dir": str(self.paths.work_dir),
                "raw_download_dir": (
                    str(self.paths.raw_download_dir) if self.paths.raw_download_dir else None
                ),
                "nfs_mount": str(self.paths.nfs_mount),
                "singularity_image": str(self.paths.singularity_image),
                "ngen_app_dir": str(self.paths.ngen_app_dir),
                "hot_start_file": (
                    str(self.paths.hot_start_file) if self.paths.hot_start_file else None
                ),
                "conda_env_name": self.paths.conda_env_name,
                "parm_dir": str(self.paths.parm_dir),
            },
            "model_config": self.model_config.to_dict(),
            "monitoring": {
                "log_level": self.monitoring.log_level,
                "log_file": (str(self.monitoring.log_file) if self.monitoring.log_file else None),
                "enable_progress_tracking": self.monitoring.enable_progress_tracking,
                "enable_timing": self.monitoring.enable_timing,
            },
            "download": {
                "enabled": self.download.enabled,
                "skip_existing": self.download.skip_existing,
                "timeout": self.download.timeout,
                "raise_on_error": self.download.raise_on_error,
                "limit_per_host": self.download.limit_per_host,
            },
        }

    def to_yaml(self, path: Path | str) -> None:
        """Write configuration to YAML file.

        Parameters
        ----------
        path : Path or str
            Path to YAML output file. Parent directories will be created
            if they don't exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False))

    def _validate_boundary_source(self) -> list[str]:
        """Validate boundary source configuration."""
        errors = []

        if self.boundary.source == "stofs":
            if not self.boundary.stofs_file and not self.download.enabled:
                errors.append(
                    "boundary.stofs_file required when using STOFS source and download is disabled"
                )
            elif (
                self.boundary.stofs_file
                and not self.boundary.stofs_file.exists()
                and not self.download.enabled
            ):
                errors.append(f"STOFS file not found: {self.boundary.stofs_file}")

        # TPXO binary (predict_tide) is inside the Singularity container at
        # DEFAULT_OTPS_DIR, so we only validate the data directory on the host
        elif self.boundary.source == "tpxo" and not self.paths.tpxo_data_dir.exists():
            errors.append(
                f"TPXO data directory not found: {self.paths.tpxo_data_dir}. "
                "TPXO tidal atlas data requires local installation."
            )

        return errors

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        from coastal_calibration.downloader import validate_date_ranges

        errors: list[str] = []

        if self.simulation.duration_hours <= 0:
            errors.append("simulation.duration_hours must be positive")

        # Model-specific validation
        errors.extend(self.model_config.validate(self))

        # Shared boundary validation
        errors.extend(self._validate_boundary_source())

        # Date range validation
        if self.download.enabled:
            sim = self.simulation
            start_time = sim.start_date
            end_time = start_time + timedelta(hours=sim.duration_hours)
            date_errors = validate_date_ranges(
                start_time,
                end_time,
                sim.meteo_source,
                self.boundary.source,
                sim.coastal_domain,
            )
            errors.extend(date_errors)

        return errors
