"""YAML configuration schema and validation for coastal calibration workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
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

# Default path templates using interpolation syntax
DEFAULT_WORK_DIR_TEMPLATE = (
    "/ngen-test/coastal/${slurm.user}/"
    "schism_${simulation.coastal_domain}_${boundary.source}_${simulation.meteo_source}/"
    "schism_${simulation.start_date}"
)
DEFAULT_RAW_DOWNLOAD_DIR_TEMPLATE = (
    "/ngen-test/coastal/${slurm.user}/"
    "schism_${simulation.coastal_domain}_${boundary.source}_${simulation.meteo_source}/"
    "raw_data"
)


@dataclass
class SlurmConfig:
    """SLURM job configuration."""

    job_name: str = "coastal_calibration"
    nodes: int = 2
    ntasks_per_node: int = 18
    partition: str = DEFAULT_SLURM_PARTITION
    exclusive: bool = True
    time_limit: str | None = None
    account: str | None = None
    qos: str | None = None
    user: str | None = None

    @property
    def total_tasks(self) -> int:
        """Total number of MPI tasks."""
        return self.nodes * self.ntasks_per_node


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
        # Convert all path fields to Path objects first
        self.work_dir = Path(self.work_dir)
        self.parm_dir = Path(self.parm_dir)
        self.nfs_mount = Path(self.nfs_mount)
        self.singularity_image = Path(self.singularity_image)
        self.ngen_app_dir = Path(self.ngen_app_dir)
        if self.raw_download_dir:
            self.raw_download_dir = Path(self.raw_download_dir)
        if self.hot_start_file:
            self.hot_start_file = Path(self.hot_start_file)

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

    def schism_mesh(self, sim: SimulationConfig) -> Path:
        """SCHISM ESMF mesh file path for the given domain."""
        return self.parm_nwm / "coastal" / sim.coastal_domain / "hgrid.nc"

    def geogrid_file(self, sim: SimulationConfig) -> Path:
        """Geogrid file path for the given domain."""
        return self.parm_nwm / sim.inland_domain / sim.geo_grid


DEFAULT_SCHISM_BINARY = "pschism_wcoss2_NO_PARMETIS_TVD-VL.openmpi"


@dataclass
class MPIConfig:
    """MPI configuration."""

    nscribes: int = 2
    omp_num_threads: int = 2
    oversubscribe: bool = False
    schism_binary: str = DEFAULT_SCHISM_BINARY


@dataclass
class MonitoringConfig:
    """Workflow monitoring configuration."""

    log_level: LogLevel = "INFO"
    log_file: Path | None = None
    enable_progress_tracking: bool = True
    enable_timing: bool = True


@dataclass
class DownloadConfig:
    """Data download configuration."""

    enabled: bool = True
    skip_existing: bool = True
    timeout: int = 600
    raise_on_error: bool = True
    limit_per_host: int = 4


@dataclass
class SfincsModelConfig:
    """SFINCS model configuration for use within :class:`CoastalCalibConfig`.

    When ``model = "sfincs"`` in the main config, this section specifies the
    pre-built SFINCS model directory and model-specific settings (observation
    points, discharge sources, etc.).  Simulation timing, domain, download,
    and boundary settings are inherited from the parent
    :class:`CoastalCalibConfig`.

    Parameters
    ----------
    model_dir : Path
        Path to the directory containing the pre-built model files
        (``sfincs.inp``, ``sfincs.nc``, ``region.geojson``, etc.).
    model_root : Path, optional
        Output directory for the built model.  Defaults to
        ``{work_dir}/sfincs_model``.
    obs_points : list, optional
        Observation point specifications as list of dicts with
        ``x``, ``y``, ``name`` keys (coordinates in model CRS).
    obs_locations : Path, optional
        Path to a GeoJSON file with observation point locations.
    obs_merge : bool
        Whether to merge with pre-existing observation points.
    src_locations : Path, optional
        Path to a SFINCS ``.src`` or GeoJSON with discharge source point
        locations.
    src_merge : bool
        Whether to merge with pre-existing discharge source points.
    precip_dataset : str, optional
        Precipitation dataset name in the data catalog.
    docker_tag : str
        Tag for the ``deltares/sfincs-cpu`` Docker image.
    sif_path : Path, optional
        Path to a pre-pulled Singularity SIF file.
    """

    model_dir: Path
    model_root: Path | None = None
    obs_points: list[dict[str, Any]] = field(default_factory=list)
    obs_locations: Path | None = None
    obs_merge: bool = False
    src_locations: Path | None = None
    src_merge: bool = False
    precip_dataset: str | None = None
    docker_tag: str = "latest"
    sif_path: Path | None = None

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        if self.model_root is not None:
            self.model_root = Path(self.model_root)
        if self.obs_locations is not None:
            self.obs_locations = Path(self.obs_locations)
        if self.src_locations is not None:
            self.src_locations = Path(self.src_locations)
        if self.sif_path is not None:
            self.sif_path = Path(self.sif_path)


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


@dataclass
class CoastalCalibConfig:
    """Complete coastal calibration workflow configuration.

    Supports both SCHISM and SFINCS models.  Set ``model`` to ``"sfincs"``
    and provide the ``sfincs`` section to use the SFINCS pipeline.  The
    default is ``"schism"`` for backward compatibility.
    """

    slurm: SlurmConfig
    simulation: SimulationConfig
    boundary: BoundaryConfig
    paths: PathConfig
    model: ModelType = "schism"
    mpi: MPIConfig = field(default_factory=MPIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    sfincs: SfincsModelConfig | None = None
    _base_config: Path | None = field(default=None, repr=False)

    @classmethod
    def _from_dict(
        cls, data: dict[str, Any], base_config_path: Path | None = None
    ) -> CoastalCalibConfig:
        """Create config from dictionary."""
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

        mpi_data = data.get("mpi", {})
        mpi = MPIConfig(**mpi_data)

        monitoring_data = data.get("monitoring", {})
        if monitoring_data.get("log_file"):
            monitoring_data["log_file"] = Path(monitoring_data["log_file"])
        monitoring = MonitoringConfig(**monitoring_data)

        download_data = data.get("download", {})
        download = DownloadConfig(**download_data)

        model = data.get("model", "schism")

        sfincs = None
        sfincs_data = data.get("sfincs")
        if sfincs_data is not None:
            sfincs = SfincsModelConfig(**sfincs_data)

        return cls(
            slurm=slurm,
            simulation=simulation,
            boundary=boundary,
            paths=paths,
            model=model,
            mpi=mpi,
            monitoring=monitoring,
            download=download,
            sfincs=sfincs,
            _base_config=base_config_path,
        )

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> CoastalCalibConfig:
        """Load configuration from YAML file with optional inheritance.

        Supports variable interpolation using ${section.key} syntax.
        Variables are resolved from other config values, e.g.:
        - ${slurm.user} -> value of slurm.user
        - ${simulation.coastal_domain} -> value of simulation.coastal_domain
        - ${boundary.source} -> value of boundary.source

        Parameters
        ----------
        config_path : Path or str
            Path to YAML configuration file.

        Returns
        -------
        CoastalCalibConfig
            Loaded configuration.

        Examples
        --------
        >>> # In YAML:
        >>> # slurm:
        >>> #   user: john
        >>> # simulation:
        >>> #   coastal_domain: hawaii
        >>> # paths:
        >>> #   work_dir: /data/${slurm.user}/${simulation.coastal_domain}
        >>> # Results in work_dir = "/data/john/hawaii"

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
        result: dict[str, Any] = {
            "slurm": {
                "job_name": self.slurm.job_name,
                "nodes": self.slurm.nodes,
                "ntasks_per_node": self.slurm.ntasks_per_node,
                "partition": self.slurm.partition,
                "exclusive": self.slurm.exclusive,
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
                "stofs_file": str(self.boundary.stofs_file) if self.boundary.stofs_file else None,
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
            "mpi": {
                "nscribes": self.mpi.nscribes,
                "omp_num_threads": self.mpi.omp_num_threads,
                "oversubscribe": self.mpi.oversubscribe,
                "schism_binary": self.mpi.schism_binary,
            },
            "monitoring": {
                "log_level": self.monitoring.log_level,
                "log_file": str(self.monitoring.log_file) if self.monitoring.log_file else None,
                "enable_progress_tracking": self.monitoring.enable_progress_tracking,
                "enable_timing": self.monitoring.enable_timing,
            },
            "model": self.model,
            "download": {
                "enabled": self.download.enabled,
                "skip_existing": self.download.skip_existing,
                "timeout": self.download.timeout,
                "raise_on_error": self.download.raise_on_error,
                "limit_per_host": self.download.limit_per_host,
            },
        }
        if self.sfincs is not None:
            result["sfincs"] = {
                "model_dir": str(self.sfincs.model_dir),
                "model_root": str(self.sfincs.model_root) if self.sfincs.model_root else None,
                "obs_points": self.sfincs.obs_points,
                "obs_locations": (
                    str(self.sfincs.obs_locations) if self.sfincs.obs_locations else None
                ),
                "obs_merge": self.sfincs.obs_merge,
                "src_locations": (
                    str(self.sfincs.src_locations) if self.sfincs.src_locations else None
                ),
                "src_merge": self.sfincs.src_merge,
                "precip_dataset": self.sfincs.precip_dataset,
                "docker_tag": self.sfincs.docker_tag,
                "sif_path": str(self.sfincs.sif_path) if self.sfincs.sif_path else None,
            }
        return result

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

    def _validate_sfincs(self) -> list[str]:
        """Validate SFINCS-specific configuration."""
        errors: list[str] = []

        if self.sfincs is None:
            errors.append("sfincs section is required when model='sfincs'")
            return errors

        if not self.sfincs.model_dir.exists():
            errors.append(f"sfincs.model_dir not found: {self.sfincs.model_dir}")
        else:
            # Check for essential pre-built model files
            required = ["sfincs.inp"]
            errors.extend(
                f"Required file missing in sfincs.model_dir: {fname}"
                for fname in required
                if not (self.sfincs.model_dir / fname).exists()
            )

        if self.sfincs.obs_locations and not self.sfincs.obs_locations.exists():
            errors.append(f"sfincs.obs_locations not found: {self.sfincs.obs_locations}")

        if self.sfincs.src_locations and not self.sfincs.src_locations.exists():
            errors.append(f"sfincs.src_locations not found: {self.sfincs.src_locations}")

        if self.sfincs.sif_path and not self.sfincs.sif_path.exists():
            errors.append(f"sfincs.sif_path not found: {self.sfincs.sif_path}")

        return errors

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        from coastal_calibration.downloader import validate_date_ranges

        errors = []

        if self.simulation.duration_hours <= 0:
            errors.append("simulation.duration_hours must be positive")

        # SFINCS-specific validation — skip SCHISM-only checks
        if self.model == "sfincs":
            errors.extend(self._validate_sfincs())

            # Only validate date ranges when download is enabled — if the
            # user is supplying their own data files, the dates don't need
            # to match the remote data source availability windows.
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

        # --- SCHISM-specific validation below ---
        if not self.slurm.user:
            errors.append("slurm.user is required")

        if self.slurm.nodes < 1:
            errors.append("slurm.nodes must be at least 1")

        if self.slurm.ntasks_per_node < 1:
            errors.append("slurm.ntasks_per_node must be at least 1")

        if self.mpi.nscribes >= self.slurm.total_tasks:
            errors.append("mpi.nscribes must be less than total MPI tasks")

        errors.extend(self._validate_boundary_source())

        if self.paths.hot_start_file and not self.paths.hot_start_file.exists():
            errors.append(f"Hot start file not found: {self.paths.hot_start_file}")

        if not self.paths.raw_download_dir:
            errors.append("paths.raw_download_dir is required")

        if not self.paths.singularity_image.exists():
            errors.append(f"Singularity image not found: {self.paths.singularity_image}")

        # Validate simulation dates against available data source ranges
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
