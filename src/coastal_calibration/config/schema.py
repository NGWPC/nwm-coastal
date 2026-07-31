"""YAML configuration schema and validation for coastal calibration workflow."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
import yaml

# ``nwm_retro`` / ``nwm_ana`` are downloaded from public archives.
# ``ngen_forecast`` means the meteorological forcing is produced ahead of time
# by the ngen forecast forcing engine and left on disk as a single
# multi-timestep file (see ``paths.forecast_meteo_file``); nothing is
# downloaded for it.
MeteoSource = Literal["nwm_retro", "nwm_ana", "ngen_forecast"]
CoastalDomain = Literal["prvi", "hawaii", "atlgulf", "pacific", "alaska"]
# ``harmonic`` predicts boundary elevations from harmonic constituents via
# pyTMD against ``tidal_atlas_dir`` (TPXO, FES, GOT, EOT, ...). ``tpxo``
# is accepted for backward compatibility with older config files and is
# normalized to ``harmonic`` in :meth:`BoundaryConfig.__post_init__`.
BoundarySource = Literal["harmonic", "tpxo", "stofs"]
ModelType = Literal["schism", "sfincs"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


# Projection of the NWM CONUS meteorological (LDASIN) grid, shared by the
# coastal domains that are carved out of it.
NWM_CONUS_METEO_CRS = (
    "+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs"
)


@dataclass
class SimulationConfig:
    """Simulation time and domain configuration.

    ``start_date`` is normalized to **naive UTC** in ``__post_init__``
    so the rest of the pipeline can compare and serialize it without
    crossing the naive/aware boundary. Tz-aware values are converted to
    UTC then stripped; tz-naive values are passed through (assumed UTC
    by the project's data contract — NWM/STOFS are published on UTC
    days).
    """

    start_date: datetime
    duration_hours: int
    coastal_domain: CoastalDomain
    meteo_source: MeteoSource
    # Model integration timestep in seconds: SCHISM's ``dt`` and the
    # tick that ``nspool``/``ihfskip``/``nhot_write`` are counted in.
    # The default (200) matches the Pacific/Hawaii forecast templates
    # and is a stable choice for coastal mesh resolutions of ~100-1000 m.
    # Note: the tidal boundary (``make_tidal_boundary``) writes
    # ``elev2D.th.nc`` at a separate cadence (default 3600 s); SCHISM
    # interpolates from that file to its own ``dt`` at runtime.
    timestep_seconds: int = 200

    _INLAND_DOMAIN: ClassVar[dict[str, str]] = {
        "prvi": "domain_puertorico",
        "hawaii": "domain_hawaii",
        "atlgulf": "domain",
        "pacific": "domain",
        "alaska": "domain_alaska",
    }
    _NWM_DOMAIN: ClassVar[dict[str, str]] = {
        "prvi": "prvi",
        "hawaii": "hawaii",
        "atlgulf": "conus",
        "pacific": "conus",
        "alaska": "alaska",
    }
    _GEO_GRID: ClassVar[dict[str, str]] = {
        "prvi": "geo_em_PRVI.nc",
        "hawaii": "geo_em_HI.nc",
        "atlgulf": "geo_em_CONUS.nc",
        "pacific": "geo_em_CONUS.nc",
        "alaska": "geo_em_AK.nc",
    }
    # Every NWM domain projects its LDASIN forcing differently: CONUS,
    # Hawaii, and PRVI are Lambert Conformal Conic about different origins,
    # while Alaska is polar stereographic.  These mirror the ``crs``
    # variable carried inside the LDASIN files themselves.
    _METEO_CRS: ClassVar[dict[str, str]] = {
        "prvi": "+proj=lcc +lat_0=18.1 +lon_0=-65.91 +lat_1=18.1 +lat_2=18.1 +x_0=0 +y_0=0 +R=6370000 +nadgrids=@null +units=m +no_defs",
        "hawaii": "+proj=lcc +lat_0=20.6 +lon_0=-157.42 +lat_1=10 +lat_2=30 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs",
        "atlgulf": NWM_CONUS_METEO_CRS,
        "pacific": NWM_CONUS_METEO_CRS,
        "alaska": "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-135 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs",
    }

    def __post_init__(self) -> None:
        from coastal_calibration.utils import to_naive_utc

        self.start_date = to_naive_utc(self.start_date)

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

    @classmethod
    def nwm_domain_for(cls, coastal_domain: str) -> str:
        """NWM domain identifier for an arbitrary coastal domain name.

        Same mapping as :attr:`nwm_domain`, but callable without a
        configured simulation and lenient about names outside
        :data:`CoastalDomain` (``"conus"`` is passed straight through, as
        the downloader accepts it).  Used for naming the download cache,
        where the NWM domain is the right key: ``atlgulf`` and ``pacific``
        pull byte-identical CONUS forcing and should share one copy.
        """
        return cls._NWM_DOMAIN.get(coastal_domain, coastal_domain)

    @property
    def geo_grid(self) -> str:
        """Geogrid filename for this coastal domain."""
        return self._GEO_GRID[self.coastal_domain]

    @property
    def meteo_crs(self) -> str:
        """PROJ string for this domain's NWM meteorological forcing grid."""
        return self._METEO_CRS[self.coastal_domain]


@dataclass
class BoundaryConfig:
    """Boundary condition configuration.

    Parameters
    ----------
    source : {"harmonic", "stofs"}
        Boundary forcing source. ``harmonic`` predicts tides locally
        via pyTMD against the atlas at
        :attr:`PathConfig.tidal_atlas_dir`; ``stofs`` regrids the NOAA
        STOFS product (and falls back to ``harmonic`` past the STOFS
        180 h window when the simulation runs longer).
        ``"tpxo"`` is accepted as a deprecated alias for ``"harmonic"``
        and is normalized at construction time.
    stofs_file : Path, optional
        STOFS NetCDF (only used when ``source == "stofs"``).
    tidal_model : str
        pyTMD model identifier (see ``pyTMD.io.load_database()``).
        Defaults to TPXO10-atlas-v2 in netcdf form. Set to e.g.
        ``"FES2014"`` or ``"GOT4.10"`` to predict against another atlas
        without code changes — only the files under
        :attr:`PathConfig.tidal_atlas_dir` change.
    """

    source: BoundarySource = "harmonic"
    stofs_file: Path | None = None
    tidal_model: str = "TPXO10-atlas-v2-nc"

    def __post_init__(self) -> None:
        # Normalize the deprecated "tpxo" alias upfront so every consumer
        # downstream only ever has to check for "harmonic".
        if self.source == "tpxo":
            self.source = "harmonic"
        if self.stofs_file is not None:
            self.stofs_file = Path(self.stofs_file).expanduser().resolve()


@dataclass
class PathConfig:
    """Path configuration for data and executables.

    Only ``work_dir`` is required. All other fields are optional and
    only needed by specific workflow stages.
    """

    METEO_SUBDIR: ClassVar[str] = "meteo"
    HYDRO_SUBDIR: ClassVar[str] = "hydro"
    COASTAL_SUBDIR: ClassVar[str] = "coastal"

    work_dir: Path
    raw_download_dir: Path | None = None
    hot_start_file: Path | None = None
    # Path to the meteorological forcing file produced by the ngen forecast
    # forcing engine (a single multi-timestep netCDF on the WRF-Hydro
    # geogrid, e.g. ``Hawaii_202509150000.nc``). Required when
    # ``simulation.meteo_source == "ngen_forecast"``; ignored otherwise.
    forecast_meteo_file: Path | None = None
    # Path to the t-route output netCDF (``troute_output_*.nc``) produced by
    # the ngen forecast's routing step, keyed by NextGen hydrofabric
    # ``feature_id``. Used for river discharge when
    # ``simulation.meteo_source == "ngen_forecast"``; only needed when the
    # model is configured with a discharge crosswalk. Ignored otherwise.
    troute_file: Path | None = None
    # Legacy create-workflow fields — not used by the run workflow.
    parm_dir: Path | None = None
    nwm_dir: Path | None = None
    # Directory containing the pyTMD tidal atlas files for
    # :attr:`BoundaryConfig.tidal_model`. The path is read directly:
    # users point at whatever folder holds the atlas constituent + grid
    # netCDFs, regardless of the subdirectory convention pyTMD's
    # bundled database uses internally.
    tidal_atlas_dir: Path | None = None

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir).expanduser().resolve()
        if self.raw_download_dir:
            self.raw_download_dir = Path(self.raw_download_dir).expanduser().resolve()
        if self.hot_start_file:
            self.hot_start_file = Path(self.hot_start_file).expanduser().resolve()
        if self.forecast_meteo_file:
            self.forecast_meteo_file = Path(self.forecast_meteo_file).expanduser().resolve()
        if self.troute_file:
            self.troute_file = Path(self.troute_file).expanduser().resolve()
        if self.parm_dir is not None:
            self.parm_dir = Path(self.parm_dir).expanduser().resolve()
        if self.nwm_dir is not None:
            self.nwm_dir = Path(self.nwm_dir).expanduser().resolve()
        if self.tidal_atlas_dir is not None:
            self.tidal_atlas_dir = Path(self.tidal_atlas_dir).expanduser().resolve()

    @property
    def parm_nwm(self) -> Path:
        """Parameter files directory (requires ``parm_dir``)."""
        if self.parm_dir is None:
            raise ValueError("paths.parm_dir is required for NWM parameter lookup")
        return self.parm_dir / "parm"

    @property
    def download_dir(self) -> Path:
        """Effective download directory (fallback to work_dir/downloads)."""
        return self.raw_download_dir or self.work_dir / "downloads"

    @classmethod
    def meteo_subdir(cls, meteo_source: str, coastal_domain: str) -> Path:
        """Relative meteo path, ``meteo/<source>/<nwm domain>``.

        Every NWM domain names its hourly forcing ``YYYYMMDDHH.LDASIN_DOMAIN1``,
        so files from different domains collide unless each domain gets its
        own directory: a cached Hawaii file would otherwise be served for a
        PRVI run covering the same hour.
        """
        return (
            Path(cls.METEO_SUBDIR) / meteo_source / SimulationConfig.nwm_domain_for(coastal_domain)
        )

    def meteo_dir(self, meteo_source: str, coastal_domain: str) -> Path:
        """Directory for meteorological data."""
        return self.download_dir / self.meteo_subdir(meteo_source, coastal_domain)

    @classmethod
    def streamflow_subdir(cls, coastal_domain: str) -> Path:
        """Relative streamflow path, ``hydro/nwm/<nwm domain>``.

        Keyed the same way as :meth:`meteo_subdir`, so a domain reads the
        same everywhere under the download directory even though NWM
        spells it differently in its own URLs (``puertorico`` there,
        ``prvi`` here).

        Only ``nwm_ana`` streamflow is downloaded; ``nwm_retro`` is read
        straight from the S3 Zarr store, so it has no directory here.
        """
        return Path(cls.HYDRO_SUBDIR) / "nwm" / SimulationConfig.nwm_domain_for(coastal_domain)

    def streamflow_dir(self, coastal_domain: str = "conus") -> Path:
        """Directory for downloaded ``nwm_ana`` streamflow data."""
        return self.download_dir / self.streamflow_subdir(coastal_domain)

    def coastal_dir(self, coastal_source: str) -> Path:
        """Directory for coastal boundary data."""
        return self.download_dir / self.COASTAL_SUBDIR / coastal_source

    def geogrid_file(self, sim: SimulationConfig) -> Path:
        """Geogrid file path for the given domain (requires ``parm_dir``)."""
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
            self.log_file = Path(self.log_file).expanduser().resolve()


@dataclass
class DownloadConfig:
    """Data download configuration."""

    enabled: bool = True
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

    Attributes
    ----------
    omp_num_threads : int
        Number of OpenMP threads per process.
    runtime_env : dict[str, str]
        Extra environment variables for the model run subprocess.
        Merged last so they can override any auto-detected value.
        Only used by model run stages (``schism_run``, ``sfincs_run``).
    """

    omp_num_threads: int
    runtime_env: dict[str, str]

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string (e.g. ``'schism'``, ``'sfincs'``)."""

    @abstractmethod
    def build_environment(self, env: dict[str, str], config: CoastalCalibConfig) -> dict[str, str]:
        """Add model-specific environment variables to *env* (mutating).

        Called by :meth:`WorkflowStage.build_environment` after shared
        variables (OpenMP pinning, HDF5 file locking) have been populated.
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
    def to_dict(self) -> dict[str, Any]:
        """Serialize model-specific fields to a dictionary."""


@dataclass
class SchismModelConfig(ModelConfig):
    """SCHISM model configuration.

    Contains compute parameters (MPI layout, SCHISM binary), the path
    to a prebuilt model directory, and the geogrid file used for
    atmospheric forcing regridding.

    Parameters
    ----------
    prebuilt_dir : Path
        Path to the directory containing the pre-built SCHISM model
        files (``hgrid.gr3``, ``vgrid.in``, ``param.nml``, etc.).
    geogrid_file : Path
        Path to the WRF geogrid file (e.g. ``geo_em_HI.nc``) used by
        the atmospheric forcing regridding stage.
    nodes : int
        Number of SLURM nodes. Defaults to ``1``; set higher for multi-node
        HPC jobs.
    ntasks_per_node : int
        MPI tasks per node. When ``<= 0`` (the default), this is auto-set
        to ``get_cpu_count() // omp_num_threads`` so a single-node run
        fills the available physical cores
        (see :func:`~coastal_calibration.utils.get_cpu_count`).
    exclusive : bool
        Request exclusive node access.
    nscribes : int
        Number of SCHISM scribe processes. When ``<= 0`` (the default),
        this is auto-detected from the prebuilt's ``param.nml``: a count
        of uncommented ``iof_*(N) = 1`` flags plus one for ``iout_sta``
        when ``include_noaa_gages`` is enabled. SCHISM aborts at init
        when nscribes is below the actual number of output variables.
    omp_num_threads : int
        OpenMP threads per MPI rank. Defaults to ``2`` (typical SCHISM
        hybrid layout); combined with the auto-detected ``ntasks_per_node``
        this fills one node's physical cores.
    oversubscribe : bool
        Pass ``--oversubscribe`` to ``mpiexec``. Only honored under
        OpenMPI; silently ignored under MPICH (see
        :func:`~coastal_calibration.utils.build_mpi_cmd`). Defaults to
        ``False``; set ``True`` when intentionally launching more MPI
        ranks than physical cores.
    schism_exe : Path, optional
        Path to a compiled SCHISM executable.  When set, the
        ``schism_run`` stage uses this binary instead of discovering
        ``pschism`` on ``PATH``.  Normally not needed -- SCHISM is
        compiled automatically when activating a pixi environment
        with the ``schism`` feature.  Set this to a system-compiled
        binary on WCOSS2 or other clusters where the model is built
        against system MPI/HDF5/NetCDF.
    include_noaa_gages : bool
        When True, automatically query NOAA CO-OPS for water level
        stations within the model domain (computed from the concave
        hull of open boundary nodes in ``hgrid.gr3``), write a
        ``station.in`` file, set ``iout_sta = 1`` in ``param.nml``,
        and generate sim-vs-obs comparison plots after the run.
        Requires the ``plot`` optional dependencies.
    discharge_file : Path, optional
        Path to a ``nwmReaches.csv`` file mapping NWM reach feature IDs
        to SCHISM source/sink elements.  When ``None`` (default), the
        discharge stage is skipped and no river forcing is generated.
    create_water_level_animation : bool
        When True, the ``schism_plot`` stage loads the 2-D elevation
        field from ``outputs/out2d_*.nc`` and renders an MP4 animation
        to ``figs/water_level.mp4`` using
        :func:`coastal_calibration.plotting.animate_water_level`.
        Requires an ``ffmpeg`` binary on PATH.  Independent of
        ``include_noaa_gages``.  Defaults to False.
    animation_fps : int
        Frames per second for the animation output.
    animation_time_stride : int
        Keep every ``animation_time_stride``-th frame from the model
        time series; useful for long runs.
    obs_points_csv : Path, optional
        Path to a CSV with columns ``id, lon, lat`` specifying extra
        observation points for water-level extraction after the run.
        The ``schism_plot`` stage interpolates water-surface elevation
        at each point (and at any NOAA CO-OPS gauges when
        ``include_noaa_gages`` is enabled) and writes the combined time
        series to ``obs_water_level.parquet`` in the work directory.
    output_freq_hours : float
        How often SCHISM writes field outputs, in hours. Translated into
        the ``nspool`` parameter of ``param.nml``. Defaults to 1.0
        (hourly output, matching the previous hardcoded behavior).
    single_output_file : bool
        When True, set ``ihfskip`` to the full simulation length so
        SCHISM keeps appending to a single output file instead of
        rotating to a new file every ``nspool`` steps. Useful on shared
        filesystems where each file rotation costs an MPI barrier and
        metadata round-trips. Defaults to False (matching the previous
        ``ihfskip = nspool`` behavior).
    run_param_overrides : dict
        Arbitrary key/value pairs written into ``param.nml`` after the
        template values, ``output_freq_hours``, and ``single_output_file``
        have been applied. Use this to override any other namelist
        parameter (e.g. ``{"dt": 100, "iwbl": 1}``). Mirrors the SFINCS
        ``run_param_overrides`` option. Validation catches mismatches
        between ``ihfskip``, ``nhot_write``, and ``nspool_sta`` before
        SCHISM is launched.
    """

    prebuilt_dir: Path | None = None
    geogrid_file: Path | None = None
    nodes: int = 1
    ntasks_per_node: int = 0
    exclusive: bool = True
    nscribes: int = 0
    omp_num_threads: int = 2
    oversubscribe: bool = False
    schism_exe: Path | None = None
    include_noaa_gages: bool = False
    discharge_file: Path | None = None
    create_water_level_animation: bool = False
    animation_fps: int = 10
    animation_time_stride: int = 1
    obs_points_csv: Path | None = None
    output_freq_hours: float = 1.0
    single_output_file: bool = False
    run_param_overrides: dict[str, Any] = field(default_factory=dict)
    runtime_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prebuilt_dir is not None:
            self.prebuilt_dir = Path(self.prebuilt_dir).expanduser().resolve()
        if self.geogrid_file is not None:
            self.geogrid_file = Path(self.geogrid_file).expanduser().resolve()
        if self.schism_exe is not None:
            self.schism_exe = Path(self.schism_exe).expanduser().resolve()
        if self.discharge_file is not None:
            self.discharge_file = Path(self.discharge_file).expanduser().resolve()
        if self.obs_points_csv is not None:
            self.obs_points_csv = Path(self.obs_points_csv).expanduser().resolve()
        if self.ntasks_per_node <= 0:
            from coastal_calibration.utils import get_cpu_count

            self.ntasks_per_node = max(get_cpu_count() // max(self.omp_num_threads, 1), 1)
        if self.nscribes <= 0:
            from coastal_calibration.schism.prep import count_required_scribes

            detected: int | None = None
            if self.prebuilt_dir is not None:
                detected = count_required_scribes(
                    self.prebuilt_dir / "param.nml", self.include_noaa_gages
                )
            self.nscribes = detected if detected and detected > 0 else 2

    @property
    def model_name(self) -> str:  # noqa: D102
        return "schism"

    @property
    def total_tasks(self) -> int:
        """Total number of MPI tasks (nodes * ntasks_per_node)."""
        return self.nodes * self.ntasks_per_node

    @property
    def coastal_parm(self) -> Path:
        """Directory containing prebuilt SCHISM model files."""
        if self.prebuilt_dir is None:
            raise ValueError("model_config.prebuilt_dir is not set")
        return self.prebuilt_dir

    @property
    def geogrid_path(self) -> Path:
        """WRF geogrid file used for atmospheric regridding."""
        if self.geogrid_file is None:
            raise ValueError("model_config.geogrid_file is not set")
        return self.geogrid_file

    @property
    def schism_mesh(self) -> Path:
        """SCHISM ESMF mesh file path."""
        return self.coastal_parm / "hgrid.nc"

    def resolved_discharge_file(self, meteo_source: str = "nwm_ana") -> Path | None:
        """Resolve the discharge crosswalk CSV, or ``None`` to skip discharge.

        The crosswalk maps SCHISM source/sink elements to routed-reach IDs.
        Which file is used depends on *meteo_source*: ``ngen_forecast`` runs
        use ``ngenReaches.csv`` (NextGen hydrofabric feature_ids, matching
        the t-route output), while ``nwm_retro`` / ``nwm_ana`` use
        ``nwmReaches.csv`` (NWM COMIDs).

        Resolution order:

        1. ``discharge_file`` explicitly set → use it if it exists, else
           ``None``. An explicit configuration is treated as exclusive: it
           does not silently fall back to the prebuilt-directory convention.
        2. ``discharge_file`` unset → look for the source-appropriate
           reaches file (``ngenReaches.csv`` or ``nwmReaches.csv``) next to
           the prebuilt model. Use it if present.
        3. Otherwise → ``None`` (river forcing is skipped and SCHISM is
           configured with ``if_source = 0``).

        A missing optional file degrades gracefully — the caller skips
        discharge rather than aborting.
        """
        if self.discharge_file is not None:
            return self.discharge_file if self.discharge_file.exists() else None
        if self.prebuilt_dir is None:
            return None
        fname = "ngenReaches.csv" if meteo_source == "ngen_forecast" else "nwmReaches.csv"
        candidate = self.prebuilt_dir / fname
        return candidate if candidate.exists() else None

    @property
    def elevation_correction_csv(self) -> Path | None:
        """Return ``elevation_correction.csv`` next to the prebuilt model.

        Returns ``None`` when ``prebuilt_dir`` is unset *or* when the
        correction file is absent. Callers can pass the result straight
        into readers that accept an optional path without re-doing the
        ``exists()`` check.
        """
        if self.prebuilt_dir is None:
            return None
        candidate = self.prebuilt_dir / "elevation_correction.csv"
        return candidate if candidate.exists() else None

    @property
    def stage_order(self) -> list[str]:  # noqa: D102
        return [
            "download",
            "schism_forcing_prep",
            "schism_forcing",
            "schism_sflux",
            "schism_params",
            "schism_obs",
            "schism_boundary",
            "schism_discharge",
            "schism_prep",
            "schism_run",
            "schism_postprocess",
            "schism_plot",
        ]

    def build_environment(  # noqa: D102
        self,
        env: dict[str, str],
        config: CoastalCalibConfig,  # noqa: ARG002
    ) -> dict[str, str]:
        from coastal_calibration.utils import build_mpi_env

        build_mpi_env(env)
        return env

    def validate(self, config: CoastalCalibConfig) -> list[str]:  # noqa: D102
        errors: list[str] = []

        if self.nodes < 1:
            errors.append("model_config.nodes must be at least 1")

        if self.ntasks_per_node < 1:
            errors.append("model_config.ntasks_per_node must be at least 1")

        if self.nscribes >= self.total_tasks:
            errors.append(
                f"model_config: nscribes ({self.nscribes}) leaves no compute ranks "
                f"with total MPI tasks {self.total_tasks} "
                f"(nodes={self.nodes} * ntasks_per_node={self.ntasks_per_node}). "
                "Either lower omp_num_threads (typical: 1) to widen ntasks_per_node, "
                "raise ntasks_per_node explicitly (set oversubscribe: true if it "
                "exceeds physical cores), or reduce iof_* outputs in the prebuilt "
                "param.nml."
            )

        if self.schism_exe and not self.schism_exe.exists():
            errors.append(f"model_config.schism_exe not found: {self.schism_exe}")

        if config.paths.hot_start_file and not config.paths.hot_start_file.exists():
            errors.append(f"Hot start file not found: {config.paths.hot_start_file}")

        if self.prebuilt_dir is None:
            errors.append("model_config.prebuilt_dir is required")
        elif not self.prebuilt_dir.exists():
            errors.append(f"model_config.prebuilt_dir not found: {self.prebuilt_dir}")
        else:
            required = [
                "hgrid.gr3",
                "vgrid.in",
                "param.nml",
                "bctides.in",
            ]
            errors.extend(
                f"Required file missing in model_config.prebuilt_dir: {fname}"
                for fname in required
                if not (self.prebuilt_dir / fname).exists()
            )

        if self.discharge_file and not self.discharge_file.exists():
            errors.append(f"model_config.discharge_file not found: {self.discharge_file}")

        if self.geogrid_file is None:
            errors.append(
                "model_config.geogrid_file is required for atmospheric forcing regridding"
            )
        elif not self.geogrid_file.exists():
            errors.append(f"model_config.geogrid_file not found: {self.geogrid_file}")

        return errors

    def create_stages(  # noqa: D102
        self, config: CoastalCalibConfig, monitor: Any
    ) -> dict[str, Any]:
        from coastal_calibration.data.download_stage import DownloadStage
        from coastal_calibration.schism.boundary import (
            BoundaryConditionStage,
            UpdateParamsStage,
        )
        from coastal_calibration.schism.forcing import (
            NWMForcingStage,
            PostForcingStage,
            PreForcingStage,
        )
        from coastal_calibration.schism.stages import (
            PostSCHISMStage,
            PreSCHISMStage,
            SchismDischargeStage,
            SchismObservationStage,
            SchismPlotStage,
            SCHISMRunStage,
        )

        return {
            "download": DownloadStage(config, monitor),
            "schism_forcing_prep": PreForcingStage(config, monitor),
            "schism_forcing": NWMForcingStage(config, monitor),
            "schism_sflux": PostForcingStage(config, monitor),
            "schism_params": UpdateParamsStage(config, monitor),
            "schism_obs": SchismObservationStage(config, monitor),
            "schism_boundary": BoundaryConditionStage(config, monitor),
            "schism_discharge": SchismDischargeStage(config, monitor),
            "schism_prep": PreSCHISMStage(config, monitor),
            "schism_run": SCHISMRunStage(config, monitor),
            "schism_postprocess": PostSCHISMStage(config, monitor),
            "schism_plot": SchismPlotStage(config, monitor),
        }

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        d: dict[str, Any] = {
            "prebuilt_dir": str(self.prebuilt_dir) if self.prebuilt_dir else None,
            "geogrid_file": str(self.geogrid_file) if self.geogrid_file else None,
            "nodes": self.nodes,
            "ntasks_per_node": self.ntasks_per_node,
            "exclusive": self.exclusive,
            "nscribes": self.nscribes,
            "omp_num_threads": self.omp_num_threads,
            "oversubscribe": self.oversubscribe,
            "schism_exe": (str(self.schism_exe) if self.schism_exe else None),
            "include_noaa_gages": self.include_noaa_gages,
            "discharge_file": (str(self.discharge_file) if self.discharge_file else None),
            "create_water_level_animation": self.create_water_level_animation,
            "animation_fps": self.animation_fps,
            "animation_time_stride": self.animation_time_stride,
            "obs_points_csv": (str(self.obs_points_csv) if self.obs_points_csv else None),
            "output_freq_hours": self.output_freq_hours,
            "single_output_file": self.single_output_file,
            "run_param_overrides": self.run_param_overrides,
            "runtime_env": self.runtime_env,
        }
        return d


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
    meteo_res : float, optional
        Output resolution (m) for gridded meteorological forcing
        (precipitation, wind, pressure).  When *None* (default) the
        resolution is determined from the SFINCS quadtree grid — it
        equals the base cell size (coarsest level) so that the meteo
        grid is never finer than needed.  Setting an explicit value
        (e.g. ``2000``) overrides the automatic calculation.

        .. note::

           Without this parameter the HydroMT ``reproject`` call
           retains the source-data resolution (≈ 1 km for NWM), and
           the LCC → UTM reprojection can inflate the output to the
           full CONUS extent, producing multi-GB files and very slow
           simulations.
    forcing_to_mesh_offset_m : float
        Vertical offset in meters *added* to the boundary-condition water
        levels before they enter SFINCS.

        Tidal-only sources (harmonic prediction) provide oscillations centered on
        zero (MSL) but carry no information about where MSL sits on the
        mesh's vertical datum.  This parameter anchors the forcing signal
        to the correct geodetic height on the mesh.  Set it to the
        elevation of MSL in the mesh datum obtained from VDatum
        (e.g. ``0.171`` for a NAVD88 mesh on the Texas Gulf coast, where
        MSL is 0.171 m above NAVD88).

        For sources that already report water levels in the mesh datum
        (e.g. STOFS on a NAVD88 mesh) set this to ``0.0``.

        Defaults to ``0.0``.
    vdatum_mesh_to_msl_m : float
        Vertical offset in meters *added* to the simulated water level
        before comparison with NOAA CO-OPS observations (which are in
        MSL).  The model output inherits the mesh vertical datum, so
        this converts it to MSL (e.g. ``0.171`` for a NAVD88 mesh on
        the Texas Gulf coast).

        Defaults to ``0.0``.
    sfincs_exe : Path, optional
        Path to a compiled SFINCS executable.  When set, the
        ``sfincs_run`` stage uses this binary instead of discovering
        ``sfincs`` on ``PATH``.  Normally not needed -- SFINCS is
        compiled automatically when activating a pixi environment
        with the ``sfincs`` feature.
    omp_num_threads : int
        Number of OpenMP threads.  Defaults to the number of physical CPU
        cores on the current machine (see :func:`~coastal_calibration.utils.get_cpu_count`).
        On HPC nodes this auto-detects correctly; on a local laptop it
        avoids over-subscribing the system.
    run_param_overrides : dict
        Arbitrary key/value pairs written to ``sfincs.inp`` just before the
        model is written to disk.  Use this to override physics parameters
        that HydroMT-SFINCS sets by default (e.g. ``advection: 0``,
        ``nuvisc: 0.01``).  Keys must be valid ``sfincs.inp`` parameter
        names.  Mirrors the SCHISM ``run_param_overrides`` option.
    create_water_level_animation : bool
        When True, the ``sfincs_plot`` stage loads the time-dependent
        water level field from ``sfincs_map.nc`` and renders an MP4
        animation to ``figs/water_level.mp4`` using
        :func:`coastal_calibration.plotting.animate_water_level`.
        Requires an ``ffmpeg`` binary on PATH.  Defaults to False.
    animation_fps : int
        Frames per second for the animation output.
    animation_time_stride : int
        Keep every ``animation_time_stride``-th frame from the model
        time series; useful for long runs.
    obs_points_csv : Path, optional
        Path to a CSV with columns ``id, lon, lat`` specifying extra
        observation points for water-level extraction after the run.
        The ``sfincs_plot`` stage interpolates water-surface elevation
        at each point (and at any NOAA CO-OPS gauges found in
        ``obs_station_map.json``) and writes the combined time series
        to ``obs_water_level.parquet`` alongside ``sfincs_map.nc``.
    """

    # Known sfincs.inp parameter names parsed by the SFINCS binary
    # (extracted from SFINCS/source/src/sfincs_input.f90).  Used to
    # catch typos in ``run_param_overrides`` early — SFINCS silently
    # ignores unrecognized parameters.
    _KNOWN_INP_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "advection",
            "advection_mask",
            "advection_scheme",
            "advlim",
            "alpha",
            "amprblock",
            "ampfile",
            "amprfile",
            "amufile",
            "amvfile",
            "baro",
            "bcafile",
            "bdrfile",
            "bndfile",
            "bndtype",
            "btfilter",
            "btrelax",
            "bzifile",
            "bzsfile",
            "cdnrb",
            "cdval",
            "cdwnd",
            "coriolis",
            "crsgeo",
            "crsfile",
            "cstfile",
            "debug",
            "depfile",
            "disfile",
            "drnfile",
            "dtmax",
            "dtmaxout",
            "dtmapout",
            "dtout",
            "dtrstout",
            "dthisout",
            "dtwave",
            "dtwnd",
            "dx",
            "dy",
            "epsg",
            "f0file",
            "factor_pres",
            "factor_prcp",
            "factor_spw_size",
            "factor_wind",
            "fcfile",
            "freqmaxig",
            "freqminig",
            "friction2d",
            "gapres",
            "global",
            "h73table",
            "hmin_cfl",
            "horton_kr_kd",
            "huthresh",
            "indexfile",
            "inifile",
            "inputformat",
            "kdfile",
            "ksfile",
            "latitude",
            "manning",
            "manning_land",
            "manning_sea",
            "manningfile",
            "mmax",
            "mskfile",
            "nc_deflate_level",
            "ncinifile",
            "netamprfile",
            "netampfile",
            "netamuamvfile",
            "netbndbzsbzifile",
            "netsrcdisfile",
            "netspwfile",
            "nfreqsig",
            "nmax",
            "nonh",
            "nh_fnudge",
            "nh_itermax",
            "nh_tol",
            "nh_tstop",
            "nuvisc",
            "nuviscfac",
            "obsfile",
            "outputformat",
            "outputtype_his",
            "outputtype_map",
            "pavbnd",
            "percentage_done",
            "precipfile",
            "prcfile",
            "psifile",
            "qinf",
            "qinf_zmin",
            "qinffile",
            "qtrfile",
            "radstr",
            "regular_output_on_mesh",
            "rgh_lev_land",
            "rhoa",
            "rhow",
            "rotation",
            "rstfile",
            "rugdepth",
            "rugfile",
            "sbgfile",
            "scsfile",
            "sefffile",
            "sfacinf",
            "sigmafile",
            "slopelim",
            "smaxfile",
            "snapwave",
            "snapwave_use_nearest",
            "snapwave_wind",
            "spwfile",
            "spwmergefrac",
            "srcfile",
            "store_dynamic_bed_level",
            "store_tsunami_arrival_time",
            "storecumprcp",
            "storefluxmax",
            "storefw",
            "storehmean",
            "storehsubgrid",
            "storemeteo",
            "storemaxwind",
            "storeqdrain",
            "storestoragevolume",
            "storetzsmax",
            "storetwet",
            "storevel",
            "storevelmax",
            "storewavdir",
            "storezvolume",
            "structure_relax",
            "t0out",
            "t1out",
            "thdfile",
            "theta",
            "tref",
            "trstout",
            "tspinup",
            "tstart",
            "tstop",
            "tsunami_arrival_threshold",
            "twet_threshold",
            "use_bcafile",
            "usespwprecip",
            "utmzone",
            "uvlim",
            "uvmax",
            "viscosity",
            "volfile",
            "wave_enhanced_roughness",
            "waveage",
            "weirfile",
            "wfpfile",
            "whifile",
            "wiggle_factor",
            "wiggle_suppression",
            "wiggle_threshold",
            "wmfred",
            "wmsignal",
            "wmtfilter",
            "wndfile",
            "writeruntime",
            "wstfile",
            "wtifile",
            "wvmfile",
            "x0",
            "y0",
            "z0lfile",
            "zsini",
            "spinup_meteo",
            "dtoutfixed",
        }
    )

    prebuilt_dir: Path
    model_root: Path | None = None
    discharge_locations_file: Path | None = None
    merge_discharge: bool = False
    include_precip: bool = False
    include_wind: bool = False
    include_pressure: bool = False
    meteo_res: float | None = None
    forcing_to_mesh_offset_m: float = 0.0
    vdatum_mesh_to_msl_m: float = 0.0
    sfincs_exe: Path | None = None
    omp_num_threads: int = field(default=0)
    run_param_overrides: dict[str, Any] = field(default_factory=dict)
    # Obsolete: the flood-map stage reads the elevation raster recorded by
    # `coastal-calibration create`. Kept only so models built before that
    # record existed still produce a flood map; setting it otherwise logs a
    # warning and has no effect.
    floodmap_dem: Path | None = None
    floodmap_hmin: float = 0.05
    floodmap_enabled: bool = True
    # Restrict the flood map to inundated land. With this off the raster also
    # covers the permanently wet sea, where "depth" is just the water column
    # over the bathymetry. The cut is the vertical datum, so terrain lying
    # *below* the datum but hydraulically dry (leveed polders, subsided urban
    # land, dredged basins) is dropped too even when the model floods it. Turn
    # this off for such domains and mask the sea some other way.
    floodmap_land_only: bool = True
    create_water_level_animation: bool = False
    animation_fps: int = 10
    animation_time_stride: int = 1
    obs_points_csv: Path | None = None
    runtime_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prebuilt_dir = Path(self.prebuilt_dir).expanduser().resolve()
        if self.model_root is not None:
            self.model_root = Path(self.model_root).expanduser().resolve()
        if self.discharge_locations_file is not None:
            self.discharge_locations_file = (
                Path(self.discharge_locations_file).expanduser().resolve()
            )
        if self.sfincs_exe is not None:
            self.sfincs_exe = Path(self.sfincs_exe).expanduser().resolve()
        if self.floodmap_dem is not None:
            self.floodmap_dem = Path(self.floodmap_dem).expanduser().resolve()
        if self.obs_points_csv is not None:
            self.obs_points_csv = Path(self.obs_points_csv).expanduser().resolve()
        if self.omp_num_threads <= 0:
            from coastal_calibration.utils import get_cpu_count

            self.omp_num_threads = get_cpu_count()

    @staticmethod
    def _discharge_token(meteo_source: str) -> str:
        """Return the discharge-file ID-space token for *meteo_source*.

        ``ngen`` for ngen_forecast (NextGen hydrofabric IDs), ``nwm``
        otherwise (NWM COMIDs).
        """
        return "ngen" if meteo_source == "ngen_forecast" else "nwm"

    @staticmethod
    def _sibling_with_token(path: Path, token: str) -> Path:
        """Return *path* re-suffixed with ``_<token>`` before its extension.

        Swaps an existing ``_nwm`` / ``_ngen`` stem suffix, or appends one
        when absent.  e.g. ``discharge_points_nwm.geojson`` + ``ngen`` ->
        ``discharge_points_ngen.geojson``.
        """
        stem = path.stem
        for other in ("nwm", "ngen"):
            if stem.endswith(f"_{other}"):
                base = stem[: -(len(other) + 1)]
                return path.with_name(f"{base}_{token}{path.suffix}")
        return path.with_name(f"{stem}_{token}{path.suffix}")

    def resolved_discharge_locations_file(self, meteo_source: str = "nwm_ana") -> Path | None:
        """Resolve the discharge-points file for *meteo_source*.

        Discharge-point files carry the reach IDs in each point's ``name``:
        NWM COMIDs for ``nwm_retro`` / ``nwm_ana`` runs, NextGen
        hydrofabric IDs for ``ngen_forecast`` runs.  By convention the
        files are suffixed ``_nwm`` / ``_ngen`` so both can coexist.

        Resolution: if ``discharge_locations_file`` already matches the
        meteo-appropriate token it is used as-is; otherwise the sibling
        carrying that token is preferred **when it exists** (so a user can
        drop both ``*_nwm`` and ``*_ngen`` files and the correct one is
        chosen automatically).  Falls back to the configured file when no
        matching sibling exists — :meth:`validate` flags a hard mismatch.

        Returns ``None`` when ``discharge_locations_file`` is unset
        (discharge is skipped).
        """
        if self.discharge_locations_file is None:
            return None
        token = self._discharge_token(meteo_source)
        path = self.discharge_locations_file
        if path.stem.endswith(f"_{token}"):
            return path
        sibling = self._sibling_with_token(path, token)
        return sibling if sibling.exists() else path

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
            "sfincs_discharge",
            "sfincs_precip",
            "sfincs_wind",
            "sfincs_pressure",
            "sfincs_write",
            "sfincs_run",
            "sfincs_floodmap",
            "sfincs_plot",
        ]

    def build_environment(  # noqa: D102
        self,
        env: dict[str, str],
        config: CoastalCalibConfig,  # noqa: ARG002
    ) -> dict[str, str]:
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

        # Validate the discharge-points file resolved for this run's source.
        if self.discharge_locations_file is not None:
            meteo_source = config.simulation.meteo_source
            token = self._discharge_token(meteo_source)
            resolved = self.resolved_discharge_locations_file(meteo_source)
            if resolved is not None and not resolved.exists():
                errors.append(f"model_config.discharge_locations_file not found: {resolved}")
            elif resolved is not None:
                other = "nwm" if token == "ngen" else "ngen"
                if resolved.stem.endswith(f"_{other}"):
                    errors.append(
                        f"Discharge file '{resolved.name}' is suffixed '_{other}' but "
                        f"simulation.meteo_source is '{meteo_source}', which needs "
                        f"'_{token}' reach IDs. Provide a '_{token}' discharge file "
                        "(its point names must match the streamflow source's feature_ids)."
                    )

        if self.sfincs_exe and not self.sfincs_exe.exists():
            errors.append(f"model_config.sfincs_exe not found: {self.sfincs_exe}")

        if self.run_param_overrides:
            unknown = sorted(set(self.run_param_overrides) - self._KNOWN_INP_PARAMS)
            if unknown:
                errors.append(
                    f"Unrecognized sfincs.inp parameter(s): {', '.join(unknown)}. "
                    "SFINCS silently ignores unknown parameters; check for typos."
                )

        return errors

    def create_stages(  # noqa: D102
        self, config: CoastalCalibConfig, monitor: Any
    ) -> dict[str, Any]:
        from coastal_calibration.data.download_stage import DownloadStage
        from coastal_calibration.sfincs.stages import (
            SfincsDataCatalogStage,
            SfincsDischargeStage,
            SfincsFloodMapStage,
            SfincsForcingStage,
            SfincsInitStage,
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
            "sfincs_discharge": SfincsDischargeStage(config, monitor),
            "sfincs_precip": SfincsPrecipitationStage(config, monitor),
            "sfincs_wind": SfincsWindStage(config, monitor),
            "sfincs_pressure": SfincsPressureStage(config, monitor),
            "sfincs_write": SfincsWriteStage(config, monitor),
            "sfincs_run": SfincsRunStage(config, monitor),
            "sfincs_floodmap": SfincsFloodMapStage(config, monitor),
            "sfincs_plot": SfincsPlotStage(config, monitor),
        }

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            "prebuilt_dir": str(self.prebuilt_dir),
            "model_root": str(self.model_root) if self.model_root else None,
            "discharge_locations_file": (
                str(self.discharge_locations_file) if self.discharge_locations_file else None
            ),
            "merge_discharge": self.merge_discharge,
            "include_precip": self.include_precip,
            "include_wind": self.include_wind,
            "include_pressure": self.include_pressure,
            "forcing_to_mesh_offset_m": self.forcing_to_mesh_offset_m,
            "vdatum_mesh_to_msl_m": self.vdatum_mesh_to_msl_m,
            "sfincs_exe": (str(self.sfincs_exe) if self.sfincs_exe else None),
            "omp_num_threads": self.omp_num_threads,
            "run_param_overrides": self.run_param_overrides,
            "floodmap_dem": (str(self.floodmap_dem) if self.floodmap_dem else None),
            "floodmap_hmin": self.floodmap_hmin,
            "floodmap_enabled": self.floodmap_enabled,
            "floodmap_land_only": self.floodmap_land_only,
            "create_water_level_animation": self.create_water_level_animation,
            "animation_fps": self.animation_fps,
            "animation_time_stride": self.animation_time_stride,
            "obs_points_csv": (str(self.obs_points_csv) if self.obs_points_csv else None),
            "runtime_env": self.runtime_env,
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
        Flat dictionary of available variables (e.g., {"user": "john"}).

    Returns
    -------
    Any
        The interpolated value.

    Examples
    --------
    >>> ctx = {"user": "john", "simulation.coastal_domain": "hawaii"}
    >>> _interpolate_value("/data/${user}/${simulation.coastal_domain}", ctx)
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
        Flat dictionary with keys like "user", "simulation.coastal_domain".
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
    # Resolve ${user} from $USER env var for default path templates.
    if "user" not in context:
        context["user"] = os.environ.get("USER", "unknown")
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

    simulation: SimulationConfig
    boundary: BoundaryConfig
    paths: PathConfig
    model_config: SchismModelConfig | SfincsModelConfig
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)

    @property
    def model(self) -> str:
        """Model identifier string (convenience accessor)."""
        return self.model_config.model_name

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoastalCalibConfig:
        """Create config from a plain dictionary.

        Parameters
        ----------
        data : dict
            Configuration dictionary with the same structure as the YAML
            file (see :meth:`to_dict` for the expected keys). The dict
            is read but not mutated.

        Returns
        -------
        CoastalCalibConfig
        """
        if "model" not in data:
            raise ValueError("'model' is required (e.g., model: schism or model: sfincs)")
        model_type: str = data["model"]

        # Read but do not mutate the caller's dict.
        model_config_data = data.get("model_config") or {}

        sim_data = data.get("simulation", {})
        if "start_date" in sim_data:
            sim_data["start_date"] = pd.to_datetime(sim_data["start_date"]).to_pydatetime()
        simulation = SimulationConfig(**sim_data)

        boundary_data = data.get("boundary", {})
        if boundary_data.get("stofs_file"):
            boundary_data["stofs_file"] = Path(boundary_data["stofs_file"])
        boundary = BoundaryConfig(**boundary_data)

        paths_data = data.get("paths", {})
        if paths_data.get("forecast_meteo_file"):
            paths_data["forecast_meteo_file"] = Path(paths_data["forecast_meteo_file"])
        if paths_data.get("troute_file"):
            paths_data["troute_file"] = Path(paths_data["troute_file"])
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
            simulation=simulation,
            boundary=boundary,
            paths=paths,
            model_config=model_config,  # pyright: ignore[reportArgumentType]
            monitoring=monitoring,
            download=download,
        )

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> CoastalCalibConfig:
        """Load configuration from YAML file with optional inheritance.

        Supports variable interpolation using ${section.key} syntax.
        Variables are resolved from other config values, e.g.:

        - ``${user}`` -> value of ``$USER`` environment variable
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

        if "_base" in data:
            base_path = Path(data.pop("_base"))
            if not base_path.is_absolute():
                base_path = config_path.parent / base_path
            base_config = cls.from_yaml(base_path)
            data = _deep_merge(base_config.to_dict(), data)

        # Ensure model key has a default before interpolation
        data.setdefault("model", "schism")

        # Interpolate variables after merging
        data = _interpolate_config(data)

        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model,
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
                "tidal_model": self.boundary.tidal_model,
            },
            "paths": {
                "work_dir": str(self.paths.work_dir),
                "raw_download_dir": (
                    str(self.paths.raw_download_dir) if self.paths.raw_download_dir else None
                ),
                "hot_start_file": (
                    str(self.paths.hot_start_file) if self.paths.hot_start_file else None
                ),
                **(
                    {"forecast_meteo_file": str(self.paths.forecast_meteo_file)}
                    if self.paths.forecast_meteo_file
                    else {}
                ),
                **(
                    {"troute_file": str(self.paths.troute_file)}
                    if self.paths.troute_file
                    else {}
                ),
                **({"parm_dir": str(self.paths.parm_dir)} if self.paths.parm_dir else {}),
                **({"nwm_dir": str(self.paths.nwm_dir)} if self.paths.nwm_dir else {}),
                **(
                    {"tidal_atlas_dir": str(self.paths.tidal_atlas_dir)}
                    if self.paths.tidal_atlas_dir
                    else {}
                ),
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

        elif self.boundary.source == "harmonic":
            if self.paths.tidal_atlas_dir is None:
                errors.append(
                    "paths.tidal_atlas_dir is required when boundary.source is 'harmonic'"
                )
            elif not self.paths.tidal_atlas_dir.exists():
                errors.append(f"Tidal atlas directory not found: {self.paths.tidal_atlas_dir}")

        return errors

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        from coastal_calibration.data.downloader import validate_date_ranges

        errors: list[str] = []

        if self.simulation.duration_hours <= 0:
            errors.append("simulation.duration_hours must be positive")

        # ngen forecast forcing is read from a pre-generated file on disk
        # rather than downloaded, so the path must be set and present.
        if self.simulation.meteo_source == "ngen_forecast":
            fcst = self.paths.forecast_meteo_file
            if fcst is None:
                errors.append(
                    "paths.forecast_meteo_file is required when "
                    "simulation.meteo_source is 'ngen_forecast'"
                )
            elif not fcst.exists():
                errors.append(f"Forecast meteo file not found: {fcst}")

            # troute_file is optional (only needed when discharge is wired),
            # but if given it must exist.
            troute = self.paths.troute_file
            if troute is not None and not troute.exists():
                errors.append(f"t-route file not found: {troute}")

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
