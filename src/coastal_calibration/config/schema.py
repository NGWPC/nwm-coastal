"""YAML configuration schema and validation for coastal calibration workflow."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
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
            self.stofs_file = Path(self.stofs_file).expanduser().resolve()


@dataclass
class PathConfig:
    """Path configuration for data and executables.

    Only ``work_dir`` is required.  All other fields are optional and
    only needed by specific workflow stages (e.g. the *create* workflow
    for SCHISM or SFINCS boundary processing that uses TPXO/OTPSnc).
    """

    METEO_SUBDIR: ClassVar[str] = "meteo"
    STREAMFLOW_SUBDIR: ClassVar[str] = "streamflow"
    HYDRO_SUBDIR: ClassVar[str] = "hydro"
    COASTAL_SUBDIR: ClassVar[str] = "coastal"

    work_dir: Path
    raw_download_dir: Path | None = None
    hot_start_file: Path | None = None
    # Legacy create-workflow fields — not used by the run workflow.
    parm_dir: Path | None = None
    nwm_dir: Path | None = None
    otps_dir: Path | None = None

    # Maps coastal_domain → NWM product subdirectory name.
    _NWM_DOMAIN_DIR: ClassVar[dict[str, str]] = {
        "hawaii": "hawaii",
        "prvi": "puertorico",
        "alaska": "alaska",
    }

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir).expanduser().resolve()
        if self.raw_download_dir:
            self.raw_download_dir = Path(self.raw_download_dir).expanduser().resolve()
        if self.hot_start_file:
            self.hot_start_file = Path(self.hot_start_file).expanduser().resolve()
        if self.parm_dir is not None:
            self.parm_dir = Path(self.parm_dir).expanduser().resolve()
        if self.nwm_dir is not None:
            self.nwm_dir = Path(self.nwm_dir).expanduser().resolve()
        if self.otps_dir is not None:
            self.otps_dir = Path(self.otps_dir).expanduser().resolve()

    @property
    def tpxo_data_dir(self) -> Path:
        """TPXO tidal atlas data directory (requires ``parm_dir``)."""
        if self.parm_dir is None:
            raise ValueError("paths.parm_dir is required for TPXO data lookup")
        return self.parm_dir / "TPXO10_atlas_v2_nc"

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

    def meteo_dir(self, meteo_source: str) -> Path:
        """Directory for meteorological data."""
        return self.download_dir / self.METEO_SUBDIR / meteo_source

    def streamflow_dir(self, meteo_source: str, coastal_domain: str = "conus") -> Path:
        """Directory for streamflow/hydro data."""
        if meteo_source == "nwm_retro":
            return self.download_dir / self.STREAMFLOW_SUBDIR / "nwm_retro"
        nwm_dir = self._NWM_DOMAIN_DIR.get(coastal_domain, "conus")
        return self.download_dir / self.HYDRO_SUBDIR / "nwm" / nwm_dir

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
    """

    omp_num_threads: int

    runtime_env: dict[str, str]
    """Number of OpenMP threads per process."""
    """Extra environment variables applied when launching the model binary.

    These are merged last, so they can override any auto-detected value.
    Only used by model run stages (``schism_run``, ``sfincs_run``).
    """

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
    """

    prebuilt_dir: Path = field(default_factory=Path)
    geogrid_file: Path = field(default_factory=Path)
    nodes: int = 2
    ntasks_per_node: int = 18
    exclusive: bool = True
    nscribes: int = 2
    omp_num_threads: int = 2
    oversubscribe: bool = False
    schism_exe: Path | None = None
    include_noaa_gages: bool = False
    runtime_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prebuilt_dir = Path(self.prebuilt_dir).expanduser().resolve()
        self.geogrid_file = Path(self.geogrid_file).expanduser().resolve()
        if self.schism_exe is not None:
            self.schism_exe = Path(self.schism_exe).expanduser().resolve()

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
        return self.prebuilt_dir

    @property
    def schism_mesh(self) -> Path:
        """SCHISM ESMF mesh file path."""
        return self.prebuilt_dir / "hgrid.nc"

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
            "schism_prep",
            "schism_run",
            "schism_postprocess",
            "schism_plot",
        ]

    def build_environment(  # noqa: D102
        self, env: dict[str, str], config: CoastalCalibConfig
    ) -> dict[str, str]:
        from coastal_calibration.utils.mpi import build_mpi_env

        build_mpi_env(env)
        return env

    def validate(self, config: CoastalCalibConfig) -> list[str]:  # noqa: D102
        errors: list[str] = []

        if self.nodes < 1:
            errors.append("model_config.nodes must be at least 1")

        if self.ntasks_per_node < 1:
            errors.append("model_config.ntasks_per_node must be at least 1")

        if self.nscribes >= self.total_tasks:
            errors.append("model_config.nscribes must be less than total MPI tasks")

        if self.schism_exe and not self.schism_exe.exists():
            errors.append(f"model_config.schism_exe not found: {self.schism_exe}")

        if config.paths.hot_start_file and not config.paths.hot_start_file.exists():
            errors.append(f"Hot start file not found: {config.paths.hot_start_file}")

        if not self.prebuilt_dir.exists():
            errors.append(f"model_config.prebuilt_dir not found: {self.prebuilt_dir}")
        else:
            required = [
                "hgrid.gr3",
                "vgrid.in",
                "param.nml",
                "nwmReaches.csv",
                "bctides.in",
            ]
            errors.extend(
                f"Required file missing in model_config.prebuilt_dir: {fname}"
                for fname in required
                if not (self.prebuilt_dir / fname).exists()
            )

        if self.geogrid_file is None:  # pyright: ignore[reportUnnecessaryComparison]
            errors.append(
                "model_config.geogrid_file is required for atmospheric forcing regridding"
            )
        elif not self.geogrid_file.exists():
            errors.append(f"model_config.geogrid_file not found: {self.geogrid_file}")

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
            "schism_prep": PreSCHISMStage(config, monitor),
            "schism_run": SCHISMRunStage(config, monitor),
            "schism_postprocess": PostSCHISMStage(config, monitor),
            "schism_plot": SchismPlotStage(config, monitor),
        }

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        d: dict[str, Any] = {
            "prebuilt_dir": str(self.prebuilt_dir),
            "geogrid_file": str(self.geogrid_file) if self.geogrid_file else None,
            "nodes": self.nodes,
            "ntasks_per_node": self.ntasks_per_node,
            "exclusive": self.exclusive,
            "nscribes": self.nscribes,
            "omp_num_threads": self.omp_num_threads,
            "oversubscribe": self.oversubscribe,
            "schism_exe": (str(self.schism_exe) if self.schism_exe else None),
            "include_noaa_gages": self.include_noaa_gages,
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

        Tidal-only sources such as TPXO provide oscillations centered on
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
        cores on the current machine (see :func:`~coastal_calibration.utils.system.get_cpu_count`).
        On HPC nodes this auto-detects correctly; on a local laptop it
        avoids over-subscribing the system.
    inp_overrides : dict
        Arbitrary key/value pairs written to ``sfincs.inp`` just before the
        model is written to disk.  Use this to override physics parameters
        that HydroMT-SFINCS sets by default (e.g. ``advection: 0``,
        ``nuvisc: 0.01``).  Keys must be valid ``sfincs.inp`` parameter
        names.
    """

    # Known sfincs.inp parameter names parsed by the SFINCS binary
    # (extracted from SFINCS/source/src/sfincs_input.f90).  Used to
    # catch typos in ``inp_overrides`` early — SFINCS silently ignores
    # unrecognized parameters.
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
    inp_overrides: dict[str, Any] = field(default_factory=dict)
    floodmap_dem: Path | None = None
    floodmap_hmin: float = 0.05
    floodmap_enabled: bool = True
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
        self, env: dict[str, str], config: CoastalCalibConfig
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

        if self.discharge_locations_file and not self.discharge_locations_file.exists():
            errors.append(
                f"model_config.discharge_locations_file not found: {self.discharge_locations_file}"
            )

        if self.sfincs_exe and not self.sfincs_exe.exists():
            errors.append(f"model_config.sfincs_exe not found: {self.sfincs_exe}")

        if self.inp_overrides:
            unknown = sorted(set(self.inp_overrides) - self._KNOWN_INP_PARAMS)
            if unknown:
                errors.append(
                    f"Unrecognized sfincs.inp parameter(s): {', '.join(unknown)}. "
                    "SFINCS silently ignores unknown parameters — check for typos."
                )

        return errors

    def create_stages(  # noqa: D102
        self, config: CoastalCalibConfig, monitor: Any
    ) -> dict[str, Any]:
        from coastal_calibration.stages.download import DownloadStage
        from coastal_calibration.stages.sfincs_build import (
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
            "inp_overrides": self.inp_overrides,
            "floodmap_dem": (str(self.floodmap_dem) if self.floodmap_dem else None),
            "floodmap_hmin": self.floodmap_hmin,
            "floodmap_enabled": self.floodmap_enabled,
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
    # Backward compat: old templates may still reference ${slurm.user}.
    if "slurm.user" not in context:
        context["slurm.user"] = context["user"]
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
    _base_config: Path | None = field(default=None, repr=False)

    @property
    def model(self) -> str:
        """Model identifier string (convenience accessor)."""
        return self.model_config.model_name

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], base_config_path: Path | None = None
    ) -> CoastalCalibConfig:
        """Create config from a plain dictionary.

        Parameters
        ----------
        data : dict
            Configuration dictionary with the same structure as the YAML
            file (see :meth:`to_dict` for the expected keys).
        base_config_path : Path, optional
            Path to a base configuration file (for YAML inheritance).
            Only needed when the config was loaded via ``_base`` key.

        Returns
        -------
        CoastalCalibConfig
        """
        if "model" not in data:
            raise ValueError("'model' is required (e.g., model: schism or model: sfincs)")
        model_type: str = data["model"]

        model_config_data = data.pop("model_config", {}) or {}

        sim_data = data.get("simulation", {})
        if "start_date" in sim_data:
            sim_data["start_date"] = _parse_datetime(sim_data["start_date"])
        simulation = SimulationConfig(**sim_data)

        boundary_data = data.get("boundary", {})
        if boundary_data.get("stofs_file"):
            boundary_data["stofs_file"] = Path(boundary_data["stofs_file"])
        boundary = BoundaryConfig(**boundary_data)

        paths_data = data.get("paths", {})
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
            _base_config=base_config_path,
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

        base_config = None
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

        return cls.from_dict(data, base_config_path=config_path if base_config else None)

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
            },
            "paths": {
                "work_dir": str(self.paths.work_dir),
                "raw_download_dir": (
                    str(self.paths.raw_download_dir) if self.paths.raw_download_dir else None
                ),
                "hot_start_file": (
                    str(self.paths.hot_start_file) if self.paths.hot_start_file else None
                ),
                **({"parm_dir": str(self.paths.parm_dir)} if self.paths.parm_dir else {}),
                **({"nwm_dir": str(self.paths.nwm_dir)} if self.paths.nwm_dir else {}),
                **({"otps_dir": str(self.paths.otps_dir)} if self.paths.otps_dir else {}),
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

        # TPXO data directory is derived from paths.parm_dir
        elif (
            self.boundary.source == "tpxo"
            and self.paths.parm_dir is not None
            and not self.paths.tpxo_data_dir.exists()
        ):
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
