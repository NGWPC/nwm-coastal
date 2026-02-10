# coastal-calibration: Design Documentation

## Overview

The `coastal-calibration` Python package is a complete redesign and rewrite of the
original bash-based SCHISM model calibration workflow. This document details the
architectural improvements, design decisions, and substantial enhancements made over the
original implementation.

______________________________________________________________________

## Table of Contents

1. [Executive Summary](#executive-summary)
1. [Original Implementation Analysis](#original-implementation-analysis)
1. [New Architecture](#new-architecture)
1. [Key Design Decisions](#key-design-decisions)
1. [Substantial Improvements](#substantial-improvements)
1. [API Reference](#api-reference)
1. [Potential Future Developments](#potential-future-developments)

______________________________________________________________________

## Executive Summary

The `coastal-calibration` package provides a modern Python interface for running SCHISM
and SFINCS coastal model calibration workflows on HPC clusters. It wraps the existing
operational workflow scripts with a clean, type-safe API while establishing the
foundation for incremental improvements.

### Design Goals

The primary objectives of this rewrite are to create a workflow that is:

1. **Intuitive and user-friendly** - Simple YAML configuration, clear CLI commands,
    helpful error messages
1. **Less prone to errors** - Type-safe configuration, comprehensive validation,
    structured logging
1. **Extensible** - Polymorphic model architecture that supports SCHISM, SFINCS, and
    future models via a common `ModelConfig` ABC

### Architectural Strategy

The package is designed with a **stable public API** that shields users from internal
changes. This enables:

- **Immediate usability** - Users get a clean interface today, even while internals are
    being improved
- **Incremental rewriting** - Embedded bash scripts can be replaced with pure Python one
    stage at a time
- **Safe evolution** - Internal rewrites don't break user-facing code or configurations

The long-term goal is to **completely rewrite** all embedded bash scripts in Python, but
doing so incrementally allows the package to be useful immediately while that work
proceeds.

### Key Features

- **Type-safe configuration** via `dataclasses` with runtime validation
- **Modular stage-based architecture** for maintainability and extensibility
- **Native Python datetime handling** replacing fragile shell date arithmetic
- **Async data downloading** with built-in source validation
- **CLI and programmatic APIs** for both interactive and automated use
- **SLURM job management** with status monitoring
- **Progress tracking** and structured logging
- **Configuration inheritance** for DRY multi-run setups
- **Smart default paths** with variable interpolation

______________________________________________________________________

## Original Implementation Analysis

### File Structure (20+ scripts)

```console
calib_org/
├── sing_run.bash                     # Main entry point (258 lines)
├── schism_calib.cfg                  # Configuration file
├── pre_nwm_forcing_coastal.bash      # Forcing preparation
├── post_nwm_forcing_coastal.bash     # Forcing post-processing
├── make_tpxo_ocean.bash              # TPXO boundary conditions
├── pre_regrid_stofs.bash             # STOFS pre-processing
├── post_regrid_stofs.bash            # STOFS post-processing
├── update_param.bash                 # Parameter file updates (249 lines)
├── pre_schism.bash                   # SCHISM input preparation
├── post_schism.bash                  # SCHISM output processing
├── merge_source_sink.bash            # Discharge file merging
├── initial_discharge.bash            # Initial discharge creation
├── combine_sink_source.bash          # Sink/source combination
└── run_sing_coastal_workflow_*.bash  # 8+ Singularity wrappers
```

### Critical Issues in Original Implementation

#### 1. Fragile Date Arithmetic

The original workflow relied on external scripts for date calculations:

```bash
# Original: External script calls for every date operation
export FORCING_END_DATE=$(${USHnwm}/utils/advance_time.sh $PDY$cyc $LENGTH_HRS)'00'
pdycyc=$(${USHnwm}/utils/advance_time.sh $PDY$cyc $hr)
```

This approach had several problems:

- Required external `advance_time.sh` and `advance_cymdh.pl` scripts
- Shell spawning overhead for each date operation
- Inconsistent handling of edge cases (leap years, month boundaries)
- No error handling for invalid dates

#### 2. Environment Variable Pitfalls

The original scripts passed dozens of environment variables between scripts:

```bash
# Original configuration (schism_calib.cfg)
export STARTPDY=20230611
export STARTCYC=00
export FCST_LENGTH_HRS=3.0
export HOT_START_FILE=''
export USE_TPXO="NO"
export COASTAL_DOMAIN=pacific
export METEO_SOURCE=NWM_RETRO
export COASTAL_WORK_DIR=/efs/schism_use_case/...

# Plus 40+ more in sing_run.bash
export NGWPC_COASTAL_PARM_DIR=/ngen-test/coastal/ngwpc-coastal
export NGEN_APP_DIR=/ngen-app
export FCST_TIMESTEP_LENGTH_SECS=3600
export OTPSDIR=$NGEN_APP_DIR/OTPSnc
# ... etc
```

Problems:

- No validation of variable values
- Easy to have typos that fail silently
- Difficult to track variable dependencies
- No documentation of which variables are required vs optional

#### 3. String-Based Domain Mapping

```bash
# Original: Repeated in multiple files
declare -A coastal_domain_to_inland_domain=( \
    [prvi]="domain_puertorico" \
    [hawaii]="domain_hawaii" \
    [atlgulf]="domain" \
    [pacific]="domain" )

declare -A coastal_domain_to_nwm_domain=( \
    [prvi]="prvi" \
    [hawaii]="hawaii" \
    [atlgulf]="conus" \
    [pacific]="conus" )

declare -A coastal_domain_to_geo_grid=( \
    [prvi]="geo_em_PRVI.nc" \
    [hawaii]="geo_em_HI.nc" \
    [atlgulf]="geo_em_CONUS.nc" \
    [pacific]="geo_em_CONUS.nc" )
```

Problems:

- Duplicated across multiple scripts
- No compile-time type checking
- Silent failures on unknown domains

#### 4. No Data Download Integration

The original workflow required manual data downloading via a separate workflow. That
workflow had no date validation, no source awareness, and no progress tracking.

#### 5. Minimal Error Handling

```bash
# Original: Scripts would continue on failure
singularity exec -B $BINDINGS --pwd ${work_dir} $SIF_PATH \
    ./run_sing_coastal_workflow_pre_forcing_coastal.bash
# No error check here

${MPICOMMAND3} singularity exec -B $BINDINGS \
    --pwd ${work_dir} \
    $SIF_PATH \
    $CONDA_ENVS_PATH/$CONDA_ENV_NAME/bin/python \
    $USHnwm/wrf_hydro_workflow_dev/forcings/WrfHydroFECPP/workflow_driver.py
# No error check here either
```

______________________________________________________________________

## New Architecture

### Package Structure

```console
src/coastal_calibration/
├── __init__.py                  # Package exports
├── cli.py                       # Command-line interface
├── runner.py                    # Main workflow orchestrator
├── downloader.py                # Async data downloading
├── scripts_path.py              # Script path management
│
├── config/
│   ├── __init__.py
│   └── schema.py                # YAML config dataclasses + ModelConfig ABC
│
├── stages/                      # Workflow stages
│   ├── __init__.py
│   ├── base.py                  # Abstract WorkflowStage base class
│   ├── download.py              # Data download stage
│   ├── forcing.py               # NWM forcing stages
│   ├── boundary.py              # Boundary condition stages
│   ├── schism.py                # SCHISM execution stages
│   ├── sfincs.py                # SFINCS data catalog & symlinks
│   └── sfincs_build.py          # SFINCS model build stages (HydroMT)
│
├── scripts/                     # Embedded bash scripts
│   ├── tpxo_to_open_bnds_hgrid/ # TPXO Python utilities
│   └── wrf_hydro_workflow_dev/  # WRF-Hydro forcing code
│
└── utils/
    ├── __init__.py
    ├── logging.py               # Workflow monitoring
    ├── slurm.py                 # SLURM job management
    ├── time.py                  # Datetime utilities
    └── workflow.py              # Workflow helper functions
```

### Core Components

#### 1. Configuration System (`config/schema.py`)

The new configuration system uses Python `dataclasses` with full type hints:

```python
from dataclasses import dataclass
from typing import Literal

CoastalDomain = Literal["prvi", "hawaii", "atlgulf", "pacific"]
MeteoSource = Literal["nwm_retro", "nwm_ana"]
BoundarySource = Literal["tpxo", "stofs"]


@dataclass
class SimulationConfig:
    """Simulation time and domain configuration."""

    start_date: datetime
    duration_hours: int
    coastal_domain: CoastalDomain
    meteo_source: MeteoSource
    timestep_seconds: int = 3600

    # Domain mappings as class variables
    _INLAND_DOMAIN: ClassVar[dict[str, str]] = {
        "prvi": "domain_puertorico",
        "hawaii": "domain_hawaii",
        "atlgulf": "domain",
        "pacific": "domain",
    }

    @property
    def start_pdy(self) -> str:
        """Return start date as YYYYMMDD string."""
        return self.start_date.strftime("%Y%m%d")

    @property
    def inland_domain(self) -> str:
        """Inland domain directory name for this coastal domain."""
        return self._INLAND_DOMAIN[self.coastal_domain]
```

Benefits:

- **Type safety**: IDE autocompletion, static analysis with `pyright`
- **Self-documenting**: Property names and docstrings explain purpose
- **Validation**: Runtime checks with helpful error messages
- **DRY**: Domain mappings defined once

#### 2. YAML Configuration with Inheritance

```yaml
# base.yaml - Shared defaults
slurm:
  partition: c5n-18xlarge

paths:
  nfs_mount: /ngen-test
  singularity_image: /ngencerf-app/singularity/ngen-coastal.sif

---
# hawaii_run.yaml - Inherits from base
_base: base.yaml

simulation:
  start_date: '2023-06-11T00:00:00'
  duration_hours: 24
  coastal_domain: hawaii
  meteo_source: nwm_retro

paths:
  work_dir: /ngen-test/coastal_runs/${simulation.coastal_domain}
```

Features:

- **Variable interpolation**: `${section.key}` syntax
- **Inheritance**: `_base` field for configuration reuse
- **Deep merging**: Override only what changes
- **Smart defaults**: Minimal configuration required

When paths are not specified, they are automatically generated using templates that
include the `${model}` variable for model-aware directory naming:

```python
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
```

```mermaid
flowchart TD
    base[base.yaml] --> hawaii[hawaii_run.yaml]
    base --> pacific[pacific_run.yaml]
    base --> prvi[prvi_run.yaml]
```

#### 3. Stage-Based Workflow Architecture

The stage pipeline is model-specific. Each `ModelConfig` subclass defines its own
`stage_order` and `create_stages()`.

**SCHISM pipeline:**

```mermaid
flowchart TD
    A[download] --> B[pre_forcing]
    B --> C[nwm_forcing]
    C --> D[post_forcing]
    D --> E[update_params]
    E --> F[boundary_conditions]
    F --> G[pre_schism]
    G --> H[schism_run]
    H --> I[post_schism]
```

**SFINCS pipeline:**

```mermaid
flowchart TD
    A[download] --> B[sfincs_symlinks]
    B --> C[sfincs_data_catalog]
    C --> D[sfincs_init]
    D --> E[sfincs_timing]
    E --> F[sfincs_forcing]
    F --> G[sfincs_obs]
    G --> H[sfincs_discharge]
    H --> I[sfincs_precip]
    I --> J[sfincs_write]
    J --> K[sfincs_run]
```

Each stage is a Python class inheriting from `WorkflowStage`:

```mermaid
classDiagram
    class WorkflowStage {
        <<abstract>>
        +run() dict
        +validate() list
    }
    WorkflowStage <|-- DownloadStage
    WorkflowStage <|-- ForcingStage
    WorkflowStage <|-- BoundaryStage
    WorkflowStage <|-- SCHISMStage
    WorkflowStage <|-- SFINCSBuildStage
```

The base class implementation:

```python
class WorkflowStage(ABC):
    """Abstract base class for workflow stages."""

    name: str = "base"
    description: str = "Base workflow stage"

    def __init__(self, config: CoastalCalibConfig, monitor: WorkflowMonitor | None):
        self.config = config
        self.monitor = monitor

    def build_environment(self) -> dict[str, str]:
        """Build environment variables for the stage."""
        # Converts config to env vars for bash scripts
        env = os.environ.copy()
        env["STARTPDY"] = self.config.simulation.start_pdy
        env["STARTCYC"] = self.config.simulation.start_cyc
        # ... all precomputed, no shell date arithmetic needed
        return env

    def run_singularity_command(
        self,
        command: list[str],
        use_mpi: bool = False,
        mpi_tasks: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command inside the Singularity container."""
        # Handles all Singularity setup, bindings, error checking
        pass

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the stage and return results."""
        pass

    def validate(self) -> list[str]:
        """Validate stage prerequisites. Return list of errors."""
        return []
```

#### 4. Workflow Runner Orchestration

```python
class CoastalCalibRunner:
    """Main workflow runner for coastal model calibration."""

    @property
    def STAGE_ORDER(self) -> list[str]:
        """Stage order is delegated to the model config."""
        return self.config.model_config.stage_order

    def run(
        self,
        start_from: str | None = None,
        stop_after: str | None = None,
        dry_run: bool = False,
    ) -> WorkflowResult:
        """Execute the calibration workflow."""
        # Validation, stage sequencing, error handling, result collection
        pass

    def submit(self, wait: bool = False) -> WorkflowResult:
        """Submit workflow as a SLURM job.

        Parameters
        ----------
        wait : bool
            If True, wait for job completion with status updates.
            If False (default), return immediately after submission.
        """
        pass
```

The `submit()` method execution flow is shown in the sequence diagram below:

```mermaid
sequenceDiagram
    participant User
    participant Runner
    participant Slurm

    User->>Runner: submit(config)
    Runner->>Runner: validate()
    Runner->>Slurm: submit_job()
    Slurm-->>Runner: job_id
    Runner-->>User: WorkflowResult
```

______________________________________________________________________

## Key Design Decisions

### 1. Python-Native Date Arithmetic

**Decision**: Replace all bash/Perl date scripts with Python `datetime`.

**Rationale**:

- Python's `datetime` and `timedelta` handle all edge cases correctly
- No external dependencies or shell spawning
- Type-safe with IDE support

**Implementation** (`utils/time.py`):

```python
_DATE_RE = re.compile(r"^\d{10}$")


def _parse_date(date_string: str) -> datetime:
    """Parse a YYYYMMDDHH string into a datetime, with strict validation."""
    if not isinstance(date_string, str) or not _DATE_RE.match(date_string):
        raise ValueError(
            f"date_string must be exactly 10 digits in YYYYMMDDHH format, got {date_string!r}"
        )
    return datetime.strptime(date_string, "%Y%m%d%H")


def advance_time(date_string: str, hours: int) -> str:
    """Advance a date string by a specified number of hours.

    Replaces advance_time.sh and advance_cymdh.pl with native Python.
    Handles leap years, month boundaries, DST, etc.
    """
    dt = _parse_date(date_string) + timedelta(hours=hours)
    return dt.strftime("%Y%m%d%H")
```

The module also consolidates `parse_datetime()` (flexible datetime parsing, previously
duplicated in `config.schema` and `downloader`) and `iter_hours()` (hour-range
iteration, previously in `downloader`).

**Impact**: The `build_environment()` method precomputes shared date-derived values,
then delegates model-specific env vars to `model_config.build_environment()`:

```python
# Shared dates computed once in Python, passed to bash scripts
env["FORCING_BEGIN_DATE"] = f"{pdycyc}00"
env["FORCING_END_DATE"] = forcing_end_dt.strftime("%Y%m%d%H00")
env["END_DATETIME"] = forcing_end_dt.strftime("%Y%m%d%H")

# Model-specific env vars (e.g., SCHISM_BEGIN_DATE, OMP_NUM_THREADS)
env = self.config.model_config.build_environment(env, self.config)
```

### 2. Integrated Data Downloading with Validation

**Decision**: Build a comprehensive downloader with source awareness and date range
validation.

**Rationale**:

- Different data sources have different availability windows
- Users shouldn't waste time on downloads that will fail
- Async downloading is faster than sequential

**Implementation** (`downloader.py`):

```python
DATA_SOURCE_DATE_RANGES: dict[str, dict[str, DateRange]] = {
    "nwm_retro": {
        "conus": DateRange(
            start=datetime(1979, 2, 1),
            end=datetime(2023, 1, 31),
            description="NWM Retrospective 3.0 (CONUS)",
        ),
        "hawaii": DateRange(
            start=datetime(1994, 1, 1),
            end=datetime(2013, 12, 31),
            description="NWM Retrospective 3.0 (Hawaii)",
        ),
        # ...
    },
    "stofs": {
        "_default": DateRange(
            start=datetime(2020, 12, 30),
            end=None,  # operational, no end date
            description="STOFS (operational)",
        ),
    },
}


def download_data(
    start_time: datetime,
    end_time: datetime,
    output_dir: Path,
    domain: Domain,
    meteo_source: MeteoSource = "nwm_retro",
    coastal_source: CoastalSource = "stofs",
) -> DownloadResults:
    """Download with validation and progress tracking."""
    # Validates dates before downloading
    errors = _validate_date_ranges(start, end, meteo_source, coastal_source, domain)
    if errors:
        raise ValueError("Date range validation failed:\n" + "\n".join(errors))

    # Uses tiny_retriever for async parallel downloads
    download(urls, paths, timeout=timeout)
```

### 3. Configuration Over Convention

**Decision**: Use explicit YAML configuration with sensible defaults.

**Rationale**:

- Original relied on implicit conventions (file locations, naming patterns)
- Explicit configuration is self-documenting
- Easier to version control and share

**Example SCHISM configuration**:

```yaml
slurm:
  job_name: coastal_calibration
  partition: c5n-18xlarge

simulation:
  start_date: '2023-06-11T00:00:00'
  duration_hours: 24
  coastal_domain: pacific
  meteo_source: nwm_retro

boundary:
  source: tpxo  # or: source: stofs

paths:
  work_dir: /ngen-test/coastal_runs/my_run
  raw_download_dir: /ngen-test/data/downloads

# SCHISM compute parameters (model_config defaults to SchismModelConfig)
model_config:
  nodes: 2
  ntasks_per_node: 18
  nscribes: 2
  omp_num_threads: 2

download:
  enabled: true
  skip_existing: true
```

**Example SFINCS configuration**:

```yaml
model: sfincs

slurm:
  job_name: sfincs_texas
  user: your_username

simulation:
  start_date: 2025-06-01
  duration_hours: 168
  coastal_domain: atlgulf
  meteo_source: nwm_ana

boundary:
  source: stofs

model_config:
  prebuilt_dir: /path/to/texas/model
  omp_num_threads: 36

download:
  enabled: true
  skip_existing: true
```

### 4. Non-Interactive Default with Interactive Flag

**Decision**: The `submit` command returns immediately by default, with an optional
`--interactive` (`-i`) flag to wait for completion.

**Rationale**:

- Matches standard `sbatch` behavior that users expect
- Allows users to submit jobs and continue working
- Interactive mode available when monitoring is desired

**CLI behavior**:

```bash
# Default: Submit and return immediately (like sbatch)
coastal-calibration submit config.yaml

# Interactive: Wait for completion with status updates
coastal-calibration submit config.yaml --interactive
coastal-calibration submit config.yaml -i
```

### 5. Stable Public API with Incremental Internal Rewrite

**Decision**: Establish a clean, stable public API while embedding existing scripts as a
transitional measure.

**Rationale**:

The primary goal of this rewrite is to create an **intuitive, user-friendly, and
extensible** workflow system. The existing bash and Python scripts are difficult to
maintain and not performant. However, rewriting everything at once would:

- Delay delivery of a usable tool to users
- Risk introducing regressions without a baseline
- Require extensive testing before any release

**Strategy**:

The architecture deliberately separates **public API** from **private implementation**:

| Layer                      | Components                                                   | Stability |
| -------------------------- | ------------------------------------------------------------ | --------- |
| **Public API**             | `CoastalCalibConfig`, `CoastalCalibRunner`, CLI              | Stable    |
| **Stage Interface**        | `WorkflowStage.run()`, `.validate()`, `.build_environment()` | Stable    |
| **Private Implementation** | Bash scripts → Pure Python                                   | Evolving  |

This allows:

1. **Users get a stable interface today** - The CLI and Python API won't change as
    internals evolve
1. **Incremental rewriting** - Each stage can be rewritten independently without
    affecting others
1. **Testing baseline first** - Establish test coverage against current behavior before
    changes
1. **Performance optimization** - Replace bash subprocess calls with native Python as
    needed

**Current State**:

- Package includes `scripts/` directory with embedded bash scripts
- `WorkflowStage.run_singularity_command()` provides abstraction layer
- Python precomputes all environment variables, minimizing bash complexity

**Future Direction**:

1. Add comprehensive integration tests capturing current behavior
1. Incrementally rewrite stages in pure Python (starting with simpler stages)
1. Deprecate bash scripts as Python replacements are validated
1. Optimize performance-critical paths (file I/O, data processing)

### 6. Strict Type Checking with `pyright`

**Decision**: Use strict `pyright` mode for static type analysis.

**Rationale**:

- Catches errors before runtime
- Enables IDE features (autocomplete, refactoring)
- Self-documents function signatures

**Configuration** (`pyproject.toml`):

```toml
[tool.pyright]
typeCheckingMode = "strict"
include = ["src/coastal_calibration"]
```

______________________________________________________________________

## Substantial Improvements

### 1. Error Handling and Validation

| Aspect                   | Original        | New                                            |
| ------------------------ | --------------- | ---------------------------------------------- |
| Configuration validation | None            | 12+ checks in `CoastalCalibConfig.validate()`  |
| Stage validation         | None            | Each stage has `validate()` method             |
| Error messages           | Exit codes only | Detailed, actionable messages                  |
| Recovery                 | Manual restart  | Partial workflow execution with `--start-from` |

**Validation examples**:

```python
def validate(self) -> list[str]:
    errors = []

    # Shared validation
    if self.simulation.duration_hours <= 0:
        errors.append("simulation.duration_hours must be positive")

    if (
        self.boundary.source == "stofs"
        and not self.boundary.stofs_file
        and not self.download.enabled
    ):
        errors.append(
            "boundary.stofs_file required when using STOFS source and download is disabled"
        )

    # Model-specific validation (delegated to ModelConfig subclass)
    errors.extend(self.model_config.validate(self))

    return errors
```

### 2. Progress Tracking and Monitoring

**Original**: No progress tracking, just log messages scattered in bash scripts.

**New**: Structured monitoring with stage context:

```python
class WorkflowMonitor:
    """Monitors and logs workflow execution progress."""

    def register_stages(self, stages: list[str]) -> None:
        """Register stages for progress tracking."""

    @contextmanager
    def stage_context(self, stage_name: str, description: str):
        """Context manager for stage execution with timing."""
        self.info(f"Starting stage: {stage_name} - {description}")
        start = time.perf_counter()
        try:
            yield
            duration = time.perf_counter() - start
            self.info(f"Completed stage: {stage_name} in {duration:.1f}s")
            self.progress[stage_name] = "completed"
        except Exception as e:
            self.progress[stage_name] = "failed"
            raise

    def save_progress(self, path: Path) -> None:
        """Save progress to JSON for resumption."""
```

### 3. SLURM Integration

**Original**: Manual SLURM script writing, no job tracking.

**New**: Full `SlurmManager` class:

```python
class SlurmManager:
    """Manage SLURM job submission and monitoring."""

    def submit_job(self, script_path: Path) -> str:
        """Submit and return job ID."""

    def get_job_status(self, job_id: str) -> JobStatus:
        """Query job status from sacct/squeue."""

    def wait_for_job(self, job_id: str, poll_interval: int = 30) -> JobStatus:
        """Block until job completes, logging state transitions."""

    def generate_job_script(self, output_path: Path) -> Path:
        """Generate SLURM script from configuration."""
```

### 4. CLI with Multiple Entry Points

```bash
# Initialize configuration for a domain
coastal-calibration init config.yaml --domain hawaii

# Validate configuration
coastal-calibration validate config.yaml

# Run directly (for testing)
coastal-calibration run config.yaml --dry-run

# Submit to SLURM cluster
coastal-calibration submit config.yaml

# Run partial workflow
coastal-calibration run config.yaml --start-from update_params --stop-after boundary_conditions

# List available stages
coastal-calibration stages
```

### 5. Dual API: CLI and Programmatic

```python
# Python API
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

config = CoastalCalibConfig.from_yaml("config.yaml")
runner = CoastalCalibRunner(config)

# Validate first
errors = runner.validate()
if errors:
    print("Validation failed:", errors)
else:
    result = runner.submit()
    print(f"Job {result.job_id}: {result.success}")
```

### 6. Comprehensive Downloader

| Feature           | Original       | New                               |
| ----------------- | -------------- | --------------------------------- |
| Data sources      | Manual AWS CLI | NWM Retro, NWM Ana, STOFS, GLOFS  |
| Date validation   | None           | Checks against known availability |
| Parallel download | None           | Async with `tiny_retriever`       |
| Skip existing     | None           | `skip_existing=True` option       |
| Progress tracking | None           | Success/failure counts            |
| Domain awareness  | Manual         | Automatic URL building            |

### 7. Results Serialization

```python
@dataclass
class WorkflowResult:
    success: bool
    job_id: str | None
    start_time: datetime
    end_time: datetime | None
    stages_completed: list[str]
    stages_failed: list[str]
    outputs: dict[str, Any]
    errors: list[str]

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def save(self, path: Path) -> None:
        """Save result to JSON for post-processing."""
```

______________________________________________________________________

## API Reference

### Configuration Classes

| Class                | Purpose                                        |
| -------------------- | ---------------------------------------------- |
| `CoastalCalibConfig` | Root configuration container                   |
| `SlurmConfig`        | SLURM scheduling parameters                    |
| `SimulationConfig`   | Time, domain, and source settings              |
| `BoundaryConfig`     | TPXO vs STOFS selection                        |
| `PathConfig`         | All file and directory paths                   |
| `ModelConfig`        | ABC for model-specific configuration           |
| `SchismModelConfig`  | SCHISM compute, MPI, and stage settings        |
| `SfincsModelConfig`  | SFINCS model paths, OpenMP, and stage settings |
| `MonitoringConfig`   | Logging and progress tracking                  |
| `DownloadConfig`     | Data download settings                         |

### SCHISM Workflow Stages

| Stage                 | Class                    | Description                              |
| --------------------- | ------------------------ | ---------------------------------------- |
| `download`            | `DownloadStage`          | Download NWM/STOFS/GLOFS data            |
| `pre_forcing`         | `PreForcingStage`        | Prepare forcing directories and symlinks |
| `nwm_forcing`         | `NWMForcingStage`        | Run WRF-Hydro forcing engine (MPI)       |
| `post_forcing`        | `PostForcingStage`       | Post-process forcing files               |
| `update_params`       | `UpdateParamsStage`      | Generate SCHISM `param.nml`              |
| `boundary_conditions` | `BoundaryConditionStage` | TPXO or STOFS boundary generation        |
| `pre_schism`          | `PreSCHISMStage`         | Prepare SCHISM inputs                    |
| `schism_run`          | `SCHISMRunStage`         | Execute `pschism` binary (MPI)           |
| `post_schism`         | `PostSCHISMStage`        | Validate and post-process outputs        |

### SFINCS Workflow Stages

| Stage                 | Class                    | Description                        |
| --------------------- | ------------------------ | ---------------------------------- |
| `download`            | `DownloadStage`          | Download NWM/STOFS data            |
| `sfincs_symlinks`     | `SFINCSSymlinksStage`    | Create `.nc` symlinks for NWM data |
| `sfincs_data_catalog` | `SFINCSDataCatalogStage` | Generate HydroMT data catalog      |
| `sfincs_init`         | `SFINCSInitStage`        | Initialize SFINCS model            |
| `sfincs_timing`       | `SFINCSTimingStage`      | Set SFINCS timing                  |
| `sfincs_forcing`      | `SFINCSForcingStage`     | Add water level forcing            |
| `sfincs_obs`          | `SFINCSObsStage`         | Add observation points             |
| `sfincs_discharge`    | `SFINCSDischargeStage`   | Add discharge sources              |
| `sfincs_precip`       | `SFINCSPrecipStage`      | Add precipitation forcing          |
| `sfincs_write`        | `SFINCSWriteStage`       | Write SFINCS model                 |
| `sfincs_run`          | `SFINCSRunStage`         | Run SFINCS (Singularity/OpenMP)    |

______________________________________________________________________

## Potential Future Developments

### Near-Term: Complete Python Rewrite

The highest priority is incrementally replacing embedded bash scripts with pure Python
implementations:

1. **Establish Testing Baseline**

    - Add integration tests that capture current workflow behavior
    - Create reference outputs for regression testing
    - Measure performance benchmarks for comparison

1. **Incremental Stage Rewriting**

    - Start with simpler stages (`pre_forcing`, `post_forcing`, `update_params`)
    - Replace bash file operations with Python `pathlib` and `shutil`
    - Convert `sed`-based `param.nml` editing to Python template/parsing
    - Rewrite forcing symlink creation in native Python

1. **Performance Optimization**

    - Profile current workflow to identify bottlenecks
    - Replace subprocess calls with direct Python implementations
    - Optimize file I/O patterns (batch operations, memory mapping)
    - Consider parallel processing for independent operations

1. **Deprecate Bash Scripts**

    - Once Python replacements are validated, remove bash dependencies
    - Simplify Singularity container requirements
    - Reduce external tool dependencies

### Medium-Term: Feature Expansion

1. **Hot Start Chain Automation**

    - Automatic hot-start file discovery
    - Multi-run chaining for long simulations

1. **Ensemble Runs**

    - Multiple configurations from single base
    - Parallel SLURM array jobs

### Long-Term: Platform Evolution

1. **Result Analysis Integration**

    - Post-run validation against observations
    - Time series extraction and plotting

1. **Cloud-Native Deployment**

    - AWS Batch support
    - Container-native execution (no Singularity)

1. **Multi-Model Coupling**

    - SCHISM + SFINCS workflows
    - Nesting support

______________________________________________________________________

## Conclusion

The `coastal-calibration` package represents a substantial modernization of the original
bash-based workflow:

| Metric          | Original              | New                     | Improvement      |
| --------------- | --------------------- | ----------------------- | ---------------- |
| Lines of bash   | ~2,500                | ~500 (embedded)         | 80% reduction    |
| Lines of Python | ~200 (scattered)      | ~4,000 (structured)     | Full rewrite     |
| Configuration   | Environment variables | Typed YAML              | Type-safe        |
| Error handling  | Exit codes            | Exceptions + validation | Comprehensive    |
| Testing         | None                  | `pytest` + `pyright`    | CI-ready         |
| Documentation   | Comments only         | Docstrings + types      | Self-documenting |
| Extensibility   | Copy & modify scripts | Inherit `WorkflowStage` | Object-oriented  |
| Model support   | SCHISM only           | SCHISM + SFINCS         | Polymorphic      |

The architecture is designed for maintainability, extensibility, and correctness while
supporting multiple coastal models (SCHISM and SFINCS) through a polymorphic
`ModelConfig` ABC and preserving compatibility with the existing HPC infrastructure.
