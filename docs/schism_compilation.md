# SCHISM De-containerization: Pixi Native Build & Execution

**Goal:** Migrate SCHISM compilation and execution from Singularity containers to
pixi-managed native builds, following the same pattern used for SFINCS.

**Status:** Complete. Validated end-to-end on macOS arm64 with the Hawaii domain.
Codebase cleanup (Phase 5) completed: unified logging, env-var elimination, Singularity
removal, stage renaming to match SFINCS conventions.

**Last updated:** 2026-03-19

______________________________________________________________________

## Table of contents

1. [Architecture overview](#architecture-overview)
1. [Build system](#build-system)
1. [SCHISM run workflow](#schism-run-workflow)
1. [Stage inventory](#stage-inventory)
1. [New Python modules](#new-python-modules)
1. [Logging](#logging)
1. [Environment variables](#environment-variables)
1. [ESMF regridding modules](#esmf-regridding-modules)
1. [ESMF and esmpy compatibility](#esmf-and-esmpy-compatibility)
1. [Parallel netCDF4 and MPI](#parallel-netcdf4-and-mpi)
1. [SCHISM version compatibility](#schism-version-compatibility)
1. [Hawaii example notebook](#hawaii-example-notebook)
1. [Known issues and workarounds](#known-issues-and-workarounds)
1. [Decision log](#decision-log)
1. [File reference](#file-reference)
1. [Cleanup changelog (Phase 5)](#cleanup-changelog-phase-5)

______________________________________________________________________

## Architecture overview

### Before (Singularity)

```
CoastalCalibRunner
  └─ WorkflowStage.run_singularity_command()
       └─ singularity exec -B ... ngen-coastal.sif bash run_sing_*.bash
            └─ pre_schism.bash / post_schism.bash / ...
                 └─ ${EXECnwm}/pschism_wcoss2_NO_PARMETIS_TVD-VL.openmpi
```

Every stage that needed SCHISM binaries or MPI wrapped its command inside
`singularity exec` with bind-mount paths to NFS volumes. The container included the
SCHISM binary, METIS tools, and the full conda environment.

### After (pixi native)

```
CoastalCalibRunner
  └─ WorkflowStage.run()
       ├─ Direct Python calls (no subprocess)
       │    schism_prep.py    → make_sflux(), correct_elevation(), ...
       │    sflux.py          → make_atmo_sflux()
       │    tides/             → make_otps_input(), otps_to_open_bnds(), generate_ocean_tide()
       ├─ MPI modules via mpiexec -m (ESMF/MPI-dependent)
       │    regridding/regrid_estofs.py    → ESTOFS boundary regridding
       │    regridding/regrid_forcings.py  → NWM atmospheric forcing regridding
       └─ Fortran binaries via subprocess
            pschism, combine_sink_source, combine_hotstart7,
            metis_prep, gpmetis, predict_tide
```

All Singularity invocations and Python-script subprocess calls have been replaced by
either:

- **Direct Python function calls**: for all logic that was previously in bash scripts or
    standalone Python scripts (makeAtmo.py, correct_elevation.py, TPXO scripts,
    makeOceanTide.py, etc.). Functions take explicit parameters; no environment variable
    reading.
- **`mpiexec -m <module>`**: for ESMF/MPI-dependent regridding (regrid_estofs,
    regrid_forcings). Invoked as proper Python modules with CLI arguments (no env-var
    passing).
- **Subprocess for Fortran binaries**: `pschism`, `metis_prep`, `gpmetis`,
    `combine_sink_source`, `combine_hotstart7`, `predict_tide`.

The legacy bash and Python scripts have been moved to `tests/legacy_scripts/` as
reference implementations for regression testing. The `scripts_path.py` module has been
removed.

Binaries are compiled from source by the `coastal_models/schism/` pixi-build package
(rattler-build backend) and installed into the active pixi environment.

______________________________________________________________________

## Build system

### Binaries installed to `$CONDA_PREFIX/bin/`

| Binary                | Source                                                                                           | Build method                          |
| --------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------- |
| `pschism`             | `coastal_models/schism/repo/src/Driver/schism_driver.F90`                                        | CMake (`-DBLD_STANDALONE=ON`)         |
| `combine_hotstart7`   | `coastal_models/schism/repo/src/Utility/Combining_Scripts/`                                      | CMake (`BUILD_TOOLS=ON`)              |
| `combine_sink_source` | `coastal_models/schism/repo/src/Utility/Pre-Processing/NWM/NWM_coupling/combine_sink_source.F90` | Standalone `gfortran`                 |
| `metis_prep`          | `coastal_models/schism/repo/src/Utility/Grid_Scripts/metis_prep.f90`                             | Standalone `gfortran`                 |
| `gpmetis`             | `coastal_models/schism/repo/src/metis-5.1.0/`                                                    | `make config && make && make install` |

### CMake flags

| Flag                     | Value   | Reason                                               |
| ------------------------ | ------- | ---------------------------------------------------- |
| `NO_PARMETIS`            | ON      | NWM config uses external METIS, not ParMETIS         |
| `TVD_LIM`                | VL      | Van Leer limiter, matches NWM production config      |
| `BLD_STANDALONE`         | ON      | Standalone build (not coupled to ESMF)               |
| `BUILD_TOOLS`            | ON      | Builds utility targets including `combine_hotstart7` |
| `OLDIO`                  | OFF     | New I/O layer; requires `nscribes >= 2` at runtime   |
| `CMAKE_Fortran_COMPILER` | mpifort | MPI Fortran wrapper from pixi openmpi                |
| `CMAKE_C_COMPILER`       | mpicc   | MPI C wrapper                                        |

### Pixi-build package: `coastal_models/schism/`

SCHISM is built as a **pixi-build package** using the `rattler-build` backend. The
recipe lives at `coastal_models/schism/` with three files:

| File          | Purpose                                                                                 |
| ------------- | --------------------------------------------------------------------------------------- |
| `pixi.toml`   | Package metadata and build backend declaration                                          |
| `recipe.yaml` | Conda recipe (source path, build/host/run dependencies, tests)                          |
| `build.sh`    | Build script (CMake for pschism, standalone gfortran utilities, METIS, macOS SDK probe) |

The recipe declares MPI-linked `hdf5` and `netcdf-fortran` as host dependencies
(`* mpi_openmpi_*` build string) so the compiled binaries link against the same
MPI-enabled libraries used by ESMF/esmpy at runtime.

On first `pixi install`, the package is built and cached as a `.conda` archive.
Subsequent runs reuse the cache (~0.3 s) unless the submodule source or recipe changes.

The `build.sh` includes:

- **macOS SDK probe** (`scripts/find_compatible_sdk.sh`): dynamically tests each
    installed SDK and picks the newest one compatible with conda-forge's linker
- **`--preprocess` fix**: on macOS, SCHISM's CMake detects GNU+Clang and sets
    `C_PREPROCESS_FLAG=--preprocess`, but gfortran doesn't understand this flag. The
    build script patches `flags.make` to replace `--preprocess` with `-cpp`

### Pixi feature: `[tool.pixi.feature.schism]`

```toml
[tool.pixi.feature.schism.dependencies]
schism = { path = "coastal_models/schism" }
# Force MPI (OpenMPI) build variants so ESMF/esmpy runs with parallel support.
mpi = { version = "*", build = "openmpi" }
openmpi = "*"
esmf = { version = "*", build = "mpi_openmpi_*" }
esmpy = "*"
mpi4py = "*"
```

Build-time dependencies (compilers, cmake, ninja) are declared in `recipe.yaml` and
resolved automatically by the rattler-build backend.

______________________________________________________________________

## SCHISM run workflow

The SCHISM run workflow follows the same pattern as the SFINCS run workflow: it is given
a **prebuilt model directory** and executes the simulation. A separate create workflow
(future work) will handle model setup from scratch.

The workflow expects a prebuilt model directory containing at minimum:

- `hgrid.gr3` / `hgrid.ll`: horizontal grid
- `vgrid.in`: vertical grid
- `param.nml.template`: parameter template
- `drag.gr3`, `manning.gr3`, etc.: ancillary grid files

All SCHISM stage names are prefixed with `schism_` and follow the same concise naming
convention as SFINCS stages (`sfincs_run`, `sfincs_forcing`, `sfincs_plot`, etc.).

### SFINCS stage naming (reference pattern)

| Stage name            | Class                    | Description               |
| --------------------- | ------------------------ | ------------------------- |
| `sfincs_symlinks`     | SfincsSymlinksStage      | Create NC symlinks        |
| `sfincs_data_catalog` | SfincsDataCatalogStage   | Generate HydroMT catalog  |
| `sfincs_init`         | SfincsInitStage          | Initialize SFINCS model   |
| `sfincs_timing`       | SfincsTimingStage        | Set timing parameters     |
| `sfincs_forcing`      | SfincsForcingStage       | Apply boundary forcing    |
| `sfincs_discharge`    | SfincsDischargeStage     | Process discharge sources |
| `sfincs_precip`       | SfincsPrecipitationStage | Add precipitation         |
| `sfincs_wind`         | SfincsWindStage          | Add wind forcing          |
| `sfincs_pressure`     | SfincsPressureStage      | Add pressure forcing      |
| `sfincs_write`        | SfincsWriteStage         | Write model files         |
| `sfincs_run`          | SfincsRunStage           | Execute SFINCS binary     |
| `sfincs_floodmap`     | SfincsFloodMapStage      | Generate flood maps       |
| `sfincs_plot`         | SfincsPlotStage          | Plot results              |

### SCHISM stage naming (current)

| Stage name            | Class                  | File          | Description                                  |
| --------------------- | ---------------------- | ------------- | -------------------------------------------- |
| `download`            | DownloadStage          | `base.py`     | Download input data                          |
| `schism_forcing_prep` | PreForcingStage        | `forcing.py`  | Stage LDASIN files                           |
| `schism_forcing`      | NWMForcingStage        | `forcing.py`  | Regrid NWM forcing via MPI                   |
| `schism_sflux`        | PostForcingStage       | `forcing.py`  | Generate sflux atmospheric files             |
| `schism_params`       | UpdateParamsStage      | `boundary.py` | Create param.nml, symlink mesh               |
| `schism_obs`          | SchismObservationStage | `schism.py`   | Query NOAA stations, write station.in        |
| `schism_boundary`     | BoundaryConditionStage | `boundary.py` | Generate boundary conditions (STOFS or TPXO) |
| `schism_prep`         | PreSCHISMStage         | `schism.py`   | Discharge, sink/source, mesh partition       |
| `schism_run`          | SCHISMRunStage         | `schism.py`   | Execute SCHISM with MPI                      |
| `schism_postprocess`  | PostSCHISMStage        | `schism.py`   | Combine hotstart files                       |
| `schism_plot`         | SchismPlotStage        | `schism.py`   | Plot sim vs observed water levels            |

______________________________________________________________________

## Stage inventory

### Stage details

**`schism_forcing_prep`** (PreForcingStage):

- Calls `schism_prep.stage_ldasin_files()` (pure Python)
- Creates `forcing_input/` and `forcing_output/` directories

**`schism_forcing`** (NWMForcingStage):

- Runs `mpiexec -m coastal_calibration.regridding.regrid_forcings`
- All parameters passed via CLI arguments (`--job-index`, `--job-count` for HPC
    job-array parallelism)

**`schism_sflux`** (PostForcingStage):

- Calls `schism_prep.make_sflux()` → `sflux.make_atmo_sflux()`
- Verifies sflux output was produced

**`schism_params`** (UpdateParamsStage):

- Calls `schism_prep.update_params()` (pure Python)
- Creates param.nml from template, removes deprecated params, adds mandatory params,
    symlinks mesh files

**`schism_obs`** (SchismObservationStage):

- Queries NOAA CO-OPS stations within domain
- Writes `station.in` for SCHISM output extraction
- Only runs when `include_noaa_gages=True`

**`schism_boundary`** (BoundaryConditionStage):

- Wrapper that delegates to `TPXOBoundaryStage` or `STOFSBoundaryStage` based on
    `config.boundary.source`
- STOFS: `mpiexec -m regrid_estofs` (CLI args) + `generate_ocean_tide()` +
    `correct_elevation()`
- TPXO: `make_otps_input()` + `predict_tide` (binary) + `otps_to_open_bnds()` +
    `correct_elevation()`

**`schism_prep`** (PreSCHISMStage):

- `stage_chrtout_files()` + `make_discharge()` + `run_combine_sink_source()` (binary)
- `merge_source_sink()` + `partition_mesh()` (binary)

**`schism_run`** (SCHISMRunStage):

- `mpiexec pschism <nscribes>` (Fortran binary)
- MPI/OMP env vars set by `SchismModelConfig.build_environment()`

**`schism_postprocess`** (PostSCHISMStage):

- `combine_hotstart7` binary for reanalysis/chained runs

**`schism_plot`** (SchismPlotStage):

- Plots simulation vs observed water levels from NOAA stations

### `schism_prep.py`: pure Python functions

All bash-script logic was extracted into testable Python functions in
`src/coastal_calibration/schism_prep.py`. Each function takes explicit parameters (no
environment variable reading), making them unit-testable.

| Function                    | Replaces                            | Description                                                                                                   |
| --------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `stage_ldasin_files()`      | `pre_nwm_forcing_coastal.bash`      | Symlinks LDASIN files into `forcing_input/`                                                                   |
| `make_sflux()`              | `post_nwm_forcing_coastal.bash`     | Calls `sflux.make_atmo_sflux()` directly                                                                      |
| `update_params()`           | `update_param.bash`                 | Creates param.nml, removes deprecated params, adds mandatory params, symlinks mesh                            |
| `make_tpxo_boundary()`      | `make_tpxo_ocean.bash`              | Calls `tides.make_otps_input()` + `predict_tide` binary + `tides.otps_to_open_bnds()` + `correct_elevation()` |
| `make_stofs_boundary()`     | `pre/regrid/post_regrid_stofs.bash` | Calls `regrid_estofs` via `mpiexec -m` + `tides.generate_ocean_tide()` + `correct_elevation()`                |
| `correct_elevation()`       | `correct_elevation.py`              | Subtracts datum correction from elev2D.th.nc in-place                                                         |
| `stage_chrtout_files()`     | Part of `pre_schism.bash`           | Symlinks NWM CHRTOUT files                                                                                    |
| `make_discharge()`          | Part of `pre_schism.bash`           | Creates discharge files from NWM data                                                                         |
| `run_combine_sink_source()` | `combine_sink_source.bash`          | Calls `combine_sink_source` binary                                                                            |
| `merge_source_sink()`       | `merge_source_sink.py` script       | Merges source/sink into precipitation data                                                                    |
| `partition_mesh()`          | Part of `pre_schism.bash`           | Runs `metis_prep` + `gpmetis`                                                                                 |
| `combine_hotstart()`        | `post_schism.bash`                  | Runs `combine_hotstart7` binary                                                                               |

______________________________________________________________________

## New Python modules

| Module                                                  | Replaces                                            | Description                                                         |
| ------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------- |
| `src/coastal_calibration/sflux.py`                      | `makeAtmo.py`                                       | `make_atmo_sflux()`: creates sflux netCDF from LDASIN files         |
| `src/coastal_calibration/regridding/regrid_estofs.py`   | `wrf_hydro/.../regrid_estofs.py`                    | ESTOFS → SCHISM boundary regridding via ESMF                        |
| `src/coastal_calibration/regridding/regrid_forcings.py` | `WrfHydroFECPP/workflow_driver.py`                  | NWM atmospheric forcing regridding via ESMF                         |
| `src/coastal_calibration/regridding/esmf_utils.py`      | (new)                                               | ESMF Grid/LocStream/Regridder utilities with MPI-aware partitioning |
| `src/coastal_calibration/tides/_otps.py`                | `make_otps_input.py` + `otps_to_open_bnds_hgrid.py` | TPXO boundary file creation                                         |
| `src/coastal_calibration/tides/_ocean_tide.py`          | `makeOceanTide.py`                                  | Tidal prediction for long forecasts (>180h)                         |
| `src/coastal_calibration/tides/_tpxo_out.py`            | `TPXOOut.py`                                        | Parser for OTPSnc predict_tide output                               |
| `src/coastal_calibration/tides/pytides/`                | `Tides/pytides/`                                    | Tidal harmonic analysis library                                     |
| `schism_prep.correct_elevation()`                       | `correct_elevation.py`                              | In-place datum correction on elev2D.th.nc                           |

______________________________________________________________________

## Logging

All new SCHISM modules use the central package logger from
`coastal_calibration.logging`:

```python
from coastal_calibration.logging import logger
```

The central logger is `logging.getLogger("coastal_calibration")` and is configured with
`RichHandler` for console output and optional file logging via `configure_logger()`.

**Modules using the central logger:**

| Module                                                                          | Logger source                                    |
| ------------------------------------------------------------------------------- | ------------------------------------------------ |
| `stages/base.py`, `stages/schism.py`, `stages/boundary.py`, `stages/forcing.py` | `WorkflowMonitor` (wraps central logger)         |
| `schism_prep.py`                                                                | `from coastal_calibration.logging import logger` |
| `sflux.py`                                                                      | `from coastal_calibration.logging import logger` |
| `tides/_otps.py`, `tides/_ocean_tide.py`                                        | `from coastal_calibration.logging import logger` |
| `regridding/regrid_estofs.py`, `regrid_forcings.py`, `esmf_utils.py`            | `from coastal_calibration.logging import logger` |
| `utils/floodmap.py`                                                             | `from coastal_calibration.logging import logger` |

No module creates its own `logging.getLogger(__name__)` logger; all go through the
central `coastal_calibration` logger hierarchy, ensuring consistent formatting, levels,
and file output.

### Log indentation convention

Stage-level logs (from `_log()` in stage classes) are prepended with 2 spaces (`"  "`).
Function-level logs in `schism_prep.py`, `sflux.py`, and other modules called by stages
use 4 spaces (`"    "`) to indicate they are sub-task detail messages nested under
stage-level messages:

```
Stage: schism_prep
  Start Time: 2026-03-19 11:33:58
  Prepare SCHISM inputs (discharge, partitioning)
  Symlinking NWM CHRTOUT files
  Running make_discharge
    Processing 16 CHRTOUT files
    Wrote vsource.th (5 rows), vsink.th, source_sink.in (731 sources, 149 sinks)
  Running combine_sink_source
    combine_sink_source completed
  Running merge_source_sink
    Wrote source.nc: 487 sources (from 487), 149 sinks, 4 timesteps
  Partitioning for 4 tasks (2 scribes)
    Partitioned mesh into 2 compute ranks → .../partition.prop
```

______________________________________________________________________

## Environment variables

### Design principle

Environment variables are **not** used to pass arguments to functions. All function
parameters are passed explicitly as Python arguments. The only remaining environment
variables are for:

1. **MPI / OpenMP runtime configuration**: set by
    `SchismModelConfig.build_environment()` in `config/schema.py`
1. **HDF5 NFS reliability**: set by `WorkflowStage.build_environment()` in
    `stages/base.py`
1. **PATH / LD_LIBRARY_PATH**: for subprocess binary discovery

### Complete list of env vars set by the codebase

#### `WorkflowStage.build_environment()` (base.py)

Common variables set for **all** models:

```python
env["HDF5_USE_FILE_LOCKING"] = "FALSE"  # NFS reliability
env["OMP_NUM_THREADS"] = str(model_config.omp_num_threads)
env["OMP_PLACES"] = "cores"
env["OMP_PROC_BIND"] = "close"
```

#### `SchismModelConfig.build_environment()` (schema.py)

Delegates to `build_mpi_env()` from `coastal_calibration.utils`, which auto-detects the
active MPI implementation (OpenMPI or MPICH) and sets the appropriate tuning variables.
Settings are layered: general settings apply on all clusters, EFA-specific settings are
added only when AWS EFA devices are detected.

**OpenMPI — general** (all clusters):

```python
env["OMPI_MCA_mpi_warn_on_fork"] = "0"  # suppress NFS fork warnings
env["OMPI_MCA_orte_tmpdir_base"] = "/tmp"  # shared-memory on local disk
```

**OpenMPI — EFA only** (when `/sys/class/infiniband/efa*` detected):

```python
env["OMPI_MCA_mtl"] = "ofi"
env["OMPI_MCA_pml"] = "cm"
env["OMPI_MCA_btl"] = "^openib"
```

**MPICH / Cray MPICH** (all clusters, including WCOSS2 via `schism_exe`):

```python
env["MPICH_OFI_STARTUP_CONNECT"] = "1"
env["MPICH_COLL_SYNC"] = "MPI_Bcast"
env["MPICH_REDUCE_NO_SMP"] = "1"
```

**Libfabric tuning — EFA only** (when EFA devices detected, any MPI impl):

```python
env["FI_OFI_RXM_SAR_LIMIT"] = "3145728"
env["FI_MR_CACHE_MAX_COUNT"] = "0"
env["FI_EFA_RECVWIN_SIZE"] = "65536"
```

#### `NWMForcingStage.run()` (forcing.py)

No environment variables. HPC job-array parallelism parameters (`--job-index`,
`--job-count`) are passed as CLI arguments to `regrid_forcings.py`.

### Env vars that were removed

| Removed env var                          | Was set in                   | Replacement                                                                       |
| ---------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------- |
| `CYCLE_DATE`, `CYCLE_TIME`, `LENGTH_HRS` | `base.py` / `schism_prep.py` | CLI arguments to `regrid_estofs` (`--cycle-date`, `--cycle-time`, `--length-hrs`) |
| `FECPP_JOB_INDEX`, `FECPP_JOB_COUNT`     | `forcing.py`                 | CLI arguments to `regrid_forcings` (`--job-index`, `--job-count`)                 |
| `COASTAL_ROOT_DIR`                       | `schism_prep.py`             | Explicit `tidal_constants_dir` parameter                                          |
| `PYTHONPATH` (for fecpp)                 | `forcing.py`                 | Direct module invocation via `mpiexec -m`                                         |
| `PYTHON_GIL`                             | `base.py`                    | Removed (not needed)                                                              |
| All `get_script_environment_vars()`      | `scripts_path.py`            | Module removed entirely; paths resolved via Python imports                        |

______________________________________________________________________

## ESMF regridding modules

### `src/coastal_calibration/regridding/`

This package contains the ESMF-based regridding implementations that replaced the legacy
subprocess scripts. All modules handle the ESMF/esmpy import fallback internally:

```python
try:
    import ESMF
except ImportError:
    import esmpy as ESMF
```

#### `esmf_utils.py`: MPI-aware ESMF utilities

Key abstractions:

- **`build_grid(lon, lat, ...)`** → `(ESMF.Grid, GridBounds)` for 2D structured grids.
- **`build_locstream(lon, lat)`** → `ESMF.LocStream` for unstructured point data.
    Properly partitions the *global* coordinate array across MPI ranks (see
    [LocStream partitioning](#locstream-partitioning) below).
- **`MaskedRegridder`**: callable class for time-varying source masks. Recomputes
    weights each call because the ESTOFS wet/dry mask changes every timestep.
- **`gather_reduce()`**, **`gatherv_1d()`**, **`allreduce_minmax()`**: MPI collective
    helpers.

#### `regrid_estofs.py`: ESTOFS boundary regridding

Regrids ESTOFS `zeta` (water level) from the global unstructured STOFS grid (~12.8M
nodes) to SCHISM open boundary nodes (~4K nodes) using ESMF
nearest-source-to-destination interpolation.

Invoked via:
`mpiexec -n <N> python -m coastal_calibration.regridding.regrid_estofs <nc_in> <nc_grid> <nc_out> --cycle-date PDY --cycle-time CYC00 --length-hrs N`

All parameters are passed via CLI arguments; no env-var reading.

#### `regrid_forcings.py`: NWM atmospheric forcing regridding

Regrids NWM LDASIN atmospheric variables (U2D, V2D, T2D, Q2D, PSFC, LWDOWN, SWDOWN,
LQFRAC) from the WRF lat-lon grid to SCHISM's lat-lon output grid, and converts RAINRATE
to volumetric flux (m³/s) on the SCHISM mesh.

Invoked via: `mpiexec -n <N> python -m coastal_calibration.regridding.regrid_forcings`

### LocStream partitioning

`ESMF.LocStream(n)` creates *n* points **locally** on the calling rank. If every rank
passes the full global count, the total LocStream has `nranks × n` points (wrong).

`build_locstream()` handles this by computing a contiguous partition of the global array
across MPI ranks:

```python
n_global = len(lon)
base, remainder = divmod(n_global, size)
# rank i gets (base+1) points if i < remainder, else base points
locstream = ESMF.LocStream(local_count, ...)
locstream["ESMF:Lon"] = lon[local_start : local_start + local_count]
```

The global index range is stored on the LocStream as `_global_lower` / `_global_upper`
so callers can slice their data arrays to match the local ESMF partition.

______________________________________________________________________

## ESMF and esmpy compatibility

### Problem

The Python package `ESMF` was renamed to `esmpy` in v8.4.0. The new regridding modules
handle both names via try/except imports. The ESMF compatibility shim
(`esmf_compat/ESMF.py`) remains on `PYTHONPATH` for any legacy subprocess that still
uses `import ESMF`.

### Solution: `src/coastal_calibration/esmf_compat/ESMF.py`

A compatibility shim that re-exports `esmpy` as `ESMF`:

```python
from esmpy import *
from esmpy import constants

Manager(debug=False)  # auto-initialize (old ESMF did this implicitly)
```

### Why auto-initialize Manager?

The ESMF compat shim calls `Manager(debug=False)` because some legacy code (in
`tests/legacy_scripts/`) calls `ESMF.local_pet()` at **module level** before any
explicit `Manager()` call. The old ESMF package auto-initialized; esmpy does not.
`Manager` is a singleton; subsequent calls are harmless no-ops.

______________________________________________________________________

## Parallel netCDF4 and MPI

### Problem

The PyPI `netcdf4` wheel bundles its own non-MPI HDF5 library. Under MPI, all ranks can
open a netCDF file with `netCDF4.Dataset()`, but non-root ranks receive empty arrays
from variable reads because the bundled HDF5 has no parallel I/O support.

This caused `regrid_estofs` to fail when running with multiple MPI ranks: rank 0 read
the 12.8M-node ESTOFS array correctly, but rank 1 received shape `(0,)`.

### Solution

Install `netcdf4` from **conda-forge** with the OpenMPI build variant instead of from
PyPI. This is done at the base Python-version feature level so all environments get it:

```toml
[tool.pixi.feature.py313.dependencies]
netcdf4 = { version = "*", build = "mpi_openmpi_*" }
```

The schism feature also pins it (redundantly but explicitly):

```toml
[tool.pixi.feature.schism.dependencies]
netcdf4 = { version = "*", build = "mpi_openmpi_*" }
```

After this change, `netCDF4.__has_parallel4_support__ == 1` and all MPI ranks can read
netCDF files independently.

______________________________________________________________________

## SCHISM version compatibility

### Deprecated parameters (must be removed from param.nml)

The following parameters were removed in newer SCHISM versions and cause
`Fortran runtime error: Cannot match namelist object name` if present:

| Parameter          | Removed in | Commit    |
| ------------------ | ---------- | --------- |
| `impose_net_flux`  | Aug 2022   | `ed26f60` |
| `isconsv`          | Same era   | -         |
| `isav`             | Same era   | -         |
| `vclose_surf_frac` | Same era   | -         |

The `update_params()` function in `schism_prep.py` strips these automatically using
regex:

```python
for deprecated in ("impose_net_flux", "isconsv", "isav", "vclose_surf_frac"):
    text = re.sub(rf"(?m)^\s*{deprecated}\s*=.*\n", "", text)
```

### Mandatory parameters (must be added to param.nml)

Since May 2024 (commit `0fec598`), SCHISM requires these parameters in the `&CORE`
namelist:

| Parameter        | Value | Purpose                            |
| ---------------- | ----- | ---------------------------------- |
| `nbins_veg_vert` | 1     | Vertical bins for vegetation model |
| `nmarsh_types`   | 1     | Number of marsh types              |

### OLDIO=OFF requires nscribes >= 2

SCHISM compiled with `OLDIO=OFF` (the new I/O layer) requires at least 2 scribe
processes. The `SchismModelConfig` default is `nscribes=2`. The Hawaii example uses
`ntasks_per_node=4, nscribes=2` (2 compute + 2 scribes).

### MPI oversubscription

For local development where the number of MPI ranks exceeds physical cores, pass
`--oversubscribe` to `mpiexec`. This is controlled by `SchismModelConfig.oversubscribe`
(default `False`, set to `True` for local testing).

### combine_sink_source output naming

Newer versions of the `combine_sink_source` binary produce `source_sink.in` (without the
`.1` suffix that older versions used). The `merge_source_sink()` function handles both.

### sflux file naming

`make_atmo_sflux()` produces `sflux_air_1.0001.nc` (4-digit zero-padded) but SCHISM
expects `sflux_air_1.1.nc` (no leading zeros). The `make_sflux()` function renames files
after generation.

______________________________________________________________________

## Hawaii example notebook

### Location

- Source: `docs/examples/notebooks/schism-hawaii.py` (jupytext percent format)
- Notebook: `docs/examples/notebooks/schism-hawaii.ipynb` (generated via
    `jupytext --to ipynb`)
- Working directory: `docs/examples/hawaii/`
- Shared downloads: `docs/examples/downloads/`

### Configuration

```python
CoastalCalibConfig.from_dict(
    {
        "model": "schism",
        "simulation": {
            "start_date": "2024-01-09",
            "duration_hours": 3,
            "coastal_domain": "hawaii",
            "meteo_source": "nwm_ana",
            "timestep_seconds": 200,
        },
        "boundary": {"source": "stofs"},
        "paths": {
            "work_dir": "./run",
            "raw_download_dir": "../downloads",
            "parm_dir": ".",
        },
        "download": {"enabled": True},
        "model_config": {
            "nodes": 1,
            "ntasks_per_node": 4,
            "nscribes": 2,
            "oversubscribe": True,
            "include_noaa_gages": True,
        },
    }
)
```

### Hawaii model details

- ~878K nodes, ~1.7M elements
- 2D barotropic (2 vertical levels via vgrid.in)
- 7 NOAA CO-OPS tide gauge stations detected within domain
- 3-hour simulation completes in ~11 minutes on Apple M2 Ultra (4 MPI ranks)

### Data availability constraints

| Data source       | Hawaii availability      | Notes                          |
| ----------------- | ------------------------ | ------------------------------ |
| NWM Retrospective | 1994-01-02 to 2013-12-31 | Cannot be used with STOFS      |
| NWM Analysis      | 2021-04-21 to present    | Use `meteo_source: "nwm_ana"`  |
| STOFS             | 2020-12-30 to present    | Use `boundary.source: "stofs"` |

______________________________________________________________________

## Known issues and workarounds

### Pacific domain too large for local machine

The Pacific mesh (~6M elements) causes stack overflow on local machines with 3 compute
ranks. The Pacific domain requires a cluster with sufficient memory per rank.

### `-fno-automatic` flag breaks SCHISM

Adding `-DCMAKE_Fortran_FLAGS="-fno-automatic"` to the CMake configuration caused SCHISM
to crash immediately. Do not use it.

### STOFS boundary for non-CONUS domains

STOFS (`stofs_2d_glo`) provides global coverage, so it works for Hawaii. TPXO requires
the `predict_tide` binary which has additional dependencies. Use `source: "stofs"` for
Hawaii and PRVI domains when running locally.

### NWM analysis 2-hour lag

NWM Analysis/Assimilation data has a 2-hour lag. The downloader automatically accounts
for this by fetching `tm02` files offset by +2 hours from the simulation time.

______________________________________________________________________

## Decision log

| Decision                                                            | Rationale                                                                               |
| ------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Migrate from activation scripts to pixi-build (rattler-build)       | Proper conda packages with dependency tracking; no hand-rolled fingerprinting           |
| Move submodules under `coastal_models/` with `./repo` source paths  | In-tree `./` paths avoid pixi-build cache invalidation bug ([#4837][pixi4837])          |
| Pin MPI variants for both SFINCS and SCHISM                         | All models share the same env; MPI-linked netcdf/hdf5 avoids runtime library conflicts  |
| Dynamic macOS SDK probe instead of hardcoded version                | Newer macOS SDKs may have incompatible TBD stubs; probe tests each SDK at build time    |
| Use existing SCHISM submodule, not `nwm.v3.0.6_no_svn`              | Cleaner; NWM-specific settings are just CMake cache flags                               |
| Build `combine_sink_source` and `metis_prep` standalone             | Not wired into SCHISM's CMake; simple single-file Fortran programs                      |
| Extract bash logic into `schism_prep.py` Python functions           | Testable; explicit parameters instead of env vars; removes script maintenance burden    |
| Replace all Python-script subprocess calls with direct imports      | Eliminates env-var passing, subprocess overhead, and fragile path resolution            |
| Use `mpiexec -m <module>` for ESMF regridding                       | Proper Python module invocation; no path resolution needed                              |
| Create `sflux.py` module from `makeAtmo.py`                         | Direct function call; inline SLP calculation; no env vars                               |
| Create `tides/` package from TPXO scripts + makeOceanTide           | Consolidates tidal boundary logic; explicit parameters; testable                        |
| Move legacy scripts to `tests/legacy_scripts/`                      | Serves as reference implementations for regression testing                              |
| Remove `scripts_path.py`                                            | No longer needed; all paths resolved via Python imports                                 |
| Pin `netcdf4 = { build = "mpi_openmpi_*" }` in py311/py313 features | PyPI wheels bundle nompi HDF5; conda-forge MPI variant needed for parallel reads        |
| Partition LocStream coordinates in `build_locstream()`              | `ESMF.LocStream(n)` treats `n` as local count; must divide global array across ranks    |
| Create ESMF compat shim instead of patching scripts                 | Avoids modifying legacy scripts; single-point fix                                       |
| Auto-init Manager in ESMF shim                                      | Legacy scripts call `ESMF.local_pet()` at module level before explicit `Manager()`      |
| Drop `singularity_image` config field                               | Singularity workflow fully removed; field silently dropped during config loading        |
| Use `nscribes=2` minimum                                            | `OLDIO=OFF` compilation requires at least 2 scribes                                     |
| Use STOFS instead of TPXO for local testing                         | `predict_tide` binary not available in pixi env                                         |
| All new modules use central package logger                          | Consistent formatting, levels, and file output across all modules                       |
| No env vars for function arguments                                  | All functions take explicit Python parameters; env vars only for MPI/OMP runtime config |
| SCHISM stage names prefixed with `schism_`                          | Consistent with SFINCS convention; meaningful and concise                               |
| SCHISM run workflow takes prebuilt model dir                        | Same pattern as SFINCS; create workflow is separate (future work)                       |
| Convert FECPP env vars to CLI args                                  | Eliminates last env-var-as-argument pattern; `--job-index`/`--job-count` CLI args       |
| Add `prebuilt_dir` to `SchismModelConfig`                           | Mirrors `SfincsModelConfig.prebuilt_dir`; `coastal_parm()` centralizes resolution       |
| 4-space indent for function-level logs                              | Aligns with stage-level 2-space indent from `_log()` / `WorkflowMonitor`                |

______________________________________________________________________

## File reference

### New files

| File                                                     | Description                                                                               |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `coastal_models/schism/{pixi.toml,recipe.yaml,build.sh}` | Pixi-build package that compiles SCHISM binaries                                          |
| `scripts/find_compatible_sdk.sh`                         | macOS SDK compatibility probe (shared by SFINCS and SCHISM builds)                        |
| `src/coastal_calibration/schism_prep.py`                 | Pure-Python SCHISM pre/post-processing functions                                          |
| `src/coastal_calibration/sflux.py`                       | sflux atmospheric forcing generation (replaces makeAtmo.py)                               |
| `src/coastal_calibration/regridding/`                    | ESMF-based regridding (ESTOFS + NWM forcing) with MPI support                             |
| `src/coastal_calibration/tides/`                         | TPXO/ocean-tide boundary utilities (replaces tpxo_to_open_bnds_hgrid/ + makeOceanTide.py) |
| `src/coastal_calibration/esmf_compat/ESMF.py`            | `import ESMF` → `esmpy` compatibility shim                                                |
| `tests/legacy_scripts/`                                  | Legacy bash+Python scripts (reference implementations for regression tests)               |
| `docs/examples/notebooks/schism-hawaii.py`               | Hawaii SCHISM tutorial notebook (jupytext)                                                |

### Modified files

| File                                             | What changed                                                                                                        |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `pyproject.toml`                                 | Schism feature uses pixi-build path dep, `netcdf4` MPI build for all envs, `preview = ["pixi-build"]`               |
| `src/coastal_calibration/config/schema.py`       | `SchismModelConfig`: binary default → `pschism`, `singularity_image` dropped, MPI/OMP env vars only                 |
| `src/coastal_calibration/stages/base.py`         | Removed PYTHON_GIL, removed scripts_path env vars, removed `run_singularity_command()`, kept ESMF compat PYTHONPATH |
| `src/coastal_calibration/stages/forcing.py`      | Uses `regrid_forcings` via `mpiexec -m`; removed fecpp PYTHONPATH hack                                              |
| `src/coastal_calibration/stages/boundary.py`     | Replaced Singularity with `update_params()`, `make_stofs_boundary()` calls                                          |
| `src/coastal_calibration/stages/schism.py`       | Replaced Singularity with `schism_prep.*` function calls + native MPI                                               |
| `src/coastal_calibration/stages/sfincs_build.py` | Uses `tides.TIDES_DATA_DIR` for TPXO config files                                                                   |

### Removed files

| File                                      | Reason                                           |
| ----------------------------------------- | ------------------------------------------------ |
| `src/coastal_calibration/scripts_path.py` | All script paths resolved via Python imports now |
| `src/coastal_calibration/scripts/`        | Moved to `tests/legacy_scripts/`                 |
| `tests/common/test_scripts_path.py`       | Tested the removed `scripts_path.py`             |
| `dockerfiles/Dockerfile.ngencoastal`      | Container build file, no longer needed           |
| `base.py: run_singularity_command()`      | Singularity execution method removed             |
| `base.py: _get_default_bindings()`        | Singularity bind-mount paths removed             |

______________________________________________________________________

## Cleanup changelog (Phase 5)

Phase 5 was a deep cleanup pass performed after the initial de-containerization (Phases
1–4) was validated end-to-end.

### Singularity removal

- Removed `run_singularity_command()` and `_get_default_bindings()` from
    `stages/base.py`
- Removed `singularity_image` field from `SchismModelConfig` (silently dropped during
    YAML config loading for backward compat)
- Removed `dockerfiles/Dockerfile.ngencoastal`
- Removed all `run_sing_*.bash` wrapper scripts (moved to `tests/legacy_scripts/`)
- No remaining references to Singularity in the active codebase

### Environment variable cleanup

- Converted `CYCLE_DATE`, `CYCLE_TIME`, `LENGTH_HRS` env-var passing in `regrid_estofs`
    to CLI arguments (`--cycle-date`, `--cycle-time`, `--length-hrs`)
- Converted `FECPP_JOB_INDEX`, `FECPP_JOB_COUNT` env-var passing in `regrid_forcings` to
    CLI arguments (`--job-index`, `--job-count`)
- Removed `COASTAL_ROOT_DIR` env-var usage; `tidal_constants_dir` is now an explicit
    function parameter
- Removed `PYTHON_GIL` from `base.py`
- Removed all `get_script_environment_vars()` calls (module deleted)
- Only remaining env vars: MPI/OMP runtime config in `schema.py`, HDF5/PATH in `base.py`

### Unified logging

- All modules (`schism_prep.py`, `sflux.py`, `tides/`, `regridding/`,
    `utils/floodmap.py`) switched from `logging.getLogger(__name__)` to the central
    package logger: `from coastal_calibration.logging import logger`
- No `logging.getLogger(__name__)` calls remain in `src/coastal_calibration/`
- Stage classes use `WorkflowMonitor` which wraps the same central logger
- Function-level logs use 4-space indent (`"    "`) for proper alignment with
    stage-level 2-space indent (`"  "`)

### Stage naming

- All SCHISM stage names now prefixed with `schism_` for consistency with the SFINCS
    `sfincs_*` convention
- Stage names are concise and descriptive: `schism_forcing_prep`, `schism_forcing`,
    `schism_sflux`, `schism_params`, `schism_obs`, `schism_boundary`, `schism_prep`,
    `schism_run`, `schism_postprocess`, `schism_plot`

### Workflow pattern alignment

- SCHISM run workflow follows the same pattern as SFINCS: given a prebuilt model
    directory, it executes the simulation
- Added `prebuilt_dir` field to `SchismModelConfig` (analogous to
    `SfincsModelConfig.prebuilt_dir`) with `coastal_parm()` method for centralized model
    directory resolution
- SCHISM create workflow (model setup from scratch) is planned as future work, following
    the `sfincs_create.py` pattern

### Remaining cleanup TODO

- Implement SCHISM create workflow (`schism_create.py`) following the `sfincs_create.py`
    pattern
- Update `schism_prep.py` functions to accept `coastal_parm: Path` directly instead of
    `parm_nwm + coastal_domain` (now that `SchismModelConfig.coastal_parm()` resolves
    this centrally)

______________________________________________________________________

## Test coverage

### Unit tests (`tests/common/`)

| Test file             | Tests | What's covered                                                                                     |
| --------------------- | ----- | -------------------------------------------------------------------------------------------------- |
| `test_schism_prep.py` | 13    | Symlinks, discharge, combine_sink_source, partition, LDASIN staging, param.nml updates             |
| `test_stages.py`      | 40+   | Stage initialization, `requires_container == False`, MPI command construction, boundary validation |

### Regridding tests (`tests/regridding/`)

| Test file                 | Tests | What's covered                                                                |
| ------------------------- | ----- | ----------------------------------------------------------------------------- |
| `test_regrid_estofs.py`   | 5     | Synthetic output structure, value plausibility, optional real-data comparison |
| `test_regrid_forcings.py` | 10    | SLP formula, synthetic vsource structure, optional real-data comparison       |

### Integration test

The Hawaii example notebook serves as an end-to-end integration test: all 11 stages run
sequentially, producing comparison plots at 7 NOAA stations. Run with:

```bash
cd docs/examples/notebooks
pixi run -e dev python schism-hawaii.py
```

[pixi4837]: https://github.com/prefix-dev/pixi/issues/4837
