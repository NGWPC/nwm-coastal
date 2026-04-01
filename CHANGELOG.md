# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **MPI detection module** (`utils/mpi.py`): auto-detects the active MPI implementation
    (OpenMPI or MPICH/Cray MPICH) at runtime via `mpiexec --version` and sets the
    correct tuning environment variables and launcher flags for each. On AWS EFA
    instances, libfabric transport and buffer tuning are applied automatically; on plain
    NFS/Lustre clusters only general settings (shared-memory on `/tmp`, fork-warning
    suppression) are used.
- **`schism_exe` config option**: replaced the `binary` field on `SchismModelConfig`
    with `schism_exe: Path | None` (default `None`), matching the existing `sfincs_exe`
    pattern. When set, the `schism_run` stage uses the given binary and isolates the
    subprocess environment from conda libraries for system MPI compatibility.
- **`runtime_env` config option**: both `SchismModelConfig` and `SfincsModelConfig`
    accept a `runtime_env: dict[str, str]` for injecting extra environment variables
    into the model run subprocess. Applied last, so it can override auto-detected MPI
    tuning values.
- **Type stubs**: added `types-geopandas` and `types-shapely` to the typecheck
    environment for better third-party type coverage.

### Changed

- **Lazy imports**: converted `__init__.py` files (package root, `config/`, `utils/`)
    and `cli.py` to use deferred imports via `__getattr__`. The CLI
    (`coastal-calibration --version`) no longer pulls the full dependency tree at
    startup, reducing cold-start time from ~12 s to under 1 s on NFS.
- **System MPI isolation**: when `schism_exe` or `sfincs_exe` is set to a
    system-compiled binary, the run stage strips conda library paths from the subprocess
    environment so the binary finds system MPI/HDF5/NetCDF. A new `runtime_env` config
    option allows injecting extra env vars for the model run. Python MPI stages (ESMF
    regridding) continue using conda OpenMPI.
- **OpenMP tuning**: moved common OpenMP variables (`OMP_NUM_THREADS`, `OMP_PLACES`,
    `OMP_PROC_BIND`) from model-specific `build_environment()` methods to the shared
    `WorkflowStage.build_environment()` in `stages/base.py`.
- **SFINCS run stage**: `SfincsRunStage.run()` now uses `self.build_environment()`
    instead of constructing an inline env dict, ensuring `HDF5_USE_FILE_LOCKING` and
    OpenMP pinning are applied consistently.

### Fixed

- **Type checking**: resolved all 895 pyright strict-mode errors across 35 source files.
    Real type issues (wrong argument types, unnecessary isinstance checks, missing
    annotations, None-safety) were fixed in code; remaining noise from third-party
    libraries without type stubs is suppressed via targeted pyright config rules.
- **Script permissions**: marked `scripts/find_compatible_sdk.sh`,
    `coastal_models/schism/build.sh`, and `coastal_models/sfincs/build.sh` as executable
    to satisfy the pre-commit shebang check.
- **Cluster install verification**: replaced `shutil.which` check with `ls` in
    `CLUSTER_INSTALL.md` — the old command returned `None` because running the Python
    binary directly bypasses the wrapper's `PATH` setup.

### Changed

- **Build system**: migrated SFINCS and SCHISM compilation from activation scripts
    (`scripts/ensure-sfincs.sh`, `scripts/ensure-schism.sh`) to pixi-build packages
    using the `rattler-build` backend. Build recipes live in `coastal_models/sfincs/`
    and `coastal_models/schism/` with `recipe.yaml` + `build.sh`. Builds are cached as
    conda packages and only recompile when the submodule source or recipe changes.
- **Submodule layout**: moved git submodules from `SFINCS/` and `schism/` to
    `coastal_models/sfincs/repo/` and `coastal_models/schism/repo/`. The in-tree
    `./repo` source paths avoid a pixi-build cache invalidation bug with out-of-tree
    paths ([prefix-dev/pixi#4837](https://github.com/prefix-dev/pixi/issues/4837)).
- **MPI variant consistency**: both SFINCS and SCHISM recipes now pin `hdf5` and
    `libnetcdf`/`netcdf-fortran` to `mpi_openmpi_*` build variants, matching ESMF/esmpy
    runtime expectations and avoiding library conflicts in shared environments.
- **macOS SDK probe**: replaced hardcoded `MacOSX15*` SDK fallback with a dynamic probe
    (`scripts/find_compatible_sdk.sh`) that tests each installed SDK against the active
    linker and picks the newest compatible one.
- **Cluster install**: rewrote `CLUSTER_INSTALL.md` around full-repo clone with
    pixi-build. The wrapper script (`nwm-coastal`) fully activates the environment so
    pixi is not needed at runtime on compute nodes.
- **mkdocs hooks**: `docs/hooks.py` now strips ANSI escape codes and absolute repo paths
    from rendered notebook HTML at build time, complementing the existing
    `scripts/clean_notebooks.py` pre-commit task.

### Removed

- `scripts/ensure-sfincs.sh` and `scripts/ensure-schism.sh` (replaced by pixi-build
    recipes).
- Build-tool dependencies (`c-compiler`, `fortran-compiler`, `cxx-compiler`, `cmake`,
    `make`, `autoconf`, `automake`, `libtool`, `m4`) from pixi feature sections in
    `pyproject.toml` (now declared in `recipe.yaml` and resolved by the build backend).

### Added

- `coastal_calibration.plotting` module with reusable visualization utilities:
    - `SfincsGridInfo` dataclass with `from_model_root()` factory for loading and
        summarizing SFINCS grid metadata (quadtree and regular grids).
    - `plot_mesh()` for visualizing the SFINCS mesh colored by refinement level with
        optional satellite basemap via `contextily`.
    - `plot_floodmap()` for reading and plotting flood-depth Cloud Optimized GeoTIFFs with
        automatic overview-level selection and basemap overlay.
    - `plot_station_comparison()` for generating 2×2 simulated vs observed water-level
        comparison plots, consolidated from the former `sfincs_plot` and `schism_plot`
        stage internals.
    - `plotable_stations()` helper for filtering stations that have both simulated and
        observed data.
- `_KNOWN_INP_PARAMS` allowlist on `SfincsModelConfig` that validates `inp_overrides`
    keys against all ~170 recognized `sfincs.inp` parameters, catching typos early
    (SFINCS silently ignores unknown parameters).
- `SfincsDischargeStage` now assigns real NWM `CHRTOUT` discharge timeseries to source
    points via `_assign_discharge_timeseries`, using the shared `read_streamflow`
    utility (`nwm_retro` reads from S3 Zarr, `nwm_ana` reads from local CHRTOUT via
    `netCDF4`).
- `tests/test_floodmap.py` with unit tests for `_write_floodmap_cog`,
    `_ensure_overviews`, and an integration test for `create_flood_depth_map`.
- QGIS plugin: optional NWM Flowlines Override in the basemap dialog, allowing users to
    load flowpaths from a separate NWM GeoPackage with configurable layer name and
    stream order column.
- QGIS plugin: "Export Selected Flowpaths" toolbar button that saves the current
    selection on the flowpaths layer to a GeoJSON file.
- QGIS plugin: stream order validation against the actual min/max range in the data,
    with a clear error when the user-specified value is out of range.
- QGIS plugin: auto-enable labels for gages (`site_no`) and CO-OPS (`station_id`) layers
    with font size 15 and text buffer.
- Narragansett Bay, RI example notebook showing compound forcing (ocean + river
    discharge + meteo) with NWM streamflow.
- SCHISM de-containerization: all Singularity container invocations replaced with native
    Python function calls, MPI module invocations (`mpiexec -m`), and subprocess calls
    for Fortran binaries (`pschism`, `combine_hotstart7`, `combine_sink_source`,
    `metis_prep`, `gpmetis`).
- `coastal_calibration.schism_prep` module with 12 pure-Python functions replacing bash
    scripts: `stage_ldasin_files`, `make_sflux`, `update_params`, `make_tpxo_boundary`,
    `make_stofs_boundary`, `correct_elevation`, `stage_chrtout_files`, `make_discharge`,
    `run_combine_sink_source`, `merge_source_sink`, `partition_mesh`,
    `combine_hotstart`.
- `coastal_calibration.sflux` module replacing `makeAtmo.py` for sflux atmospheric
    forcing generation with inline sea-level pressure reduction.
- `coastal_calibration.tides` package replacing `tpxo_to_open_bnds_hgrid/` and
    `makeOceanTide.py` with TPXO boundary utilities and bundled `pytides` harmonic
    library.
- `coastal_calibration.regridding` package with ESMF-based regridding for STOFS boundary
    conditions (`regrid_estofs`) and NWM atmospheric forcing (`regrid_forcings`) with
    MPI-aware LocStream partitioning.
- `coastal_calibration.utils.streamflow` module supporting NWM retrospective (S3 Zarr)
    and analysis (local CHRTOUT) streamflow reads via unified `read_streamflow()`
    function.
- `scripts/ensure-schism.sh` pixi activation script for native SCHISM binary compilation
    with SHA256-based rebuild detection.
- ESMF/esmpy compatibility shim (`esmf_compat/ESMF.py`) for `import ESMF` → `esmpy`
    transition with auto-initialized `Manager`.
- Hawaii SCHISM example notebook demonstrating end-to-end workflow with 3-hour
    simulation on ~878K node mesh.
- `prepare-topobathy` and `update-dem-index` CLI commands for DEM management.
- `docs/schism_compilation.md`: comprehensive SCHISM de-containerization guide covering
    architecture, build system, stage inventory, and known issues.
- `docs/stofs_tpxo_improvements.md`: analysis of STOFS/TPXO boundary condition pipeline
    issues and improvement plan.
- SCHISM compilation and unit tests in `tests/schism/` and regridding tests in
    `tests/regridding/` with synthetic grid generation.

### Changed

- Centralize station comparison plotting into `coastal_calibration.plotting.stations`,
    removing duplicate `_plotable_stations` and `_plot_figures` code from
    `SfincsPlotStage` and `SchismPlotStage`.
- Refactor Lavaca and Narragansett example notebooks to use the new `plotting` module
    (`SfincsGridInfo`, `plot_mesh`, `plot_floodmap`) instead of inline visualization
    code.
- **Breaking:** Rename `nwm_discharge` config section to `river_discharge` and add
    `max_snap_distance_m` field (default 2000 m). Discharge points that must move
    farther than this threshold to reach an active grid cell are dropped with a warning.
    Previously these points could be silently written with out-of-bounds coordinates.
- **Breaking:** Simplify `river_discharge` config from 5 fields (`hydrofabric_gpkg`,
    `flowpaths_layer`, `flowpath_id_column`, `flowpath_ids`, `coastal_domain`) to 2
    fields (`flowlines`, `nwm_id_column`). Users now provide a GeoJSON file of selected
    flowpaths (e.g., exported from the QGIS plugin) instead of a full NWM hydrofabric
    GeoPackage with explicit IDs.
- QGIS plugin: "Export Selected Flowpaths" now writes GeoJSON (`.geojson`) instead of
    GeoPackage (`.gpkg`).
- Rename example directory `texas-lavaca/` to `lavaca-tx/` and update notebook paths
    accordingly.
- Consolidate example notebooks: remove CLI variants and `_api`/`_cli` suffixes, keeping
    one notebook per region (`lavaca.ipynb`, `narragansett.ipynb`).
- Rewrite `create_flood_depth_map` to read the DEM and index COG at full resolution and
    write a flood-depth Cloud Optimized GeoTIFF block-by-block, bypassing an upstream
    `hydromt_sfincs` bug where `downscale_floodmap` opened rasters with
    `overview_level=0` and silently halved the output resolution.
- Separate discharge concerns between `create` and `run` stages: the create stage now
    only writes the `.src` file with snapped locations, while the run stage
    (`SfincsDischargeStage`) adds points to the model and attaches real NWM streamflow
    data.
- Consolidate stdout/logging suppression for `hydromt-sfincs` into a shared
    `suppress_hydromt_output()` context manager in `utils.logging`, replacing duplicated
    `_suppress_stdout` helpers in `creator.py` and `sfincs_create.py`.
- `SfincsGridInfo.from_model_root()` no longer accepts a `base_resolution` parameter;
    the coarsest cell size is now derived automatically from the quadtree grid data.
- Bump `pyproject-fmt` pre-commit hook from v2.16.2 to v2.18.1.
- `sfincs_symlinks` stage now reports when symlinks already exist instead of silently
    showing "Created 0 … symlinks".
- Add `jupytext` pre-commit hook (`--sync`) to keep `.ipynb` and `.py` notebook pairs in
    sync automatically.
- Configure `ty` type-checker to resolve third-party imports from the pixi `typecheck`
    environment via `python` path instead of `extra-paths`.
- Exclude `docs/examples/downloads/`, `examples/hawaii/run/`, `examples/lavaca-tx/run/`,
    and example `cache/` directories from MkDocs static file copying to prevent
    `No space left on device` errors from large simulation data files.
- Update all documentation to reflect SCHISM de-containerization: new stage names
    (`schism_forcing_prep`, `schism_sflux`, `schism_params`, `schism_boundary`,
    `schism_prep`, `schism_postprocess`), removal of `singularity_image` config field,
    addition of `prebuilt_dir` and `geogrid_file` fields, and removal of all
    Singularity/container references from user-facing docs.
- Remove `sfincs_obs` from the SFINCS stage pipeline documentation.
- Add Hawaii SCHISM example card with correct thumbnail to the examples index page.
- Fix jupytext `formats` config in `pyproject.toml` (trailing slash in the prefix key
    caused a triple-slash path that resolved to the read-only root filesystem on macOS).
- Scope the jupytext pre-commit hook to `docs/examples/notebooks/` and exclude `.ipynb`
    files from `pretty-format-json` to prevent hook conflicts.
- Improve writing quality and consistency across all documentation files.
- Discharge snapping now always snaps to the nearest active cell (previously kept points
    as-is when they happened to land on an active cell) and converts KDTree distances
    from CRS units to meters before comparing with `max_snap_distance_m`.
- Simplify `SfincsDischargeStage` discharge loading: remove the HydroMT `get_geodataset`
    fallback chain and call `read_streamflow` directly. Resolves CHRTOUT files from
    `config.paths.streamflow_dir` instead of the HydroMT data catalog.
- **Breaking:** All SCHISM stages now run natively without Singularity containers.
    Stages call Python functions directly instead of shelling out to bash scripts inside
    `singularity exec`.
- **Breaking:** SCHISM stage names renamed to `schism_*` prefix convention (matching
    SFINCS `sfincs_*` pattern): `schism_forcing_prep`, `schism_forcing`, `schism_sflux`,
    `schism_params`, `schism_obs`, `schism_boundary`, `schism_prep`, `schism_run`,
    `schism_postprocess`, `schism_plot`.
- **Breaking:** `singularity_image` config field removed from `SchismModelConfig`.
- All environment-variable-based argument passing in SCHISM stages replaced with
    explicit Python function parameters or CLI arguments (`--cycle-date`,
    `--cycle-time`, `--length-hrs`, `--job-index`, `--job-count`).
- ESMF regridding modules invoked via `mpiexec -m coastal_calibration.regridding.*` with
    CLI arguments instead of environment variables.
- Unified logging: all SCHISM modules use central `coastal_calibration` logger with
    4-space indent for function-level logs under 2-space stage-level logs.
- Tests reorganized from flat `tests/` directory into `tests/common/`, `tests/schism/`,
    `tests/sfincs/`, `tests/regridding/` subdirectories.
- Pixi `netcdf4` dependency pinned to MPI-aware conda-forge build variant
    (`mpi_openmpi_*`) for parallel HDF5 support across all environments.

### Fixed

- Clean all generated files (discharge, boundary, sflux, partitioning, outputs, status)
    from the run directory at the start of a full workflow run via
    `clean_run_directory()`. When resuming with `start_from`, no wipe occurs so
    prerequisites and earlier outputs are preserved. Consolidates the per-stage cleanup
    previously in `make_sflux()` and `PreForcingStage`.
- Use `CONSERVE` regrid method for Grid-to-Mesh(ELEMENT) regridding in
    `_regrid_to_schism`. ESMF `BILINEAR` only supports `MeshLoc.NODE` destinations; the
    `BILINEAR` + `ELEMENT` combination was silently accepted by some ESMF builds (macOS)
    but correctly rejected with `ESMC_RC_ARG_VALUE` (rc=509) on others (Ubuntu CI).
- `_read_staout` now returns empty arrays when `staout_1` has no data (e.g., station
    output not enabled), preventing `IndexError` on 1-dimensional array indexing.
- `PostSCHISMStage` filters known non-fatal `QUICKSEARCH` dry-node warnings from
    `fatal.error` instead of treating all content as a hard failure.
- `SchismPlotStage` skips gracefully when `staout_1` is empty instead of crashing.
- Skip ESMF regridding tests on platforms where `ESMF.Mesh` is unavailable or
    `ESMC_FieldRegridStore` fails with rc=509. Enlarge synthetic ESMF grids to avoid
    BILINEAR regrid failures on small domains.
- Exclude regridding tests from CI catch-all test environments that lack MPI/ESMF.
- Replace `assert` statements with proper exceptions (`ValueError`, `RuntimeError`,
    `FileNotFoundError`) throughout `src/` modules.
- Use `ty: ignore[rule-name]` with specific rule names instead of blanket `type: ignore`
    comments.
- Add `*.ESMF_LogFile` to `.gitignore`.
- Configure `schism` submodule to track only HEAD (no internal changes tracked),
    matching the SFINCS submodule pattern.
- SFINCS activation script now detects stale `Makefile` configured for a different
    environment prefix and reconfigures automatically, fixing `sfincs` binary not found
    when switching between pixi environments.
- Flood depth map generation for regular (non-quadtree) SFINCS grids: the `zsmax` index
    lookup in `_reduce_zsmax` used C-order (row-major) flattening but
    `SfincsGrid.get_indices_at_points` returns Fortran-order (column-major) linearized
    indices. Changed to Fortran-order to match, fixing incorrect flood depth values on
    regular grids.
- `sfincs_floodmap` stage now reads `zsmax` via `SfincsModel` with `apply_all_patches()`
    instead of `xu.open_dataset`, fixing a crash on quadtree models without refinement
    levels (where the output lacks UGRID topology).
- STOFS data catalog entry previously used a recursive glob (`stofs/**/*.fields.cwl.nc`)
    that matched all cached STOFS files. When the cache held files from different STOFS
    mesh versions with incompatible dimensions, `xarray` concatenation failed. A new
    `_stofs_uri()` helper now builds an exact file path for the simulation's date and
    cycle hour, avoiding multi-file collision.
- Expand the STOFS `drop_variables` list to also drop `nvell`, `ibtype`, `nbvv`,
    `max_nvell`, and `depth`, reducing memory for the ~12-million-node STOFS mesh.
- `_downstream_endpoint` now compares both endpoints of NWM hydrofabric flowpath
    linestrings and returns whichever is closest to the AOI boundary, fixing incorrect
    discharge point placement when flowpath direction is reversed.
- Quadtree mesh plotting (`SfincsGridInfo`) now masks fill values (-1) in face-node
    connectivity before computing cell widths, fixing bogus level counts and distorted
    mesh visualizations.
- QGIS plugin: use `mActionVertexToolActiveLayer.svg` icon for the Edit Polygon toolbar
    button (previously used the removed `mActionNodeTool.svg`).
- NWM data catalog globs (`*.LDASIN_DOMAIN1.nc`, `*.CHRTOUT_DOMAIN1.nc`) now include the
    simulation year-month prefix (e.g., `202506*.LDASIN_DOMAIN1.nc`). Previously, stale
    files from other runs or domains in the shared download directory were loaded,
    causing `xr.open_mfdataset` to fail with "non-monotonic global indexes along
    dimension x" when combining grids of different sizes (e.g., Hawaii 390×590 vs CONUS
    3840×4608).
- `clip_and_reproject` now sorts spatial coordinates after the `nearest_index` reproject
    step, preventing "non-monotonic global indexes" errors from floating-point drift in
    reprojected coordinates.
- `_read_from_chrtout` in `utils/streamflow.py` now handles CHRTOUT files with 2D
    `streamflow(time, feature_id)` arrays by squeezing the time dimension, fixing
    compatibility with test fixtures that write multi-dimensional streamflow data.

### Removed

- `scripts_path.py` module. All script paths now resolved via Python imports.
- `src/coastal_calibration/scripts/` directory. Legacy bash and Python scripts moved to
    `tests/legacy_scripts/` as reference implementations for regression testing.
- `run_singularity_command()` and `_get_default_bindings()` from `stages/base.py`.
    Singularity execution infrastructure fully removed.
- `requires_container` class attribute from `WorkflowStage`. All stages now run
    natively.
- `Dockerfile.ngencoastal`. Container build file no longer needed.
- All `run_sing_*.bash` Singularity wrapper scripts.
- `MPIConfig` class (fields absorbed into `SchismModelConfig`).

## [3.1.1.0.0] - 2026-02-19

### Added

- Initial release of NWM Coastal
- SCHISM coastal model workflow support
- YAML configuration with variable interpolation
- Configuration inheritance with `_base` field
- CLI commands: `init`, `validate`, `submit`, `run`, `stages`
- Python API for programmatic workflow control
- Automatic data download from NWM and STOFS sources
- Support for TPXO and STOFS boundary conditions
- Support for four coastal domains: Hawaii, PRVI, Atlantic/Gulf, Pacific
- Interactive and non-interactive job submission modes
- Partial workflow execution with `--start-from` and `--stop-after`
- Smart default paths with interpolation templates
- Comprehensive configuration validation
- MkDocs documentation with Material theme
- Per-stage completion tracking in the `submit` path's generated runner script: each
    container stage is recorded in `.pipeline_status.json` as it finishes, enabling
    mid-pipeline restarts after a SLURM job failure without re-running expensive stages
    (e.g., `predict_tide`). On resubmit, completed stages are automatically skipped.
- `meteo_res` option in `SfincsModelConfig` to control the output resolution (m) of
    gridded meteorological forcing (precipitation, wind, pressure). When not set, the
    resolution is derived from the SFINCS quadtree grid base cell size.
- Meteo grid clipping (`_clip_meteo_to_domain`) that trims reprojected meteo grids to
    the model domain extent, preventing the LCC → UTM reprojection from inflating grids
    to CONUS scale, **reducing SFINCS runtime from 15 h+ to under 15 min**.
- Stale netCDF file cleanup in `SfincsInitStage` to prevent HDF5 segfaults when
    re-running a pipeline over an existing model directory.
- `GeoDataset`-based water-level forcing with IDW interpolation to boundary points,
    replacing the built-in `model.water_level.create(geodataset=...)` which passed all
    source stations incompatibly with `.bnd` files.
- Active-cell filtering for discharge source points to prevent a SFINCS Fortran segfault
    when a source point falls on an inactive grid cell.
- `apply_all_patches()` convenience function in `_hydromt_compat` that applies all
    `hydromt`/`hydromt-sfincs` compatibility patches in one call, with logging.
- `quiet` parameter on `WorkflowMonitor.mark_stage_completed()` to control whether a
    visible COMPLETED log line is emitted for externally-executed stages.
- Unified `run` and `submit` execution pipelines. Both commands now execute the same
    stage pipeline. `submit` automatically partitions stages into login-node
    (Python-only) and SLURM job (container) groups.
- `--start-from` and `--stop-after` options for `submit` command, matching `run`
- `requires_container` class attribute on `WorkflowStage` for automatic stage
    classification (Python-only vs container)
- `schism_obs` stage: automatic NOAA CO-OPS water level station discovery via concave
    hull of open boundary nodes, writing `station.in` and `station_noaa_ids.txt`
- `schism_plot` stage: post-run comparison plots of simulated vs NOAA-observed water
    levels with MLLW→MSL datum conversion
- `COOPSAPIClient` for querying the NOAA CO-OPS API (station metadata, water levels,
    datums) with local caching of station metadata
- `include_noaa_gages` option in `SchismModelConfig` (defaults to `false`) that enables
    the `schism_obs` and `schism_plot` stages
- Automatic `param.nml` patching (`iout_sta = 1`, `nspool_sta = 18`) when `station.in`
    exists, ensuring `mod(nhot_write, nspool_sta) == 0` across all domain templates
- `forcing_to_mesh_offset_m` option in `SfincsModelConfig` to apply a vertical offset to
    boundary-condition water levels before they enter SFINCS. For tidal-only sources
    like TPXO, this anchors the tidal signal to the correct geodetic height of MSL on
    the mesh.
- `vdatum_mesh_to_msl_m` option in `SfincsModelConfig` to convert SFINCS output from the
    mesh vertical datum to MSL for comparison with NOAA CO-OPS observations.
- Sanity-check warning in `sfincs_forcing` when adjusted boundary water levels fall
    outside the ±15 m range, indicating a possible sign or magnitude error in
    `forcing_to_mesh_offset_m`.
- `sfincs_wind`, `sfincs_pressure`, and `sfincs_plot` stages to SFINCS workflow
- SFINCS coastal model workflow with full pipeline (download through `sfincs_run`)
- Polymorphic `ModelConfig` ABC with `SchismModelConfig` and `SfincsModelConfig`
    concrete implementations
- `MODEL_REGISTRY` for automatic model dispatch from YAML `model:` key
- `--model` option for `init` and `stages` CLI commands
- Model-specific compute parameters (SCHISM: multi-node MPI; SFINCS: single-node OpenMP)
- `${model}` variable in default path templates for model-aware directory naming

### Changed

- `DownloadStage.description` is now a property that derives its text from the
    configured data sources (e.g. "Download input data (NWM, TPXO)") instead of a static
    string.
- `hydromt` compatibility patches consolidated into `apply_all_patches()` with per-patch
    logging; individual imports replaced by a single call.
- `CoastalCalibConfig` now takes `model_config: ModelConfig` instead of separate
    `model`, `mpi`, and `sfincs` parameters
- `SlurmConfig` now contains only scheduling parameters (`job_name`, `partition`,
    `time_limit`, `account`, `qos`, `user`); compute resources (`nodes`,
    `ntasks_per_node`, `exclusive`) moved to `SchismModelConfig`
- Default path templates use `${model}_` prefix instead of hardcoded `schism_`
- Stage order and stage creation delegated to `ModelConfig` subclasses
- SFINCS datum handling split into two separate offsets: the former single
    `navd88_to_msl_m` field is replaced by `forcing_to_mesh_offset_m` (applied to
    boundary forcing before simulation) and `vdatum_mesh_to_msl_m` (applied to model
    output for observation comparison). The two offsets serve fundamentally different
    purposes and may have different values depending on the boundary source.
- SFINCS field renames: `model_dir` -> `prebuilt_dir`, `obs_points` ->
    `observation_points`, `obs_merge` -> `merge_observations`, `src_locations` ->
    `discharge_locations_file`, `src_merge` -> `merge_discharge`

### Fixed

- Call `expanduser()` before `resolve()` on all path config fields so that paths
    containing `~` are correctly expanded to the user's home directory.
- Call `monitor.end_workflow()` before returning early in no-wait mode (`submit` with
    `wait=False`), so that the workflow timing summary is always closed.
- Set `HDF5_USE_FILE_LOCKING=FALSE` in container environment to prevent
    `PermissionError` on NFS-mounted filesystems.
- Add conda environment paths (`PATH`, `LD_LIBRARY_PATH`) to the `run` path's
    `build_environment()` so that `mpiexec` and MPI shared libraries from the conda
    environment are found, matching the environment set up by the generated `submit`
    scripts. Without these paths, the `run` path could not locate `mpiexec`, causing MPI
    stages to hang or fail.
- Add MPI/EFA fabric tuning variables (`MPICH_OFI_STARTUP_CONNECT`,
    `FI_OFI_RXM_SAR_LIMIT`, etc.) to the `run` path's SCHISM environment, matching the
    `submit` path and preventing hangs on AWS `c5n` nodes.
- Suppress ESMF diagnostic output from SLURM logs by redirecting stdout to `/dev/null`
    for MPI stages and setting `ESMF.Manager(debug=False)`.
- Redirect container stdout/stderr to temporary files instead of pipes to prevent
    pipe-buffer deadlocks with MPI process trees (`mpiexec` → `singularity`), where
    inherited pipe file-descriptors in child processes can fill the OS pipe buffer (64
    KB on Linux) and deadlock the entire tree.
- Use `$COASTAL_DOMAIN` instead of hardcoded `prvi` in `make_tpxo_ocean.bash` so the
    correct open-boundary mesh is used for all domains.
- Add missing `$` in `${PDY}` variable expansion in `post_regrid_stofs.bash` log
    filename.
- Correct malformed shebangs (`#/usr/bin/evn`) in `pre_nwm_forcing_coastal.bash` and
    `post_nwm_forcing_coastal.bash`.
- Use integer division (`//`) for the netCDF array index in `WrfHydroFECPP/fecpp/app.py`
    to avoid `float` index errors.
- Use numeric comparison (`-gt`) instead of string comparison (`>`) for `LENGTH_HRS` in
    `update_param.bash`.
- Add missing sub-hourly `CHRTOUT` symlinks for Hawaii in the last-timestep block of
    `initial_discharge.bash`.
- Read `NSCRIBES` from the environment with a fallback default instead of hardcoding it
    in `pre_schism.bash` and `run_sing_coastal_workflow_post_schism.bash`.
- Compute `LENGTH_HRS` in `STOFSBoundaryStage` directly instead of parsing stdout from
    the pre-script, which was silently lost after the `Popen.communicate()` fix
    redirected stdout to `/dev/null`.
- Remove duplicate domain-to-inland/geogrid mappings in `runner.py` and use the
    canonical properties from `SimulationConfig` to prevent the two copies from drifting
    out of sync.
- Correct shebangs (`#!/usr/bin/bash` → `#!/usr/bin/env bash`) in
    `pre_regrid_stofs.bash` and `post_regrid_stofs.bash` for consistency and
    portability.
- Source inner bash scripts from `$SCRIPTS_DIR` instead of `./` in all wrapper scripts,
    so that the bind-mounted (package) versions are used rather than the stale copies
    baked into the container image.
- Export `COASTAL_SCRIPTS_DIR`, `WRF_HYDRO_DIR`, `TPXO_SCRIPTS_DIR`, and
    `FORCINGS_SCRIPTS_DIR` in the `submit` path's generated runner script. These
    variables were only set in the `run` path, causing
    `$COASTAL_SCRIPTS_DIR/makeAtmo.py` (and similar) to resolve to just `/makeAtmo.py`
    and fail silently.
- Export date-component variables (`FORCING_START_YEAR`, `FORCING_START_MONTH`,
    `FORCING_START_DAY`, `FORCING_START_HOUR`, `PDY`, `cyc`, `FORCING_BEGIN_DATE`,
    `FORCING_END_DATE`, `END_DATETIME`) in the `submit` path header so that
    `makeAtmo.py`, `makeDischarge.py`, and other Python scripts inside the container
    have access to them across all stages.
- Add `set -e` to all inner bash scripts (`post_nwm_forcing_coastal.bash`,
    `initial_discharge.bash`, `merge_source_sink.bash`, `combine_sink_source.bash`,
    `pre_nwm_forcing_coastal.bash`, `post_regrid_stofs.bash`, `pre_regrid_stofs.bash`,
    `make_tpxo_ocean.bash`, `pre_schism.bash`, `post_schism.bash`, `update_param.bash`)
    so that command failures (e.g., `python` file-not-found or import errors) propagate
    instead of being silently swallowed.
- Correct shebang in `make_tpxo_ocean.bash` and `pre_schism.bash` (`#!/usr/bin/bash` →
    `#!/usr/bin/env bash`).
- Copy `setup_tpxo.txt` and `Model_tpxo10_atlas` from `$SCRIPTS_DIR` instead of `./` in
    `make_tpxo_ocean.bash`, so the bind-mounted (package) versions are used rather than
    stale copies baked into the container image.
- Truncate discharge arrays in `merge_source_sink.py` to match the precipitation
    timestep count from `precip_source.nc`, preventing a shape-mismatch `ValueError`
    when sub-hourly `CHRTOUT` files (e.g., Hawaii) produce one extra trailing timestep.
- Export `SCHISM_BEGIN_DATE` and `SCHISM_END_DATE` in the `submit` path header so that
    `update_param.bash` can patch `param.nml` with the correct simulation start/end
    dates. Without these, `param.nml` retains its template defaults (2000-01-01) and
    SCHISM aborts with a time mismatch against `sflux` forcing files.
- Report accurately which container stages completed vs failed when a SLURM job ends
    with a non-zero exit status, instead of marking all container stages as failed.

### Removed

- `MPIConfig` class (fields absorbed into `SchismModelConfig`)

[3.1.1.0.0]: https://github.com/NGWPC/nwm-coastal/releases/tag/3.1.1.0.0
[unreleased]: https://github.com/NGWPC/nwm-coastal/compare/3.1.1.0.0...HEAD
