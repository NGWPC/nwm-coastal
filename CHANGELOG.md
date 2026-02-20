# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-19

### Added

- Per-stage completion tracking in the `submit` path's generated runner script: each
    container stage is recorded in `.pipeline_status.json` as it finishes, enabling
    mid-pipeline restarts after a SLURM job failure without re-running expensive stages
    (e.g., `predict_tide`). On resubmit, completed stages are automatically skipped.
- `meteo_res` option in `SfincsModelConfig` to control the output resolution (m) of
    gridded meteorological forcing (precipitation, wind, pressure). When not set, the
    resolution is derived from the SFINCS quadtree grid base cell size.
- Meteo grid clipping (`_clip_meteo_to_domain`) that trims reprojected meteo grids to
    the model domain extent, preventing the LCC → UTM reprojection from inflating grids
    to CONUS scale — **reducing SFINCS runtime from 15 h+ to under 15 min**.
- Stale netCDF file cleanup in `SfincsInitStage` to prevent HDF5 segfaults when
    re-running a pipeline over an existing model directory.
- Geodataset-based water-level forcing with IDW interpolation to boundary points,
    replacing the built-in `model.water_level.create(geodataset=...)` which passed all
    source stations incompatibly with `.bnd` files.
- Active-cell filtering for discharge source points to prevent a SFINCS Fortran segfault
    when a source point falls on an inactive grid cell.
- `apply_all_patches()` convenience function in `_hydromt_compat` that applies all
    hydromt/hydromt-sfincs compatibility patches in one call, with logging.
- `quiet` parameter on `WorkflowMonitor.mark_stage_completed()` to control whether a
    visible COMPLETED log line is emitted for externally-executed stages.
- Unified `run` and `submit` execution pipelines — both commands now execute the same
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
- SFINCS coastal model workflow with full pipeline (download through sfincs_run)
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
- Hydromt compatibility patches consolidated into `apply_all_patches()` with per-patch
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
    `discharge_locations_file`, `src_merge` -> `merge_discharge`, `docker_tag` ->
    `container_tag`, `sif_path` -> `container_image`

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
- Add missing sub-hourly CHRTOUT symlinks for Hawaii in the last-timestep block of
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
    `FORCINGS_SCRIPTS_DIR` in the `submit` path's generated runner script — these
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
    when sub-hourly CHRTOUT files (e.g., Hawaii) produce one extra trailing timestep.
- Export `SCHISM_BEGIN_DATE` and `SCHISM_END_DATE` in the `submit` path header so that
    `update_param.bash` can patch `param.nml` with the correct simulation start/end
    dates — without these, `param.nml` retains its template defaults (2000-01-01) and
    SCHISM aborts with a time mismatch against `sflux` forcing files.
- Report accurately which container stages completed vs failed when a SLURM job ends
    with a non-zero exit status, instead of marking all container stages as failed.

### Removed

- `MPIConfig` class (fields absorbed into `SchismModelConfig`)

## [0.1.0] - 2026-02-06

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

### Supported Data Sources

- NWM Retrospective 3.0 (1979-02-01 to 2023-01-31)
- NWM Analysis (2018-09-17 to present)
- STOFS water levels (2020-12-30 to present)
- GLOFS (Great Lakes OFS, 2005-09-30 to present)
- TPXO tidal model (local installation required)

[0.1.0]: https://github.com/NGWPC/nwm-coastal/releases/tag/v0.1.0
[0.2.0]: https://github.com/NGWPC/nwm-coastal/compare/v0.1.0...v0.2.0
