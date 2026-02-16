All notable changes to this project will be documented in this file.
We follow the [Semantic Versioning 2.0.0](http://semver.org/) format.


## x.y.z - YYYY-MM-DD

### Added

- Lorem ipsum dolor sit amet

### Deprecated

- Nothing.

### Removed

- Nothing.

### Fixed

- Nothing.
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
- `forcing_to_mesh_offset_m` option in `SfincsModelConfig` to apply a vertical offset
    to boundary-condition water levels before they enter SFINCS. For tidal-only sources
    like TPXO, this anchors the tidal signal to the correct geodetic height of MSL on the
    mesh.
- `vdatum_mesh_to_msl_m` option in `SfincsModelConfig` to convert SFINCS output from
    the mesh vertical datum to MSL for comparison with NOAA CO-OPS observations.
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
[unreleased]: https://github.com/NGWPC/nwm-coastal/compare/v0.1.0...HEAD
