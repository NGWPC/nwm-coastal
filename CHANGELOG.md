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

- Unified `run` and `submit` execution pipelines — both commands now execute the same
    stage pipeline. `submit` automatically partitions stages into login-node
    (Python-only) and SLURM job (container) groups.
- `--start-from` and `--stop-after` options for `submit` command, matching `run`
- `requires_container` class attribute on `WorkflowStage` for automatic stage
    classification (Python-only vs container)
- `schism_obs` stage (NOAA observation station discovery) and `schism_plot` stage
    (simulated vs observed water level plots) to SCHISM workflow
- `sfincs_wind`, `sfincs_pressure`, and `sfincs_plot` stages to SFINCS workflow
- SFINCS coastal model workflow with full pipeline (download through sfincs_run)
- Polymorphic `ModelConfig` ABC with `SchismModelConfig` and `SfincsModelConfig`
    concrete implementations
- `MODEL_REGISTRY` for automatic model dispatch from YAML `model:` key
- `--model` option for `init` and `stages` CLI commands
- Model-specific compute parameters (SCHISM: multi-node MPI; SFINCS: single-node OpenMP)
- `${model}` variable in default path templates for model-aware directory naming

### Changed

- `CoastalCalibConfig` now takes `model_config: ModelConfig` instead of separate
    `model`, `mpi`, and `sfincs` parameters
- `SlurmConfig` now contains only scheduling parameters (`job_name`, `partition`,
    `time_limit`, `account`, `qos`, `user`); compute resources (`nodes`,
    `ntasks_per_node`, `exclusive`) moved to `SchismModelConfig`
- Default path templates use `${model}_` prefix instead of hardcoded `schism_`
- Stage order and stage creation delegated to `ModelConfig` subclasses
- SFINCS field renames: `model_dir` -> `prebuilt_dir`, `obs_points` ->
    `observation_points`, `obs_merge` -> `merge_observations`, `src_locations` ->
    `discharge_locations_file`, `src_merge` -> `merge_discharge`, `docker_tag` ->
    `container_tag`, `sif_path` -> `container_image`

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
