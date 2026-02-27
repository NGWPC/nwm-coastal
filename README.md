# NWM Coastal: Coastal Model Workflow

`nwm-coastal` is a collection of tools designed to implement skill assessments for
coastal model runs and execute National Water Model (NWM) hindcast/forecast runs for
standalone coastal models utilized in NWMv4 operations. This includes the Semi-implicit
Cross-scale Hydroscience Integrated System Model (SCHISM) and Super-Fast INundation of
CoastS (SFINCS) as the intended coastal models for implementation in NWMv4 operations.
This repository stores symlinks to both coastal modeling development groups, which are
used to compile and run the NWMv4 coastal models. The coastal modeling output for NWMv4
operations is the Total Water Level (TWL) fields. These fields are the focus of the
coastal tools in this repository for skill assessments across NWM domains, as well as
the end result to retrieve from tools in this repository that set up and execute NWMv4
hindcast and operational forecast runs.

## Installation

```bash
pip install coastal-calibration
```

For development, use [Pixi](https://pixi.prefix.dev/latest/). First, install Pixi
following the instructions on the Pixi website, or run the following command for
Linux/macOS and restart your terminal:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Then, clone the repository and install the `dev` dependencies:

```bash
git clone https://github.com/NGWPC/nwm-coastal
cd nwm-coastal
pixi install -e dev
```

Requires Python >= 3.11.

## Quick Start

Note that for development, all commands need to be run with `pixi r -e dev` to activate
the virtual environment. For example:

```bash
pixi r -e dev coastal-calibration run config.yaml
```

For the rest of this section, we will omit the `pixi r -e dev` prefix for brevity, but
it is required when running from the development environment.

Generate a configuration file, adjust it, then run:

```bash
# SCHISM (default)
coastal-calibration init config.yaml --domain hawaii

# SFINCS
coastal-calibration init config.yaml --domain atlgulf --model sfincs
```

Edit `config.yaml` to set your simulation parameters. A minimal SCHISM configuration
only requires the following (paths are auto-generated based on user, domain, and
source):

```yaml
simulation:
  start_date: 2021-06-11
  duration_hours: 24
  coastal_domain: hawaii
  meteo_source: nwm_ana

boundary:
  source: stofs
```

A minimal SFINCS configuration requires a `model` key and a `model_config` section
pointing to a pre-built SFINCS model:

```yaml
model: sfincs

simulation:
  start_date: 2025-06-01
  duration_hours: 168
  coastal_domain: atlgulf
  meteo_source: nwm_ana

boundary:
  source: stofs

model_config:
  prebuilt_dir: /path/to/prebuilt/sfincs/model
```

Validate and run:

```bash
coastal-calibration validate config.yaml
coastal-calibration run config.yaml
```

The `run` command executes all stages sequentially on the current node. It is designed
to be used inside a user-written `sbatch` script for full control over SLURM resource
allocation. It supports `--start-from` and `--stop-after` for partial workflows:

```bash
coastal-calibration run config.yaml --start-from boundary_conditions
coastal-calibration run config.yaml --stop-after post_forcing
```

### Running Inside a SLURM Job (`sbatch`)

Write an `sbatch` script with an inline YAML configuration and use
`coastal-calibration run` to execute it. Complete examples are provided in
[`docs/examples/`](docs/examples/):

- [`schism.sh`](docs/examples/schism.sh) — SCHISM workflow (multi-node MPI)
- [`sfincs.sh`](docs/examples/sfincs.sh) — SFINCS workflow (single-node OpenMP)

```bash
sbatch docs/examples/schism.sh
```

### Creating a SFINCS Model from Scratch

Build a new SFINCS quadtree model from an AOI polygon:

```bash
# Optionally download NWS topobathy for the AOI
coastal-calibration prepare-topobathy aoi.geojson --domain atlgulf

# Create the model
coastal-calibration create create_config.yaml
```

The `create` workflow generates a quadtree SFINCS model with elevation, mask, boundary
cells, and subgrid tables. It supports automatic NOAA DEM discovery and download.

## Python API

```python
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

config = CoastalCalibConfig.from_yaml("config.yaml")
runner = CoastalCalibRunner(config)
result = runner.run()

if result.success:
    print(f"Completed in {result.duration_seconds:.1f}s")
```

### Running Partial Workflows

```python
result = runner.run(start_from="pre_forcing", stop_after="post_forcing")
result = runner.run(start_from="pre_schism")
```

## Configuration Reference

### Model Configuration

Model-specific parameters live in `model_config`. The `model` key selects which model
type to use (`schism` or `sfincs`). SCHISM is the default when no `model` key is
present.

#### SCHISM (`SchismModelConfig`)

| Parameter            | Type | Default                                      | Description                                  |
| -------------------- | ---- | -------------------------------------------- | -------------------------------------------- |
| `singularity_image`  | path | `/ngencerf-app/singularity/ngen-coastal.sif` | Singularity/Apptainer SIF image              |
| `nodes`              | int  | 2                                            | Number of compute nodes                      |
| `ntasks_per_node`    | int  | 18                                           | MPI tasks per node                           |
| `exclusive`          | bool | true                                         | Request exclusive nodes                      |
| `nscribes`           | int  | 2                                            | Number of SCHISM I/O scribes                 |
| `omp_num_threads`    | int  | 2                                            | OpenMP threads                               |
| `oversubscribe`      | bool | false                                        | Allow MPI oversubscription                   |
| `binary`             | str  | `pschism_wcoss2_NO_PARMETIS_TVD-VL.openmpi`  | SCHISM executable name                       |
| `include_noaa_gages` | bool | false                                        | Enable NOAA station discovery and comparison |

#### SFINCS (`SfincsModelConfig`)

| Parameter                    | Type  | Default  | Description                                              |
| ---------------------------- | ----- | -------- | -------------------------------------------------------- |
| `prebuilt_dir`               | path  | required | Path to pre-built SFINCS model                           |
| `model_root`                 | path  | null     | Output directory (defaults to `{work_dir}/sfincs_model`) |
| `observation_points`         | list  | `[]`     | Observation point coordinates                            |
| `observation_locations_file` | path  | null     | Observation locations file                               |
| `merge_observations`         | bool  | false    | Merge observations into model                            |
| `discharge_locations_file`   | path  | null     | Discharge source locations file                          |
| `merge_discharge`            | bool  | false    | Merge discharge into model                               |
| `include_noaa_gages`         | bool  | false    | Enable NOAA station discovery and comparison plots       |
| `include_precip`             | bool  | false    | Add precipitation forcing                                |
| `include_wind`               | bool  | false    | Add wind forcing                                         |
| `include_pressure`           | bool  | false    | Add atmospheric pressure forcing                         |
| `forcing_to_mesh_offset_m`   | float | 0.0      | Vertical offset (m) added to boundary forcing            |
| `vdatum_mesh_to_msl_m`       | float | 0.0      | Vertical offset (m) converting model output to MSL       |
| `meteo_res`                  | float | null     | Meteo forcing output resolution (m)                      |
| `sfincs_exe`                 | path  | null     | Local SFINCS executable (bypasses container)             |
| `omp_num_threads`            | int   | auto     | OpenMP threads (defaults to CPU count)                   |
| `container_tag`              | str   | latest   | SFINCS container tag                                     |
| `container_image`            | path  | null     | Singularity image path                                   |

### Simulation Settings

| Parameter          | Type     | Options                                | Description                   |
| ------------------ | -------- | -------------------------------------- | ----------------------------- |
| `start_date`       | datetime | -                                      | Simulation start (ISO format) |
| `duration_hours`   | int      | -                                      | Simulation length in hours    |
| `coastal_domain`   | str      | `prvi`, `hawaii`, `atlgulf`, `pacific` | Coastal domain                |
| `meteo_source`     | str      | `nwm_retro`, `nwm_ana`                 | Meteorological data source    |
| `timestep_seconds` | int      | 3600                                   | Forcing time step             |

### Boundary Settings

| Parameter    | Type | Options         | Description               |
| ------------ | ---- | --------------- | ------------------------- |
| `source`     | str  | `tpxo`, `stofs` | Boundary condition source |
| `stofs_file` | path | -               | STOFS file path           |

### Path Settings

All path fields support `~` (tilde) expansion, so you can write `~/my_data` instead of
the full home directory path.

| Parameter          | Type | Default      | Description                        |
| ------------------ | ---- | ------------ | ---------------------------------- |
| `work_dir`         | path | -            | Working directory for outputs      |
| `raw_download_dir` | path | null         | Directory with downloaded NWM data |
| `nfs_mount`        | path | `/ngen-test` | NFS mount point                    |
| `hot_start_file`   | path | null         | Hot restart file for warm start    |

## Supported Domains and Data Sources

**Domains**: `atlgulf`, `pacific`, `hawaii`, `prvi`

| Source      | Date Range               | Description           |
| ----------- | ------------------------ | --------------------- |
| `nwm_retro` | 1979-02-01 to 2023-01-31 | NWM Retrospective 3.0 |
| `nwm_ana`   | 2018-09-17 to present    | NWM Analysis          |
| `stofs`     | 2020-12-30 to present    | STOFS water levels    |
| `glofs`     | 2005-09-30 to present    | Great Lakes OFS       |
| `tpxo`      | N/A (local installation) | TPXO tidal model      |

## Workflow Stages

### SCHISM Stages

Each stage is classified as Python-only or container-based (requires Singularity). The
`run` command executes all stages sequentially.

1. **`download`** - Download NWM/STOFS data _(Python-only)_
1. **`pre_forcing`** - Prepare NWM forcing data _(container)_
1. **`nwm_forcing`** - Generate atmospheric forcing (MPI) _(container)_
1. **`post_forcing`** - Post-process forcing data _(container)_
1. **`schism_obs`** - Add NOAA observation stations _(Python-only)_
1. **`update_params`** - Create SCHISM `param.nml` file _(container)_
1. **`boundary_conditions`** - Generate boundary conditions _(container)_
1. **`pre_schism`** - Prepare SCHISM inputs _(container)_
1. **`schism_run`** - Run SCHISM model (MPI) _(container)_
1. **`post_schism`** - Post-process outputs _(container)_
1. **`schism_plot`** - Plot simulated vs observed water levels _(Python-only)_

### SFINCS Stages

1. **`download`** - Download NWM/STOFS data _(Python-only)_
1. **`sfincs_symlinks`** - Create `.nc` symlinks for NWM data _(Python-only)_
1. **`sfincs_data_catalog`** - Generate HydroMT data catalog _(Python-only)_
1. **`sfincs_init`** - Initialize SFINCS model (pre-built) _(Python-only)_
1. **`sfincs_timing`** - Set SFINCS timing _(Python-only)_
1. **`sfincs_forcing`** - Add water level forcing _(Python-only)_
1. **`sfincs_obs`** - Add observation points _(Python-only)_
1. **`sfincs_discharge`** - Add discharge sources _(Python-only)_
1. **`sfincs_precip`** - Add precipitation forcing _(Python-only)_
1. **`sfincs_wind`** - Add wind forcing _(Python-only)_
1. **`sfincs_pressure`** - Add atmospheric pressure forcing _(Python-only)_
1. **`sfincs_write`** - Write SFINCS model _(Python-only)_
1. **`sfincs_run`** - Run SFINCS model (Singularity) _(container)_
1. **`sfincs_plot`** - Plot simulated vs observed water levels _(Python-only)_

### SFINCS Creation Stages

Used by the `create` command to build a new SFINCS model from an AOI polygon:

1. **`create_grid`** - Create SFINCS quadtree grid from AOI polygon
1. **`create_fetch_elevation`** - Fetch NOAA topobathy DEM for AOI
1. **`create_elevation`** - Add elevation and bathymetry data
1. **`create_mask`** - Create active cell mask
1. **`create_boundary`** - Create water level boundary cells
1. **`create_discharge`** - Add NWM discharge source points _(optional)_
1. **`create_subgrid`** - Create subgrid tables
1. **`create_write`** - Write SFINCS model to disk

## Configuration Inheritance

Use `_base` to inherit from a shared configuration. This is useful for running the same
simulation across different domains or time periods:

```yaml
# base.yaml - shared settings
simulation:
  duration_hours: 24
  meteo_source: nwm_ana

boundary:
  source: stofs
```

```yaml
# hawaii_run.yaml - Hawaii-specific run
_base: base.yaml

simulation:
  start_date: 2021-06-11
  coastal_domain: hawaii
```

```yaml
# prvi_run.yaml - Puerto Rico/Virgin Islands run
_base: base.yaml

simulation:
  start_date: 2022-09-18
  coastal_domain: prvi
```

## CLI Reference

```bash
# Generate a new configuration file
coastal-calibration init config.yaml --domain pacific
coastal-calibration init config.yaml --domain atlgulf --model sfincs

# Validate a configuration file
coastal-calibration validate config.yaml

# Run workflow (inside SLURM job or for testing)
coastal-calibration run config.yaml
coastal-calibration run config.yaml --start-from update_params
coastal-calibration run config.yaml --stop-after post_forcing

# Create a SFINCS model from an AOI polygon
coastal-calibration create create_config.yaml
coastal-calibration create create_config.yaml --start-from create_elevation

# Download NWS topobathy DEM for an AOI
coastal-calibration prepare-topobathy aoi.geojson --domain atlgulf

# Rebuild NOAA DEM spatial index
coastal-calibration update-dem-index

# List available workflow stages
coastal-calibration stages
coastal-calibration stages --model schism
coastal-calibration stages --model sfincs
coastal-calibration stages --model create
```

## Credits and references

1. [NextGen Water Modeling Framework Prototype](https://github.com/NOAA-OWP/ngen)
1. [schism-dev](https://ccrm.vims.edu/schismweb/) community
1. [Deltares](https://www.deltares.nl/en/software-and-data/products/sfincs) community

## License

BSD-2-Clause. See [LICENSE](LICENSE) for details.
