# Coastal Calibration: Coastal Model Calibration Workflow

A Python package for running SCHISM and SFINCS coastal model calibration workflows on
HPC clusters with Singularity containers and SLURM job scheduling.

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
pixi r -e dev coastal-calibration submit config.yaml -i
```

For the rest of this section, we will omit the `pixi r -e dev` prefix for brevity, but
it is required when running from the development environment.

Generate a configuration file, adjust it, then submit:

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
slurm:
  job_name: my_schism_run
  user: your_username

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

slurm:
  job_name: my_sfincs_run
  user: your_username

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

Validate and submit:

```bash
coastal-calibration validate config.yaml
coastal-calibration submit config.yaml
```

Both `run` and `submit` execute the same stage pipeline. The difference is that `run`
executes everything locally (for use inside an interactive compute session), while
`submit` runs Python-only stages on the login node and submits container stages as a
SLURM job. Both support `--start-from` and `--stop-after` for partial workflows.

By default, the CLI submits the job and returns immediately after the Python-only
pre-job stages complete (e.g., download, observation station discovery):

```console
INFO  Running download stage on login node...
INFO  meteo/nwm_ana: 4/4 [OK]
INFO  hydro/nwm: 16/16 [OK]
INFO  coastal/stofs: 1/1 [OK]
INFO  Total: 21/21 (failed: 0)
INFO  Download stage completed
INFO  Generated job script: .../submit_job.sh
INFO  Generated runner script: .../sing_run_generated.bash
INFO  Submitting job: .../submit_job.sh
INFO  Job 167 submitted.
INFO  Once the job starts running, SLURM logs will be written to: .../slurm-167.out
INFO  Check job status with: squeue -j 167
```

Use the `--interactive` (or `-i`) flag to wait and monitor the job until completion:

```bash
coastal-calibration submit config.yaml --interactive
```

Run partial pipelines with `--start-from` and `--stop-after`:

```bash
coastal-calibration submit config.yaml --start-from boundary_conditions
coastal-calibration submit config.yaml --stop-after post_forcing -i
```

## Python API

```python
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

config = CoastalCalibConfig.from_yaml("config.yaml")
runner = CoastalCalibRunner(config)
result = runner.submit()

if result.success:
    print(f"Job {result.job_id} completed in {result.duration_seconds:.1f}s")
```

### Running Partial Workflows

```python
# Both run() and submit() support start_from/stop_after
result = runner.run(start_from="pre_forcing", stop_after="post_forcing")
result = runner.run(start_from="pre_schism")
result = runner.submit(wait=True, start_from="boundary_conditions")
```

## Configuration Reference

### SLURM Settings

| Parameter    | Type | Default               | Description        |
| ------------ | ---- | --------------------- | ------------------ |
| `job_name`   | str  | `coastal_calibration` | SLURM job name     |
| `partition`  | str  | `c5n-18xlarge`        | SLURM partition    |
| `time_limit` | str  | null                  | Time limit         |
| `account`    | str  | null                  | SLURM account      |
| `qos`        | str  | null                  | Quality of Service |
| `user`       | str  | null                  | SLURM username     |

### Model Configuration

Model-specific parameters live in `model_config`. The `model` key selects which model
type to use (`schism` or `sfincs`). SCHISM is the default when no `model` key is
present.

#### SCHISM (`SchismModelConfig`)

| Parameter            | Type | Default                                     | Description                                  |
| -------------------- | ---- | ------------------------------------------- | -------------------------------------------- |
| `nodes`              | int  | 2                                           | Number of compute nodes                      |
| `ntasks_per_node`    | int  | 18                                          | MPI tasks per node                           |
| `exclusive`          | bool | true                                        | Request exclusive nodes                      |
| `nscribes`           | int  | 2                                           | Number of SCHISM I/O scribes                 |
| `omp_num_threads`    | int  | 2                                           | OpenMP threads                               |
| `oversubscribe`      | bool | false                                       | Allow MPI oversubscription                   |
| `binary`             | str  | `pschism_wcoss2_NO_PARMETIS_TVD-VL.openmpi` | SCHISM executable name                       |
| `include_noaa_gages` | bool | false                                       | Enable NOAA station discovery and comparison |

#### SFINCS (`SfincsModelConfig`)

| Parameter                    | Type  | Default  | Description                            |
| ---------------------------- | ----- | -------- | -------------------------------------- |
| `prebuilt_dir`               | path  | required | Path to pre-built SFINCS model         |
| `observation_points`         | list  | `[]`     | Observation point coordinates          |
| `observation_locations_file` | path  | null     | Observation locations file             |
| `merge_observations`         | bool  | false    | Merge observations into model          |
| `discharge_locations_file`   | path  | null     | Discharge source locations file        |
| `merge_discharge`            | bool  | false    | Merge discharge into model             |
| `meteo_res`                  | float | null     | Meteo forcing output resolution (m)    |
| `omp_num_threads`            | int   | auto     | OpenMP threads (defaults to CPU count) |
| `container_tag`              | str   | latest   | SFINCS container tag                   |
| `container_image`            | path  | null     | Singularity image path                 |

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

| Parameter           | Type | Default                                      | Description                        |
| ------------------- | ---- | -------------------------------------------- | ---------------------------------- |
| `work_dir`          | path | -                                            | Working directory for outputs      |
| `raw_download_dir`  | path | null                                         | Directory with downloaded NWM data |
| `nfs_mount`         | path | `/ngen-test`                                 | NFS mount point                    |
| `singularity_image` | path | `/ngencerf-app/singularity/ngen-coastal.sif` | Singularity image                  |
| `hot_start_file`    | path | null                                         | Hot restart file for warm start    |

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

Each stage is either Python-only (runs on login node in `submit`) or container-based
(runs inside SLURM job in `submit`). In `run` mode all stages execute locally.

1. **`download`** - Download NWM/STOFS data _(Python-only)_
1. **`pre_forcing`** - Prepare NWM forcing data _(container)_
1. **`nwm_forcing`** - Generate atmospheric forcing (MPI) _(container)_
1. **`post_forcing`** - Post-process forcing data _(container)_
1. **`update_params`** - Create SCHISM `param.nml` file _(container)_
1. **`schism_obs`** - Add NOAA observation stations _(Python-only)_
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

## Configuration Inheritance

Use `_base` to inherit from a shared configuration. This is useful for running the same
simulation across different domains or time periods:

```yaml
# base.yaml - shared settings
slurm:
  job_name: coastal_sim
  user: your_username

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

# Submit job and return immediately (default)
coastal-calibration submit config.yaml

# Submit job and wait for completion with status updates
coastal-calibration submit config.yaml --interactive
coastal-calibration submit config.yaml -i

# Submit with partial pipeline
coastal-calibration submit config.yaml --start-from boundary_conditions
coastal-calibration submit config.yaml --stop-after post_forcing -i

# Run workflow locally (inside SLURM job or for testing)
coastal-calibration run config.yaml
coastal-calibration run config.yaml --start-from update_params

# List available workflow stages
coastal-calibration stages
coastal-calibration stages --model schism
coastal-calibration stages --model sfincs
```

## License

BSD-2-Clause. See [LICENSE](LICENSE) for details.
