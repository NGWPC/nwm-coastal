# Configuration

NWM Coastal uses YAML configuration files to define simulation parameters. This page
documents all available configuration options.

## Minimal Configuration

### SCHISM (default)

The simplest valid SCHISM configuration only requires:

```yaml
slurm:
  job_name: my_run
  user: your_username

simulation:
  start_date: 2021-06-11
  duration_hours: 24
  coastal_domain: hawaii
  meteo_source: nwm_ana

boundary:
  source: stofs
```

All other parameters have sensible defaults. When no `model` key is present, SCHISM is
assumed.

### SFINCS

A minimal SFINCS configuration requires a `model` key and a `model_config` section:

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

## Variable Interpolation

Configuration values support variable interpolation using `${section.key}` syntax:

```yaml
slurm:
  user: john

simulation:
  coastal_domain: hawaii

paths:
  work_dir: /data/${slurm.user}/${simulation.coastal_domain}
  # Resolves to: /data/john/hawaii
```

### Default Path Templates

If not specified, paths are automatically generated using model-aware templates:

| Path               | Default Template                                                                                                                                         |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `work_dir`         | `/ngen-test/coastal/${slurm.user}/${model}_${simulation.coastal_domain}_${boundary.source}_${simulation.meteo_source}/${model}_${simulation.start_date}` |
| `raw_download_dir` | `/ngen-test/coastal/${slurm.user}/${model}_${simulation.coastal_domain}_${boundary.source}_${simulation.meteo_source}/raw_data`                          |

The `${model}` variable resolves to `schism` or `sfincs` based on the `model` key.

## Configuration Sections

### Model Selection

The top-level `model` key selects the model type. It defaults to `schism` if omitted.

```yaml
model: sfincs  # or "schism" (default)
```

### SLURM Settings

Configure SLURM job scheduling. Compute resources (nodes, tasks) are model-specific and
live in the `model_config` section.

```yaml
slurm:
  job_name: coastal_calibration  # Job name shown in squeue
  user: your_username            # Required: your SLURM username
  partition: c5n-18xlarge        # SLURM partition
  time_limit:                    # Time limit (HH:MM:SS), null for no limit
  account:                       # SLURM account for billing
  qos:                           # Quality of Service
```

| Parameter    | Type   | Default               | Description         |
| ------------ | ------ | --------------------- | ------------------- |
| `job_name`   | string | `coastal_calibration` | SLURM job name      |
| `user`       | string | **required**          | Your SLURM username |
| `partition`  | string | `c5n-18xlarge`        | SLURM partition     |
| `time_limit` | string | null                  | Time limit          |
| `account`    | string | null                  | SLURM account       |
| `qos`        | string | null                  | Quality of Service  |

### Simulation Settings

Define the simulation time period and domain:

```yaml
simulation:
  start_date: 2021-06-11         # Start date (ISO format)
  duration_hours: 24             # Simulation length
  coastal_domain: hawaii         # Domain name
  meteo_source: nwm_ana          # Meteorological data source
  timestep_seconds: 3600         # Forcing time step
```

| Parameter          | Type     | Default      | Options                                |
| ------------------ | -------- | ------------ | -------------------------------------- |
| `start_date`       | datetime | **required** | ISO format date/datetime               |
| `duration_hours`   | int      | **required** | Positive integer                       |
| `coastal_domain`   | string   | **required** | `hawaii`, `prvi`, `atlgulf`, `pacific` |
| `meteo_source`     | string   | **required** | `nwm_ana`, `nwm_retro`                 |
| `timestep_seconds` | int      | 3600         | Forcing time step in seconds           |

#### Supported Date Formats

The `start_date` field accepts multiple formats:

```yaml
start_date: 2021-06-11              # Date only (midnight)
start_date: 2021-06-11T00:00:00     # ISO format with time
start_date: "2021-06-11 00:00:00"   # Date with space separator
start_date: 20210611                # Compact format
```

### Boundary Settings

Configure boundary conditions:

```yaml
boundary:
  source: stofs          # Boundary condition source
  stofs_file:            # Optional: explicit STOFS file path
```

| Parameter    | Type   | Default | Options         | Description               |
| ------------ | ------ | ------- | --------------- | ------------------------- |
| `source`     | string | `tpxo`  | `tpxo`, `stofs` | Boundary condition source |
| `stofs_file` | path   | null    | -               | Override STOFS file path  |

### Path Settings

Configure file system paths:

```yaml
paths:
  work_dir: /path/to/work         # Working directory (auto-generated if not set)
  raw_download_dir: /path/to/data # Download directory (auto-generated if not set)
  nfs_mount: /ngen-test           # NFS mount point
  singularity_image: /ngencerf-app/singularity/ngen-coastal.sif
  ngen_app_dir: /ngen-app
  hot_start_file:                 # Hot restart file for warm start
  conda_env_name: ngen_forcing_coastal
  parm_dir: /ngen-test/coastal/ngwpc-coastal
```

| Parameter           | Type   | Default                                      |
| ------------------- | ------ | -------------------------------------------- |
| `work_dir`          | path   | Auto-generated from template                 |
| `raw_download_dir`  | path   | Auto-generated from template                 |
| `nfs_mount`         | path   | `/ngen-test`                                 |
| `singularity_image` | path   | `/ngencerf-app/singularity/ngen-coastal.sif` |
| `ngen_app_dir`      | path   | `/ngen-app`                                  |
| `hot_start_file`    | path   | null                                         |
| `conda_env_name`    | string | `ngen_forcing_coastal`                       |
| `parm_dir`          | path   | `/ngen-test/coastal/ngwpc-coastal`           |

### Model Configuration

Model-specific parameters live in the `model_config` section. The contents depend on the
`model` key.

#### SCHISM Model Configuration

```yaml
# model: schism (default, can be omitted)
model_config:
  nodes: 2                        # Number of compute nodes
  ntasks_per_node: 18             # MPI tasks per node
  exclusive: true                 # Request exclusive node access
  nscribes: 2                     # SCHISM I/O scribes
  omp_num_threads: 2              # OpenMP threads
  oversubscribe: false            # Allow MPI oversubscription
  binary: pschism_wcoss2_NO_PARMETIS_TVD-VL.openmpi
  include_noaa_gages: true        # Enable NOAA observation stations & comparison plots
```

| Parameter            | Type   | Default                                     | Description                                  |
| -------------------- | ------ | ------------------------------------------- | -------------------------------------------- |
| `nodes`              | int    | 2                                           | Number of compute nodes                      |
| `ntasks_per_node`    | int    | 18                                          | MPI tasks per node                           |
| `exclusive`          | bool   | true                                        | Request exclusive node access                |
| `nscribes`           | int    | 2                                           | SCHISM I/O scribes                           |
| `omp_num_threads`    | int    | 2                                           | OpenMP threads                               |
| `oversubscribe`      | bool   | false                                       | Allow MPI oversubscription                   |
| `binary`             | string | `pschism_wcoss2_NO_PARMETIS_TVD-VL.openmpi` | SCHISM executable name                       |
| `include_noaa_gages` | bool   | false                                       | Enable NOAA station discovery and comparison |

#### NOAA Observation Stations (`include_noaa_gages`)

When set to `true`, two additional stages are activated in the SCHISM pipeline:

- **`schism_obs`**: Automatically discovers NOAA CO-OPS water level stations within the
    model domain by computing a concave hull around the open boundary nodes in
    `hgrid.gr3` and querying the CO-OPS API. Writes a `station.in` file so SCHISM
    outputs time-series at those locations, and a `station_noaa_ids.txt` companion that
    maps station indices to NOAA station IDs. The `pre_schism` stage then patches
    `param.nml` to enable station output (`iout_sta = 1`, `nspool_sta = 18`).
- **`schism_plot`**: After the SCHISM run completes, reads the station output
    (`staout_1`), fetches the corresponding NOAA CO-OPS observations (converting from
    MLLW to MSL datum), and generates 2×2 comparison plots saved to `figs/`.

Both stages require network access for NOAA CO-OPS API calls and are classified as
Python-only stages (they run on the login node in `submit` mode).

#### SFINCS Model Configuration

```yaml
model: sfincs

model_config:
  prebuilt_dir: /path/to/model    # Required: pre-built SFINCS model directory
  observation_points: []          # Observation point coordinates
  observation_locations_file:     # Observation locations file
  merge_observations: false       # Merge observations into model
  discharge_locations_file:       # Discharge source locations file
  merge_discharge: false          # Merge discharge into model
  omp_num_threads: 36             # OpenMP threads
  container_tag: latest           # SFINCS container tag
  container_image:                # Singularity image path
```

| Parameter                    | Type   | Default  |
| ---------------------------- | ------ | -------- |
| `prebuilt_dir`               | path   | required |
| `observation_points`         | list   | `[]`     |
| `observation_locations_file` | path   | null     |
| `merge_observations`         | bool   | false    |
| `discharge_locations_file`   | path   | null     |
| `merge_discharge`            | bool   | false    |
| `omp_num_threads`            | int    | 36       |
| `container_tag`              | string | latest   |
| `container_image`            | path   | null     |

### Monitoring Settings

Configure logging and monitoring:

```yaml
monitoring:
  log_level: INFO                 # Logging verbosity
  log_file:                       # Optional log file path
  enable_progress_tracking: true  # Show progress bars
  enable_timing: true             # Track stage timing
```

| Parameter                  | Type   | Default | Options                             |
| -------------------------- | ------ | ------- | ----------------------------------- |
| `log_level`                | string | `INFO`  | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `log_file`                 | path   | null    | Path to log file                    |
| `enable_progress_tracking` | bool   | true    | Show progress bars                  |
| `enable_timing`            | bool   | true    | Track and report stage timing       |

### Download Settings

Configure data download behavior:

```yaml
download:
  enabled: true           # Enable automatic downloads
  skip_existing: true     # Skip already downloaded files
  timeout: 600            # Download timeout in seconds
  raise_on_error: true    # Fail on download errors
  limit_per_host: 4       # Concurrent downloads per host
```

| Parameter        | Type | Default | Description                      |
| ---------------- | ---- | ------- | -------------------------------- |
| `enabled`        | bool | true    | Enable automatic downloads       |
| `skip_existing`  | bool | true    | Skip already downloaded files    |
| `timeout`        | int  | 600     | Download timeout (seconds)       |
| `raise_on_error` | bool | true    | Fail workflow on download errors |
| `limit_per_host` | int  | 4       | Concurrent downloads per host    |

## Configuration Inheritance

Use `_base` to inherit settings from another configuration file:

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
# hawaii_run.yaml - inherits from base.yaml
_base: base.yaml

simulation:
  start_date: 2021-06-11
  coastal_domain: hawaii
```

```yaml
# prvi_run.yaml - different domain, same settings
_base: base.yaml

simulation:
  start_date: 2022-09-18
  coastal_domain: prvi
```

This allows you to:

- Share common settings across multiple runs
- Override only the parameters that differ
- Maintain consistency across related simulations

## Validation

Validate your configuration before running:

```bash
coastal-calibration validate config.yaml
```

The validation checks:

- All required fields are present
- Date ranges are valid for selected data sources
- File paths exist (for required files)
- SLURM parameters are valid
- Model-specific configuration is consistent (e.g., nscribes < total MPI tasks for
    SCHISM)
