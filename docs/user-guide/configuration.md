# Configuration

NWM Coastal uses YAML configuration files to define simulation parameters. This page
documents all available configuration options.

## Minimal Configuration

The simplest valid configuration only requires:

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

All other parameters have sensible defaults.

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

If not specified, paths are automatically generated:

| Path               | Default Template                                                                                                                                     |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `work_dir`         | `/ngen-test/coastal/${slurm.user}/schism_${simulation.coastal_domain}_${boundary.source}_${simulation.meteo_source}/schism_${simulation.start_date}` |
| `raw_download_dir` | `/ngen-test/coastal/${slurm.user}/schism_${simulation.coastal_domain}_${boundary.source}_${simulation.meteo_source}/raw_data`                        |

## Configuration Sections

### SLURM Settings

Configure SLURM job submission:

```yaml
slurm:
  job_name: coastal_calibration  # Job name shown in squeue
  user: your_username            # Required: your SLURM username
  nodes: 2                       # Number of compute nodes
  ntasks_per_node: 18            # MPI tasks per node
  partition: c5n-18xlarge        # SLURM partition
  exclusive: true                # Request exclusive node access
  time_limit:                    # Time limit (HH:MM:SS), null for no limit
  account:                       # SLURM account for billing
  qos:                           # Quality of Service
```

| Parameter         | Type   | Default               | Description             |
| ----------------- | ------ | --------------------- | ----------------------- |
| `job_name`        | string | `coastal_calibration` | SLURM job name          |
| `user`            | string | **required**          | Your SLURM username     |
| `nodes`           | int    | 2                     | Number of nodes         |
| `ntasks_per_node` | int    | 18                    | Tasks per node          |
| `partition`       | string | `c5n-18xlarge`        | SLURM partition         |
| `exclusive`       | bool   | true                  | Request exclusive nodes |
| `time_limit`      | string | null                  | Time limit (HH:MM:SS)   |
| `account`         | string | null                  | SLURM account           |
| `qos`             | string | null                  | Quality of Service      |

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

### MPI Settings

Configure MPI execution:

```yaml
mpi:
  nscribes: 2                     # SCHISM I/O scribes
  omp_num_threads: 2              # OpenMP threads
  oversubscribe: false            # Allow MPI oversubscription
  schism_binary: pschism_wcoss2_NO_PARMETIS_TVD-VL.openmpi
```

| Parameter         | Type   | Default                                     |
| ----------------- | ------ | ------------------------------------------- |
| `nscribes`        | int    | 2                                           |
| `omp_num_threads` | int    | 2                                           |
| `oversubscribe`   | bool   | false                                       |
| `schism_binary`   | string | `pschism_wcoss2_NO_PARMETIS_TVD-VL.openmpi` |

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
  nodes: 2

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
- MPI configuration is consistent
