# Quick Start

This guide walks you through running your first coastal simulation using the
`coastal-calibration` CLI.

## Prerequisites

Before starting, ensure you have:

- `coastal-calibration` installed (see [Installation](installation.md))
- Access to an HPC cluster with SLURM
- The Singularity image at `/ngencerf-app/singularity/ngen-coastal.sif`
- Access to the `/ngen-test` NFS mount

## SCHISM Quick Start

### Step 1: Generate a Configuration File

Create a new configuration file for your simulation:

```bash
coastal-calibration init config.yaml --domain hawaii
```

This generates a template configuration file with sensible defaults.

### Step 2: Edit the Configuration

Open `config.yaml` and set your SLURM username:

```yaml
slurm:
  job_name: my_schism_run
  user: your_username  # Replace with your SLURM username

simulation:
  start_date: 2021-06-11
  duration_hours: 24
  coastal_domain: hawaii
  meteo_source: nwm_ana

boundary:
  source: stofs
```

!!! tip "Minimal Configuration"

    The configuration above is all you need! Paths are automatically generated based on your
    username, domain, and data sources.

### Step 3: Validate the Configuration

Before submitting, validate your configuration:

```bash
coastal-calibration validate config.yaml
```

This checks for:

- Required fields
- Valid date ranges for data sources
- File and directory existence
- SLURM configuration validity

### Step 4: Submit the Job

#### Option A: Submit and Return Immediately (Default)

```bash
coastal-calibration submit config.yaml
```

This will:

1. Run Python-only pre-job stages on the login node (download, observation stations)
1. Generate a SLURM job script for container stages
1. Submit the job and return immediately

```console
INFO  Running download stage on login node...
INFO  meteo/nwm_ana: 4/4 [OK]
INFO  hydro/nwm: 16/16 [OK]
INFO  coastal/stofs: 1/1 [OK]
INFO  Total: 21/21 (failed: 0)
INFO  Download stage completed
INFO  Job 167 submitted.
INFO  Check job status with: squeue -j 167
```

#### Option B: Submit and Wait for Completion

Use the `--interactive` flag to monitor the job until it completes:

```bash
coastal-calibration submit config.yaml --interactive
```

#### Option C: Submit a Partial Pipeline

Use `--start-from` and `--stop-after` (same options as `run`) to submit only part of the
workflow:

```bash
coastal-calibration submit config.yaml --start-from boundary_conditions -i
coastal-calibration submit config.yaml --stop-after post_forcing
```

### Step 5: Check Results

After the job completes, find your outputs in the work directory:

```bash
ls /ngen-test/coastal/your_username/schism_hawaii_stofs_nwm_ana/schism_2021-06-11/
```

!!! tip "Compare with NOAA observations"

    Add `include_noaa_gages: true` under `model_config` to automatically discover NOAA
    CO-OPS water level stations, collect station time-series during the simulation, and
    generate comparison plots after the run completes. See
    [Configuration](../user-guide/configuration.md#noaa-observation-stations-include_noaa_gages)
    for details.

## SFINCS Quick Start

### Step 1: Generate a SFINCS Configuration

```bash
coastal-calibration init sfincs_config.yaml --domain atlgulf --model sfincs
```

### Step 2: Edit the Configuration

Set your username and the path to a pre-built SFINCS model:

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

### Step 3: Validate and Submit

```bash
coastal-calibration validate sfincs_config.yaml
coastal-calibration submit sfincs_config.yaml --interactive
```

## Using the Python API

You can also run workflows programmatically:

```python
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

# Load configuration
config = CoastalCalibConfig.from_yaml("config.yaml")

# Create runner and submit
runner = CoastalCalibRunner(config)
result = runner.submit(wait=True)

if result.success:
    print(f"Job completed in {result.duration_seconds:.1f}s")
else:
    print(f"Job failed: {result.errors}")
```

## Next Steps

- Learn about [Configuration Options](../user-guide/configuration.md)
- Explore [Workflow Stages](../user-guide/workflow-stages.md)
- See the [CLI Reference](../user-guide/cli.md)
