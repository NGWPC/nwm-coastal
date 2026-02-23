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

Open `config.yaml` and set your simulation parameters:

```yaml
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

Before running, validate your configuration:

```bash
coastal-calibration validate config.yaml
```

This checks for:

- Required fields
- Valid date ranges for data sources
- File and directory existence
- Model-specific configuration validity

### Step 4: Run the Workflow

#### Heredoc sbatch Script (Recommended)

The preferred approach on clusters is to write an `sbatch` script with an inline YAML
configuration using a heredoc. This keeps the SLURM directives and workflow
configuration in a single, self-contained file:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=coastal_schism
#SBATCH --partition=c5n-18xlarge
#SBATCH -N 2
#SBATCH --ntasks-per-node=18
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out

CONFIG_FILE="/tmp/coastal_config_${SLURM_JOB_ID}.yaml"

cat > "${CONFIG_FILE}" <<'EOF'
model: schism

simulation:
  start_date: 2021-01-01
  duration_hours: 12
  coastal_domain: hawaii
  meteo_source: nwm_retro

boundary:
  source: tpxo

model_config:
  include_noaa_gages: true
EOF

/ngen-test/coastal-calibration/coastal-calibration run "${CONFIG_FILE}"
rm -f "${CONFIG_FILE}"
```

Save this as `my_run.sh` and submit with `sbatch my_run.sh`.

!!! tip "Use the full NFS path"

    Compute nodes may not have `coastal-calibration` in their `PATH`. Using the full path to
    the wrapper on the shared filesystem
    (`/ngen-test/coastal-calibration/coastal-calibration`) ensures the command is always
    found.

!!! tip "Unique config filenames"

    The config filename uses `$SLURM_JOB_ID` to avoid collisions when multiple jobs run
    concurrently.

!!! tip "Single-quoted heredoc"

    Use `<<'EOF'` (single-quoted) to prevent the shell from expanding `$` variables inside
    the YAML content. This ensures the YAML is written exactly as written.

Complete SCHISM and SFINCS sbatch examples are provided in
[`docs/examples/`](../examples/).

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

Set the path to a pre-built SFINCS model:

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

### Step 3: Validate and Submit

```bash
coastal-calibration validate sfincs_config.yaml
```

Then write an sbatch script using `coastal-calibration run sfincs_config.yaml`.

## Using the Python API

You can also run workflows programmatically:

```python
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

# Load configuration
config = CoastalCalibConfig.from_yaml("config.yaml")

# Create runner and execute
runner = CoastalCalibRunner(config)
result = runner.run()

if result.success:
    print(f"Completed in {result.duration_seconds:.1f}s")
else:
    print(f"Failed: {result.errors}")
```

## Next Steps

- Learn about [Configuration Options](../user-guide/configuration.md)
- Explore [Workflow Stages](../user-guide/workflow-stages.md)
- See the [CLI Reference](../user-guide/cli.md)
