# Quick Start

This guide walks you through running your first coastal simulation.

## Prerequisites

- Pixi installed with the `dev` environment (see [Installation](installation.md))
- For SCHISM: a pre-built model directory and geogrid file
- For SFINCS: a pre-built model directory (or create one from an AOI polygon using the
    `create` workflow)

## SCHISM Quick Start

### Step 1: Generate a Configuration File

```bash
pixi r -e dev coastal-calibration init config.yaml --domain hawaii
```

This generates a template configuration file with sensible defaults.

### Step 2: Edit the Configuration

Open `config.yaml` and set your simulation parameters:

```yaml
simulation:
  start_date: 2025-11-26
  duration_hours: 50
  coastal_domain: hawaii
  meteo_source: nwm_ana

boundary:
  source: stofs

model_config:
  prebuilt_dir: /path/to/schism/model
  geogrid_file: /path/to/geo_em_HI.nc
  include_noaa_gages: true
```

!!! tip "Minimal Configuration"

    For SCHISM, you need to specify `prebuilt_dir` (the pre-built mesh and config files) and
    `geogrid_file` (for atmospheric forcing regridding). Everything else has sensible
    defaults.

### Step 3: Validate and Run

```bash
pixi r -e dev coastal-calibration validate config.yaml
pixi r -e dev coastal-calibration run config.yaml
```

The pipeline executes 11 stages: download, forcing preparation, atmospheric regridding,
boundary conditions, mesh partitioning, model execution, and validation against NOAA
observations.

### Running on HPC

On clusters, write an `sbatch` script with an inline YAML config:

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
  start_date: 2025-11-26
  duration_hours: 50
  coastal_domain: hawaii
  meteo_source: nwm_ana

boundary:
  source: stofs

model_config:
  include_noaa_gages: true
EOF

coastal-calibration run "${CONFIG_FILE}"
rm -f "${CONFIG_FILE}"
```

Submit with `sbatch my_run.sh`. Complete examples are provided for
[SCHISM](../examples/slurm/schism.sh) and [SFINCS](../examples/slurm/sfincs.sh).

!!! tip "Single-quoted heredoc"

    Use `<<'EOF'` (single-quoted) to prevent the shell from expanding `$` variables inside
    the YAML content.

## SFINCS Quick Start

### Step 1: Generate a SFINCS Configuration

```bash
pixi r -e dev coastal-calibration init sfincs_config.yaml --domain atlgulf --model sfincs
```

### Step 2: Edit the Configuration

Set the path to a pre-built SFINCS model:

```yaml
model: sfincs

simulation:
  start_date: 2024-01-09
  duration_hours: 60
  coastal_domain: atlgulf
  meteo_source: nwm_ana

boundary:
  source: stofs

model_config:
  prebuilt_dir: /path/to/prebuilt/sfincs/model
```

### Step 3: Validate and Run

```bash
pixi r -e dev coastal-calibration validate sfincs_config.yaml
pixi r -e dev coastal-calibration run sfincs_config.yaml
```

## SFINCS Model Creation

The `create` command builds a new SFINCS quadtree model from an AOI polygon, handling
grid generation, DEM download, elevation, masking, boundary cells, and subgrid tables.

!!! tip "Draw the AOI in QGIS"

    Use the [QGIS Plugin](../user-guide/qgis-plugin.md) to interactively draw your AOI
    polygon, snap it to watershed divides, select discharge points, and export everything as
    GeoJSON.

### Step 1: Prepare a Topobathy DEM

```bash
pixi r -e dev coastal-calibration prepare-topobathy aoi.geojson --domain atlgulf --output-dir ./dem
```

### Step 2: Write a Creation Config

```yaml
aoi: aoi.geojson
output_dir: ./my_sfincs_model

grid:
  crs: EPSG:32617

elevation:
  datasets:
    - name: nws_topobathy
      zmin: -20000

data_catalog:
  data_libs:
    - ./dem/data_catalog.yml
```

### Step 3: Run the Creation Workflow

```bash
pixi r -e dev coastal-calibration create create_config.yaml
```

The output directory will contain a ready-to-run SFINCS model that can be used as
`prebuilt_dir` in a simulation config.

## Python API

You can also run workflows programmatically:

```python
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

config = CoastalCalibConfig.from_yaml("config.yaml")
runner = CoastalCalibRunner(config)
result = runner.run()

if result.success:
    print(f"Completed in {result.duration_seconds:.1f}s")
else:
    print(f"Failed: {result.errors}")
```

### Partial Workflows

Restart from a specific stage or stop early:

```python
result = runner.run(start_from="schism_forcing_prep", stop_after="schism_sflux")
```

Or via the CLI:

```bash
pixi r -e dev coastal-calibration run config.yaml --start-from schism_boundary
pixi r -e dev coastal-calibration run config.yaml --stop-after schism_sflux
```

## Next Steps

- Learn about [Configuration Options](../user-guide/configuration.md)
- Explore [Workflow Stages](../user-guide/workflow-stages.md)
- See the [CLI Reference](../user-guide/cli.md)
- Try the example notebooks:
    [Narragansett Bay (SFINCS)](../examples/notebooks/narragansett.ipynb) and
    [Hawaii (SCHISM)](../examples/notebooks/schism-hawaii.ipynb)
