# CLI Reference

The `coastal-calibration` command-line interface provides commands for managing SCHISM
and SFINCS coastal model workflows.

## Global Options

```bash
coastal-calibration --help
coastal-calibration --version
```

## Commands

### init

Create a minimal configuration file.

```bash
coastal-calibration init OUTPUT [OPTIONS]
```

**Arguments:**

| Argument | Description                                  |
| -------- | -------------------------------------------- |
| `OUTPUT` | Path where the configuration will be written |

**Options:**

| Option          | Description                            | Default   |
| --------------- | -------------------------------------- | --------- |
| `--domain`      | Coastal domain to use                  | `pacific` |
| `--force`, `-f` | Overwrite existing file without prompt | False     |
| `--model`       | Model type (`schism` or `sfincs`)      | `schism`  |

**Examples:**

```bash
# Generate default SCHISM configuration
coastal-calibration init config.yaml

# Generate configuration for a specific domain
coastal-calibration init pacific_config.yaml --domain pacific

# Generate SFINCS configuration
coastal-calibration init sfincs_config.yaml --domain atlgulf --model sfincs

# Overwrite existing file
coastal-calibration init config.yaml --force
```

### validate

Validate a configuration file for errors and warnings.

```bash
coastal-calibration validate <config>
```

**Arguments:**

| Argument | Description                    |
| -------- | ------------------------------ |
| `config` | Path to the configuration file |

**Examples:**

```bash
coastal-calibration validate config.yaml
```

**Output:**

```console
✓ Configuration is valid
```

Or with errors:

```console
✗ Configuration has errors:
  - simulation.duration_hours must be positive
  - Simulation dates outside nwm_ana range (2018-09-17 to present)
```

### run

Run the workflow directly (inside a SLURM job or for local testing).

```bash
coastal-calibration run <config> [OPTIONS]
```

**Arguments:**

| Argument | Description                    |
| -------- | ------------------------------ |
| `config` | Path to the configuration file |

**Options:**

| Option         | Description                              | Default |
| -------------- | ---------------------------------------- | ------- |
| `--start-from` | Stage to start from                      | First   |
| `--stop-after` | Stage to stop after                      | Last    |
| `--dry-run`    | Validate configuration without executing | False   |

**Available Stages (SCHISM):**

- `download`
- `pre_forcing`
- `nwm_forcing`
- `post_forcing`
- `schism_obs`
- `update_params`
- `boundary_conditions`
- `pre_schism`
- `schism_run`
- `post_schism`
- `schism_plot`

**Available Stages (SFINCS):**

- `download`
- `sfincs_symlinks`
- `sfincs_data_catalog`
- `sfincs_init`
- `sfincs_timing`
- `sfincs_forcing`
- `sfincs_obs`
- `sfincs_discharge`
- `sfincs_precip`
- `sfincs_wind`
- `sfincs_pressure`
- `sfincs_write`
- `sfincs_run`
- `sfincs_plot`

**Examples:**

```bash
# Run entire workflow
coastal-calibration run config.yaml

# Run only forcing stages (SCHISM)
coastal-calibration run config.yaml --start-from pre_forcing --stop-after post_forcing

# Run only the model build (SFINCS)
coastal-calibration run config.yaml --stop-after sfincs_write

# Run only the model execution (SFINCS)
coastal-calibration run config.yaml --start-from sfincs_run
```

#### Using `run` Inside a SLURM Job — Heredoc (Recommended)

The recommended way to run workflows on a cluster is to write an `sbatch` script with an
inline YAML configuration using a heredoc. This keeps everything in a single,
self-contained file and gives you full control over SLURM resource allocation:

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

Key points:

- **Use the full NFS path**: Compute nodes may not have `coastal-calibration` in their
    `PATH`. Using the full path to the wrapper on the shared filesystem ensures the
    command is always found.
- **`run` executes all stages sequentially**: All stages execute locally on the
    allocated nodes.
- **Use `$SLURM_JOB_ID` in the config filename**: Ensures uniqueness when multiple jobs
    run concurrently.
- **Use `<<'EOF'`** (single-quoted heredoc): Prevents shell variable expansion inside
    the YAML content.
- **SCHISM uses multi-node MPI**: `-N 2 --ntasks-per-node=18` matches the default
    `model_config` values (2 nodes, 18 tasks/node).
- **SFINCS uses single-node OpenMP**: `-N 1 --ntasks=1` is sufficient since SFINCS is
    OpenMP-only (parallelism is controlled by `model_config.omp_num_threads`).

Complete examples for both models are available in `docs/examples/`:

- [`schism.sh`](https://github.com/NGWPC/nwm-coastal/blob/development/docs/examples/schism.sh)
    — SCHISM multi-node MPI
- [`sfincs.sh`](https://github.com/NGWPC/nwm-coastal/blob/development/docs/examples/sfincs.sh)
    — SFINCS single-node OpenMP

### create

Create a SFINCS quadtree model from an AOI polygon.

```bash
coastal-calibration create <config> [OPTIONS]
```

**Arguments:**

| Argument | Description                    |
| -------- | ------------------------------ |
| `config` | Path to the configuration file |

**Options:**

| Option         | Description                              | Default |
| -------------- | ---------------------------------------- | ------- |
| `--start-from` | Stage to start from                      | First   |
| `--stop-after` | Stage to stop after                      | Last    |
| `--dry-run`    | Validate configuration without executing | False   |

**Available Stages:**

- `create_grid` — Create SFINCS grid from AOI polygon
- `create_fetch_elevation` — Fetch NOAA topobathy DEM for AOI
- `create_elevation` — Add elevation and bathymetry data
- `create_mask` — Create active cell mask
- `create_boundary` — Create water level boundary cells
- `create_subgrid` — Create subgrid tables
- `create_write` — Write SFINCS model to disk

**Examples:**

```bash
# Run entire creation workflow
coastal-calibration create create_config.yaml

# Run only up to grid generation
coastal-calibration create create_config.yaml --stop-after create_grid

# Resume from elevation stage
coastal-calibration create create_config.yaml --start-from create_elevation

# Dry run to validate config
coastal-calibration create create_config.yaml --dry-run
```

### prepare-topobathy

Download a NWS 30 m topobathymetric DEM clipped to an AOI bounding box.

```bash
coastal-calibration prepare-topobathy <aoi> [OPTIONS]
```

**Arguments:**

| Argument | Description                                            |
| -------- | ------------------------------------------------------ |
| `aoi`    | Path to an AOI polygon file (GeoJSON, Shapefile, etc.) |

**Options:**

| Option         | Description                                               | Default              |
| -------------- | --------------------------------------------------------- | -------------------- |
| `--domain`     | Coastal domain (`atlgulf`, `hi`, `prvi`, `pacific`, `ak`) | **required**         |
| `--output-dir` | Output directory for GeoTIFF + catalog                    | Same as AOI location |
| `--buffer-deg` | BBox buffer in degrees                                    | 0.1                  |

**Examples:**

```bash
# Download DEM for Atlantic/Gulf domain
coastal-calibration prepare-topobathy aoi.geojson --domain atlgulf

# Download to a specific directory
coastal-calibration prepare-topobathy aoi.geojson --domain prvi --output-dir ./dem_data
```

### update-dem-index

Rebuild the NOAA DEM spatial index from S3 STAC metadata.

```bash
coastal-calibration update-dem-index [OPTIONS]
```

**Options:**

| Option           | Description                                               | Default           |
| ---------------- | --------------------------------------------------------- | ----------------- |
| `--output`       | Write index to this path instead of the packaged location | Packaged location |
| `--max-datasets` | Limit S3 scan to N datasets (for testing)                 | All               |

**Examples:**

```bash
# Rebuild the packaged index
coastal-calibration update-dem-index

# Write to a custom path
coastal-calibration update-dem-index --output ./my_index.json
```

### stages

List all available workflow stages.

```bash
coastal-calibration stages [OPTIONS]
```

**Options:**

| Option    | Description                               | Default  |
| --------- | ----------------------------------------- | -------- |
| `--model` | Show stages for a specific model/workflow | Show all |

Valid `--model` values: `schism`, `sfincs`, `create`.

**Examples:**

```bash
# List all stages for all workflows
coastal-calibration stages

# List only SCHISM stages
coastal-calibration stages --model schism

# List only SFINCS stages
coastal-calibration stages --model sfincs

# List only creation stages
coastal-calibration stages --model create
```

**Output (all):**

```console
SCHISM workflow stages:
  1. download: Download NWM/STOFS data (optional)
  2. pre_forcing: Prepare NWM forcing data
  3. nwm_forcing: Generate atmospheric forcing (MPI)
  4. post_forcing: Post-process forcing data
  5. schism_obs: Add NOAA observation stations
  6. update_params: Create SCHISM param.nml
  7. boundary_conditions: Generate boundary conditions (TPXO/STOFS)
  8. pre_schism: Prepare SCHISM inputs
  9. schism_run: Run SCHISM model (MPI)
  10. post_schism: Post-process SCHISM outputs
  11. schism_plot: Plot simulated vs observed water levels

SFINCS workflow stages:
  1. download: Download NWM/STOFS data (optional)
  2. sfincs_symlinks: Create .nc symlinks for NWM data
  3. sfincs_data_catalog: Generate HydroMT data catalog
  4. sfincs_init: Initialise SFINCS model (pre-built)
  5. sfincs_timing: Set SFINCS timing
  6. sfincs_forcing: Add water level forcing
  7. sfincs_obs: Add observation points
  8. sfincs_discharge: Add discharge sources
  9. sfincs_precip: Add precipitation forcing
  10. sfincs_wind: Add wind forcing
  11. sfincs_pressure: Add atmospheric pressure forcing
  12. sfincs_write: Write SFINCS model
  13. sfincs_run: Run SFINCS model (Singularity)
  14. sfincs_plot: Plot simulated vs observed water levels

SFINCS creation stages (create subcommand):
  1. create_grid: Create SFINCS grid from AOI polygon
  2. create_fetch_elevation: Fetch NOAA topobathy DEM for AOI
  3. create_elevation: Add elevation and bathymetry data
  4. create_mask: Create active cell mask
  5. create_boundary: Create water level boundary cells
  6. create_subgrid: Create subgrid tables
  7. create_write: Write SFINCS model to disk
```

## Exit Codes

| Code | Description                    |
| ---- | ------------------------------ |
| 0    | Success                        |
| 1    | Configuration validation error |
| 2    | Runtime error                  |
| 3    | Runtime error (stage failure)  |

## Environment Variables

The CLI respects these environment variables:

| Variable            | Description                    |
| ------------------- | ------------------------------ |
| `COASTAL_LOG_LEVEL` | Override default log level     |
| `SLURM_JOB_ID`      | Detected when running in SLURM |

## Shell Completion

To enable shell completion (bash/zsh):

```bash
# Bash
eval "$(_COASTAL_CALIBRATION_COMPLETE=bash_source coastal-calibration)"

# Zsh
eval "$(_COASTAL_CALIBRATION_COMPLETE=zsh_source coastal-calibration)"
```

Add this to your shell profile for persistent completion.
