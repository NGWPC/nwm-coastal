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
  - slurm.user is required
  - Simulation dates outside nwm_ana range (2018-09-17 to present)
```

### submit

Submit a workflow as a SLURM job.

```bash
coastal-calibration submit <config> [OPTIONS]
```

**Arguments:**

| Argument | Description                    |
| -------- | ------------------------------ |
| `config` | Path to the configuration file |

**Options:**

| Option                | Description                          | Default |
| --------------------- | ------------------------------------ | ------- |
| `--interactive`, `-i` | Wait for job completion with updates | False   |

**Examples:**

```bash
# Submit and return immediately
coastal-calibration submit config.yaml

# Submit and wait for completion
coastal-calibration submit config.yaml --interactive
coastal-calibration submit config.yaml -i
```

### run

Run the workflow directly (for testing or inside a SLURM job).

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
- `update_params`
- `schism_obs`
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
- `sfincs_write`
- `sfincs_run`

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

### stages

List all available workflow stages.

```bash
coastal-calibration stages [OPTIONS]
```

**Options:**

| Option    | Description                      | Default  |
| --------- | -------------------------------- | -------- |
| `--model` | Show stages for a specific model | Show all |

**Examples:**

```bash
# List all stages for both models
coastal-calibration stages

# List only SCHISM stages
coastal-calibration stages --model schism

# List only SFINCS stages
coastal-calibration stages --model sfincs
```

**Output (all):**

```console
SCHISM workflow stages:
  1. download: Download NWM/STOFS data (optional)
  2. pre_forcing: Prepare NWM forcing data
  3. nwm_forcing: Generate atmospheric forcing (MPI)
  4. post_forcing: Post-process forcing data
  5. update_params: Create SCHISM param.nml
  6. schism_obs: Add NOAA observation stations
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
  10. sfincs_write: Write SFINCS model
  11. sfincs_run: Run SFINCS model (Singularity)
```

## Exit Codes

| Code | Description                    |
| ---- | ------------------------------ |
| 0    | Success                        |
| 1    | Configuration validation error |
| 2    | Runtime error                  |
| 3    | Job submission failed          |
| 4    | Job execution failed           |

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
