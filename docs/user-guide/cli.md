# CLI Reference

The `coastal-calibration` command-line interface provides commands for managing SCHISM
coastal model workflows.

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

**Examples:**

```bash
# Generate default configuration
coastal-calibration init config.yaml

# Generate configuration for a specific domain
coastal-calibration init pacific_config.yaml --domain pacific

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

**Default Behavior (Non-Interactive):**

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

**Interactive Mode:**

```console
INFO  Running download stage on login node...
INFO  Download stage completed
INFO  Job submitted with ID: 167
INFO  Waiting for job 167 to complete...
INFO  Job 167 state: PENDING
INFO  Job 167 state: CONFIGURING
INFO  Job 167 state: RUNNING
INFO  Job 167 state: COMPLETED
INFO  Job 167 completed successfully.
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

**Available Stages:**

- `download`
- `pre_forcing`
- `nwm_forcing`
- `post_forcing`
- `update_params`
- `boundary_conditions`
- `pre_schism`
- `schism_run`
- `post_schism`

**Examples:**

```bash
# Run entire workflow
coastal-calibration run config.yaml

# Run only forcing stages
coastal-calibration run config.yaml --start-from pre_forcing --stop-after post_forcing

# Run from SCHISM preparation onwards
coastal-calibration run config.yaml --start-from pre_schism

# Run only download stage
coastal-calibration run config.yaml --stop-after download
```

### stages

List all available workflow stages.

```bash
coastal-calibration stages
```

**Output:**

```console
Available workflow stages:
  1. download            - Download NWM/STOFS data
  2. pre_forcing         - Prepare NWM forcing data
  3. nwm_forcing         - Generate atmospheric forcing (MPI)
  4. post_forcing        - Post-process forcing data
  5. update_params       - Create SCHISM param.nml file
  6. boundary_conditions - Generate boundary conditions
  7. pre_schism          - Prepare SCHISM inputs
  8. schism_run          - Run SCHISM model (MPI)
  9. post_schism         - Post-process outputs
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
