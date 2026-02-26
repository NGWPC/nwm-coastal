# Installation

## Requirements

- Python >= 3.11
- Access to an HPC cluster with SLURM and Singularity
- NFS mount point (default: `/ngen-test`)
- Singularity image with SCHISM and/or SFINCS and dependencies pre-compiled

!!! note "Model Executables"

    This package orchestrates SCHISM and SFINCS workflows on HPC clusters where the models
    are **already compiled and available** (typically inside a Singularity container). You
    do not need to install SCHISM or SFINCS locally to use this package for job submission.

## Install from PyPI

```bash
pip install coastal-calibration
```

This installs the core package with CLI and workflow orchestration capabilities.

## Install from Source

For development or to get the latest features, install from source:

```bash
git clone https://github.com/NGWPC/nwm-coastal
cd nwm-coastal
pip install -e .
```

## Development Installation with Pixi

For development, we recommend using [Pixi](https://pixi.prefix.dev/latest/) for
environment management:

```bash
# Install Pixi (Linux/macOS)
curl -fsSL https://pixi.sh/install.sh | sh
```

!!! tip "Restart Terminal"

    After installing Pixi, restart your terminal or run `source ~/.bashrc` (or
    `source ~/.zshrc` for Zsh) to make the `pixi` command available.

```bash
# Clone and install
git clone https://github.com/NGWPC/nwm-coastal
cd nwm-coastal
pixi install -e dev
```

### Available Environments

| Environment | Description                                 | Command                         |
| ----------- | ------------------------------------------- | ------------------------------- |
| `dev`       | Development with all tools                  | `pixi r -e dev <cmd>`           |
| `test311`   | Testing with Python 3.11                    | `pixi r -e test311 test`        |
| `test314`   | Testing with Python 3.14                    | `pixi r -e test314 test`        |
| `schism`    | Local development with SCHISM I/O libraries | `pixi r -e schism <cmd>`        |
| `sfincs`    | Local development with HydroMT-SFINCS       | `pixi r -e sfincs <cmd>`        |
| `typecheck` | Type checking with Pyright                  | `pixi r -e typecheck typecheck` |
| `lint`      | Linting with pre-commit                     | `pixi r lint`                   |
| `docs`      | Documentation building                      | `pixi r -e docs docs-serve`     |

## Optional Dependencies

Optional dependencies are available for **local development purposes only**. They are
useful for:

- Reading and analyzing model output files
- Debugging and testing workflow components locally
- Building SFINCS models with HydroMT

!!! warning "Not Required for Cluster Execution"

    These optional dependencies are **not required** to submit and run jobs on the cluster.
    The actual SCHISM and SFINCS executables must be pre-compiled and available on the HPC
    cluster (inside the Singularity container).

```bash
# SCHISM I/O dependencies (netCDF, numpy, etc.) - for local development
pip install coastal-calibration[schism]

# SFINCS/HydroMT dependencies - for local model building and analysis
pip install coastal-calibration[sfincs]

# Development dependencies (Jupyter, etc.)
pip install coastal-calibration[dev]

# Documentation dependencies
pip install coastal-calibration[docs]
```

## Verify Installation

After installation, verify by running:

```bash
coastal-calibration --help
```

You should see the CLI help output with available commands:

```console
Usage: coastal-calibration [OPTIONS] COMMAND [ARGS]...

  NWM Coastal: Coastal model workflow on HPC clusters.

Commands:
  init      Generate a new configuration file.
  validate  Validate a configuration file.
  submit    Submit workflow as a SLURM job.
  run       Run workflow directly.
  stages    List available workflow stages.
```
