# Installation

## Requirements

- Python >= 3.11
- Access to an HPC cluster with Singularity (SCHISM workflow only)
- NFS mount point (default: `/ngen-test`)
- Singularity image with SCHISM and dependencies pre-compiled (SCHISM workflow only)
- Compiled SFINCS binary (SFINCS workflow only — see
    [Compiling SFINCS](../sfincs_compilation.md))

!!! note "Model Executables"

    The SCHISM workflow requires a pre-compiled Singularity container on the HPC cluster.
    The SFINCS workflow uses a natively compiled binary — pixi environments with the
    `sfincs` feature build it automatically on first activation.

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
| `typecheck` | Type checking with ty                       | `pixi r -e typecheck typecheck` |
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
    SCHISM must be pre-compiled inside a Singularity container on the HPC cluster. SFINCS is
    compiled natively (see [Compiling SFINCS](../sfincs_compilation.md)).

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

## QGIS Plugin

The repository includes a QGIS plugin for interactive AOI polygon creation. See the
[QGIS Plugin guide](../user-guide/qgis-plugin.md) for installation and usage
instructions.

## Verify Installation

After installation, verify by running:

```bash
coastal-calibration --help
```

You should see the CLI help output with available commands:

```console
Usage: coastal-calibration [OPTIONS] COMMAND [ARGS]...

  Coastal calibration workflow manager (SCHISM, SFINCS).

Commands:
  create             Create a SFINCS model from an AOI polygon.
  init               Create a minimal configuration file.
  prepare-topobathy  Download NWS topobathy DEM clipped to an AOI bounding box.
  run                Run the calibration workflow.
  stages             List available workflow stages.
  update-dem-index   Rebuild the NOAA DEM spatial index from S3 STAC metadata.
  validate           Validate a configuration file.
```
