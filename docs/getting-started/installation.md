# Installation

## Prerequisites

- [Git](https://git-scm.com/)
- [Pixi](https://pixi.prefix.dev/latest/installation/)

Pixi handles all other dependencies including Python, and compiling SFINCS and SCHISM
from source. No containers are required.

## Install

```bash
git clone --recurse-submodules https://github.com/NGWPC/nwm-coastal
cd nwm-coastal
pixi install -e dev
```

The first activation compiles both SFINCS and SCHISM from source (they are included as
git submodules). Subsequent activations skip the build if dependencies have not changed.

## Running Commands

All commands must be run through Pixi to activate the environment:

```bash
pixi r -e dev coastal-calibration --help
```

You should see the CLI help output:

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

## Available Environments

| Environment | Description                          | Command                         |
| ----------- | ------------------------------------ | ------------------------------- |
| `dev`       | Every binary and dev tool            | `pixi r -e dev <cmd>`           |
| `default`   | Python package only, no model binaries | `pixi r <cmd>`                |
| `test313`   | Test suite as run in CI              | `pixi r -e test313 test`        |
| `typecheck` | Type checking                        | `pixi r -e typecheck typecheck` |
| `lint`      | Linting with pre-commit              | `pixi r lint`                   |
| `docs`      | Documentation building               | `pixi r -e docs docs-serve`     |

All environments use Python 3.13; the package requires 3.13 or newer.

`dev` installs every binary (SCHISM, SFINCS, predict_tide) and every Python dependency
needed to develop, test, and run the package. `test313` mirrors it without the notebook
and changelog tooling, and is what CI runs. `default` is the lightweight option: the
Python package and its dependencies, without SCHISM, SFINCS, or the MPI/ESMF stack, so
it installs without compiling anything.

## QGIS Plugin

The repository includes a QGIS plugin for interactive domain definition. See the
[QGIS Plugin guide](../user-guide/qgis-plugin.md) for installation and usage
instructions.
