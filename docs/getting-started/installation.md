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

## Download Model Data

The workflows need model data that is not in the repository: prebuilt SCHISM and SFINCS
models, ESMF mesh and domain files, the TPXO tidal atlas, and hydrofabric copies. With
AWS credentials for `s3://ngwpc-dev`, one script fetches all of it:

```bash
./scripts/setup_data_coastal.sh
```

It asks where each directory should go, defaulting to siblings of your `nwm-coastal`
checkout, and writes:

| Directory                  | Contents                                  |
| -------------------------- | ----------------------------------------- |
| `run_coastal/`             | Prebuilt SCHISM and SFINCS models         |
| `coastal_data/`            | TPXO tidal atlas, hydrofabric copies      |
| `run_ngen/data/esmf_mesh/` | ESMF mesh, domain, and extract files      |

Set `RUN_NGEN_ROOT` or `RUN_COASTAL_ROOT` beforehand to skip those prompts. Use
`--dry-run` to preview, and `--help` for the full options. `run_coastal` is large, so
the first run takes a while.

Point `paths.tidal_atlas_dir` at `coastal_data/TPXO10_atlas_v2_nc` in any configuration
that uses harmonic tides.

### Without AWS credentials

Most of this data has a public equivalent:

- **SCHISM models.** NOAA publishes the NWM coastal module parameters at
    [water.noaa.gov/about/nwm](https://water.noaa.gov/about/nwm), as
    [`NWM_coastal_parameters.tar.gz`](https://www.nohrsc.noaa.gov/owp_files/nwm/nwm_parameters/NWM_coastal_parameters.tar.gz).
    Use an unpacked domain directory as `model_config.prebuilt_dir`. These may not
    include the WRF geogrid file that `model_config.geogrid_file` needs; check the
    [NWM parameter files](https://www.nohrsc.noaa.gov/owp_files/nwm/nwm_parameters/NWM_parameter_files_v3.0.tar.gz)
    as well.
- **SFINCS models.** No credentials needed. The `create` workflow builds a model from an
    AOI polygon using public elevation and land-cover sources, so you can make your own
    — see the [Lavaca Bay example](../examples/notebooks/lavaca.ipynb).
- **Tides.** The TPXO10 atlas is free for academic and non-commercial use but requires
    registration at [tpxo.net](https://www.tpxo.net/). It is only needed for
    `boundary.source: harmonic`; `stofs` and `glofs` are public and need no
    registration.
- **Hydrofabric.** NextGen hydrofabric geopackages are published by
    [Lynker Spatial](https://docs.lynker-spatial.com/data-service/hydrofabric) under
    ODbL.

Retrospective and analysis runs work entirely from public data once you have a model:
NWM forcing, STOFS, and GLOFS are all open. The ecFlow forecast demo in `forecast_demo/`
is the exception — it additionally needs the `nwm-rte` repository and the staged data
above.

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
