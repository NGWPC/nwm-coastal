# forecast_demo setup (short version)

This is the setup steps in order, without the ecflow material and edge cases
covered in `README.md`. For the full reference, use that file.

Two parts: this file covers one-time environment setup. `forecast_walkthrough.py`
is the script used to actually generate a forecast once setup is complete.

## Setup (one-time per machine)

### 1. Clone nwm-coastal

```bash
git clone git@github.com:NGWPC/nwm-coastal.git
cd nwm-coastal
git submodule update --init --recursive
```

### 2. Build the pixi environment

```bash
cd nwm-coastal
pixi install -e dev
```

Compiles SFINCS and SCHISM from source. Slow on first run, cached after that.

### 3. Create the wrapper scripts

`pixi install` does not create `nwm-coastal-cli`/`nwm-coastal-py`. These must
be created manually, once. `hotstart_coastal_models.sh`, `gen_cycle_config.py`,
and the ecflow tasks all call these two files directly as executables — none
of them will run without this step. Script contents are in
`docs/getting-started/cluster-install.md`; copy them into the repo root and
`chmod +x` both.

```bash
cd nwm-coastal
INSTALL_DIR="$(pwd)"

cat > "${INSTALL_DIR}/nwm-coastal-cli" <<WRAPPER
#!/bin/bash
set -eu

_ENV="${INSTALL_DIR}/.pixi/envs/dev"

export PATH="\${_ENV}/bin:\${PATH:-}"
export LD_LIBRARY_PATH="\${_ENV}/lib:\${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="\${_ENV}"
export HDF5_USE_FILE_LOCKING=FALSE

for _script in "\${_ENV}"/etc/conda/activate.d/*.sh; do
    [ -f "\$_script" ] && . "\$_script"
done

exec python -m coastal_calibration.cli "\$@"
WRAPPER
chmod +x "${INSTALL_DIR}/nwm-coastal-cli"

cat > "${INSTALL_DIR}/nwm-coastal-py" <<WRAPPER
#!/bin/bash
set -eu

_ENV="${INSTALL_DIR}/.pixi/envs/dev"

export PATH="\${_ENV}/bin:\${PATH:-}"
export LD_LIBRARY_PATH="\${_ENV}/lib:\${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="\${_ENV}"
export HDF5_USE_FILE_LOCKING=FALSE

for _script in "\${_ENV}"/etc/conda/activate.d/*.sh; do
    [ -f "\$_script" ] && . "\$_script"
done

exec python "\$@"
WRAPPER
chmod +x "${INSTALL_DIR}/nwm-coastal-py"
```

Confirm both work:
```bash
./nwm-coastal-py -c "print('ok')"
./nwm-coastal-cli --help
```

### 4. Clone nwm-rte

```bash
git clone git@github.com:NGWPC/nwm-rte.git
```

Temporary steps: checkout the coastalforcing-pw branch of the nwm-rte.

On the same directory level as nwm-rte,

```bash
git clone git@github.com:NGWPC/ngen-forcing.git
```

And checkout the coastalforcing-pw branch of that repo as well.

### 5. Build the RTE Docker image

Temporary steps: because we are using non-development branches
to create the image, recommend exporting a target image name before
building.

```bash
export TARGET_IMAGE_NAME=ngen_rte_coastal
cd nwm-rte
./setup_clone_repos.sh https
./ngen_rte_build.sh
```

Default tag is `ngen_rte_ghcr` (from `TARGET_IMAGE_NAME` in `config.bashrc`).
If a different tag is used here, `TARGET_IMAGE_NAME` must be exported to match
in every later step, or downstream scripts default to `ngen_rte_ghcr` and will
not find the image.

### 6. Set AWS credentials

Requires read access to `s3://ngwpc-coastal` and `s3://ngwpc-dev`.

Note: these bucket names are not expected to be permanent. These data will likely 
be relocated.

### 7. Pull the coastal forecast data

```bash
./setup_data_coastal_forecast.sh
```

What it downloads, and where (all under `$RUN_NGEN_ROOT` unless noted):

- ESMF mesh domain files (`geo_em_*.nc`, `GEOGRID_LDASOUT_Spatial_Metadata_*.nc`)
  for CONUS/Alaska/Hawaii/PRVI → `data/esmf_mesh/NWM/domain/`
- Pre-computed ESMF regrid weights → `data/esmf_mesh/regrid_weights/`
- Regionalization inputs (same as `setup_data.sh -r`) — requires
  `nwm-region-mgr` to already be cloned
- SCHISM/SFINCS model directories (`sfincs_models/`, `schism_models/`) and
  the base per-cycle run templates (`schism_sims/run.yaml`,
  `sfincs_sims/run.yaml`) → `$RUN_COASTAL_ROOT`, created as a sibling of
  `nwm-rte`
- VPU03S extract polygon (`esmf_conus_03s_extract.geojson`) →
  `data/esmf_mesh/esmf_domain_extract/`

One item is not yet handled by this script and currently requires a manual
step:

- The NHF dataset required for the SCHISM crosswalk (`nwmReaches.csv` →
  `ngenReaches.csv`) is not downloaded here.

The VPU hydrofabric geopackage copy is handled by `forecast_walkthrough.py`
(step 3) instead of here -- it copies
`nwm-region-mgr/data/inputs/region/hydrofabric/gpkg_vpu/vpu_<VPU>.gpkg` to
`$RUN_NGEN_ROOT/data/hydrofabric/vpu_<VPU>.gpkg` if it isn't already there.
This is needed because the Icefabric API t-route would normally query for
this isn't reliably reachable from all networks; `nwm-region-mgr` must
already be cloned as a sibling of `nwm-coastal` (which is done in clone repos
above).

Finish the exports:

```bash
export NWM_COASTAL_ROOT=/path/to/nwm-coastal
export NWM_RTE_ROOT=/path/to/nwm-rte
export RUN_NGEN_ROOT=/path/to/run_ngen    # usually on same level as rte
export RUN_COASTAL_ROOT=/path/to/run_coastal    # also usually on same level as rte
```

Proceed to `forecast_walkthrough.py`.
