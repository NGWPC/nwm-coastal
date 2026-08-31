# forecast_demo setup (short version)

This is the setup steps in order, without the ecflow material and edge cases
covered in `README.md`. For the full reference, use that file.

Two parts: this file covers one-time environment setup. `forecast_walkthrough2.py`
is the script used to actually generate a forecast once setup is complete.

## Setup (one-time per machine)

### 1. Clone nwm-coastal

```bash
git clone git@github.com:NGWPC/nwm-coastal.git
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

### 4. Clone nwm-rte

```bash
git clone git@github.com:NGWPC/nwm-rte.git
```

### 5. Build the RTE Docker image

```bash
cd nwm-rte
./ngen_rte_build.sh
```

Default tag is `ngen_rte_ghcr` (from `TARGET_IMAGE_NAME` in `config.bashrc`).
If a different tag is used here, `TARGET_IMAGE_NAME` must be exported to match
in every later step, or downstream scripts default to `ngen_rte_ghcr` and will
not find the image.

### 6. Set AWS credentials

Requires read access to `s3://ngwpc-coastal` and `s3://ngwpc-dev`.

Note: these bucket names are not expected to be permanent. Once this pipeline
is operational, this data will likely be relocated.

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
  `ngenReaches.csv`) is not downloaded here. That crosswalk step is currently
  broken/skipped rather than functional.

The VPU hydrofabric geopackage copy is handled by `forecast_walkthrough2.py`
(step 3) instead of here -- it copies
`nwm-region-mgr/data/inputs/region/hydrofabric/gpkg_vpu/vpu_<VPU>.gpkg` to
`$RUN_NGEN_ROOT/data/hydrofabric/vpu_<VPU>.gpkg` if it isn't already there.
This is needed because the Icefabric API t-route would normally query for
this isn't reliably reachable from all networks; `nwm-region-mgr` must
already be cloned as a sibling of `nwm-coastal`.

Setup is complete after this step. Proceed to `forecast_walkthrough2.py`.
