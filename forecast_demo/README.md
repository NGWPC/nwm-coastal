# forecast_demo

Tools and infrastructure for running the hourly coastal forecast cycle
(nwm-rte forcing + t-route regionalization + SCHISM/SFINCS coastal runs).
This README is the complete, ordered runbook for everything a new user needs
to do to get from a fresh checkout to a running forecast.

## Two ways to run this

- **`server/` + `ecf_home/` + `suite_def/`** -- ecflow-orchestrated. Self-chains
  hourly, runs unattended once started. See "Usage" and "Cold start /
  initialization procedure" below.
- **`forecast_walkthrough.py`** -- runs one spinup + one AnA hour + one SR
  cycle directly, step by step, with no ecflow involved at all. Good for
  understanding exactly what each stage does and what it produces. See its
  own header comment for details.

Both share every prerequisite below (env vars, `nwm-rte` build, data
staging, wrappers) and both call the same underlying tools
(`bin/gen_cycle_config.py`, `bin/hotstart_coastal_models.sh`, the `nwm-rte`
Docker image).

## Required environment variables

Nothing in this directory hardcodes a path to a particular person's home
directory or checkout location -- every filesystem path is derived from
four environment variables, which must be set before running anything
under `server/` or `bin/`:

| Variable            | Points to                                                | Example                                  |
|----------------------|----------------------------------------------------------|-------------------------------------------|
| `NWM_COASTAL_ROOT`   | This repo's own checkout root (contains `forecast_demo/`, `nwm-coastal-cli`, `nwm-coastal-py`) | `/contrib/<you>/ngwpc/nwm-coastal` |
| `NWM_RTE_ROOT`       | `nwm-rte` repo checkout root                              | `/contrib/<you>/ngwpc/nwm-rte`            |
| `RUN_NGEN_ROOT`      | ngen forcing/regionalization working directory (`data/`, `regionalization/` live here) | `/contrib/<you>/ngwpc/run_ngen` |
| `RUN_COASTAL_ROOT`   | SCHISM/SFINCS working directory (`schism_sims/`, `sfincs_sims/` live here) | `/contrib/<you>/ngwpc/run_coastal` |

Set them once, e.g. in your shell profile or an untracked `.env` you
`source` before working in this directory:

```bash
export NWM_COASTAL_ROOT=/contrib/<you>/ngwpc/nwm-coastal
export NWM_RTE_ROOT=/contrib/<you>/ngwpc/nwm-rte
export RUN_NGEN_ROOT=/contrib/<you>/ngwpc/run_ngen
export RUN_COASTAL_ROOT=/contrib/<you>/ngwpc/run_coastal
```

`server/ecf_env.sh` requires `NWM_COASTAL_ROOT` (everything else it needs
-- `ECF_HOME`, the ecflow binary `PATH` -- derives from it or from its own
optional overrides, see below). `server/load_and_begin.sh` and
`server/seed_ring.sh` additionally require `NWM_RTE_ROOT`/`RUN_NGEN_ROOT`/
`RUN_COASTAL_ROOT`. `bin/hotstart_coastal_models.sh` requires all four
(its coastal spin-up only needs `NWM_COASTAL_ROOT`/`RUN_COASTAL_ROOT`, but
its t-route bootstrap step needs `NWM_RTE_ROOT`/`RUN_NGEN_ROOT` too, so all
four are checked up front regardless of which `--coastal-only`/
`--troute-only` flag you use). Each script fails fast with a clear error
if a variable it needs is unset.

Optional overrides:

- `ECF_PORT` (default `39411`) -- the local port the ecflow server listens on.
- `ECFLOW_BIN_DIR` (default `/contrib/software/ecflow/5.6.0/bin`) -- where the
  `ecflow_server`/`ecflow_client` 5.6.0 binaries live, if not on `PATH`
  already.

## Prerequisite: `nwm-rte` built and its coastal data staged

The coastal `.ecf` tasks and `bin/hotstart_coastal_models.sh` run everything
through the `nwm-rte` Docker image (`docker_run` in the generated `.ecf`
files) -- `nwm-rte` itself must already be cloned, its image built, and its
`run_ngen` data directory populated before any of this demo can run.

1. Clone `nwm-rte` as a sibling of `nwm-coastal` (matches `NWM_RTE_ROOT`
   above), then build its Docker image:
   ```bash
   cd "${NWM_RTE_ROOT}"
   ./ngen_rte_build.sh
   ```
   This tags the image using `TARGET_IMAGE_NAME` from `nwm-rte/config.bashrc`
   (`ngen_rte_ghcr` by default). The `.ecf` files/`hotstart_coastal_models.sh`
   export this same default, so no extra env var is needed unless you build
   under a different tag.
2. **Requires AWS credentials** with read access to `s3://ngwpc-coastal`
   (domain files, regrid weights, `run_coastal` working directory) and
   `s3://ngwpc-dev` (regionalization inputs, via `nwm-rte/setup_data.sh`'s
   `SOURCE_BUCKET_DEV`). Configure these however your organization provides
   them (e.g. `aws configure`, SSO, environment variables) *before* running
   the script below -- it will fail with a `NoCredentials`/`AccessDenied`
   error otherwise.

   Populate `RUN_NGEN_ROOT` with the data this demo needs (ESMF mesh domain
   files, pre-computed regrid weights, regionalization inputs, and the
   `run_coastal` working directory itself) by running `nwm-rte`'s coastal
   data setup script:
   ```bash
   cd "${NWM_RTE_ROOT}"
   ./setup_data_coastal_forecast.sh
   ```
   See that script's own header comment for exactly what it downloads and
   why (it intentionally skips gage-specific streamflow/calibration/RFC
   data that `setup_data.sh` pulls but this demo never uses). Pass `-s`/
   `--skip-existing` to avoid re-downloading the large domain files if
   you've already run this once.

## Prerequisite (workaround): stage the VPU hydrofabric geopackage locally

The t-route regionalization step (`run_regionalization_standalone`, called
from `troute_sr.ecf`/`troute_ana_a.ecf`/`troute_ana_b.ecf` and
`bin/hotstart_coastal_models.sh`'s troute bootstrap) normally fetches the
VPU's hydrofabric geopackage from the Icefabric API
(`edfs.test.nextgenwaterprediction.com`) at runtime. Whether that hostname
resolves depends entirely on the network the command runs from, so check it
on **each** environment you use rather than assuming one implies the other:

```bash
getent hosts edfs.test.nextgenwaterprediction.com
```

- **Resolves:** the API call succeeds on its own -- skip this section, no
  `--hydrofab_file` override needed.
- **Does not resolve** (`Name or service not known`): apply the workaround
  below. This has been observed both locally (no VPN access from this
  machine) and may also apply on a cluster if its compute/login nodes sit
  behind a firewall without a route to that host -- test independently on
  each, since cluster network egress rules are commonly different from a
  local machine's.

Until API access is confirmed working, stage the geopackage locally and pass
it explicitly:

1. Get the VPU's hydrofabric geopackage. It's already tracked in
   `nwm-region-mgr`, at
   `nwm-region-mgr/data/inputs/region/hydrofabric/gpkg_vpu/vpu_<VPU>.gpkg`
   (e.g. `vpu_03S.gpkg`).
2. Copy it into `RUN_NGEN_ROOT` so it's visible inside the `nwm-rte`
   container (the whole `RUN_NGEN_ROOT` directory is bind-mounted to
   `/ngwpc/run_ngen` -- see `nwm-rte/run.sh`):
   ```bash
   mkdir -p "${RUN_NGEN_ROOT}/data/hydrofabric"
   cp "${NWM_COASTAL_ROOT}/../nwm-region-mgr/data/inputs/region/hydrofabric/gpkg_vpu/vpu_<VPU>.gpkg" \
       "${RUN_NGEN_ROOT}/data/hydrofabric/vpu_<VPU>.gpkg"
   ```
3. The four troute invocations already pass
   `--hydrofab_file "/ngwpc/run_ngen/data/hydrofabric/vpu_%VPU%.gpkg"` (or
   `${VPU}` in `hotstart_coastal_models.sh`), which bypasses the Icefabric
   API call entirely once the file above exists (see `--hydrofab_file`'s own
   help text in `nwm-rte/bin_mounted/ngen_rte/run_config/cli_args.py`:
   "Path to local hydrofabric gpkg file. If provided, bypasses msw-mgr
   Icefabric API call.").

## Prerequisite: `nwm-coastal-py` / `nwm-coastal-cli` wrappers

`bin/hotstart_coastal_models.sh` and the generated `.ecf` task scripts call
`${NWM_COASTAL_ROOT}/nwm-coastal-py` and `${NWM_COASTAL_ROOT}/nwm-coastal-cli`
directly, as plain executables -- unconditionally, regardless of how you set
up `nwm-coastal` itself. Following `nwm-coastal`'s own
[`docs/getting-started/installation.md`](../docs/getting-started/installation.md)
(`git clone` + `pixi install -e dev`) does **not** create these two files --
that doc only documents running commands through `pixi r -e dev ...`. The
wrapper scripts are documented instead in
[`docs/getting-started/cluster-install.md`](../docs/getting-started/cluster-install.md),
which is written for a multi-node HPC deployment, but the wrapper-creation
step itself is not actually cluster-specific -- it works the same on a
single local machine.

**If `${NWM_COASTAL_ROOT}/nwm-coastal-py` doesn't already exist, create both
wrappers before running anything under `bin/` or `server/`:**

```bash
cd "${NWM_COASTAL_ROOT}"   # must have already run `pixi install -e dev` here

cat > nwm-coastal-cli <<'WRAPPER'
#!/bin/bash
set -eu
_ENV="$(dirname "$(readlink -f "$0")")/.pixi/envs/dev"
export PATH="${_ENV}/bin:${PATH:-}"
export LD_LIBRARY_PATH="${_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${_ENV}"
export HDF5_USE_FILE_LOCKING=FALSE
for _script in "${_ENV}"/etc/conda/activate.d/*.sh; do
    [ -f "$_script" ] && . "$_script"
done
exec coastal-calibration "$@"
WRAPPER
chmod +x nwm-coastal-cli

cat > nwm-coastal-py <<'WRAPPER'
#!/bin/bash
set -eu
_ENV="$(dirname "$(readlink -f "$0")")/.pixi/envs/dev"
export PATH="${_ENV}/bin:${PATH:-}"
export LD_LIBRARY_PATH="${_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${_ENV}"
export HDF5_USE_FILE_LOCKING=FALSE
for _script in "${_ENV}"/etc/conda/activate.d/*.sh; do
    [ -f "$_script" ] && . "$_script"
done
exec python "$@"
WRAPPER
chmod +x nwm-coastal-py
```

This is a one-time step per checkout, on either a local machine or a
cluster's shared filesystem.

- **Local, single machine:** stop here. `NWM_COASTAL_ROOT` already resolves
  both wrapper paths directly, so no further step is needed.
- **Cluster, multi-node:** additionally do `cluster-install.md`'s "Make it
  available to all users" step (an `/etc/profile.d/` drop-in adding the
  install directory to `PATH`). This demo's own scripts don't need it --
  they always invoke `${NWM_COASTAL_ROOT}/nwm-coastal-py`/`nwm-coastal-cli`
  by full path -- but Slurm-dispatched compute-node jobs launched *outside*
  this demo's own scripts (e.g. a user running `nwm-coastal-cli` directly
  in an `sbatch` script) won't have `NWM_COASTAL_ROOT` set unless that
  env var is also exported cluster-wide, so the `PATH` drop-in is the more
  robust way to make the wrappers reachable across nodes. See
  `cluster-install.md`'s own warning about node-local symlinks not working
  for compute nodes launched by Slurm.

## The suite definition is generated, not hand-edited

`suite_def/coastal_hourly.def.template` is the tracked source of truth --
it uses `${NWM_COASTAL_ROOT}`/`${NWM_RTE_ROOT}`/`${RUN_NGEN_ROOT}`/
`${RUN_COASTAL_ROOT}` placeholders in its `edit` block instead of literal
paths. `server/load_and_begin.sh` runs `envsubst` against it to produce
`suite_def/coastal_hourly.def` (installation-specific, gitignored) before
loading it into the server. If you change a path variable, edit the
`.template` file and re-run `load_and_begin.sh` -- never hand-edit the
generated `.def` directly, it gets overwritten.

## Usage

```bash
# one-time, per shell session (or put in your profile):
export NWM_COASTAL_ROOT=... NWM_RTE_ROOT=... RUN_NGEN_ROOT=... RUN_COASTAL_ROOT=...

server/start_server.sh       # starts ecflow_server if not already running
server/load_and_begin.sh     # generates coastal_hourly.def, loads it, begins the suite
server/stop_server.sh        # halts and terminates the server
```

`bin/gen_cycle_config.py` and `bin/hotstart_coastal_models.sh` are called
by the generated `.ecf` task scripts (via `%NWM_COASTAL_PY%`/
`%NWM_COASTAL_CLI%`/`%GEN_SCRIPT%`, all resolved from the same env vars)
and can also be run standalone -- see `hotstart_coastal_models.sh`'s own
header comment for its bootstrap/spin-up role.

## Cold start / initialization procedure

### When you need this

The suite self-chains hour-to-hour: every hour's `troute_ana_a`/
`schism_ana`/`sfincs_ana` triggers off the *same task in the previous
hour* (a true 24h ring -- h00's predecessor is `../h23`). On a fresh
`--begin`, there is no natural entry point *for any hour* -- every hour's
trigger is waiting on a predecessor that has never run. You need this
procedure the first time you ever begin the suite, or any time after an
outage long enough that no real prior-hour AnA state exists to self-chain
from.

### Procedure

1. Set the four env vars above.
2. Pick `TARGET_CYCLE` (`YYYYMMDDHH`, UTC) -- the first hour the live
   suite will run for real. Its predecessor hour (`TARGET_CYCLE` - 1h) is
   where all bootstrap state below lands.
3. Produce warm-start state for the predecessor hour:
   ```bash
   bin/hotstart_coastal_models.sh <TARGET_CYCLE>
   ```
   This is the slow step (an 18h SCHISM/SFINCS spin-up by default, plus a
   short t-route bootstrap run). Use `--dry-run` first to sanity-check env
   vars/paths without running anything, and shorten `SPINUP_HOURS`/
   `RAMP_HOURS` (positional args 2/3) for a quick smoke test. Use
   `--coastal-only`/`--troute-only` to re-run just one half if only that
   half needs redoing.
4. Start the ecflow server if it isn't already running:
   ```bash
   server/start_server.sh
   server/load_and_begin.sh
   ```
5. Seed the ring:
   ```bash
   server/seed_ring.sh <TARGET_CYCLE>
   ```
   This verifies the state from step 3 is really on disk, then
   force-completes exactly the three cross-hour-referenced nodes
   (`troute_ana_a`, `schism_ana`, `sfincs_ana`) under
   `/coastal_hourly/cycle/h<PREV_HH>/`. Use `--dry-run` first -- it's
   entirely read-only and safe to run repeatedly against a live server.
6. Confirm: `ecflow_client --get_state=/coastal_hourly/cycle/h<TARGET_HH>`
   should progress past its `time HH:00` gate on its own from here.

### Scope note

The predecessor hour's *other* tasks (`troute_ana_b`, `gen_configs_ana`,
`run_stofs_download_ana`, `gen_configs_sr`, `run_stofs_download_sr`,
`schism_sr`, `sfincs_sr`) are intentionally left untouched by
`seed_ring.sh` -- they aren't referenced by any trigger `TARGET_CYCLE`
depends on. Depending on wall-clock timing they may sit queued
indefinitely (harmless) or run for real against real data once the ring
starts moving (also harmless, just possibly redundant with the manual
bootstrap). This is expected, not a bug.

### Troubleshooting

`seed_ring.sh` fails loudly and tells you which precondition wasn't met:
- **server not reachable** -- run `server/start_server.sh` +
  `server/load_and_begin.sh` first.
- **target node not found** -- the suite isn't loaded/begun yet, or
  `TARGET_CYCLE`'s hour doesn't match what you expect.
- **missing troute/SCHISM/SFINCS state file** -- re-run
  `bin/hotstart_coastal_models.sh <TARGET_CYCLE>` (or just the missing
  half, via `--troute-only`/`--coastal-only`).
