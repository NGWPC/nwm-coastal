# ecflow_demo

Hourly ecflow suite (`coastal_hourly`) driving nwm-rte forcing + t-route
regionalization + SCHISM/SFINCS coastal runs. See
`suite_def/coastal_hourly.def.template`'s own header comment for the
pipeline shape.

## Required environment variables

Nothing in this directory hardcodes a path to a particular person's home
directory or checkout location -- every filesystem path is derived from
four environment variables, which must be set before running anything
under `server/` or `bin/`:

| Variable            | Points to                                                | Example                                  |
|----------------------|----------------------------------------------------------|-------------------------------------------|
| `NWM_COASTAL_ROOT`   | This repo's own checkout root (contains `ecflow_demo/`, `nwm-coastal-cli`, `nwm-coastal-py`) | `/contrib/<you>/ngwpc/nwm-coastal` |
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
optional overrides, see below). `server/load_and_begin.sh` additionally
requires `NWM_RTE_ROOT`/`RUN_NGEN_ROOT`/`RUN_COASTAL_ROOT`.
`bin/hotstart_coastal_models.sh` requires `NWM_COASTAL_ROOT`/
`RUN_COASTAL_ROOT` only (it never touches troute or ngen forcing --
STOFS boundary data only). Each script fails fast with a clear error if a
variable it needs is unset.

Optional overrides:

- `ECF_PORT` (default `39411`) -- the local port the ecflow server listens on.
- `ECFLOW_BIN_DIR` (default `/contrib/software/ecflow/5.6.0/bin`) -- where the
  `ecflow_server`/`ecflow_client` 5.6.0 binaries live, if not on `PATH`
  already.

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
