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
