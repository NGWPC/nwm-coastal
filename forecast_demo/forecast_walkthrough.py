# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Manual Forecast-Cycle Walkthrough (no ecflow required)
#
# This notebook runs one full spinup -> AnA hour -> short-range (SR) cycle
# of the coastal forecast pipeline by calling each underlying tool directly,
# in sequence -- the same commands `forecast_demo`'s ecflow suite
# (`server/`, `ecf_home/`, `suite_def/`) automates and self-chains hourly,
# just run here by hand, once, so every input/output/flag is visible.
#
# **This notebook does not use or require ecflow in any way.** It is fully
# standalone: it runs its own spinup, then one AnA hour, then one SR cycle.
#
# What this demonstrates:
#
# 1. **The SCHISM crosswalk** (`ngenReaches.csv`) -- a one-time, offline,
#    per-domain step, unrelated to any specific forecast cycle.
# 2. **The SFINCS crosswalk** -- baked into initial SFINCS model creation,
#    not a per-cycle step either.
# 3. **AnA and SR coastal met forcing inputs** -- the `ngen_rte.run_coastal`
#    forcing engine invocation, called once for AnA and once for SR.
# 4. **Coastal boundary forcing (STOFS)** -- downloaded via
#    `nwm-coastal-cli run ... --stop-after download`.
# 5. **Hourly cycles** -- `TARGET_CYCLE` below is the one thing you pick;
#    everything else derives from it.
# 6. **State saving and warm starts** -- SCHISM's `hotstart_it=<N>.nc` /
#    SFINCS's `.rst` restart files, and t-route's own independent
#    save/load-state mechanism, all configured (not hand-built) by
#    `gen_cycle_config.py`.
#
# `forecast_demo/` also offers an ecflow-orchestrated way to run this
# continuously and unattended -- see `../README.md`'s "Two ways to run
# this" section -- but nothing below depends on it.

# %% [markdown]
# ## Section 0 -- Setup
#
# Prerequisites (see `../README.md` for the full, ordered runbook): AWS
# credentials for `s3://ngwpc-coastal`/`s3://ngwpc-dev`, `nwm-rte` cloned +
# its Docker image built, coastal data staged
# (`setup_data_coastal_forecast.sh`), the VPU hydrofabric geopackage
# workaround staged, and the `nwm-coastal-py`/`nwm-coastal-cli` wrappers
# created. The four env vars below must already be set in the shell this
# notebook/script runs in.

# %%
import os
import subprocess
from pathlib import Path

def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set -- see ../README.md's \"Required environment "
            "variables\" section."
        )
    return value

NWM_COASTAL_ROOT = Path(_required_env("NWM_COASTAL_ROOT"))
NWM_RTE_ROOT = Path(_required_env("NWM_RTE_ROOT"))
RUN_NGEN_ROOT = Path(_required_env("RUN_NGEN_ROOT"))
RUN_COASTAL_ROOT = Path(_required_env("RUN_COASTAL_ROOT"))

VPU = "03S"

# The one hourly cycle this notebook demonstrates. YYYYMMDDHH, UTC.
TARGET_CYCLE = "2026082123"


def _prev_cycle(cycle: str) -> str:
    """Return cycle - 1h, same YYYYMMDDHH format."""
    from datetime import datetime, timedelta

    dt = datetime.strptime(cycle, "%Y%m%d%H") - timedelta(hours=1)
    return dt.strftime("%Y%m%d%H")


PREV_CYCLE = _prev_cycle(TARGET_CYCLE)
print(f"TARGET_CYCLE={TARGET_CYCLE}  PREV_CYCLE={PREV_CYCLE}  VPU={VPU}")


def run_nwm_rte(module: str, args: list[str]) -> subprocess.CompletedProcess:
    """Run an ngen_rte module inside the nwm-rte Docker container.

    Mirrors the `cd $NWM_RTE_ROOT && source config.bashrc && source run.sh
    && docker_run python -um <module> <args>` pattern used by every
    forecast_demo/ecf_home/*.ecf file -- reused here for the exact same
    invocation shape, not as a dependency on ecflow itself.
    """
    quoted_args = " ".join(f'"{a}"' for a in args)
    script = (
        f'cd "{NWM_RTE_ROOT}" && '
        f"source config.bashrc && source run.sh && "
        f'docker_run python -um "{module}" {quoted_args}'
    )
    print(f"--- run_nwm_rte: {module} {quoted_args} ---")
    result = subprocess.run(["bash", "-c", script], check=True)
    return result


def run_gen_cycle_config(model: str, run_type: str, **kwargs) -> subprocess.CompletedProcess:
    """Call forecast_demo/bin/gen_cycle_config.py directly via nwm-coastal-py.

    Reuses this script as-is; does not reimplement its config-generation
    logic.
    """
    gen_script = NWM_COASTAL_ROOT / "forecast_demo" / "bin" / "gen_cycle_config.py"
    nwm_coastal_py = NWM_COASTAL_ROOT / "nwm-coastal-py"
    args = [str(nwm_coastal_py), str(gen_script), "--model", model, "--run-type", run_type]
    for key, value in kwargs.items():
        if value is None:
            continue
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    print(f"--- run_gen_cycle_config: model={model} run_type={run_type} ---")
    return subprocess.run(args, check=True)


def find_troute_output(region_dir: Path) -> Path:
    """Locate the actual troute_output_<window-start-timestamp>.nc file under
    a region_*/<VPU>/ directory -- the real filename/timestamp isn't
    predictable from the cycle datetime alone (it's stamped with the
    window's start, not the cycle end, and troute's own -lb value
    determines how far back that is), so this is found by glob rather
    than constructed. gen_cycle_config.py runs on the HOST via
    nwm-coastal-py (not inside the RTE Docker container), so this returns
    a host-side path, not a /ngwpc/... container path.
    """
    matches = sorted((region_dir / "Output").glob("troute_output_*.nc"))
    if not matches:
        raise FileNotFoundError(f"No troute_output_*.nc found under {region_dir / 'Output'}")
    return matches[-1]


def nwm_coastal_cli(args: list[str]) -> subprocess.CompletedProcess:
    cli = NWM_COASTAL_ROOT / "nwm-coastal-cli"
    print(f"--- nwm-coastal-cli {' '.join(args)} ---")
    return subprocess.run([str(cli), *args], check=True)


SCHISM_BASE_YAML = RUN_COASTAL_ROOT / "schism_sims" / "run.yaml"
SFINCS_BASE_YAML = RUN_COASTAL_ROOT / "sfincs_sims" / "run.yaml"
SCHISM_CYCLES_DIR = RUN_COASTAL_ROOT / "schism_sims" / "cycles"
SFINCS_CYCLES_DIR = RUN_COASTAL_ROOT / "sfincs_sims" / "cycles"

# %% [markdown]
# ## Section 1 -- Spinup
#
# The suite self-chains hour-to-hour: every hour's state is derived from
# the *same task in the previous hour*. On a fresh start there is no real
# `PREV_CYCLE` state to self-chain from -- this section manufactures it
# directly by calling `bin/hotstart_coastal_models.sh`, the same standalone
# bootstrap script the ecflow front-end also uses (it is not itself an
# ecflow dependency -- its own header comment says it's meant to run
# "OUTSIDE ecflow"). A short `SPINUP_HOURS`/`RAMP_HOURS` override is used
# below (instead of the 18h/9h defaults) so this notebook completes in a
# reasonable demo runtime -- a real deployment would typically use the
# full default spin-up.

# %%
hotstart_script = NWM_COASTAL_ROOT / "forecast_demo" / "bin" / "hotstart_coastal_models.sh"
subprocess.run(
    [str(hotstart_script), TARGET_CYCLE, "4", "2"],  # 4h spin-up, 2h ramp -- demo-sized
    check=True,
    env={
        **os.environ,
        "NWM_COASTAL_ROOT": str(NWM_COASTAL_ROOT),
        "NWM_RTE_ROOT": str(NWM_RTE_ROOT),
        "RUN_NGEN_ROOT": str(RUN_NGEN_ROOT),
        "RUN_COASTAL_ROOT": str(RUN_COASTAL_ROOT),
    },
)

# %%
# Confirm the expected bootstrap state actually landed before moving on.
troute_state = RUN_NGEN_ROOT / "regionalization" / f"region_ana_a_{PREV_CYCLE}" / VPU / "state_save" / "troute"
print("troute bootstrap state:", troute_state, "exists:", troute_state.exists())

schism_ana_dir = SCHISM_CYCLES_DIR / f"ana_{PREV_CYCLE}"
sfincs_ana_dir = SFINCS_CYCLES_DIR / f"ana_{PREV_CYCLE}"
print("SCHISM spin-up cycle dir:", schism_ana_dir, "exists:", schism_ana_dir.exists())
print("SFINCS spin-up cycle dir:", sfincs_ana_dir, "exists:", sfincs_ana_dir.exists())

# %% [markdown]
# ## Section 2 -- SCHISM crosswalk: `nwmReaches.csv` -> `ngenReaches.csv`
#
# **One-time, offline, per-SCHISM-domain.** Not part of the per-cycle
# sequence below.
#
# SCHISM already ships a crosswalk from its own source/sink mesh element
# IDs to NWM COMIDs (`nwmReaches.csv`). To drive SCHISM discharge from
# NextGen/t-route output instead of raw NWM output, that crosswalk needs
# one more hop: COMID -> 16-digit NextGen flowpath ID (`fp_id`), the ID
# t-route actually keys its output on. `schism_discharge` (stage 8 of 12
# in `SchismModelConfig.stage_order`) auto-discovers the resulting
# `ngenReaches.csv` whenever `meteo_source == "ngen_forecast"`.

# %%
from coastal_calibration.schism.ngen_reaches import translate_nwm_to_ngen_reaches

schism_domain_dir = RUN_COASTAL_ROOT / "schism_models" / "atlgulf_extract_03S"
nwm_reaches_csv = schism_domain_dir / "nwmReaches.csv"
hydrofabric_gpkg = RUN_NGEN_ROOT / "data" / "hydrofabric" / f"vpu_{VPU}.gpkg"
ngen_reaches_csv = schism_domain_dir / "ngenReaches.csv"

stats = translate_nwm_to_ngen_reaches(
    nwm_reaches=nwm_reaches_csv,
    gpkg=hydrofabric_gpkg,
    output=ngen_reaches_csv,
)
print(stats)

# %%
import pandas as pd

print("nwmReaches.csv (COMID-keyed):")
print(pd.read_csv(nwm_reaches_csv).head())
print("\nngenReaches.csv (fp_id-keyed):")
print(pd.read_csv(ngen_reaches_csv).head())

# %% [markdown]
# ## Section 3 -- SFINCS crosswalk: baked into model *setup*, not per-cycle
#
# Unlike SCHISM, SFINCS's discharge-location crosswalk isn't a standalone
# script -- it's a stage (`CreateDischargeStage`, `name = "create_discharge"`)
# inside `coastal-calibration create`, the command that builds a new SFINCS
# model from an AOI polygon in the first place. It intersects NWM/NextGen
# flowpath linestrings against the AOI, snaps the resulting points to
# active SFINCS grid cells, and writes a `.src`-format discharge locations
# file. This runs once, at model creation time -- never per forecast cycle.
#
# This forecast domain's SFINCS model (`sfincs_models/tampabay/`) was
# already created previously, so this section reads the discharge
# locations file it already produced rather than re-running `create` --
# doing that live here would repeat the same slow, network-bound DEM
# fetch every time this notebook runs, for a step that only needs to
# happen once, ever, per domain. To see `create_discharge` actually run
# (e.g. when standing up a brand-new domain), use `nwm-coastal-cli create
# <your create.yaml> --stop-after create_discharge` directly.

# %%
sfincs_domain_dir = RUN_COASTAL_ROOT / "sfincs_models" / "tampabay"
src_file = sfincs_domain_dir / "sfincs_ngen.src"
print(f"Discharge locations file (already produced by 'create'): {src_file}")
print(f"exists: {src_file.exists()}")
print(src_file.read_text()[:500])

# %% [markdown]
# ## Section 4 -- One AnA hour
#
# This section shows the state-saving/warm-start mechanism once, for a
# single `TARGET_CYCLE`/`PREV_CYCLE` pair, rather than looping over many
# hours -- repeated self-chaining across many hours is exactly what the
# ecflow front-end already demonstrates; this notebook's job is to make
# each step's inputs/outputs/flags legible, once.

# %% [markdown]
# ### 4.1 -- troute AnA-A (1h window, self-chains from `PREV_CYCLE`)

# %%
# Baked into the nwm-rte Docker image itself (mswm's installed
# example_inputs, not mounted from the host) -- container-internal path,
# matching INSTALLED_REGIONALIZATION_RESULTS in nwm-rte/config.bashrc.
INSTALLED_REGIONALIZATION_RESULTS = "/ngen-app/ngen-python/lib/python3.11/site-packages/mswm/example_inputs/regionalization"
formulation_assignment_csv = f"{INSTALLED_REGIONALIZATION_RESULTS}/vpu_{VPU}/formulation_assignment.csv"
catchment_groups_csv = f"{INSTALLED_REGIONALIZATION_RESULTS}/vpu_{VPU}/catchment_groups.csv"
hydrofab_file_container = f"/ngwpc/run_ngen/data/hydrofabric/vpu_{VPU}.gpkg"

save_dir_a_host = RUN_NGEN_ROOT / "regionalization" / f"region_ana_a_{TARGET_CYCLE}" / VPU / "state_save"
save_dir_a_container = f"/ngwpc/run_ngen/regionalization/region_ana_a_{TARGET_CYCLE}/{VPU}/state_save"
prev_ana_a_save_host = RUN_NGEN_ROOT / "regionalization" / f"region_ana_a_{PREV_CYCLE}" / VPU / "state_save" / "troute"
prev_ana_a_save_container = f"/ngwpc/run_ngen/regionalization/region_ana_a_{PREV_CYCLE}/{VPU}/state_save"

end_dt = f"{TARGET_CYCLE[0:4]}-{TARGET_CYCLE[4:6]}-{TARGET_CYCLE[6:8]} {TARGET_CYCLE[8:10]}:00:00"

load_state_args = ["-lsf", prev_ana_a_save_container] if prev_ana_a_save_host.exists() else []

run_nwm_rte(
    "ngen_rte.run_regionalization_standalone",
    [
        "-n", "12",
        "-faf", str(formulation_assignment_csv),
        "-cgf", str(catchment_groups_csv),
        "-fconfig", "standard_ana",
        "-dt", end_dt,
        "-lb", "120",
        "-rname", f"region_ana_a_{TARGET_CYCLE}",
        "-v", VPU,
        "--hydrofab_file", hydrofab_file_container,
        "-outfmt", "NetCDF",
        "-ss", "-ssd", save_dir_a_container,
        *load_state_args,
    ],
)

# %% [markdown]
# ### 4.2 -- troute AnA-B (3h window, T-3->T0; loads from the same `PREV_CYCLE` ana_a save as 4.1)

# %%
cycle_dt = end_dt  # same cycle datetime

run_nwm_rte(
    "ngen_rte.run_regionalization_standalone",
    [
        "-n", "12",
        "-faf", str(formulation_assignment_csv),
        "-cgf", str(catchment_groups_csv),
        "-fconfig", "standard_ana",
        "-dt", cycle_dt,
        "-lb", "240",
        "-rname", f"region_ana_b_{TARGET_CYCLE}",
        "-v", VPU,
        "--hydrofab_file", hydrofab_file_container,
        "-outfmt", "NetCDF",
        "-ss", "-ssd", f"/ngwpc/run_ngen/regionalization/region_ana_b_{TARGET_CYCLE}/{VPU}/state_save",
        *load_state_args,
    ],
)

# %% [markdown]
# ### 4.3 -- Met forcing AnA
#
# `-lb 240 -fih 240` (AnA-only) widens the lookback window to T-3 and emits
# all 4 hourly samples T-3..T0, instead of just the current hour.

# %%
run_nwm_rte(
    "ngen_rte.run_coastal",
    [
        "-dt", cycle_dt,
        "-rname", "coastal_ana",
        "-fconfig", "standard_ana",
        "-gdomain", "vpu03s",
        "-lb", "240",
        "-fih", "240",
    ],
)

ana_forcing_file = RUN_NGEN_ROOT / "data" / "scratch" / "standard_ana_coastal" / f"vpu03s_{TARGET_CYCLE}00.nc"
print("Forcing output:", ana_forcing_file, "exists:", ana_forcing_file.exists())

# %% [markdown]
# ### 4.4 -- STOFS coastal boundary forcing (AnA)

# %%
ana_run_yaml = SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}" / "run.yaml"
# gen_configs (4.5, below) writes this file; a placeholder cycle dir must
# exist first for the download step's config to resolve against.
ana_run_yaml.parent.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### 4.5 -- gen_configs (AnA): build this cycle's SCHISM + SFINCS `run.yaml`
#
# Warm-starts from `PREV_CYCLE`'s own T-2 hotstart/rst checkpoint (not
# T0 -- that's what 4.1/4.2 just produced for T-3..T0, self-chaining
# forward one more hour).

# %%
for model, base_yaml, cycles_dir in (
    ("schism", SCHISM_BASE_YAML, SCHISM_CYCLES_DIR),
    ("sfincs", SFINCS_BASE_YAML, SFINCS_CYCLES_DIR),
):
    run_gen_cycle_config(
        model=model,
        run_type="ana",
        base_yaml=base_yaml,
        cycle=TARGET_CYCLE,
        start_date=cycle_dt,
        forecast_meteo_file=ana_forcing_file,
        troute_file=find_troute_output(RUN_NGEN_ROOT / "regionalization" / f"region_ana_b_{TARGET_CYCLE}" / VPU),
        cycle_dir=cycles_dir / f"ana_{TARGET_CYCLE}",
    )

# %% [markdown]
# ### 4.6 -- STOFS download + SCHISM/SFINCS AnA run

# %%
nwm_coastal_cli(["run", str(SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}" / "run.yaml"), "--stop-after", "download"])
nwm_coastal_cli(["run", str(SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}" / "run.yaml"), "--start-from", "schism_forcing_prep"])
nwm_coastal_cli(["run", str(SFINCS_CYCLES_DIR / f"ana_{TARGET_CYCLE}" / "run.yaml")])

# %%
schism_hotstart = sorted((SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}").glob("**/hotstart_it=*.nc"))
sfincs_rst = sorted((SFINCS_CYCLES_DIR / f"ana_{TARGET_CYCLE}").glob("**/*.rst"))
print("SCHISM hotstart files written:", schism_hotstart)
print("SFINCS restart files written:", sfincs_rst)

# %% [markdown]
# ## Section 5 -- One SR cycle (same hour)
#
# The AnA -> SR handoff is same-cycle, not cross-cycle like AnA-A/AnA-B:
# `--t0-troute-file`/`--t0-precip-source-file` backfill T0 because
# `troute_sr`/`ngen_rte.run_coastal`'s own SR windows don't reach T0 on
# their own.

# %% [markdown]
# ### 5.1 -- troute SR (loads this cycle's own ana_b save; no `-ss`/`-ssd`)

# %%
this_cycle_ana_b_save_container = f"/ngwpc/run_ngen/regionalization/region_ana_b_{TARGET_CYCLE}/{VPU}/state_save"

run_nwm_rte(
    "ngen_rte.run_regionalization_standalone",
    [
        "-n", "12",
        "-faf", str(formulation_assignment_csv),
        "-cgf", str(catchment_groups_csv),
        "-fconfig", "short_range",
        "-dt", cycle_dt,
        "-rname", f"region_sr_{TARGET_CYCLE}",
        "-v", VPU,
        "--hydrofab_file", hydrofab_file_container,
        "-outfmt", "NetCDF",
        "-lsf", this_cycle_ana_b_save_container,
    ],
)

# %% [markdown]
# ### 5.2 -- Met forcing SR (no `-lb`/`-fih` -- current-hour window only)

# %%
run_nwm_rte(
    "ngen_rte.run_coastal",
    [
        "-dt", cycle_dt,
        "-rname", "coastal_short_range",
        "-fconfig", "short_range",
        "-gdomain", "vpu03s",
    ],
)

sr_forcing_file = RUN_NGEN_ROOT / "data" / "scratch" / "short_range_coastal" / f"vpu03s_{TARGET_CYCLE}00.nc"
print("SR forcing output:", sr_forcing_file, "exists:", sr_forcing_file.exists())

# %% [markdown]
# ### 5.3 -- gen_configs (SR): backfill T0 + warm-start from this cycle's own T0 checkpoint

# %%
sr_cycle_dir = {
    "schism": SCHISM_CYCLES_DIR / f"sr_{TARGET_CYCLE}",
    "sfincs": SFINCS_CYCLES_DIR / f"sr_{TARGET_CYCLE}",
}
t0_precip_source = SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}" / "run" / "precip_source.nc"
t0_hotstart = schism_hotstart[-1] if schism_hotstart else None
t0_rst = sfincs_rst[-1] if sfincs_rst else None

run_gen_cycle_config(
    model="schism",
    run_type="sr",
    base_yaml=SCHISM_BASE_YAML,
    cycle=TARGET_CYCLE,
    start_date=cycle_dt,
    forecast_meteo_file=sr_forcing_file,
    troute_file=find_troute_output(RUN_NGEN_ROOT / "regionalization" / f"region_sr_{TARGET_CYCLE}" / VPU),
    cycle_dir=sr_cycle_dir["schism"],
    t0_troute_file=find_troute_output(RUN_NGEN_ROOT / "regionalization" / f"region_ana_b_{TARGET_CYCLE}" / VPU),
    t0_precip_source_file=t0_precip_source,
    hot_start_file=t0_hotstart,
)
run_gen_cycle_config(
    model="sfincs",
    run_type="sr",
    base_yaml=SFINCS_BASE_YAML,
    cycle=TARGET_CYCLE,
    start_date=cycle_dt,
    forecast_meteo_file=sr_forcing_file,
    troute_file=find_troute_output(RUN_NGEN_ROOT / "regionalization" / f"region_sr_{TARGET_CYCLE}" / VPU),
    cycle_dir=sr_cycle_dir["sfincs"],
    sfincs_rst_file=t0_rst,
)

# %% [markdown]
# ### 5.4 -- STOFS download + SCHISM/SFINCS SR run

# %%
nwm_coastal_cli(["run", str(sr_cycle_dir["schism"] / "run.yaml"), "--stop-after", "download"])
nwm_coastal_cli(["run", str(sr_cycle_dir["schism"] / "run.yaml"), "--start-from", "schism_forcing_prep"])
nwm_coastal_cli(["run", str(sr_cycle_dir["sfincs"] / "run.yaml")])

# %% [markdown]
# ### 5.5 -- Confirm state actually carried over
#
# Read a water-level/discharge field from both the AnA and SR outputs and
# compare at the shared T0 boundary -- the first SR timestep should be
# very close to the last AnA timestep at the same station, since the SR
# run warm-started from exactly that point.

# %%
import xarray as xr

schism_ana_out = next((SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}").glob("**/outputs/*.nc"))
schism_sr_out = next((sr_cycle_dir["schism"]).glob("**/outputs/*.nc"))

with xr.open_dataset(schism_ana_out) as ana_ds, xr.open_dataset(schism_sr_out) as sr_ds:
    ana_last = ana_ds["elevation"].isel(time=-1, node=0).item()
    sr_first = sr_ds["elevation"].isel(time=0, node=0).item()
    print(f"AnA last-timestep elevation (station 0): {ana_last:.4f}")
    print(f"SR first-timestep elevation (station 0):  {sr_first:.4f}")
    print(f"Difference: {abs(ana_last - sr_first):.4f} (should be small -- confirms warm-start continuity)")

# %% [markdown]
# ## Section 6 -- Recap
#
# This notebook manually ran one full spinup -> AnA hour -> SR cycle
# sequence, with no ecflow involved. `forecast_demo/`'s ecflow front-end
# (`server/` + `ecf_home/` + `suite_def/`) automates exactly this sequence
# hourly and continuously -- see `../README.md` for that path.
