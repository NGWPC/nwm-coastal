"""Run one AnA hour + one SR cycle by hand, no ecflow.

Setup steps are in README2.md. This script assumes setup is already done.

Run with: nwm-coastal-py forecast_walkthrough2.py
"""

import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


# ---------------------------------------------------------------------------
# 1. Env vars (see README2.md step 7 for what populates these)
# ---------------------------------------------------------------------------

NWM_COASTAL_ROOT = Path(_required_env("NWM_COASTAL_ROOT"))
NWM_RTE_ROOT = Path(_required_env("NWM_RTE_ROOT"))
RUN_NGEN_ROOT = Path(_required_env("RUN_NGEN_ROOT"))
RUN_COASTAL_ROOT = Path(_required_env("RUN_COASTAL_ROOT"))

VPU = "03S"

# Pick the hourly cycle to run. YYYYMMDDHH, UTC.
TARGET_CYCLE = "2026082812"


def _prev_cycle(cycle: str) -> str:
    from datetime import datetime, timedelta

    dt = datetime.strptime(cycle, "%Y%m%d%H") - timedelta(hours=1)
    return dt.strftime("%Y%m%d%H")


PREV_CYCLE = _prev_cycle(TARGET_CYCLE)
CYCLE_DT = f"{TARGET_CYCLE[0:4]}-{TARGET_CYCLE[4:6]}-{TARGET_CYCLE[6:8]} {TARGET_CYCLE[8:10]}:00:00"
print(f"TARGET_CYCLE={TARGET_CYCLE}  PREV_CYCLE={PREV_CYCLE}  VPU={VPU}")

SCHISM_BASE_YAML = RUN_COASTAL_ROOT / "schism_sims" / "run.yaml"
SFINCS_BASE_YAML = RUN_COASTAL_ROOT / "sfincs_sims" / "run.yaml"
SCHISM_CYCLES_DIR = RUN_COASTAL_ROOT / "schism_sims" / "cycles"
SFINCS_CYCLES_DIR = RUN_COASTAL_ROOT / "sfincs_sims" / "cycles"

# Container-side paths for troute (baked into the RTE image, not host paths).
INSTALLED_REGIONALIZATION_RESULTS = "/ngen-app/ngen-python/lib/python3.11/site-packages/mswm/example_inputs/regionalization"
FORMULATION_ASSIGNMENT_CSV = f"{INSTALLED_REGIONALIZATION_RESULTS}/vpu_{VPU}/formulation_assignment.csv"
CATCHMENT_GROUPS_CSV = f"{INSTALLED_REGIONALIZATION_RESULTS}/vpu_{VPU}/catchment_groups.csv"
HYDROFAB_FILE_CONTAINER = f"/ngwpc/run_ngen/data/hydrofabric/vpu_{VPU}.gpkg"


def run_nwm_rte(module: str, args: list[str]) -> subprocess.CompletedProcess:
    """cd into nwm-rte, source config, call an ngen_rte module in the RTE container."""
    quoted_args = " ".join(f'"{a}"' for a in args)
    script = (
        f'cd "{NWM_RTE_ROOT}" && '
        f"source config.bashrc && source run.sh && "
        f'docker_run python -um "{module}" {quoted_args}'
    )
    print(f"--- run_nwm_rte: {module} {quoted_args} ---")
    return subprocess.run(["bash", "-c", script], check=True)


def run_gen_cycle_config(model: str, run_type: str, **kwargs) -> subprocess.CompletedProcess:
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
    """troute_output_*.nc's timestamp is the window start, not the cycle -- glob for it."""
    matches = sorted((region_dir / "Output").glob("troute_output_*.nc"))
    if not matches:
        raise FileNotFoundError(f"No troute_output_*.nc found under {region_dir / 'Output'}")
    return matches[-1]


def nwm_coastal_cli(args: list[str]) -> subprocess.CompletedProcess:
    cli = NWM_COASTAL_ROOT / "nwm-coastal-cli"
    print(f"--- nwm-coastal-cli {' '.join(args)} ---")
    return subprocess.run([str(cli), *args], check=True)


# ---------------------------------------------------------------------------
# 2. SCHISM crosswalk (nwmReaches.csv -> ngenReaches.csv)
#
# One-time, per SCHISM domain. Not a per-cycle step. Left commented out --
# uncomment only when standing up a new SCHISM domain, or if ngenReaches.csv
# is missing/needs regenerating. Currently broken -- see README2.md.
# ---------------------------------------------------------------------------

# from coastal_calibration.schism.ngen_reaches import translate_nwm_to_ngen_reaches
#
# schism_domain_dir = RUN_COASTAL_ROOT / "schism_models" / "atlgulf_extract_03S"
# stats = translate_nwm_to_ngen_reaches(
#     nwm_reaches=schism_domain_dir / "nwmReaches.csv",
#     gpkg=RUN_NGEN_ROOT / "data" / "hydrofabric" / f"vpu_{VPU}.gpkg",
#     output=schism_domain_dir / "ngenReaches.csv",
# )
# print(stats)

# ---------------------------------------------------------------------------
# 3. VPU hydrofabric geopackage + ESMF mesh
#
# One-time per VPU -- only needs re-running if you want a different
# domain/VPU, or the source CONUS grid changed. The gpkg copy is needed
# because the Icefabric API t-route would normally query for this isn't
# reliably reachable from all networks; skipped if the file already exists.
# ---------------------------------------------------------------------------

hydrofab_file_host = RUN_NGEN_ROOT / "data" / "hydrofabric" / f"vpu_{VPU}.gpkg"
if not hydrofab_file_host.exists():
    hydrofab_file_host.parent.mkdir(parents=True, exist_ok=True)
    nwm_region_mgr_gpkg = (
        NWM_COASTAL_ROOT.parent / "nwm-region-mgr" / "data" / "inputs" / "region"
        / "hydrofabric" / "gpkg_vpu" / f"vpu_{VPU}.gpkg"
    )
    shutil.copy(nwm_region_mgr_gpkg, hydrofab_file_host)

# Skip the ESMF mesh extract if geo_em_vpu03s.nc already exists under
# $RUN_NGEN_ROOT/data/esmf_mesh/NWM/domain/.
subprocess.run(
    [
        str(NWM_COASTAL_ROOT / "nwm-coastal-py"),
        str(NWM_COASTAL_ROOT / "forecast_demo" / "bin" / "extract_esmf_domain.py"),
        "--source-domain", "CONUS",
        "--extract-geojson", str(RUN_NGEN_ROOT / "data" / "esmf_mesh" / "esmf_domain_extract" / "esmf_conus_03s_extract.geojson"),
        "--output-name", "vpu03s",
    ],
    check=True,
)

# ---------------------------------------------------------------------------
# 4. hotstart_coastal_models.sh -- spin up SCHISM/SFINCS, bootstrap troute AnA-A
#
# Produces state for PREV_CYCLE. hotstart_coastal_models.sh's own default
# is 18h spinup / 9h ramp; this demo uses 24h/6h instead.
# ---------------------------------------------------------------------------

SPINUP_HOURS = 24
RAMP_HOURS = 6

subprocess.run(
    [
        str(NWM_COASTAL_ROOT / "forecast_demo" / "bin" / "hotstart_coastal_models.sh"),
        TARGET_CYCLE, str(SPINUP_HOURS), str(RAMP_HOURS),
    ],
    check=True,
    env={
        **os.environ,
        "NWM_COASTAL_ROOT": str(NWM_COASTAL_ROOT),
        "NWM_RTE_ROOT": str(NWM_RTE_ROOT),
        "RUN_NGEN_ROOT": str(RUN_NGEN_ROOT),
        "RUN_COASTAL_ROOT": str(RUN_COASTAL_ROOT),
    },
)

# Spinup check: simulated vs observed over the spinup window itself.
# This is the actual evidence that spinup worked -- as the tidal signal
# propagates in from the boundary, simulated should converge toward
# observed by the end of the window. If it hasn't, spinup needs to be
# longer before trusting AnA/SR off of it.
import pandas as pd

from coastal_calibration.data.coops_api import query_coops_byids
from coastal_calibration.plotting import plot_station_comparison

spinup_schism_series = pd.read_parquet(SCHISM_CYCLES_DIR / f"ana_{PREV_CYCLE}" / "run" / "obs_water_level.parquet")
spinup_sfincs_series = pd.read_parquet(SFINCS_CYCLES_DIR / f"ana_{PREV_CYCLE}" / "run" / "sfincs_model" / "obs_water_level.parquet")
spinup_station_ids = sorted(set(spinup_schism_series.columns) & set(spinup_sfincs_series.columns))
if not spinup_station_ids:
    print("Spinup check: no shared stations, skipping")
else:
    # hotstart_coastal_models.sh anchors the spin-up's start-date at
    # TARGET_CYCLE-3h, not PREV_CYCLE -- see SPINUP_END_DT in that script.
    spinup_end = datetime.strptime(TARGET_CYCLE, "%Y%m%d%H") - timedelta(hours=3)
    spinup_start = spinup_end - timedelta(hours=SPINUP_HOURS)
    spinup_obs_ds = query_coops_byids(
        spinup_station_ids,
        spinup_start.strftime("%Y%m%d %H:%M"),
        spinup_end.strftime("%Y%m%d %H:%M"),
        product="water_level",
        datum="MSL",
        units="metric",
        time_zone="gmt",
    )
    spinup_figs_dir = RUN_COASTAL_ROOT / "comparison_plots" / f"spinup_{PREV_CYCLE}"
    spinup_figs_dir.mkdir(parents=True, exist_ok=True)
    plot_station_comparison(
        {
            "SCHISM": (spinup_schism_series.index.to_numpy(), spinup_schism_series[spinup_station_ids].to_numpy()),
            "SFINCS": (spinup_sfincs_series.index.to_numpy(), spinup_sfincs_series[spinup_station_ids].to_numpy()),
        },
        spinup_station_ids,
        spinup_figs_dir,
        obs_ds=spinup_obs_ds,
        stations_per_figure=1,
    )

# ---------------------------------------------------------------------------
# 5. AnA cycle
#
# gen_cycle_config.py needs a troute output and a met forcing file as inputs
# -- both are produced by separate ngen_rte calls first, then handed to it.
# ---------------------------------------------------------------------------

prev_ana_a_save_host = RUN_NGEN_ROOT / "regionalization" / f"region_ana_a_{PREV_CYCLE}" / VPU / "state_save" / "troute"
load_state_args = (
    ["-lsf", f"/ngwpc/run_ngen/regionalization/region_ana_a_{PREV_CYCLE}/{VPU}/state_save"]
    if prev_ana_a_save_host.exists()
    else []
)

# troute AnA-A: 1h window, self-chains from PREV_CYCLE
run_nwm_rte(
    "ngen_rte.run_regionalization_standalone",
    [
        "-n", "12",
        "-faf", FORMULATION_ASSIGNMENT_CSV,
        "-cgf", CATCHMENT_GROUPS_CSV,
        "-fconfig", "standard_ana",
        "-dt", CYCLE_DT,
        "-lb", "120",
        "-rname", f"region_ana_a_{TARGET_CYCLE}",
        "-v", VPU,
        "--hydrofab_file", HYDROFAB_FILE_CONTAINER,
        "-outfmt", "NetCDF",
        "-ss", "-ssd", f"/ngwpc/run_ngen/regionalization/region_ana_a_{TARGET_CYCLE}/{VPU}/state_save",
        *load_state_args,
    ],
)

# troute AnA-B: 3h window, T-3 -> T0
run_nwm_rte(
    "ngen_rte.run_regionalization_standalone",
    [
        "-n", "12",
        "-faf", FORMULATION_ASSIGNMENT_CSV,
        "-cgf", CATCHMENT_GROUPS_CSV,
        "-fconfig", "standard_ana",
        "-dt", CYCLE_DT,
        "-lb", "240",
        "-rname", f"region_ana_b_{TARGET_CYCLE}",
        "-v", VPU,
        "--hydrofab_file", HYDROFAB_FILE_CONTAINER,
        "-outfmt", "NetCDF",
        "-ss", "-ssd", f"/ngwpc/run_ngen/regionalization/region_ana_b_{TARGET_CYCLE}/{VPU}/state_save",
        *load_state_args,
    ],
)

# Met forcing AnA: -lb/-fih 240 widens to T-3, emits all 4 hourly samples
run_nwm_rte(
    "ngen_rte.run_coastal",
    [
        "-dt", CYCLE_DT,
        "-rname", "coastal_ana",
        "-fconfig", "standard_ana",
        "-gdomain", "vpu03s",
        "-lb", "240",
        "-fih", "240",
    ],
)
ana_forcing_file = RUN_NGEN_ROOT / "data" / "scratch" / "standard_ana_coastal" / f"vpu03s_{TARGET_CYCLE}00.nc"

for model, base_yaml, cycles_dir in (
    ("schism", SCHISM_BASE_YAML, SCHISM_CYCLES_DIR),
    ("sfincs", SFINCS_BASE_YAML, SFINCS_CYCLES_DIR),
):
    run_gen_cycle_config(
        model=model,
        run_type="ana",
        base_yaml=base_yaml,
        cycle=TARGET_CYCLE,
        start_date=CYCLE_DT,
        forecast_meteo_file=ana_forcing_file,
        troute_file=find_troute_output(RUN_NGEN_ROOT / "regionalization" / f"region_ana_b_{TARGET_CYCLE}" / VPU),
        cycle_dir=cycles_dir / f"ana_{TARGET_CYCLE}",
    )

nwm_coastal_cli(["run", str(SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}" / "run.yaml"), "--stop-after", "download"])
nwm_coastal_cli(["run", str(SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}" / "run.yaml"), "--start-from", "schism_forcing_prep"])
nwm_coastal_cli(["run", str(SFINCS_CYCLES_DIR / f"ana_{TARGET_CYCLE}" / "run.yaml")])

schism_hotstart = sorted((SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}").glob("**/hotstart_it=*.nc"))
sfincs_rst = sorted((SFINCS_CYCLES_DIR / f"ana_{TARGET_CYCLE}").glob("**/*.rst"))
print("SCHISM hotstart files:", schism_hotstart)
print("SFINCS restart files:", sfincs_rst)

# SCHISM + SFINCS + observed, one figure per shared station
ana_schism_series = pd.read_parquet(SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}" / "run" / "obs_water_level.parquet")
ana_sfincs_series = pd.read_parquet(SFINCS_CYCLES_DIR / f"ana_{TARGET_CYCLE}" / "run" / "sfincs_model" / "obs_water_level.parquet")
ana_station_ids = sorted(set(ana_schism_series.columns) & set(ana_sfincs_series.columns))
if not ana_station_ids:
    print("AnA comparison: no shared stations, skipping")
else:
    ana_obs_ds = query_coops_byids(
        ana_station_ids,
        (datetime.strptime(TARGET_CYCLE, "%Y%m%d%H") - timedelta(hours=3)).strftime("%Y%m%d %H:%M"),
        datetime.strptime(TARGET_CYCLE, "%Y%m%d%H").strftime("%Y%m%d %H:%M"),
        product="water_level",
        datum="MSL",
        units="metric",
        time_zone="gmt",
    )
    ana_figs_dir = RUN_COASTAL_ROOT / "comparison_plots" / f"ana_{TARGET_CYCLE}"
    ana_figs_dir.mkdir(parents=True, exist_ok=True)
    plot_station_comparison(
        {
            "SCHISM": (ana_schism_series.index.to_numpy(), ana_schism_series[ana_station_ids].to_numpy()),
            "SFINCS": (ana_sfincs_series.index.to_numpy(), ana_sfincs_series[ana_station_ids].to_numpy()),
        },
        ana_station_ids,
        ana_figs_dir,
        obs_ds=ana_obs_ds,
        stations_per_figure=1,
    )

# ---------------------------------------------------------------------------
# 6. SR cycle
#
# Same-cycle handoff from AnA, not cross-cycle -- t0-troute-file/
# t0-precip-source-file backfill T0 since SR's own window doesn't reach it.
# ---------------------------------------------------------------------------

# troute SR: loads this cycle's own ana_b save, no -ss/-ssd
run_nwm_rte(
    "ngen_rte.run_regionalization_standalone",
    [
        "-n", "12",
        "-faf", FORMULATION_ASSIGNMENT_CSV,
        "-cgf", CATCHMENT_GROUPS_CSV,
        "-fconfig", "short_range",
        "-dt", CYCLE_DT,
        "-rname", f"region_sr_{TARGET_CYCLE}",
        "-v", VPU,
        "--hydrofab_file", HYDROFAB_FILE_CONTAINER,
        "-outfmt", "NetCDF",
        "-lsf", f"/ngwpc/run_ngen/regionalization/region_ana_b_{TARGET_CYCLE}/{VPU}/state_save",
    ],
)

# Met forcing SR: current-hour window only, no -lb/-fih
run_nwm_rte(
    "ngen_rte.run_coastal",
    [
        "-dt", CYCLE_DT,
        "-rname", "coastal_short_range",
        "-fconfig", "short_range",
        "-gdomain", "vpu03s",
    ],
)
sr_forcing_file = RUN_NGEN_ROOT / "data" / "scratch" / "short_range_coastal" / f"vpu03s_{TARGET_CYCLE}00.nc"

sr_cycle_dir = {
    "schism": SCHISM_CYCLES_DIR / f"sr_{TARGET_CYCLE}",
    "sfincs": SFINCS_CYCLES_DIR / f"sr_{TARGET_CYCLE}",
}
run_gen_cycle_config(
    model="schism",
    run_type="sr",
    base_yaml=SCHISM_BASE_YAML,
    cycle=TARGET_CYCLE,
    start_date=CYCLE_DT,
    forecast_meteo_file=sr_forcing_file,
    troute_file=find_troute_output(RUN_NGEN_ROOT / "regionalization" / f"region_sr_{TARGET_CYCLE}" / VPU),
    cycle_dir=sr_cycle_dir["schism"],
    t0_troute_file=find_troute_output(RUN_NGEN_ROOT / "regionalization" / f"region_ana_b_{TARGET_CYCLE}" / VPU),
    t0_precip_source_file=SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}" / "run" / "precip_source.nc",
    hot_start_file=schism_hotstart[-1] if schism_hotstart else None,
)
run_gen_cycle_config(
    model="sfincs",
    run_type="sr",
    base_yaml=SFINCS_BASE_YAML,
    cycle=TARGET_CYCLE,
    start_date=CYCLE_DT,
    forecast_meteo_file=sr_forcing_file,
    troute_file=find_troute_output(RUN_NGEN_ROOT / "regionalization" / f"region_sr_{TARGET_CYCLE}" / VPU),
    cycle_dir=sr_cycle_dir["sfincs"],
    sfincs_rst_file=sfincs_rst[-1] if sfincs_rst else None,
)

nwm_coastal_cli(["run", str(sr_cycle_dir["schism"] / "run.yaml"), "--stop-after", "download"])
nwm_coastal_cli(["run", str(sr_cycle_dir["schism"] / "run.yaml"), "--start-from", "schism_forcing_prep"])
nwm_coastal_cli(["run", str(sr_cycle_dir["sfincs"] / "run.yaml")])

# SCHISM + SFINCS + observed, one figure per shared station
# SR run-type is a fixed 18h window (gen_cycle_config.py's own hardcoded duration_hours)
sr_schism_series = pd.read_parquet(sr_cycle_dir["schism"] / "run" / "obs_water_level.parquet")
sr_sfincs_series = pd.read_parquet(sr_cycle_dir["sfincs"] / "run" / "sfincs_model" / "obs_water_level.parquet")
sr_station_ids = sorted(set(sr_schism_series.columns) & set(sr_sfincs_series.columns))
if not sr_station_ids:
    print("SR comparison: no shared stations, skipping")
else:
    sr_obs_ds = query_coops_byids(
        sr_station_ids,
        datetime.strptime(TARGET_CYCLE, "%Y%m%d%H").strftime("%Y%m%d %H:%M"),
        (datetime.strptime(TARGET_CYCLE, "%Y%m%d%H") + timedelta(hours=18)).strftime("%Y%m%d %H:%M"),
        product="water_level",
        datum="MSL",
        units="metric",
        time_zone="gmt",
    )
    sr_figs_dir = RUN_COASTAL_ROOT / "comparison_plots" / f"sr_{TARGET_CYCLE}"
    sr_figs_dir.mkdir(parents=True, exist_ok=True)
    plot_station_comparison(
        {
            "SCHISM": (sr_schism_series.index.to_numpy(), sr_schism_series[sr_station_ids].to_numpy()),
            "SFINCS": (sr_sfincs_series.index.to_numpy(), sr_sfincs_series[sr_station_ids].to_numpy()),
        },
        sr_station_ids,
        sr_figs_dir,
        obs_ds=sr_obs_ds,
        stations_per_figure=1,
    )

print("Done. AnA + SR outputs are under:")
print(" ", SCHISM_CYCLES_DIR / f"ana_{TARGET_CYCLE}")
print(" ", SFINCS_CYCLES_DIR / f"ana_{TARGET_CYCLE}")
print(" ", sr_cycle_dir["schism"])
print(" ", sr_cycle_dir["sfincs"])
