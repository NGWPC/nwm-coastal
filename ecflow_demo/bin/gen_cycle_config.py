#!/usr/bin/env python3
"""Generate a per-cycle SCHISM/SFINCS run.yaml from a base template.

Invoked by the ecflow 'gen_configs' task via nwm-coastal-py (the
coastal-calibration pixi env's Python + PyYAML).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["schism", "sfincs"])
    p.add_argument("--base-yaml", required=True, type=Path)
    p.add_argument("--cycle", required=True, help="YYYYMMDDHH")
    p.add_argument(
        "--start-date",
        required=True,
        help=(
            "This cycle's hour, YYYY-MM-DD HH:MM:SS (e.g. the ngen_forcing_sr/"
            "troute_sr cycle time -- always the cycle's T0, regardless of "
            "--run-type). NOT the model's own start_date -- see --run-type "
            "for how it's shifted."
        ),
    )
    p.add_argument("--forecast-meteo-file", required=True, type=Path)
    p.add_argument("--troute-file", required=True, type=Path)
    p.add_argument(
        "--cycle-dir",
        required=True,
        type=Path,
        help=(
            "Per-cycle directory, e.g. run_coastal/schism_sims/cycles/<YYYYMMDDHH>. "
            "The generated config is written to <cycle-dir>/run.yaml and "
            "paths.work_dir is set to <cycle-dir>/run."
        ),
    )
    p.add_argument(
        "--run-type",
        choices=["sr", "ana"],
        default="sr",
        help=(
            "sr (default): model start_date = --start-date + 1h (no true T+0 "
            "driving data -- see the shift comment in main()), duration_hours "
            "left at base_yaml's value (17h). "
            "ana: model start_date = --start-date - 3h (the AnA window's true "
            "start, T-3 -- troute_ana_b/ngen_forcing_ana's own output already "
            "covers this whole window with real values, so there's no T+0 gap "
            "to compensate for), duration_hours forced to 3 (T-3 -> T0). For "
            "sfincs, ana also sets dtrstout=3600 in run_param_overrides so "
            "SFINCS actually writes hourly restarts during the run -- "
            "--sfincs-rst-file alone only wires the read side."
        ),
    )
    p.add_argument(
        "--sfincs-rst-file",
        type=Path,
        default=None,
        help=(
            "(--model sfincs only) Path to a previous cycle's SFINCS restart "
            "(.rst) file to warm-start from, via sfincs.inp's rstfile key. "
            "Optional -- if omitted, or the path doesn't exist (true bootstrap: "
            "the very first cycle in the suite's history), the model just "
            "cold-starts; this is not an error. The ecflow trigger graph is "
            "responsible for never invoking this with a not-yet-produced path "
            "otherwise -- this flag does not distinguish 'no prior cycle "
            "exists' from 'the producing cycle hasn't finished yet'."
        ),
    )
    p.add_argument(
        "--hot-start-file",
        type=Path,
        default=None,
        help=(
            "(--model schism only) Path to a previous cycle's combined SCHISM "
            "hotstart (hotstart_it=<N>.nc) file to warm-start from, via "
            "paths.hot_start_file -- already fully wired end-to-end "
            "(boundary.py -> make_param_nml sets ihot=1 and copies the file). "
            "Same optional/fallback semantics as --sfincs-rst-file above."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    for label, path in (
        ("base config", args.base_yaml),
        ("forcing file", args.forecast_meteo_file),
        ("t-route file", args.troute_file),
    ):
        if not path.is_file():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 1

    with args.base_yaml.open() as fh:
        cfg = yaml.safe_load(fh)

    if cfg.get("model") != args.model:
        print(
            f"ERROR: {args.base_yaml} has model={cfg.get('model')!r}, "
            f"expected {args.model!r}",
            file=sys.stderr,
        )
        return 1

    out = args.cycle_dir / "run.yaml"
    work_dir = args.cycle_dir / "run"

    cycle_time = datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S")
    sim_cfg = cfg.setdefault("simulation", {})

    if args.run_type == "sr":
        # NWM short_range forecast forcing and t-route's routed streamflow
        # both start their first real output at cycle-time + 1h, not at the
        # cycle time itself -- there is no real T+0 driving data for either
        # source. Using the raw cycle time as the model's start_date forced
        # every downstream stage to fabricate a synthetic t=0 by persisting
        # the first real value backward one hour, which is what caused SCHISM
        # to abort mid-run (confirmed live). Setting the model's actual
        # start_date to cycle time + 1h means it starts exactly where the
        # real data starts -- no fabrication anywhere. This also means the
        # model runs for one hour less than base_yaml's duration_hours
        # nominally suggests (17 real hours of driving data, not 18) --
        # duration_hours in the base templates reflects that already.
        model_start_date = cycle_time + timedelta(hours=1)
    else:
        # AnA: troute_ana_b/ngen_forcing_ana already cover the full T-3->T0
        # window with real values (that's the whole point of the AnA
        # lookback), so there's no T+0 gap to shift past -- start_date is
        # T-3 directly. duration_hours is forced to 3 regardless of
        # base_yaml's SR-oriented value.
        model_start_date = cycle_time - timedelta(hours=3)
        sim_cfg["duration_hours"] = 3

    sim_cfg["start_date"] = model_start_date.strftime("%Y-%m-%d %H:%M:%S")
    paths = cfg.setdefault("paths", {})

    # coastal-calibration's PathConfig resolves every relative path against
    # the invoking process's CWD, not the YAML's location (see
    # PathConfig.__post_init__). The base templates use relative paths
    # (e.g. raw_download_dir: ../stofs_download) that only work because
    # they were hand-run from inside schism_sims/sfincs_sims. ecflow tasks
    # don't cd there, so make every relative paths.* entry absolute here,
    # resolved against the base YAML's own directory (matching what the
    # relative path meant when the template was written).
    base_dir = args.base_yaml.parent
    for key, value in list(paths.items()):
        if isinstance(value, str) and value and not value.startswith("/"):
            paths[key] = str((base_dir / value).resolve())

    paths["work_dir"] = str(work_dir)
    paths["forecast_meteo_file"] = str(args.forecast_meteo_file)
    paths["troute_file"] = str(args.troute_file)

    # SFINCS warm start: rstfile is a plain sfincs.inp key (no dedicated
    # schema field like SCHISM's paths.hot_start_file), set via the
    # existing run_param_overrides passthrough that SfincsWriteStage
    # already applies verbatim before writing sfincs.inp. Look for the
    # file; if it's not there (true bootstrap -- the first cycle in the
    # suite's history, with no prior AnA cycle to have produced one), just
    # don't set rstfile and let SFINCS use its own default (cold start).
    # The ecflow trigger graph, not this check, is what must guarantee
    # this is never invoked while the producing AnA cycle is still running.
    if args.model == "sfincs" and args.sfincs_rst_file is not None:
        if args.sfincs_rst_file.is_file():
            model_cfg = cfg.setdefault("model_config", {})
            overrides = model_cfg.setdefault("run_param_overrides", {})
            overrides["rstfile"] = str(args.sfincs_rst_file)
            print(f"Warm-starting SFINCS from {args.sfincs_rst_file}", file=sys.stderr)
        else:
            print(
                f"No SFINCS restart file at {args.sfincs_rst_file}; cold-starting",
                file=sys.stderr,
            )

    # SFINCS AnA state *writing*: independent of the read-side rstfile
    # above. dtrstout=3600 makes SFINCS write an hourly restart
    # (sfincs.<YYYYMMDD.HHMMSS>.rst -- tref + elapsed seconds, tref ==
    # this run's start_date == T-3) at T-2, T-1, and T0. The .ecf scripts
    # locate the T-2 file (self-chain to next cycle's AnA) and the T0 file
    # (this cycle's SR warm start) by computing the expected filename
    # directly, since the naming is a deterministic function of real time.
    if args.model == "sfincs" and args.run_type == "ana":
        model_cfg = cfg.setdefault("model_config", {})
        overrides = model_cfg.setdefault("run_param_overrides", {})
        overrides.setdefault("dtrstout", 3600)

    # SCHISM warm start: paths.hot_start_file is already a first-class
    # schema field (unlike SFINCS's raw config-key override above) --
    # boundary.py already reads it and make_param_nml already sets
    # ihot=1 / copies the file. Same fallback semantics as SFINCS's check.
    if args.model == "schism" and args.hot_start_file is not None:
        if args.hot_start_file.is_file():
            paths["hot_start_file"] = str(args.hot_start_file)
            print(f"Warm-starting SCHISM from {args.hot_start_file}", file=sys.stderr)
        else:
            print(
                f"No SCHISM hotstart file at {args.hot_start_file}; cold-starting",
                file=sys.stderr,
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    with out.open("w") as fh:
        fh.write(
            f"# AUTO-GENERATED by gen_cycle_config.py -- do not hand-edit.\n"
            f"# model={args.model} cycle={args.cycle} base={args.base_yaml}\n"
        )
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)

    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
