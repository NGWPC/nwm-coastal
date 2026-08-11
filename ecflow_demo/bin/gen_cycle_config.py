#!/usr/bin/env python3
"""Generate a per-cycle SCHISM/SFINCS run.yaml from a base template.

Invoked by the ecflow 'gen_configs' task via nwm-coastal-py (the
coastal-calibration pixi env's Python + PyYAML).
"""
from __future__ import annotations

import argparse
import json
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
    p.add_argument(
        "--forecast-meteo-file",
        type=Path,
        default=None,
        help=(
            "Required for --run-type ana/sr. Optional for --run-type spinup -- "
            "omit it entirely when the spin-up has both discharge and wind"
            "forcing disabled via model_config overrides."
        ),
    )
    p.add_argument(
        "--troute-file",
        type=Path,
        default=None,
        help=(
            "Required for --run-type ana/sr. Optional for --run-type spinup -- "
            "same reasoning as --forecast-meteo-file above (discharge disabled "
            "means nothing reads it)."
        ),
    )
    p.add_argument(
        "--t0-troute-file",
        type=Path,
        default=None,
        help=(
            "Optional second t-route output, used only when --troute-file's "
            "own window doesn't reach back to the model's start_date (e.g. "
            "--run-type sr, where troute's own SR run starts 1h after T0 by "
            "mswm design) -- just the T0 row is pulled from here (typically "
            "the ana run) instead of leaving that timestep at zero. Ignored "
            "for --run-type ana, as ana --troute-file already covers T0 "
            "directly."
        ),
    )
    p.add_argument(
        "--t0-precip-source-file",
        type=Path,
        default=None,
        help=(
            "Optional second precip_source.nc, used only when this  run's"
            "own gridded meteo forcing has no valid T0 sample "
            "(e.g. --run-type sr, same class of gap as --t0-troute-file "
            "above) -- typically that hour's own AnA schism_ana run's "
            "precip_source.nc. Ignored for --run-type ana, whose own "
            "forcing already covers T0 directly."
        ),
    )
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
        choices=["sr", "ana", "spinup"],
        default="ana",
        help=(
            "sr: model start_date = --start-date (T0) directly."
            "ana (default): model start_date = --start-date - 3h (the AnA 
            "window's true start.)"
            "For sfincs, ana also sets dtrstout=3600 in run_param_overrides so "
            "SFINCS writes hourly restarts during the run -- "
            "spinup: standalone boundary/tide equilibration run, used by "
            "hotstart_coastal_models.py -- model start_date = --start-date minus "
            "--duration-hours (required for this type; default=18h). Always"
            "cold-starts (no --hot-start-file/--sfincs-rst-file)."
        ),
    )
    p.add_argument(
        "--duration-hours",
        type=int,
        default=None,
        help="(--run-type spinup only, required) Length of the spin-up run in hours.",
    )
    p.add_argument(
        "--extra-run-param-overrides",
        type=str,
        default=None,
        help=(
            "Optional JSON object merged into model_config.run_param_overrides "
            "(e.g. '{\"dramp\": 0.25, \"nramp_elev\": 1}' for SCHISM or "
            "'{\"tspinup\": 21600}' for SFINCS) -- used by hotstart_coastal_models.py "
            "to enable each model's physics ramp for the spin-up run. Merged after "
            "any other overrides this script itself sets, so it can override those too."
        ),
    )
    p.add_argument(
        "--sfincs-rst-file",
        type=Path,
        default=None,
        help=(
            "(--model sfincs only) Optional  path to a previous cycle's SFINCS "
            "restart (.rst) file to warm-start from, via sfincs.inp's rstfile key. "
        ),
    )
    p.add_argument(
        "--hot-start-file",
        type=Path,
        default=None,
        help=(
            "(--model schism only) Path to a previous cycle's combined SCHISM "
            "hotstart (hotstart_it=<N>.nc) file to warm-start from, via "
            "paths.hot_start_file"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.base_yaml.is_file():
        print(f"ERROR: base config not found: {args.base_yaml}", file=sys.stderr)
        return 1

    # --forecast-meteo-file/--troute-file are required for ana/sr;
    # optional for spinup, where discharge and (SCHISM) wind/
    # (SFINCS) precip+wind+pressure are all disabled and nothing
    # ever reads them
    for label, flag, path in (
        ("forcing file", "--forecast-meteo-file", args.forecast_meteo_file),
        ("t-route file", "--troute-file", args.troute_file),
    ):
        if path is None:
            if args.run_type != "spinup":
                print(f"ERROR: {flag} is required for --run-type {args.run_type}", file=sys.stderr)
                return 1
            continue
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

    if args.run_type == "spinup":
        if args.duration_hours is None:
            print("ERROR: --run-type spinup requires --duration-hours", file=sys.stderr)
            return 1
        if (args.hot_start_file is not None) or (args.sfincs_rst_file is not None):
            print(
                "ERROR: --run-type spinup always cold-starts; "
                "--hot-start-file/--sfincs-rst-file are not accepted",
                file=sys.stderr,
            )
            return 1
        model_start_date = cycle_time - timedelta(hours=args.duration_hours)
        sim_cfg["duration_hours"] = args.duration_hours

        # Spin-up is boundary(STOFS)-forcing only -- config overrides, not
        # base-template edits, so the real ana/sr cycles that share these
        # same base_yaml files are completely unaffected.
        spinup_model_cfg = cfg.setdefault("model_config", {})
        if args.model == "schism":
            # River/precip forcing is only ever injected via source.nc,
            # which SchismDischargeStage (and the precip regridding that
            # feeds it, schism_forcing) skips entirely whenever
            # discharge_file doesn't resolve to a real, existing file (see
            # PathConfig.resolved_discharge_file). Just unsetting
            # discharge_file isn't enough -- it falls back to
            # auto-discovering ngenReaches.csv/nwmReaches.csv next to
            # prebuilt_dir (which exists here) -- so point it at a
            # deliberately nonexistent path instead, forcing the "explicit
            # but missing -> skip" branch of that resolution order.
            # include_wind=False additionally skips schism_sflux and sets
            # nws=0 in param.nml (see UpdateParamsStage/update_params),
            # so --forecast-meteo-file's actual content is now never read
            # by anything
            spinup_model_cfg["discharge_file"] = (
                "/nonexistent/hotstart_coastal_models-spinup-no-discharge.csv"
            )
            spinup_model_cfg["include_wind"] = False
        else:
            spinup_model_cfg["include_precip"] = False
            spinup_model_cfg["include_wind"] = False
            spinup_model_cfg["include_pressure"] = False
            spinup_model_cfg["merge_discharge"] = False
            spinup_model_cfg["discharge_locations_file"] = None
    elif args.run_type == "sr":
        # SCHISM/SFINCS load their T0 state directly from this same hour's
        # own AnA coastal-model checkpoint (see --hot-start-file/
        # --sfincs-rst-file)
        model_start_date = cycle_time
        sim_cfg["duration_hours"] = 18
    else:
        # AnA: troute_ana_b/ngen_forcing_ana already cover the full T-3->T0
        # window with real values -- start_date is
        # T-3 directly. duration_hours is forced to 3 regardless of
        # base_yaml's SR-oriented value.
        model_start_date = cycle_time - timedelta(hours=3)
        sim_cfg["duration_hours"] = 3

    sim_cfg["start_date"] = model_start_date.strftime("%Y-%m-%d %H:%M:%S")
    paths = cfg.setdefault("paths", {})

    # coastal-calibration's PathConfig resolves every relative path against
    # the invoking process's CWD, not the YAML's location (see
    # PathConfig.__post_init__). The base templates use relative paths
    # because it is uncertain where a user will install them. ecflow tasks
    # don't cd there, so make every relative paths.* entry absolute here,
    # resolved against the base YAML's own directory (matching what the
    # relative path meant when the template was written).
    base_dir = args.base_yaml.parent
    for key, value in list(paths.items()):
        if isinstance(value, str) and value and not value.startswith("/"):
            paths[key] = str((base_dir / value).resolve())

    # Same problem, same fix, for model_config's own path-valued keys (e.g.
    # prebuilt_dir: ../sfincs_models/tampabay)
    _MODEL_CONFIG_PATH_KEYS = {
        "schism": ("prebuilt_dir", "geogrid_file", "schism_exe", "obs_points_csv"),
        "sfincs": (
            "prebuilt_dir",
            "model_root",
            "discharge_locations_file",
            "sfincs_exe",
            "floodmap_dem",
            "obs_points_csv",
        ),
    }
    model_cfg_for_paths = cfg.get("model_config", {})
    for key in _MODEL_CONFIG_PATH_KEYS.get(args.model, ()):
        value = model_cfg_for_paths.get(key)
        if isinstance(value, str) and value and not value.startswith("/"):
            model_cfg_for_paths[key] = str((base_dir / value).resolve())

    paths["work_dir"] = str(work_dir)
    # Explicit None (not just "leave unset"), since the base template
    # itself may already carry a literal example/stale value here that a
    # simple omission would otherwise silently fall back to.
    paths["forecast_meteo_file"] = str(args.forecast_meteo_file) if args.forecast_meteo_file else None
    paths["troute_file"] = str(args.troute_file) if args.troute_file else None
    if args.t0_troute_file is not None:
        paths["t0_troute_file"] = str(args.t0_troute_file)
    if args.t0_precip_source_file is not None and args.model == "schism":
        paths["t0_precip_source_file"] = str(args.t0_precip_source_file)

    # SFINCS warm start: rstfile is a plain sfincs.inp key set via the
    # existing run_param_overrides passthrough that SfincsWriteStage
    # already applies verbatim before writing sfincs.inp. Look for the
    # file; if it's not there (the first cycle in the
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

    # SFINCS AnA state writing: independent of the read-side rstfile
    # above. dtrstout=3600 makes SFINCS write an hourly restart
    # (sfincs.<YYYYMMDD.HHMMSS>.rst -- tref + elapsed seconds, tref ==
    # this run's start_date == T-3) at T-2, T-1, and T0. The .ecf scripts
    # locate the T-2 file (self-chain to next cycle's AnA) and the T0 file
    # (this cycle's SR warm start) by computing the expected filename
    # directly, since the naming is a deterministic function of real time.
    if args.model == "sfincs" and args.run_type in ("ana", "spinup"):
        model_cfg = cfg.setdefault("model_config", {})
        overrides = model_cfg.setdefault("run_param_overrides", {})
        overrides.setdefault("dtrstout", 3600)

    # SCHISM warm start: paths.hot_start_file is already a first-class
    # schema field boundary.py already reads it and make_param_nml already sets
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

    if args.extra_run_param_overrides is not None:
        try:
            extra = json.loads(args.extra_run_param_overrides)
        except json.JSONDecodeError as exc:
            print(f"ERROR: --extra-run-param-overrides is not valid JSON: {exc}", file=sys.stderr)
            return 1
        model_cfg = cfg.setdefault("model_config", {})
        overrides = model_cfg.setdefault("run_param_overrides", {})
        overrides.update(extra)

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
