"""Python implementations of bash workflow functions.

This module replaces several bash scripts with native Python implementations:
- pre_nwm_forcing_coastal.bash
- post_nwm_forcing_coastal.bash
- merge_source_sink.bash
- post_schism.bash

These functions perform file operations, symlink creation, and environment setup
that are more naturally expressed in Python.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from coastal_calibration.utils.time import advance_time, format_forcing_date, parse_date_components

_DATE_RE = re.compile(r"^\d{10}$")


def _validate_date_string(date_string: str) -> None:
    """Validate that ``date_string`` is exactly 10 digits (YYYYMMDDHH)."""
    if not isinstance(date_string, str) or not _DATE_RE.match(date_string):
        raise ValueError(
            f"date_string must be exactly 10 digits in YYYYMMDDHH format, got {date_string!r}"
        )


logger = logging.getLogger(__name__)


def pre_nwm_forcing_coastal(
    date_string: str,
    coastal_forcing_output_dir: str | Path,
    length_hrs: int,
    nwm_forcing_retro_dir: str | Path,
    data_exec: str | Path,
    meteo_source: str = "nwm_retro",
    coastal_domain: str = "conus",
) -> dict[str, str]:
    """Prepare NWM forcing data for coastal model.

    Creates symlinks to LDASIN forcing files and sets up environment variables
    for the coastal forcing engine.  Both ``nwm_retro`` and ``nwm_ana`` store
    downloaded files as ``YYYYMMDDHH.LDASIN_DOMAIN1``, so this function
    creates symlinks using that convention for all meteo sources.

    This replaces pre_nwm_forcing_coastal.bash.

    Parameters
    ----------
    date_string : str
        Date string in YYYYMMDDHH format.
    coastal_forcing_output_dir : str or Path
        Output directory for coastal forcing files.
    length_hrs : int
        Number of hours of forcing data.
    nwm_forcing_retro_dir : str or Path
        Directory containing NWM forcing files (``YYYYMMDDHH.LDASIN_DOMAIN1``).
    data_exec : str or Path
        Working directory for execution.
    meteo_source : str
        Meteorological data source (``"nwm_retro"`` or ``"nwm_ana"``).
        Kept for API compatibility; both sources now use the same naming.
    coastal_domain : str
        Coastal domain name (e.g. ``"hawaii"``, ``"conus"``).
        Kept for API compatibility.

    Returns
    -------
    dict
        Environment variables that were set.
    """
    _validate_date_string(date_string)

    data_exec = Path(data_exec)
    coastal_forcing_output_dir = Path(coastal_forcing_output_dir)
    nwm_forcing_retro_dir = Path(nwm_forcing_retro_dir)

    date_parts = parse_date_components(date_string)

    forcing_begin_date = format_forcing_date(date_string)
    forcing_end_date = format_forcing_date(advance_time(date_string, length_hrs))

    nwm_forcing_output_dir = data_exec / "forcing_input"
    forcing_input_subdir = nwm_forcing_output_dir / forcing_begin_date[:10]
    forcing_input_subdir.mkdir(parents=True, exist_ok=True)

    current_date = date_string
    for i in range(length_hrs + 1):
        filename = f"{current_date}.LDASIN_DOMAIN1"
        source = nwm_forcing_retro_dir / filename
        target = forcing_input_subdir / filename

        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)

        current_date = advance_time(date_string, i + 1)

    coastal_forcing_output_dir.mkdir(parents=True, exist_ok=True)

    env_vars = {
        "FORCING_BEGIN_DATE": forcing_begin_date,
        "NWM_FORCING_OUTPUT_DIR": str(nwm_forcing_output_dir),
        "FORCING_END_DATE": forcing_end_date,
        "COASTAL_FORCING_INPUT_DIR": str(forcing_input_subdir),
        "COASTAL_WORK_DIR": str(data_exec),
        "FORCING_START_YEAR": date_parts["year"],
        "FORCING_START_MONTH": date_parts["month"],
        "FORCING_START_DAY": date_parts["day"],
        "FORCING_START_HOUR": date_parts["hour"],
        "COASTAL_FORCING_OUTPUT_DIR": str(coastal_forcing_output_dir),
        "LENGTH_HRS": str(length_hrs),
        "FECPP_JOB_INDEX": "0",
        "FECPP_JOB_COUNT": "1",
    }

    return env_vars


def post_nwm_forcing_coastal(
    date_string: str,
    coastal_forcing_output_dir: str | Path,
    length_hrs: int,
    data_exec: str | Path,
    data_logs: str | Path,
    coastal_scripts_dir: str | Path,
) -> dict[str, str]:
    """Post-process NWM forcing data for coastal model.

    Creates symlinks to forcing output and runs makeAtmo.py to generate
    final SCHISM inputs.

    This replaces post_nwm_forcing_coastal.bash.

    Parameters
    ----------
    date_string : str
        Date string in YYYYMMDDHH format
    coastal_forcing_output_dir : str or Path
        Directory containing coastal forcing output
    length_hrs : int
        Number of hours of forcing data
    data_exec : str or Path
        Working directory for execution
    data_logs : str or Path
        Directory for log files
    coastal_scripts_dir : str or Path
        Directory containing coastal Python scripts

    Returns
    -------
    dict
        Environment variables that were set
    """
    _validate_date_string(date_string)

    data_exec = Path(data_exec)
    coastal_forcing_output_dir = Path(coastal_forcing_output_dir)
    data_logs = Path(data_logs)
    coastal_scripts_dir = Path(coastal_scripts_dir)

    date_parts = parse_date_components(date_string)
    pdy = date_parts["pdy"]
    cyc = date_parts["cyc"]

    forcing_begin_date = format_forcing_date(date_string)
    forcing_end_date = format_forcing_date(advance_time(date_string, length_hrs))

    nwm_forcing_output_dir = data_exec / "forcing_input"
    env_vars = {
        "FORCING_BEGIN_DATE": forcing_begin_date,
        "NWM_FORCING_OUTPUT_DIR": str(nwm_forcing_output_dir),
        "FORCING_END_DATE": forcing_end_date,
        "COASTAL_FORCING_INPUT_DIR": str(nwm_forcing_output_dir / forcing_begin_date[:10]),
        "COASTAL_WORK_DIR": str(data_exec),
        "FORCING_START_YEAR": date_parts["year"],
        "FORCING_START_MONTH": date_parts["month"],
        "FORCING_START_DAY": date_parts["day"],
        "FORCING_START_HOUR": date_parts["hour"],
        "COASTAL_FORCING_OUTPUT_DIR": str(coastal_forcing_output_dir),
        "LENGTH_HRS": str(length_hrs),
    }

    precip_source = coastal_forcing_output_dir / "precip_source.nc"
    precip_link = data_exec / "precip_source.nc"
    if precip_source.exists():
        if precip_link.is_symlink() or precip_link.exists():
            precip_link.unlink()
        precip_link.symlink_to(precip_source)

    sflux_dir = data_exec / "sflux"
    sflux_dir.mkdir(parents=True, exist_ok=True)

    log_file = data_logs / f"nwm_forcing_{pdy}{cyc}.log"
    make_atmo_script = coastal_scripts_dir / "makeAtmo.py"

    # Update environment for subprocess
    run_env = os.environ.copy()
    run_env.update(env_vars)

    with Path.open(log_file, "a") as log_f:
        result = subprocess.run(
            [sys.executable, "-u", str(make_atmo_script)],
            env=run_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(data_exec),
            check=False,
        )

    if result.returncode != 0:
        log_tail = ""
        if log_file.exists():
            lines = log_file.read_text().splitlines()
            log_tail = "\n".join(lines[-20:])
        raise RuntimeError(
            f"makeAtmo.py failed (exit code {result.returncode}).\nLog file: {log_file}\n{log_tail}"
        )

    return env_vars


def nwm_coastal_merge_source_sink(
    nwm_cycle: str,
    nwm_base_cycle: str,
    data_exec: str | Path,
    data_logs: str | Path,
    coastal_scripts_dir: str | Path,
    parm_nwm: str | Path,
    coastal_domain: str,
    coastal_root_dir: str | Path | None = None,
) -> None:
    """Merge source and sink data for coastal model.

    This replaces merge_source_sink.bash.

    Parameters
    ----------
    nwm_cycle : str
        NWM cycle identifier
    nwm_base_cycle : str
        NWM base cycle identifier
    data_exec : str or Path
        Working directory for execution
    data_logs : str or Path
        Directory for log files
    coastal_scripts_dir : str or Path
        Directory containing coastal Python scripts
    parm_nwm : str or Path
        NWM parameter directory
    coastal_domain : str
        Coastal domain name
    coastal_root_dir : str or Path, optional
        Coastal root directory (defaults to data_exec)
    """
    data_exec = Path(data_exec)
    data_logs = Path(data_logs)
    coastal_scripts_dir = Path(coastal_scripts_dir)
    parm_nwm = Path(parm_nwm)

    cycle = nwm_cycle

    if nwm_base_cycle != cycle:
        source_nc = parm_nwm / "coastal" / coastal_domain / "source.nc"
        target_nc = data_exec / "source.nc"
        if source_nc.exists():
            if target_nc.is_symlink() or target_nc.exists():
                target_nc.unlink()
            target_nc.symlink_to(source_nc)

    env_vars = {
        "COASTAL_ROOT_DIR": str(coastal_root_dir or data_exec),
        "COASTAL_WORK_DIR": str(data_exec),
    }

    run_env = os.environ.copy()
    run_env.update(env_vars)

    log_file = data_logs / "merge_source_sink.log"
    merge_script = coastal_scripts_dir / "merge_source_sink.py"

    with Path.open(log_file, "a") as log_f:
        result = subprocess.run(
            [sys.executable, str(merge_script)],
            env=run_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(data_exec),
            check=False,
        )

    if result.returncode != 0:
        log_tail = ""
        if log_file.exists():
            lines = log_file.read_text().splitlines()
            log_tail = "\n".join(lines[-20:])
        raise RuntimeError(
            f"merge_source_sink.py failed (exit code {result.returncode}).\n"
            f"Log file: {log_file}\n{log_tail}"
        )


def post_nwm_coastal(
    data_exec: str | Path,
    length_hrs: int,
    restart_write_hr: int | None = None,
    exec_nwm: str | Path | None = None,
    chained_reanalysis: bool = False,
) -> None:
    """Post-process SCHISM coastal model output.

    Checks for errors and optionally combines hotstarts for analysis.

    This replaces post_schism.bash.

    Parameters
    ----------
    data_exec : str or Path
        Working directory for execution
    length_hrs : int
        Simulation length in hours
    restart_write_hr : int, optional
        Hour at which restart was written
    exec_nwm : str or Path, optional
        Directory containing NWM executables
    chained_reanalysis : bool
        Whether running chained reanalysis

    Raises
    ------
    RuntimeError
        If fatal.error file exists and has content
    """
    data_exec = Path(data_exec)
    outputs_dir = data_exec / "outputs"

    outputs_dir.mkdir(parents=True, exist_ok=True)

    fatal_error = outputs_dir / "fatal.error"
    if fatal_error.exists() and fatal_error.stat().st_size > 0:
        raise RuntimeError(f"SCHISM program failed. See {fatal_error} file for more detail.")

    # Combine hotstarts for analysis or chained reanalysis
    if (
        (length_hrs < 0 or chained_reanalysis)
        and restart_write_hr is not None
        and exec_nwm is not None
    ):
        exec_nwm = Path(exec_nwm)
        combine_binary = exec_nwm / "combine_hotstart7"
        if combine_binary.exists():
            target = outputs_dir / "combine_hotstart7"
            if target.exists() or target.is_symlink():
                target.unlink()
            # Copy the binary
            import shutil

            shutil.copy2(combine_binary, target)
