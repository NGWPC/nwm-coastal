"""SCHISM pre- and post-processing functions.

Pure-Python replacements for the bash scripts that previously
orchestrated SCHISM pre/post processing (``initial_discharge.bash``,
``combine_sink_source.bash``, ``merge_source_sink.bash``,
``pre_schism.bash``, ``post_schism.bash``).

All functions accept explicit paths/values rather than reading
``os.environ``, making them testable in isolation.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import netCDF4
import numpy as np

from coastal_calibration._nc_io import write_var
from coastal_calibration.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def _symlink(src: Path, dst: Path) -> None:
    """Create a symlink, replacing an existing one."""
    dst.unlink(missing_ok=True)
    dst.symlink_to(src)


def _format_namelist_value(value: Any) -> str:
    """Render *value* as a Fortran namelist literal."""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, (int, float)):
        return repr(value)
    return str(value)


def _apply_namelist_overrides(text: str, overrides: dict[str, Any]) -> str:
    """Replace ``key = ...`` lines in a Fortran namelist with new values.

    Keys not already present in *text* are appended just before the
    closing ``/`` of the ``&OPT`` block (SCHISM's general physics/
    numerics namelist -- home to the vast majority of ad-hoc override
    keys, e.g. ``nramp``/``nrampbc`` alongside their already-present
    siblings ``dramp``/``drampbc``). SCHISM's real ``param.nml`` has
    three blocks (``&CORE``, ``&OPT``, ``&SCHOUT``); inserting into
    whichever block happens to close last (the old behavior) landed
    physics keys inside ``&SCHOUT`` -- an output-control namelist that
    doesn't recognize them -- corrupting it enough to break parsing of
    unrelated array members like ``iof_hydro`` at SCHISM startup. Falls
    back to the last block in the file if no ``&OPT`` block is found
    (e.g. a differently-structured namelist file). Raises ``KeyError``
    if no namelist block is found at all.
    """
    # The closing line's whitespace must be horizontal-only ([ \t], not
    # \s): \s also matches \n, so a greedy \s*$ silently swallows the
    # blank lines *between* this block's "/" and the next block's "&NAME",
    # moving the match's end() past where "/" actually is.
    opt_close_re = re.compile(r"(?mi)^&OPT[ \t]*$.*?(^[ \t]*/[ \t]*$)", flags=re.DOTALL)

    out = text
    for key, value in overrides.items():
        rendered = _format_namelist_value(value)
        new_text, n = re.subn(
            rf"(?mi)^(\s*){re.escape(key)}\s*=.*$",
            rf"\g<1>{key} = {rendered}",
            out,
        )
        if n == 0:
            opt_match = opt_close_re.search(out)
            insert_at = opt_match.start(1) if opt_match else out.rfind("/")
            if insert_at < 0:
                raise KeyError(f"No namelist closing '/' found in param.nml; cannot insert {key}")
            out = out[:insert_at] + f"  {key} = {rendered}\n" + out[insert_at:]
        else:
            out = new_text
    return out


def _read_namelist_int(text: str, key: str) -> int | None:
    """Return the integer value of ``key`` in a Fortran namelist, if present."""
    m = re.search(rf"(?mi)^\s*{re.escape(key)}\s*=\s*(-?\d+)", text)
    return int(m.group(1)) if m else None


def validate_param_nml(param_path: Path) -> list[str]:
    """Check the SCHISM-required relationships between output parameters.

    Returns a list of human-readable error strings (empty if the file
    is internally consistent). Catches the same constraints SCHISM
    enforces at startup so the user sees the misconfiguration before
    the MPI binary aborts.
    """
    text = param_path.read_text()

    nspool = _read_namelist_int(text, "nspool")
    ihfskip = _read_namelist_int(text, "ihfskip")
    nhot = _read_namelist_int(text, "nhot")
    nhot_write = _read_namelist_int(text, "nhot_write")
    iout_sta = _read_namelist_int(text, "iout_sta")
    nspool_sta = _read_namelist_int(text, "nspool_sta")

    errors: list[str] = []
    if nspool is not None and nspool <= 0:
        errors.append(f"param.nml: nspool must be > 0 (got {nspool})")
    if ihfskip is not None and ihfskip <= 0:
        errors.append(f"param.nml: ihfskip must be > 0 (got {ihfskip})")
    if nspool is not None and ihfskip is not None and ihfskip % nspool != 0:
        errors.append(f"param.nml: ihfskip ({ihfskip}) must be a multiple of nspool ({nspool})")
    if nhot == 1 and nhot_write is not None and ihfskip is not None and nhot_write % ihfskip != 0:
        errors.append(
            f"param.nml: nhot_write ({nhot_write}) must be a multiple of ihfskip ({ihfskip}) "
            "when nhot = 1"
        )
    if (
        iout_sta is not None
        and iout_sta != 0
        and nhot_write is not None
        and nspool_sta is not None
        and nhot_write % nspool_sta != 0
    ):
        errors.append(
            f"param.nml: nhot_write ({nhot_write}) must be a multiple of nspool_sta "
            f"({nspool_sta}) when iout_sta = {iout_sta}"
        )
    return errors


def clean_run_directory(work_dir: Path) -> None:
    """Remove generated files from a previous SCHISM run.

    Preserves symlinks to prebuilt model files and log files.
    Safe to call on a fresh directory (no-ops if files don't exist).
    """
    generated_files = [
        "vsource.th",
        "vsource.th.1",
        "vsink.th",
        "vsink.th.1",
        "source_sink.in",
        "source_sink.in.1",
        "i_sink_source.txt",
        "source.nc",
        "precip_source.nc",
        "elev2D.th.nc",
        "param.nml",
        "partition.prop",
        "nwmReaches.csv",
        "otps_lat_lon_time.txt",
        "otps_out.txt",
        "station_noaa_ids.txt",
        ".pipeline_status.json",
    ]
    for name in generated_files:
        (work_dir / name).unlink(missing_ok=True)

    for f in work_dir.glob("graphinfo*"):
        f.unlink(missing_ok=True)

    sflux_dir = work_dir / "sflux"
    if sflux_dir.is_dir():
        for f in sflux_dir.glob("*.nc"):
            f.unlink(missing_ok=True)

    for dirname in [
        "outputs",
        "coastal_forcing_output",
        "forcing_input",
        "nwm_output",
        "nwm_output_ana",
        "figs",
    ]:
        d = work_dir / dirname
        if d.exists():
            shutil.rmtree(d)

    logger.info("Cleaned generated files from %s", work_dir)


# ---------------------------------------------------------------------------
# 1. Stage CHRTOUT files  (was initial_discharge.bash symlink logic)
# ---------------------------------------------------------------------------


def stage_chrtout_files(
    *,
    work_dir: Path,
    start_date: datetime,
    duration_hours: int,
    coastal_domain: str,
    streamflow_dir: Path,
) -> tuple[Path, Path | None]:
    """Symlink NWM CHRTOUT files into staging directories.

    Returns ``(nwm_output_dir, nwm_ana_dir)`` so that
    :func:`make_discharge` can find them.

    Both directories are scoped to *start_date* and emptied first, the same
    way :func:`stage_ldasin_files` scopes its output.  ``make_discharge``
    reads whatever it finds by glob and derives each timestamp from the file
    itself, so a link left behind by an earlier run in the same ``work_dir``
    would silently extend the discharge series with another run's dates.
    """
    stamp = start_date.strftime("%Y%m%d%H")
    nwm_output_dir = work_dir / "nwm_output" / stamp
    nwm_ana_dir = work_dir / "nwm_output_ana" / stamp
    for staging in (nwm_output_dir, nwm_ana_dir):
        staging.mkdir(parents=True, exist_ok=True)
        for stale in staging.glob("*CHRTOUT*"):
            stale.unlink()

    is_hawaii = "hawaii" in coastal_domain
    sub_steps = (15, 30, 45) if is_hawaii else ()

    cycle_length_hrs = duration_hours - 1

    for i in range(cycle_length_hrs + 2):
        dt = start_date + timedelta(hours=i)
        pdycyc = dt.strftime("%Y%m%d%H")

        if i == 0:
            # First timestep → analysis dir
            fname = f"{pdycyc}00.CHRTOUT_DOMAIN1"
            _symlink(streamflow_dir / fname, nwm_ana_dir / fname)
            for m in sub_steps:
                fname = f"{pdycyc}{m:02d}.CHRTOUT_DOMAIN1"
                _symlink(streamflow_dir / fname, nwm_output_dir / fname)
        else:
            fname = f"{pdycyc}00.CHRTOUT_DOMAIN1"
            _symlink(streamflow_dir / fname, nwm_output_dir / fname)
            for m in sub_steps:
                fname = f"{pdycyc}{m:02d}.CHRTOUT_DOMAIN1"
                _symlink(streamflow_dir / fname, nwm_output_dir / fname)

    return nwm_output_dir, nwm_ana_dir


def _parse_reach_rows(rows: list[str], count: int, kind: str) -> tuple[list[int], list[int]]:
    """Split ``elem_id feature_id`` rows into element IDs and feature IDs."""
    parts = [r.split() for r in rows]
    if len(parts) != count:
        raise ValueError(f"nwmReaches.csv declares {count} {kind} rows but has {len(parts)}")
    if any(len(p) < 2 for p in parts):
        raise ValueError(f"nwmReaches.csv has a {kind} row that is not 'elem_id feature_id'")
    return [int(p[0]) for p in parts], [int(p[1]) for p in parts]


def _write_th_file(path: Path, data: NDArray[np.floating[Any]], times: NDArray[np.floating[Any]]) -> None:
    """Write a SCHISM time-history (.th) file.

    *times* are elapsed seconds since the simulation start, one per row of
    *data*, and must already include a t=0 row -- SCHISM requires the first
    row of a .th file to be at t=0 (see the t=0 padding in ``make_discharge``).
    """
    with path.open("w") as f:
        for i in range(data.shape[0]):
            parts = [str(times[i])]
            parts.extend(str(data[i, j]) for j in range(data.shape[1]))
            f.write("\t".join(parts) + "\n")


# ---------------------------------------------------------------------------
# 2. Make discharge  (was makeDischarge.py)
# ---------------------------------------------------------------------------


def make_discharge(  # noqa: PLR0912
    *,
    work_dir: Path,
    nwm_output_dir: Path,
    nwm_ana_dir: Path | None = None,
    is_analysis: bool = False,
    meteo_source: str = "nwm_ana",
    domain: str = "conus",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    troute_file: Path | None = None,
    t0_troute_file: Path | None = None,
    reaches_filename: str = "nwmReaches.csv",
) -> None:
    """Create discharge files from routed-streamflow output.

    Writes ``vsource.th``, ``vsink.th``, and ``source_sink.in`` into
    *work_dir*.  The sink block of ``nwmReaches.csv`` is optional, since a
    mesh subset can hold sources and no sinks; ``source_sink.in`` still
    declares a sink count of ``0`` in that case.

    For ``nwm_retro`` the streamflow is read directly from the S3 Zarr
    store (requires *start_date* and *end_date*).  For ``nwm_ana`` the
    streamflow is read from local CHRTOUT netCDF files.  For
    ``ngen_forecast`` it is read from the t-route output netCDF
    (*troute_file*, requires *start_date* and *end_date*).

    *t0_troute_file* is optional and only relevant for ``ngen_forecast``:
    some callers' primary *troute_file* doesn't reach back to *start_date*
    by design (e.g. an SR forecast warm-started from an AnA cycle, where
    troute's own SR run starts 1h after T0). When given and *troute_file*'s
    own data doesn't cover *start_date*, just the T0 row is pulled from
    this second source instead of leaving that row at zero. Every other
    caller simply omits it and behavior is unchanged.
    """
    from coastal_calibration.data.streamflow import read_streamflow

    reaches_path = work_dir / reaches_filename
    # Blank separator lines are dropped, which makes the trailing sink
    # block optional: a mesh subset can contain sources and no sinks, and
    # such a file may end right after the source block instead of writing
    # the "0" count.  Both spellings read as zero sinks.
    lines = [ln for ln in reaches_path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{reaches_path} is empty")

    nso = int(lines[0])
    soelems, soids = _parse_reach_rows(lines[1 : 1 + nso], nso, "source")

    # The whole remainder is handed over so an under-declared count or a
    # trailing block raises instead of silently dropping sink forcing.
    sink_lines = lines[1 + nso :]
    nsi = int(sink_lines[0]) if sink_lines else 0
    sielems, siids = _parse_reach_rows(sink_lines[1:], nsi, "sink")

    all_fids = soids + siids

    if meteo_source == "nwm_retro":
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required for nwm_retro")
        df = read_streamflow(
            all_fids,
            start_date,
            end_date,
            meteo_source="nwm_retro",
            domain=domain,
        )
    elif meteo_source == "ngen_forecast":
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required for ngen_forecast")
        if troute_file is None:
            raise ValueError("troute_file is required for ngen_forecast discharge")
        df = read_streamflow(
            all_fids,
            start_date,
            end_date,
            meteo_source="ngen_forecast",
            troute_file=troute_file,
        )
        if t0_troute_file is not None and start_date not in df.index:
            import pandas as pd

            df_t0 = read_streamflow(
                all_fids,
                start_date,
                start_date,
                meteo_source="ngen_forecast",
                troute_file=t0_troute_file,
            )
            if not df_t0.empty:
                # df and df_t0 come from two different troute regionalization
                # runs (region_sr vs region_ana_b) and can each cover a
                # different reach subset -- concat unions their columns, so
                # any fid present in one but absent from the other becomes
                # NaN at that row. Confirmed live: this NaN then overwrote
                # the zero-initialized vsource/vsink array below at the T0
                # row for every one of this run's 269 source elements
                # (region_ana_b's reach set didn't overlap this run's at
                # all), producing a raw netCDF fill-value-scale discharge
                # that blew up SCHISM within its first timestep. Zero-fill
                # to match the "unmatched slots stay at zero" convention
                # documented below, instead of leaking NaN into the model.
                df = pd.concat([df_t0, df]).sort_index().fillna(0.0)
    else:
        # Gather local CHRTOUT files for nwm_ana
        chrtout_files: list[Path] = []
        if not is_analysis and nwm_ana_dir is not None:
            ana_files = sorted(nwm_ana_dir.glob("*CHRTOUT*"))
            if ana_files:
                chrtout_files.append(ana_files[-1])
        chrtout_files.extend(sorted(nwm_output_dir.glob("*CHRTOUT*")))

        if not chrtout_files:
            raise FileNotFoundError(f"No CHRTOUT files found in {nwm_output_dir}")

        logger.info("    Processing %d CHRTOUT files", len(chrtout_files))

        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required for nwm_ana")
        df = read_streamflow(
            all_fids,
            start_date,
            end_date,
            meteo_source="nwm_ana",
            chrtout_files=chrtout_files,
        )

    # Resample sub-hourly data to hourly (e.g., Hawaii 15-min data)
    if len(df) > 1:
        freq = df.index.to_series().diff().median()
        if freq < timedelta(hours=1):  # pyright: ignore[reportOperatorIssue]
            df = df.resample("h").mean()

    # SCHISM reads vsource/vsink from source.nc purely positionally -- row i
    # is always taken to mean "elapsed i*3600s from start_date" (see
    # schism_init.F90's "nc" branch and schism_step.F90's STEP: vsource
    # read: both index by time/th_dt3(1), never by the stored time_vsource
    # values). merge_source_sink() later adds discharge into precip
    # row-for-row by array position too, not by matching time labels. So
    # writing df's rows in arrival order (the old behavior) only happened
    # to be correct when df's first row landed exactly on start_date with
    # no gaps -- any mismatch between troute's real window and start_date
    # would silently misalign every row downstream. Scatter into a
    # gapless hourly grid spanning [start_date, end_date] by each row's
    # real elapsed time instead, so row i is genuinely i*3600s regardless
    # of what troute actually returned. Unmatched slots stay at zero --
    # deliberately not backfilled (e.g. hot-started runs load their t=0
    # state from the restart file and don't need forcing to describe it).
    n_hours = round((end_date - start_date).total_seconds() / 3600)
    elapsed = np.arange(n_hours + 1, dtype=float) * 3600.0
    vsource = np.zeros((n_hours + 1, len(soids)))
    vsink = np.zeros((n_hours + 1, len(siids)))

    row_idx = np.round((df.index - start_date).total_seconds().to_numpy() / 3600).astype(int)
    in_range = (row_idx >= 0) & (row_idx <= n_hours)
    if not in_range.all():
        logger.warning(
            "    make_discharge: dropping %d row(s) outside [start_date, end_date]",
            int((~in_range).sum()),
        )
    row_idx = row_idx[in_range]
    df = df.iloc[in_range]

    for i, sid in enumerate(soids):
        if sid in df.columns:
            vsource[row_idx, i] = df[sid].to_numpy()

    for i, sid in enumerate(siids):
        if sid in df.columns:
            vsink[row_idx, i] = -1.0 * df[sid].to_numpy()

    _write_th_file(work_dir / "vsource.th", vsource, elapsed)
    _write_th_file(work_dir / "vsink.th", vsink, elapsed)

    # source_sink.in
    with (work_dir / "source_sink.in").open("w") as f:
        f.write(f"{len(soelems)}\n")
        for e in soelems:
            f.write(f"{e}\n")
        f.write("\n")
        f.write(f"{len(sielems)}\n")
        for e in sielems:
            f.write(f"{e}\n")

    logger.info(
        "    Wrote vsource.th (%d rows), vsink.th, source_sink.in (%d sources, %d sinks)",
        vsource.shape[0],
        len(soelems),
        len(sielems),
    )


# ---------------------------------------------------------------------------
# 3. Combine sink/source  (Fortran binary, stdin-driven)
# ---------------------------------------------------------------------------


def run_combine_sink_source(work_dir: Path) -> None:
    """Run ``combine_sink_source`` binary with required stdin."""
    result = subprocess.run(
        ["combine_sink_source"],
        input="1\n2\n",
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"combine_sink_source failed (exit {result.returncode}): {result.stderr[-2000:]}"
        )
    logger.info("    combine_sink_source completed")


# ---------------------------------------------------------------------------
# 4. Merge source/sink  (was merge_source_sink.py)
# ---------------------------------------------------------------------------


def merge_source_sink(  # noqa: PLR0915
    *,
    work_dir: Path,
    element_areas: NDArray[np.floating[Any]],
    prebuilt_dir: Path | None = None,
    t0_precip_source_file: Path | None = None,
) -> None:
    """Merge river discharge into precipitation source and write ``source.nc``.

    Parameters
    ----------
    work_dir : Path
        SCHISM working directory containing discharge output files.
    element_areas : ndarray
        Per-element geodesic areas (m²) used to compute minimum
        source thresholds.  Typically obtained from
        ``NWMSCHISMProject.element_areas``.
    prebuilt_dir : Path | None
        If given, the reference ``source.nc`` is symlinked from this
        directory into *work_dir* first.
    t0_precip_source_file : Path | None
        Optional path to a second ``precip_source.nc`` (typically that
        hour's own AnA ``schism_ana`` run) to pull a valid T0 sample
        from when this run's own ``precip_source.nc`` has none.
        Some callers' precip regridding doesn't reach back to T0 (e.g.
        an SR forecast, where the underlying gridded forcing's own T0
        sample doesn't exist by BMI design -- confirmed live: its
        ``vsource`` array is masked in full across the whole domain at
        row 0, `time_vsource[0]` too). ``np.ma`` addition propagates
        that mask through the source/sink merge below regardless of
        real discharge, and the masked cells are then serialized to
        the raw netCDF fill sentinel (~1e37) -- confirmed live as the
        cause of an SR SCHISM run destabilizing within its first
        timestep. When given and row 0 is masked, that row is replaced
        with the T0 sample (its own last row -- both files share the
        same mesh-element ordering/shape) from this second source
        instead of writing it out corrupted. Every other caller simply
        omits it and behavior is unchanged.
    """
    # Optionally stage source.nc from prebuilt model directory
    if prebuilt_dir is not None:
        src_nc = prebuilt_dir / "source.nc"
        dst_nc = work_dir / "source.nc"
        if src_nc.exists() and not dst_nc.exists():
            _symlink(src_nc, dst_nc)

    # --- read discharge files from combine_sink_source output ---
    # Newer versions of combine_sink_source produce "source_sink.in"
    # (without the ".1" suffix).  Fall back to the legacy name.
    soel1: list[int] = []
    siel: list[int] = []
    ss_path = work_dir / "source_sink.in"
    if not ss_path.exists():
        ss_path = work_dir / "source_sink.in.1"
    with ss_path.open() as f:
        nsoel1 = int(f.readline())
        soel1.extend(int(f.readline()) for _ in range(nsoel1))
        next(f)
        nsiel = int(f.readline())
        siel.extend(int(f.readline()) for _ in range(nsiel))

    # Read vsink.th.1
    vsink_lines = (work_dir / "vsink.th.1").read_text().splitlines()
    count = len(vsink_lines)
    si = np.zeros((count, nsiel + 1))
    for j, line in enumerate(vsink_lines):
        if line:
            si[j, :] = np.array(line.split(), dtype=float)
    time = si[:, 0]
    si = si[:, 1:]

    # Read vsource.th.1
    vsource_lines = (work_dir / "vsource.th.1").read_text().splitlines()
    so1 = np.zeros((count, nsoel1 + 1))
    for j, line in enumerate(vsource_lines):
        if line:
            so1[j, :] = np.array(line.split(), dtype=float)
    so1 = so1[:, 1:]

    # Read precipitation source
    with netCDF4.Dataset(str(work_dir / "precip_source.nc"), "r") as precip:
        so2 = precip.variables["vsource"][:]

    if (
        t0_precip_source_file is not None
        and np.ma.is_masked(so2)
        and np.ma.getmaskarray(so2)[0].any()
    ):
        with netCDF4.Dataset(str(t0_precip_source_file), "r") as t0precip:
            t0_row = np.ma.filled(t0precip.variables["vsource"][-1, :], 0.0)
        so2[0, :] = t0_row
        so2.mask[0, :] = False

    # Truncate discharge arrays to match precipitation time dimension
    ntime = so2.shape[0]
    if so1.shape[0] > ntime:
        so1 = so1[:ntime, :]
        si = si[:ntime, :]
        time = time[:ntime]

    # Merge river discharge into precipitation
    for i, elem in enumerate(soel1):
        so2[:, elem - 1] = so2[:, elem - 1] + so1[:, i]

    # Apply minimum value threshold based on element areas
    threshold = (0.01 * element_areas) / (3600.0 * (len(so2) - 1))

    md = np.max(so2, axis=0)
    keep = np.argwhere(md > threshold).ravel()
    so2 = so2[:, keep]
    keep += 1  # convert to 1-based element numbers

    # Write source.nc
    out_path = work_dir / "source.nc"
    with netCDF4.Dataset(str(out_path), "w", format="NETCDF4") as ncout:
        ncout.set_fill_off()

        ncout.createDimension("time_vsource", len(time))
        ncout.createDimension("time_vsink", len(time))
        ncout.createDimension("time_msource", len(time))
        ncout.createDimension("nsources", len(keep))
        ncout.createDimension("nsinks", nsiel)
        ncout.createDimension("ntracers", 2)
        ncout.createDimension("one", 1)

        ncso = ncout.createVariable("source_elem", "i4", ("nsources",))
        ncsi = ncout.createVariable("sink_elem", "i4", ("nsinks",))
        ncvso = ncout.createVariable(
            "vsource",
            "f8",
            ("time_vsource", "nsources"),
            zlib=True,
        )
        ncvsi = ncout.createVariable(
            "vsink",
            "f8",
            ("time_vsink", "nsinks"),
            zlib=True,
        )
        ncvmo = ncout.createVariable(
            "msource",
            "i4",
            ("time_msource", "ntracers", "nsources"),
            zlib=True,
        )
        nctso = ncout.createVariable("time_vsource", "f8", ("time_vsource",))
        nctsi = ncout.createVariable("time_vsink", "f8", ("time_vsink",))
        nctmo = ncout.createVariable("time_msource", "f8", ("time_msource",))
        ncvsos = ncout.createVariable("time_step_vsource", "f4", ("one",))
        ncvsis = ncout.createVariable("time_step_vsink", "f4", ("one",))
        ncvmos = ncout.createVariable("time_step_msource", "f4", ("one",))

        write_var(ncso, keep)
        write_var(ncsi, np.asarray(siel))
        write_var(ncvso, so2)
        write_var(ncvsi, si)
        write_var(nctso, time)
        write_var(nctsi, time)
        write_var(nctmo, time)
        ncvsos[:] = time[1] - time[0]
        ncvsis[:] = time[1] - time[0]
        ncvmos[:] = time[1] - time[0]

        fill_val = np.full((len(time), len(keep)), -9999.0)
        ncvmo[:, 0, :] = fill_val
        ncout.sync()

        fill_val.fill(0)
        ncvmo[:, 1, :] = fill_val
        ncout.sync()
    logger.info(
        "    Wrote source.nc: %d sources (from %d), %d sinks, %d timesteps",
        len(keep),
        so2.shape[1],
        nsiel,
        len(time),
    )


# ---------------------------------------------------------------------------
# 5. Mesh partitioning  (was create_offline_partition in pre_schism.bash)
# ---------------------------------------------------------------------------


def partition_mesh(
    *,
    work_dir: Path,
    total_tasks: int,
    nscribes: int,
) -> Path:
    """Run ``metis_prep`` + ``gpmetis`` and write ``partition.prop``.

    Returns the path to the generated ``partition.prop``.
    """
    n_compute = total_tasks - nscribes

    # metis_prep: converts hgrid.gr3 + vgrid.in → graphinfo
    result = subprocess.run(
        ["metis_prep", "./hgrid.gr3", "./vgrid.in"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"metis_prep failed (exit {result.returncode}): {result.stderr[-2000:]}")

    # gpmetis: partition graphinfo into n_compute parts
    result = subprocess.run(
        [
            "gpmetis",
            "./graphinfo",
            str(n_compute),
            "-ufactor=1.01",
            "-seed=15",
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gpmetis failed (exit {result.returncode}): {result.stderr[-2000:]}")

    # Convert graphinfo.part.N → partition.prop  (awk '{print NR,$0}')
    part_file = work_dir / f"graphinfo.part.{n_compute}"
    prop_file = work_dir / "partition.prop"
    lines = part_file.read_text().splitlines()
    with prop_file.open("w") as f:
        for i, line in enumerate(lines, start=1):
            f.write(f"{i} {line}\n")

    logger.info(
        "    Partitioned mesh into %d compute ranks → %s",
        n_compute,
        prop_file,
    )
    return prop_file


# ---------------------------------------------------------------------------
# 6. Combine hotstart  (was the conditional in post_schism.bash)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. Stage LDASIN forcing files  (was pre_nwm_forcing_coastal.bash)
# ---------------------------------------------------------------------------


def stage_ldasin_files(
    *,
    work_dir: Path,
    start_date: datetime,
    duration_hours: int,
    nwm_forcing_dir: Path,
) -> tuple[Path, Path]:
    """Stage LDASIN forcing files into the expected directory structure.

    Creates ``forcing_input/<forcing_begin_date>/`` with symlinks to
    the downloaded LDASIN files, and ``coastal_forcing_output/``.

    Returns ``(forcing_input_dir, coastal_forcing_output_dir)``.
    """
    pdy = start_date.strftime("%Y%m%d")
    cyc = start_date.strftime("%H")
    forcing_begin = f"{pdy}{cyc}"

    nwm_forcing_output = work_dir / "forcing_input"
    forcing_subdir = nwm_forcing_output / forcing_begin[:10]
    forcing_subdir.mkdir(parents=True, exist_ok=True)

    coastal_forcing_output = work_dir / "coastal_forcing_output"
    coastal_forcing_output.mkdir(parents=True, exist_ok=True)

    length_hrs = int(duration_hours)
    base_dt = start_date.replace(tzinfo=UTC) if start_date.tzinfo is None else start_date

    for i in range(abs(length_hrs) + 1):
        dt = base_dt + timedelta(hours=i)
        pdycyc = dt.strftime("%Y%m%d%H")
        fname = f"{pdycyc}.LDASIN_DOMAIN1"
        src = nwm_forcing_dir / fname
        dst = forcing_subdir / fname
        if src.exists():
            _symlink(src, dst)
        else:
            logger.warning("    Missing LDASIN file: %s", src)

    logger.info(
        "    Staged LDASIN files in %s (%d hours from %s)",
        forcing_subdir,
        abs(length_hrs),
        forcing_begin,
    )
    return nwm_forcing_output, coastal_forcing_output


def stage_forecast_forcing(
    *,
    work_dir: Path,
    start_date: datetime,
    forecast_file: Path,
) -> tuple[Path, Path]:
    """Stage a pre-generated ngen forecast forcing file for regridding.

    Unlike :func:`stage_ldasin_files`, which symlinks one file per hour,
    the ngen forecast forcing engine emits a single multi-timestep file on
    the WRF-Hydro geogrid.  A single symlink is placed in
    ``forcing_input/<YYYYMMDDHH>/`` with a ``*.LDASIN_DOMAIN1`` name so the
    (slab-aware) regridder and sflux generator discover it via their
    existing globs and iterate its timesteps in place — no copy needed.

    Returns ``(forcing_input_dir, coastal_forcing_output_dir)``.
    """
    pdy = start_date.strftime("%Y%m%d")
    cyc = start_date.strftime("%H")
    forcing_begin = f"{pdy}{cyc}"

    nwm_forcing_output = work_dir / "forcing_input"
    forcing_subdir = nwm_forcing_output / forcing_begin[:10]
    forcing_subdir.mkdir(parents=True, exist_ok=True)

    coastal_forcing_output = work_dir / "coastal_forcing_output"
    coastal_forcing_output.mkdir(parents=True, exist_ok=True)

    if not forecast_file.exists():
        raise FileNotFoundError(f"Forecast meteo file not found: {forecast_file}")

    dst = forcing_subdir / f"{forecast_file.stem}.LDASIN_DOMAIN1"
    _symlink(forecast_file, dst)

    logger.info("    Staged forecast forcing %s -> %s", forecast_file, dst)
    return nwm_forcing_output, coastal_forcing_output


# ---------------------------------------------------------------------------
# 8. Generate sflux from LDASIN  (was post_nwm_forcing_coastal → makeAtmo.py)
# ---------------------------------------------------------------------------


def _read_hgrid_bbox(hgrid_path: Path) -> tuple[float, float, float, float]:
    """Return ``(lon_min, lat_min, lon_max, lat_max)`` of nodes in a SCHISM hgrid file.

    Reads the node block of a ``.gr3``/``.ll`` mesh file and returns the
    coordinate extent.  Used by :func:`make_sflux` to subset NWM
    atmospheric forcing to the SCHISM mesh footprint.  Coordinates are
    interpreted as ``(lon, lat)`` — the caller is expected to point at a
    geographic mesh (``hgrid.ll`` for projected setups).
    """
    with hgrid_path.open("r") as f:
        f.readline()  # description line
        header = f.readline().split()
        if len(header) < 2:
            raise ValueError(f"Malformed hgrid header in {hgrid_path}: {header!r}")
        n_nodes = int(header[1])
        coords = np.zeros((n_nodes, 2), dtype=np.float64)
        for i in range(n_nodes):
            parts = f.readline().split()
            coords[i, 0] = float(parts[1])
            coords[i, 1] = float(parts[2])
    return (
        float(coords[:, 0].min()),
        float(coords[:, 1].min()),
        float(coords[:, 0].max()),
        float(coords[:, 1].max()),
    )


def make_sflux(
    *,
    work_dir: Path,
    forcing_input_dir: Path,
    start_date: datetime,
    geogrid_file: Path,
    bbox_buffer_deg: float = 0.5,
) -> Path:
    """Generate sflux atmospheric forcing from LDASIN files.

    This is the pure-Python equivalent of ``makeAtmo.py`` — reads LDASIN
    files and writes ``sflux/sflux_air_1.0001.nc``.

    The output is subset to the SCHISM mesh footprint when an hgrid file
    (``hgrid.ll`` preferred, then ``hgrid.gr3``) is present in
    *work_dir*.  This avoids writing the full CONUS forcing grid into
    sflux for mesh subdomains, which can be a 100x I/O reduction on
    multi-node MPI runs that read the file concurrently from NFS.  Set
    *bbox_buffer_deg* to control the padding around the mesh extent.

    When no hgrid file is found a warning is logged and the full
    geogrid is written (the historical behavior).

    Returns the path to the sflux output directory.
    """
    pdy = start_date.strftime("%Y%m%d")
    cyc = start_date.strftime("%H")
    forcing_begin = f"{pdy}{cyc}"

    forcing_subdir = forcing_input_dir / forcing_begin[:10]
    sflux_dir = work_dir / "sflux"
    sflux_dir.mkdir(parents=True, exist_ok=True)

    sflux_out = sflux_dir / "sflux_air_1.0001.nc"

    # Symlink precip_source.nc if it was generated
    precip_nc = work_dir / "coastal_forcing_output" / "precip_source.nc"
    if precip_nc.exists():
        dst = work_dir / "precip_source.nc"
        if not dst.exists():
            _symlink(precip_nc, dst)

    mesh_bbox: tuple[float, float, float, float] | None = None
    hgrid_ll = work_dir / "hgrid.ll"
    hgrid_gr3 = work_dir / "hgrid.gr3"
    if hgrid_ll.exists():
        mesh_bbox = _read_hgrid_bbox(hgrid_ll)
        logger.info("    Subsetting sflux to hgrid.ll extent: %s", mesh_bbox)
    elif hgrid_gr3.exists():
        mesh_bbox = _read_hgrid_bbox(hgrid_gr3)
        logger.info("    Subsetting sflux to hgrid.gr3 extent: %s", mesh_bbox)
    else:
        logger.warning(
            "    No hgrid.ll or hgrid.gr3 in %s; writing full geogrid (no subsetting).",
            work_dir,
        )

    from coastal_calibration.schism.sflux import make_atmo_sflux

    make_atmo_sflux(
        forcing_input_dir=forcing_subdir,
        work_dir=work_dir,
        start_dt=start_date,
        geogrid_file=geogrid_file,
        mesh_bbox=mesh_bbox,
        bbox_buffer_deg=bbox_buffer_deg,
    )

    if not sflux_out.exists():
        raise RuntimeError(f"make_atmo_sflux did not produce {sflux_out}.")

    # SCHISM expects sflux_air_1.{n}.nc (no leading zeros) but makeAtmo
    # produces sflux_air_1.0001.nc (4-digit zero-padded).  Rename files
    # to match the expected naming convention.
    for f in sflux_dir.glob("sflux_air_*.nc"):
        m = re.match(r"(sflux_air_\d+)\.(\d+)\.nc", f.name)
        if m and len(m.group(2)) > 1 and m.group(2).startswith("0"):
            new_name = f"{m.group(1)}.{int(m.group(2))}.nc"
            new_path = f.parent / new_name
            new_path.unlink(missing_ok=True)
            f.rename(new_path)
            logger.info("    Renamed %s → %s", f.name, new_name)

    logger.info("    Generated sflux in %s", sflux_dir)
    return sflux_dir


# ---------------------------------------------------------------------------
# 9. Update param.nml  (was update_param.bash)
# ---------------------------------------------------------------------------


def count_required_scribes(param_nml: Path, include_noaa_gages: bool) -> int | None:
    """Count the SCHISM scribes implied by an active ``param.nml``.

    Sums uncommented ``iof_*(N) = 1`` flags plus the effective
    ``iout_sta`` value: when ``include_noaa_gages`` is True the
    ``schism_obs`` stage will flip ``iout_sta`` to 1, so it counts as 1
    regardless of the template; otherwise we read the template's own
    ``iout_sta`` value. SCHISM aborts at init when its CLI ``nscribes``
    argument is below this number.

    Returns
    -------
    int or None
        Scribes needed, or ``None`` when *param_nml* can't be read
        (caller should fall back to a safe default).
    """
    try:
        text = param_nml.read_text()
    except OSError:
        return None
    iof_count = len(re.findall(r"(?m)^\s*iof_\w+\(\d+\)\s*=\s*1\b", text))
    if include_noaa_gages:
        iout_sta = 1
    else:
        iout_sta = 1 if re.search(r"(?m)^\s*iout_sta\s*=\s*1\b", text) else 0
    return iof_count + iout_sta


def update_params(  # noqa: PLR0912, PLR0915
    *,
    work_dir: Path,
    prebuilt_dir: Path,
    start_date: datetime,
    duration_hours: int,
    timestep_seconds: int = 200,
    hot_start_file: Path | None = None,
    output_freq_hours: float = 1.0,
    single_output_file: bool = False,
    run_param_overrides: dict[str, Any] | None = None,
    discharge_enabled: bool = True,
    wind_enabled: bool = True,
) -> Path:
    """Create ``param.nml`` and symlink static mesh files.

    This is the pure-Python equivalent of ``update_param.bash``.
    Copies the template ``param.nml`` from the prebuilt model
    directory, updates date/time/duration parameters, and symlinks
    mesh files.

    ``timestep_seconds`` is SCHISM's integration timestep (``dt`` in
    ``param.nml``) and the time unit that ``nspool``/``ihfskip``/
    ``nhot_write`` are counted in. Defaults to 200, the value used by
    the Pacific and Hawaii forecast templates.

    ``output_freq_hours`` sets how often SCHISM writes field outputs
    (translated into ``nspool``). ``single_output_file`` controls
    whether SCHISM rotates to a new output file after each write
    (``ihfskip = nspool``, the historical behavior) or keeps appending
    to one file across the whole run (``ihfskip = total_timesteps``).
    The latter matters on shared filesystems where every rotation costs
    an MPI barrier and metadata round-trips.

    ``run_param_overrides`` is applied last and overrides any namelist
    key set above. Values are written verbatim (no quoting), so callers
    are responsible for matching the namelist syntax (numbers as
    numbers, strings without spaces).

    Returns the path to the generated ``param.nml``.
    """
    coastal_parm = prebuilt_dir

    # Copy template param.nml
    param_path = work_dir / "param.nml"
    shutil.copy2(coastal_parm / "param.nml", param_path)
    text = param_path.read_text()

    # Compute date parameters
    pdy = start_date.strftime("%Y%m%d")
    cyc = start_date.strftime("%H")

    length_hrs = int(duration_hours)
    rnhours = -length_hrs if length_hrs <= 0 else length_hrs

    start_year = pdy[:4]
    start_month = pdy[4:6]
    start_day = pdy[6:8]
    start_hour_val = int(cyc)
    start_minute = start_date.minute
    # SCHISM uses fractional hour
    start_hour_frac = start_hour_val + start_minute / 60.0

    # Update date parameters
    text = re.sub(r"(?m)^(\s*)start_year\s*=.*$", rf"\g<1>start_year = {start_year}", text)
    text = re.sub(r"(?m)^(\s*)start_month\s*=.*$", rf"\g<1>start_month = {start_month}", text)
    text = re.sub(r"(?m)^(\s*)start_day\s*=.*$", rf"\g<1>start_day = {start_day}", text)
    text = re.sub(r"(?m)^(\s*)start_hour\s*=.*$", rf"\g<1>start_hour = {start_hour_frac:.2f}", text)

    # nspool, ihfskip, and nhot_write are counted in timesteps and are
    # coupled via SCHISM's divisibility constraints, so we resolve them
    # together: the user's ``run_param_overrides`` (if any) take
    # precedence for each key, and the auto-derived defaults for the
    # keys they did *not* override adapt to whatever they did override.
    # This means ``run_param_overrides={"ihfskip": 324}`` produces a
    # sensible ``nhot_write`` automatically.
    overrides_remaining = dict(run_param_overrides) if run_param_overrides else {}

    nspool_default = max(1, round(output_freq_hours * 3600 / timestep_seconds))
    nspool = int(overrides_remaining.pop("nspool", nspool_default))

    if single_output_file:
        # Round up so the last write fits inside one file.
        total_timesteps = -(-int(rnhours) * 3600 // timestep_seconds)
        ihfskip_default = max(nspool, total_timesteps)
    else:
        ihfskip_default = nspool
    ihfskip = int(overrides_remaining.pop("ihfskip", ihfskip_default))

    # SCHISM requires ``nhot_write`` to be a multiple of ihfskip when
    # nhot=1. We aim for "hotstart every simulated hour" (nhot_target =
    # 18 timesteps at the canonical dt=200) so a hotstart lands at every
    # hour boundary regardless of the run's total length -- this is what
    # AnA cycling needs (a checkpoint at the 1h mark and another at the
    # run's end), and it degrades gracefully for longer forecast runs
    # (just more, unused, intermediate hotstarts). Round up to the next
    # multiple of ihfskip so the divisibility constraint holds. The user
    # can still override ``nhot_write`` directly.
    nhot_target = max(1, round(1 * 3600 / timestep_seconds))
    nhot_write_default = max(1, -(-nhot_target // ihfskip)) * ihfskip
    nhot_write = int(overrides_remaining.pop("nhot_write", nhot_write_default))

    text = re.sub(r"(?m)^(\s*)nspool\s*=.*$", rf"\g<1>nspool = {nspool}", text)
    text = re.sub(r"(?m)^(\s*)ihfskip\s*=.*$", rf"\g<1>ihfskip = {ihfskip}", text)
    text = re.sub(
        r"(?m)^(\s*)nhot_write\s*=.*$",
        rf"\g<1>nhot_write = {nhot_write} !must be a multiple of ihfskip if nhot=1",
        text,
    )

    # if_source = -1 reads netCDF source/sink forcing (source.nc), produced
    # by the discharge stage. When discharge is disabled (no discharge_file
    # configured) the stage skips and source.nc is never written — leaving
    # if_source = -1 would make SCHISM abort at init reading the missing
    # file. Set to 0 in that case so SCHISM runs without river forcing.
    if_source_val = -1 if discharge_enabled else 0
    text = re.sub(r"(?m)^(\s*)if_source\s*=.*$", rf"\g<1>if_source = {if_source_val}", text)

    # nws = 2 reads sflux atmospheric files, produced by the schism_sflux
    # stage. When wind is disabled (include_wind=False) that stage skips
    # and no sflux files are ever written -- leaving nws = 2 would make
    # SCHISM abort at init looking for them. nws = 0 means no atmospheric
    # forcing is applied at all (confirmed in the template's own comment).
    if not wind_enabled:
        text = re.sub(r"(?m)^(\s*)nws\s*=.*$", r"\g<1>nws = 0", text)

    # rnday is fractional day
    rnday = rnhours / 24.0
    text = re.sub(r"(?m)^(\s*)rnday\s*=.*$", rf"\g<1>rnday = {rnday:.8f}", text)

    # Timestep and atmospheric timestep
    text = re.sub(r"(?m)^(\s*)dt\s*=.*$", rf"\g<1>dt = {timestep_seconds}", text)
    text = re.sub(r"(?m)^(\s*)wtiminc\s*=.*$", r"\g<1>wtiminc = 600", text)

    # Hot start handling
    if hot_start_file and hot_start_file.exists():
        text = re.sub(r"(?m)^(\s*)ihot\s*=.*$", r"\g<1>ihot = 1", text)
        shutil.copy2(hot_start_file, work_dir / "hotstart.nc")
    else:
        text = re.sub(r"(?m)^(\s*)ihot\s*=.*$", r"\g<1>ihot = 0", text)

    # Remove deprecated parameters that are incompatible with newer SCHISM
    for deprecated in ("impose_net_flux", "isconsv", "isav", "vclose_surf_frac"):
        text = re.sub(rf"(?m)^\s*{deprecated}\s*=.*\n", "", text)

    # Add mandatory parameters (SCHISM >= May 2024, commit 0fec598)
    if "nbins_veg_vert" not in text:
        text = re.sub(
            r"(?m)(^\s*ihfskip\s*=.*$)",
            r"\1\n  nbins_veg_vert = 1\n  nmarsh_types = 1",
            text,
        )

    # Apply any remaining user-supplied namelist overrides (those not
    # consumed earlier by the nspool/ihfskip/nhot_write resolver).
    if overrides_remaining:
        text = _apply_namelist_overrides(text, overrides_remaining)

    param_path.write_text(text)

    # Symlink static mesh files
    static_files = [
        "hgrid.gr3",
        "hgrid.ll",
        "manning.gr3",
        "vgrid.in",
        "bctides.in",
        "windrot_geo2proj.gr3",
        "hgrid.utm",
        "hgrid.cpp",
        "elev.ic",
    ]
    for fname in static_files:
        src = coastal_parm / fname
        dst = work_dir / fname
        if src.exists():
            dst.unlink(missing_ok=True)
            dst.symlink_to(src)

    # Optional files
    for fname in ("station.in", "open_bnds_hgrid.nc", "hgrid.nc"):
        src = coastal_parm / fname
        dst = work_dir / fname
        if src.exists():
            dst.unlink(missing_ok=True)
            dst.symlink_to(src)

    # Copy sflux directory (only the small template files — never the
    # ``sflux_air_*.nc`` forcing arrays). Those are either stale leftovers
    # from a previous run or about to be regenerated by ``schism_sflux``;
    # copying a multi-GB stale .nc here can stall this stage for minutes.
    sflux_src = coastal_parm / "sflux"
    sflux_dst = work_dir / "sflux"
    sflux_dst.mkdir(exist_ok=True)
    if sflux_src.exists():
        for f in sflux_src.iterdir():
            if f.suffix == ".nc":
                continue
            shutil.copy2(f, sflux_dst / f.name)

    logger.info("    Created param.nml and symlinked mesh files in %s", work_dir)
    return param_path


# ---------------------------------------------------------------------------
# 10. Elevation datum correction  (was correct_elevation.py)
# ---------------------------------------------------------------------------


def correct_elevation(
    elev_file: Path,
    correction_file: Path,
    n_open_boundary_nodes: int | None = None,
) -> None:
    """Subtract datum corrections from ``elev2D.th.nc`` in-place.

    Parameters
    ----------
    elev_file : Path
        SCHISM boundary forcing netCDF4 file with a ``time_series``
        variable (modified in-place).
    correction_file : Path
        CSV file with correction values in the 6th column (0-indexed: 5),
        one value per open-boundary node, with one header row to skip.
    n_open_boundary_nodes : int, optional
        Expected number of open boundary nodes.  When provided the CSV
        row count is validated against this value before applying the
        correction.
    """
    import netCDF4
    import numpy as np

    elev_correct = np.loadtxt(str(correction_file), delimiter=",", skiprows=1, usecols=5)

    if n_open_boundary_nodes is not None and len(elev_correct) != n_open_boundary_nodes:
        raise ValueError(
            f"elevation_correction.csv has {len(elev_correct)} rows but the mesh "
            f"has {n_open_boundary_nodes} open boundary nodes"
        )

    with netCDF4.Dataset(elev_file, "r+") as ds:
        elev_var = ds["time_series"]
        for t in range(elev_var.shape[0]):
            elev_var[t] = elev_var[t].ravel() - elev_correct


# ---------------------------------------------------------------------------
# 11. Harmonic-tide boundary conditions
# ---------------------------------------------------------------------------


def make_tidal_boundary(
    *,
    work_dir: Path,
    start_date: datetime,
    duration_hours: int,
    prebuilt_dir: Path,
    atlas_dir: Path,
    tidal_model: str = "TPXO10-atlas-v2-nc",
    time_step_seconds: int = 3600,
    correction_file: Path | None = None,
    n_open_boundary_nodes: int | None = None,
) -> Path:
    """Generate tidal boundary forcing via pyTMD harmonic prediction.

    Predicts elevations at the SCHISM open-boundary nodes against
    *tidal_model* (TPXO/FES/GOT/EOT — any model in pyTMD's database) at
    *time_step_seconds* cadence and writes ``elev2D.th.nc``. SCHISM
    interpolates between rows at its own integration dt, so any
    positive cadence is valid.

    Returns the path to ``elev2D.th.nc``.
    """
    from coastal_calibration.data.tides import write_schism_boundary

    elev_file = work_dir / "elev2D.th.nc"
    write_schism_boundary(
        grid_file=prebuilt_dir / "open_bnds_hgrid.nc",
        output_file=elev_file,
        start_dt=start_date,
        duration_hours=duration_hours,
        atlas_dir=atlas_dir,
        tidal_model=tidal_model,
        time_step_seconds=time_step_seconds,
    )

    if correction_file is not None and correction_file.exists():
        logger.info("    Applying elevation datum correction")
        correct_elevation(
            elev_file,
            correction_file,
            n_open_boundary_nodes=n_open_boundary_nodes,
        )

    logger.info("    Tidal boundary created: %s", elev_file)
    return elev_file


# ---------------------------------------------------------------------------
# 11. STOFS boundary conditions  (was pre/regrid/post_regrid_stofs.bash)
# ---------------------------------------------------------------------------


def make_stofs_boundary(
    *,
    work_dir: Path,
    start_date: datetime,
    duration_hours: int,
    stofs_file: Path,
    prebuilt_dir: Path,
    mpi_tasks: int,
    correction_file: Path | None = None,
    n_open_boundary_nodes: int | None = None,
    runtime_env: dict[str, str] | None = None,
    atlas_dir: Path | None = None,
    tidal_model: str = "TPXO10-atlas-v2-nc",
) -> Path:
    """Generate boundary forcing from STOFS data via ESMF regridding.

    Runs ``regrid_estofs.py`` via MPI and optionally ``makeOceanTide.py``
    for medium-range runs.

    Parameters
    ----------
    runtime_env : dict[str, str], optional
        Cluster-specific env-var overrides (typically MPI / fabric
        tuning supplied via ``SchismModelConfig.runtime_env``). Applied
        on top of the default ``os.environ`` + ``HDF5_USE_FILE_LOCKING``,
        so user values always win.

    Returns the path to ``elev2D.th.nc``.
    """
    import os
    import sys

    pdy = start_date.strftime("%Y%m%d")
    cyc = start_date.strftime("%H")
    coastal_parm = prebuilt_dir

    # Pre-process: symlink STOFS and hgrid files
    estofs_data = work_dir / f"stofs_2d_glo.t{cyc}z.fields.cwl.nc"
    _symlink(stofs_file, estofs_data)

    hgrid_file = work_dir / "open_bnds_hgrid.nc"
    open_bnds_src = coastal_parm / "open_bnds_hgrid.nc"
    if not hgrid_file.exists() and open_bnds_src.exists():
        _symlink(open_bnds_src, hgrid_file)

    output_file = work_dir / "elev2D.th.nc"
    length_hrs = abs(int(duration_hours)) + 1

    # Run regrid_estofs via MPI using the regridding module

    env = os.environ.copy()
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    if runtime_env:
        env.update(runtime_env)

    from coastal_calibration.utils import build_mpi_cmd

    cmd = [
        *build_mpi_cmd(mpi_tasks),
        sys.executable,
        "-m",
        "coastal_calibration.regridding.regrid_estofs",
        str(estofs_data),
        str(hgrid_file),
        str(output_file),
        "--cycle-date",
        pdy,
        "--cycle-time",
        f"{cyc}00",
        "--length-hrs",
        str(length_hrs),
    ]

    result = subprocess.run(
        cmd,
        env=env,
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"regrid_estofs failed (exit {result.returncode}): {result.stderr[-2000:]}"
        )

    # Post-process: tidal fill for medium-range runs (>180h)
    raw_length = abs(int(duration_hours))
    if raw_length > 180:
        if atlas_dir is None:
            logger.warning(
                "    Duration %dh > 180h but no tidal atlas configured; "
                "STOFS boundary extends only 180 h. Set paths.tidal_atlas_dir "
                "to enable pyTMD tidal fill.",
                raw_length,
            )
        else:
            from coastal_calibration.data.tides import extend_schism_boundary

            try:
                extend_schism_boundary(
                    grid_file=coastal_parm / "open_bnds_hgrid.nc",
                    output_file=output_file,
                    start_dt=start_date,
                    duration_hours=raw_length,
                    atlas_dir=atlas_dir,
                    tidal_model=tidal_model,
                )
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                # Non-fatal: tidal fill is a best-effort augmentation past
                # the 180 h STOFS forecast window. Log loudly so the user
                # knows their boundary is truncated, but continue — the
                # first 180 h of forcing is still valid.
                logger.error(
                    "    pyTMD tidal fill failed (%s); STOFS boundary "
                    "extends only %d h instead of %d h",
                    exc,
                    180,
                    raw_length,
                )

    # Apply elevation correction if available
    if correction_file is not None and correction_file.exists():
        try:
            correct_elevation(
                output_file,
                correction_file,
                n_open_boundary_nodes=n_open_boundary_nodes,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            # Datum correction is required for accuracy when supplied.
            # Re-raise so the user sees a clear failure rather than a
            # silently uncorrected boundary that produces a quiet datum
            # offset in all downstream comparisons.
            msg = (
                f"correct_elevation failed for {output_file} using "
                f"{correction_file}: {exc}. The STOFS boundary is in "
                "the wrong vertical datum; remove the correction file "
                "to skip datum correction or fix the input."
            )
            raise RuntimeError(msg) from exc

    if not output_file.exists():
        raise RuntimeError("STOFS boundary: elev2D.th.nc was not produced")

    logger.info("    STOFS boundary created: %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# 12. Combine hotstart  (was the conditional in post_schism.bash)
# ---------------------------------------------------------------------------


def combine_hotstart(outputs_dir: Path) -> list[Path]:
    """Run ``combine_hotstart7`` for every hotstart iteration in *outputs_dir*.

    SCHISM writes one set of per-rank ``hotstart_<rank>_<iteration>.nc``
    files at each ``nhot_write`` interval (see ``make_param_nml``).
    ``combine_hotstart7`` requires an explicit ``-i <iteration>`` to know
    which set to merge -- it does not auto-detect it -- and produces
    ``hotstart_it=<iteration>.nc``. This finds every iteration actually
    present (via rank 0's files) and combines each one in turn.

    Returns the combined file paths, in iteration order.
    """
    iterations = sorted(
        int(m.group(1))
        for f in outputs_dir.glob("hotstart_000000_*.nc")
        if (m := re.match(r"hotstart_000000_(\d+)\.nc$", f.name))
    )
    if not iterations:
        logger.info("    No hotstart_000000_*.nc files found in %s; nothing to combine", outputs_dir)
        return []

    combined_paths = []
    for iteration in iterations:
        result = subprocess.run(
            ["combine_hotstart7", "-i", str(iteration)],
            cwd=outputs_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"combine_hotstart7 -i {iteration} failed (exit {result.returncode}): "
                f"{result.stderr[-2000:]}"
            )
        combined_path = outputs_dir / f"hotstart_it={iteration}.nc"
        if not combined_path.exists():
            raise RuntimeError(f"combine_hotstart7 -i {iteration} did not produce {combined_path}")
        combined_paths.append(combined_path)
        logger.info("    combine_hotstart7 -i %d completed -> %s", iteration, combined_path.name)

    return combined_paths
