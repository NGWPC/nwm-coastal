"""SCHISM pre- and post-processing functions.

Pure-Python replacements for the bash scripts that previously
orchestrated SCHISM pre/post processing (``initial_discharge.bash``,
``combine_sink_source.bash``, ``merge_source_sink.bash``,
``pre_schism.bash``, ``post_schism.bash``).

All functions accept explicit paths/values rather than reading
``os.environ``, making them testable in isolation.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np

from coastal_calibration.utils.logging import logger


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
    prebuilt_dir: Path,
) -> tuple[Path, Path | None]:
    """Symlink NWM CHRTOUT files into staging directories.

    Parameters
    ----------
    prebuilt_dir : Path
        Prebuilt SCHISM model directory containing ``nwmReaches.csv``.

    Returns ``(nwm_output_dir, nwm_ana_dir)`` so that
    :func:`make_discharge` can find them.
    """
    nwm_output_dir = work_dir / "nwm_output"
    nwm_ana_dir = work_dir / "nwm_output_ana"
    nwm_output_dir.mkdir(parents=True, exist_ok=True)
    nwm_ana_dir.mkdir(parents=True, exist_ok=True)

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

    # Copy nwmReaches.csv from prebuilt model directory
    reaches_src = prebuilt_dir / "nwmReaches.csv"
    shutil.copy2(reaches_src, work_dir / "nwmReaches.csv")

    return nwm_output_dir, nwm_ana_dir


def _symlink(src: Path, dst: Path) -> None:
    """Create a symlink, replacing an existing one."""
    dst.unlink(missing_ok=True)
    dst.symlink_to(src)


# ---------------------------------------------------------------------------
# 2. Make discharge  (was makeDischarge.py)
# ---------------------------------------------------------------------------


def make_discharge(
    *,
    work_dir: Path,
    nwm_output_dir: Path,
    nwm_ana_dir: Path | None = None,
    is_analysis: bool = False,
    meteo_source: str = "nwm_ana",
    domain: str = "conus",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> None:
    """Create discharge files from NWM CHRT output.

    Writes ``vsource.th``, ``vsink.th``, and ``source_sink.in`` into
    *work_dir*.

    For ``nwm_retro`` the streamflow is read directly from the S3 Zarr
    store (requires *start_date* and *end_date*).  For ``nwm_ana`` the
    streamflow is read from local CHRTOUT netCDF files.
    """
    from coastal_calibration.utils.streamflow import read_streamflow

    reaches_path = work_dir / "nwmReaches.csv"
    soelems: list[int] = []
    soids: list[int] = []
    sielems: list[int] = []
    siids: list[int] = []
    with reaches_path.open() as f:
        nso = int(f.readline())
        for _ in range(nso):
            parts = f.readline().split()
            soelems.append(int(parts[0]))
            soids.append(int(parts[1]))
        next(f)
        nsi = int(f.readline())
        for _ in range(nsi):
            parts = f.readline().split()
            sielems.append(int(parts[0]))
            siids.append(int(parts[1]))

    all_fids = soids + siids

    if meteo_source == "nwm_retro":
        if start_date is None or end_date is None:
            raise ValueError(
                "start_date and end_date are required for nwm_retro"
            )
        df = read_streamflow(
            all_fids,
            start_date,
            end_date,
            meteo_source="nwm_retro",
            domain=domain,
        )
    else:
        # Gather local CHRTOUT files for nwm_ana
        chrtout_files: list[Path] = []
        if not is_analysis and nwm_ana_dir is not None:
            ana_files = sorted(nwm_ana_dir.glob("*CHRTOUT*"))
            if ana_files:
                chrtout_files.append(ana_files[-1])
        chrtout_files.extend(sorted(nwm_output_dir.glob("*CHRTOUT*")))

        if not chrtout_files:
            raise FileNotFoundError(
                f"No CHRTOUT files found in {nwm_output_dir}"
            )

        logger.info("    Processing %d CHRTOUT files", len(chrtout_files))

        if start_date is None or end_date is None:
            raise ValueError(
                "start_date and end_date are required for nwm_ana"
            )
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
        if freq < timedelta(hours=1):
            df = df.resample("h").mean()

    # Build vsource / vsink arrays from the DataFrame
    n_rows = len(df)
    vsource = np.zeros((n_rows, len(soids)))
    vsink = np.zeros((n_rows, len(siids)))

    for i, sid in enumerate(soids):
        if sid in df.columns:
            vsource[:, i] = df[sid].to_numpy()

    for i, sid in enumerate(siids):
        if sid in df.columns:
            vsink[:, i] = -1.0 * df[sid].to_numpy()

    tstep = 3600.0

    _write_th_file(work_dir / "vsource.th", vsource, tstep)
    _write_th_file(work_dir / "vsink.th", vsink, tstep)

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
        "    Wrote vsource.th (%d rows), vsink.th, source_sink.in "
        "(%d sources, %d sinks)",
        vsource.shape[0],
        len(soelems),
        len(sielems),
    )


def _write_th_file(path: Path, data: np.ndarray, tstep: float) -> None:
    """Write a SCHISM time-history (.th) file."""
    t = 0.0
    with path.open("w") as f:
        for i in range(data.shape[0]):
            parts = [str(t)]
            parts.extend(str(data[i, j]) for j in range(data.shape[1]))
            f.write("\t".join(parts) + "\n")
            t += tstep


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
            f"combine_sink_source failed (exit {result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    logger.info("    combine_sink_source completed")


# ---------------------------------------------------------------------------
# 4. Merge source/sink  (was merge_source_sink.py)
# ---------------------------------------------------------------------------


def merge_source_sink(
    *,
    work_dir: Path,
    root_dir: Path,
    prebuilt_dir: Path | None = None,
) -> None:
    """Merge river discharge into precipitation source and write ``source.nc``.

    If *prebuilt_dir* is given, the reference ``source.nc`` is
    symlinked from that directory into *work_dir* first.
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
        for _ in range(nsoel1):
            soel1.append(int(f.readline()))
        next(f)
        nsiel = int(f.readline())
        for _ in range(nsiel):
            siel.append(int(f.readline()))

    # Read vsink.th.1
    vsink_lines = (work_dir / "vsink.th.1").read_text().splitlines()
    count = len(vsink_lines)
    si = np.zeros((count, nsiel + 1))
    for j, line in enumerate(vsink_lines):
        if line:
            si[j, :] = np.fromstring(line, dtype=float, sep=" ")
    time = si[:, 0]
    si = si[:, 1:]

    # Read vsource.th.1
    vsource_lines = (work_dir / "vsource.th.1").read_text().splitlines()
    so1 = np.zeros((count, nsoel1 + 1))
    for j, line in enumerate(vsource_lines):
        if line:
            so1[j, :] = np.fromstring(line, dtype=float, sep=" ")
    so1 = so1[:, 1:]

    # Read precipitation source
    precip = nc.Dataset(str(work_dir / "precip_source.nc"), "r")
    so2 = precip.variables["vsource"][:]

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
    threshold = np.genfromtxt(str(root_dir / "element_areas.txt"))
    threshold = (0.01 * threshold) / (3600.0 * (len(so2) - 1))

    md = np.max(so2, axis=0)
    keep = np.argwhere(md > threshold).ravel()
    so2 = so2[:, keep]
    keep += 1  # convert to 1-based element numbers

    # Write source.nc
    out_path = work_dir / "source.nc"
    ncout = nc.Dataset(str(out_path), "w", format="NETCDF4")
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
        "vsource", "f8", ("time_vsource", "nsources"), zlib=True,
    )
    ncvsi = ncout.createVariable(
        "vsink", "f8", ("time_vsink", "nsinks"), zlib=True,
    )
    ncvmo = ncout.createVariable(
        "msource", "i4", ("time_msource", "ntracers", "nsources"), zlib=True,
    )
    nctso = ncout.createVariable("time_vsource", "f8", ("time_vsource",))
    nctsi = ncout.createVariable("time_vsink", "f8", ("time_vsink",))
    nctmo = ncout.createVariable("time_msource", "f8", ("time_msource",))
    ncvsos = ncout.createVariable("time_step_vsource", "f4", ("one",))
    ncvsis = ncout.createVariable("time_step_vsink", "f4", ("one",))
    ncvmos = ncout.createVariable("time_step_msource", "f4", ("one",))

    ncso[:] = keep
    ncsi[:] = siel
    ncvso[:] = so2
    ncvsi[:] = si
    nctso[:] = time
    nctsi[:] = time
    nctmo[:] = time
    ncvsos[:] = time[1] - time[0]
    ncvsis[:] = time[1] - time[0]
    ncvmos[:] = time[1] - time[0]

    fill_val = np.full((len(time), len(keep)), -9999.0)
    ncvmo[:, 0, :] = fill_val
    ncout.sync()

    fill_val.fill(0)
    ncvmo[:, 1, :] = fill_val
    ncout.sync()

    ncout.close()
    logger.info(
        "    Wrote source.nc — %d sources (from %d), %d sinks, %d timesteps",
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
        raise RuntimeError(
            f"metis_prep failed (exit {result.returncode}): {result.stderr[-2000:]}"
        )

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
        raise RuntimeError(
            f"gpmetis failed (exit {result.returncode}): {result.stderr[-2000:]}"
        )

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
    base_dt = start_date.replace(tzinfo=timezone.utc) if start_date.tzinfo is None else start_date

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


# ---------------------------------------------------------------------------
# 8. Generate sflux from LDASIN  (was post_nwm_forcing_coastal → makeAtmo.py)
# ---------------------------------------------------------------------------


def make_sflux(
    *,
    work_dir: Path,
    forcing_input_dir: Path,
    start_date: datetime,
    duration_hours: int,
    geogrid_file: Path,
) -> Path:
    """Generate sflux atmospheric forcing from LDASIN files.

    This is the pure-Python equivalent of ``makeAtmo.py`` — reads
    LDASIN files and writes ``sflux/sflux_air_1.0001.nc``.

    Returns the path to the sflux output file.
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

    from coastal_calibration.sflux import make_atmo_sflux

    make_atmo_sflux(
        forcing_input_dir=forcing_subdir,
        work_dir=work_dir,
        start_dt=start_date,
        duration_hours=duration_hours,
        geogrid_file=geogrid_file,
    )

    if not sflux_out.exists():
        raise RuntimeError(f"make_atmo_sflux did not produce {sflux_out}.")

    # SCHISM expects sflux_air_1.{n}.nc (no leading zeros) but makeAtmo
    # produces sflux_air_1.0001.nc (4-digit zero-padded).  Rename files
    # to match the expected naming convention.
    import re

    for f in sflux_dir.glob("sflux_air_*.nc"):
        m = re.match(r"(sflux_air_\d+)\.(\d+)\.nc", f.name)
        if m and len(m.group(2)) > 1 and m.group(2).startswith("0"):
            new_name = f"{m.group(1)}.{int(m.group(2))}.nc"
            new_path = f.parent / new_name
            if not new_path.exists():
                f.rename(new_path)
                logger.info("    Renamed %s → %s", f.name, new_name)

    logger.info("    Generated sflux in %s", sflux_dir)
    return sflux_dir


# ---------------------------------------------------------------------------
# 9. Update param.nml  (was update_param.bash)
# ---------------------------------------------------------------------------


def update_params(
    *,
    work_dir: Path,
    prebuilt_dir: Path,
    start_date: datetime,
    duration_hours: int,
    hot_start_file: Path | None = None,
) -> Path:
    """Create ``param.nml`` and symlink static mesh files.

    This is the pure-Python equivalent of ``update_param.bash``.
    Copies the template ``param.nml`` from the prebuilt model
    directory, updates date/time/duration parameters, and symlinks
    mesh files.

    Returns the path to the generated ``param.nml``.
    """
    import re

    coastal_parm = prebuilt_dir

    # Copy template param.nml
    param_path = work_dir / "param.nml"
    shutil.copy2(coastal_parm / "param.nml", param_path)
    text = param_path.read_text()

    # Compute date parameters
    pdy = start_date.strftime("%Y%m%d")
    cyc = start_date.strftime("%H")

    length_hrs = int(duration_hours)
    if length_hrs <= 0:
        rnhours = -length_hrs
    else:
        rnhours = length_hrs

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

    # Update run parameters
    text = re.sub(r"(?m)^(\s*)nspool\s*=.*$", r"\g<1>nspool = 18", text)
    text = re.sub(r"(?m)^(\s*)ihfskip\s*=.*$", r"\g<1>ihfskip = 18", text)

    # Use netCDF source/sink forcing
    text = re.sub(r"(?m)^(\s*)if_source\s*=.*$", r"\g<1>if_source = -1", text)

    # rnday is fractional day
    rnday = rnhours / 24.0
    text = re.sub(r"(?m)^(\s*)rnday\s*=.*$", rf"\g<1>rnday = {rnday:.8f}", text)

    # Timestep and atmospheric timestep
    text = re.sub(r"(?m)^(\s*)dt\s*=.*$", r"\g<1>dt = 200", text)
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

    param_path.write_text(text)

    # Symlink static mesh files
    static_files = [
        "hgrid.gr3", "hgrid.ll", "manning.gr3", "vgrid.in",
        "bctides.in", "windrot_geo2proj.gr3", "hgrid.utm",
        "hgrid.cpp", "elev.ic", "element_areas.txt",
    ]
    for fname in static_files:
        src = coastal_parm / fname
        dst = work_dir / fname
        if src.exists():
            dst.unlink(missing_ok=True)
            dst.symlink_to(src)

    # Optional files
    for fname in ("station.in", "elevation_correction.csv", "open_bnds_hgrid.nc", "hgrid.nc"):
        src = coastal_parm / fname
        dst = work_dir / fname
        if src.exists():
            dst.unlink(missing_ok=True)
            dst.symlink_to(src)

    # Copy sflux directory (contains sflux_inputs.txt)
    sflux_src = coastal_parm / "sflux"
    sflux_dst = work_dir / "sflux"
    sflux_dst.mkdir(exist_ok=True)
    if sflux_src.exists():
        for f in sflux_src.iterdir():
            shutil.copy2(f, sflux_dst / f.name)

    logger.info("    Created param.nml and symlinked mesh files in %s", work_dir)
    return param_path


# ---------------------------------------------------------------------------
# 10. Elevation datum correction  (was correct_elevation.py)
# ---------------------------------------------------------------------------


def correct_elevation(elev_file: Path, correction_file: Path) -> None:
    """Subtract datum corrections from ``elev2D.th.nc`` in-place.

    Parameters
    ----------
    elev_file : Path
        SCHISM boundary forcing netCDF4 file with a ``time_series``
        variable (modified in-place).
    correction_file : Path
        CSV file with correction values in the 6th column (0-indexed: 5),
        one value per open-boundary node, with one header row to skip.
    """
    import netCDF4 as nc
    import numpy as np

    elev_correct = np.loadtxt(str(correction_file), delimiter=",", skiprows=1, usecols=5)
    with nc.Dataset(elev_file, "r+") as ds:
        elev_var = ds["time_series"]
        for t in range(elev_var.shape[0]):
            elev_var[t] = elev_var[t].ravel() - elev_correct


# ---------------------------------------------------------------------------
# 11. TPXO boundary conditions  (was make_tpxo_ocean.bash)
# ---------------------------------------------------------------------------


def make_tpxo_boundary(
    *,
    work_dir: Path,
    start_date: datetime,
    duration_hours: int,
    timestep_seconds: int,
    prebuilt_dir: Path,
    otps_dir: Path | None,
    tpxo_data_dir: Path,
) -> Path:
    """Generate tidal boundary forcing from TPXO atlas.

    Runs ``predict_tide`` binary and Python scripts to produce
    ``elev2D.th.nc``.

    Returns the path to ``elev2D.th.nc``.
    """
    from coastal_calibration.tides import make_otps_input, otps_to_open_bnds

    coastal_parm = prebuilt_dir
    pdy = start_date.strftime("%Y%m%d")
    cyc = start_date.strftime("%H")

    end_dt = start_date + timedelta(hours=abs(int(duration_hours)))

    # 1. Create OTPS input file
    open_bnds = coastal_parm / "open_bnds_hgrid.nc"
    otps_input = work_dir / "otps_lat_lon_time.txt"

    make_otps_input(
        grid_file=open_bnds,
        output_file=otps_input,
        start_dt=start_date,
        end_dt=end_dt,
        timestep_s=timestep_seconds,
    )

    # 2. Copy OTPS setup files and link TPXO atlas data
    from coastal_calibration.tides import TIDES_DATA_DIR

    for fname in ("setup_tpxo.txt", "Model_tpxo10_atlas"):
        src = TIDES_DATA_DIR / fname
        if src.exists():
            shutil.copy2(src, work_dir / fname)

    tpxo_link = work_dir / "TPXO10_atlas_v2_nc"
    tpxo_link.unlink(missing_ok=True)
    tpxo_link.symlink_to(tpxo_data_dir)

    # 3. Run predict_tide
    if otps_dir is not None:
        predict_tide_bin = otps_dir / "predict_tide"
    else:
        found = shutil.which("predict_tide")
        if found is None:
            raise RuntimeError(
                "predict_tide not found on PATH.  Set paths.otps_dir or "
                "install predict_tide via pixi."
            )
        predict_tide_bin = Path(found)
    result = subprocess.run(
        [str(predict_tide_bin)],
        stdin=(work_dir / "setup_tpxo.txt").open(),
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"predict_tide failed (exit {result.returncode}): "
            f"{result.stderr[-2000:]}"
        )

    # 4. Convert OTPS output to elev2D.th.nc
    otps_to_open_bnds(
        otps_output_file=work_dir / "otps_out.txt",
        grid_file=open_bnds,
        elev_output_file=work_dir / "elev2D.th.nc",
    )

    # 5. Apply elevation correction if available
    correction_file = work_dir / "elevation_correction.csv"
    if correction_file.exists():
        logger.info("    Applying elevation datum correction")
        correct_elevation(work_dir / "elev2D.th.nc", correction_file)

    elev_file = work_dir / "elev2D.th.nc"
    if not elev_file.exists():
        raise RuntimeError("TPXO boundary: elev2D.th.nc was not produced")

    logger.info("    TPXO boundary created: %s", elev_file)
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
) -> Path:
    """Generate boundary forcing from STOFS data via ESMF regridding.

    Runs ``regrid_estofs.py`` via MPI and optionally ``makeOceanTide.py``
    for medium-range runs.

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
    import sys

    env = os.environ.copy()
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    cmd = [
        "mpiexec", "-n", str(mpi_tasks),
        sys.executable, "-m", "coastal_calibration.regridding.regrid_estofs",
        str(estofs_data), str(hgrid_file), str(output_file),
        "--cycle-date", pdy,
        "--cycle-time", f"{cyc}00",
        "--length-hrs", str(length_hrs),
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
            f"regrid_estofs failed (exit {result.returncode}): "
            f"{result.stderr[-2000:]}"
        )

    # Post-process: tidal fill for medium-range runs (>180h)
    raw_length = abs(int(duration_hours))
    if raw_length > 180:
        from coastal_calibration.tides import generate_ocean_tide

        tidal_constants_dir = Path(
            os.environ.get("COASTAL_ROOT_DIR", "")
        ) / "Tides" / "TidalConst"
        try:
            generate_ocean_tide(
                hgrid_gr3=coastal_parm / "hgrid.gr3",
                output_file=output_file,
                start_dt=start_date,
                duration_hours=raw_length,
                tidal_constants_dir=tidal_constants_dir,
            )
        except Exception:
            logger.warning("    generate_ocean_tide failed", exc_info=True)

    # Apply elevation correction if available
    correction_file = work_dir / "elevation_correction.csv"
    if correction_file.exists():
        try:
            correct_elevation(output_file, correction_file)
        except Exception:
            logger.warning("    correct_elevation failed", exc_info=True)

    if not output_file.exists():
        raise RuntimeError("STOFS boundary: elev2D.th.nc was not produced")

    logger.info("    STOFS boundary created: %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# 12. Combine hotstart  (was the conditional in post_schism.bash)
# ---------------------------------------------------------------------------


def combine_hotstart(outputs_dir: Path) -> None:
    """Run ``combine_hotstart7`` in the outputs directory."""
    result = subprocess.run(
        ["combine_hotstart7"],
        cwd=outputs_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"combine_hotstart7 failed (exit {result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    logger.info("    combine_hotstart7 completed")
