"""Tidal boundary-condition generation via pyTMD.

One predictor, one atlas layout — serves SCHISM ``elev2D.th.nc``, the
>180 h ESTOFS fill, and SFINCS boundary forcing. Works against any
model in ``pyTMD.io.load_database()`` that exposes a netcdf elevation
group (TPXO10, TPXO9, FES2014, GOT, EOT, ...).

Public functions
----------------
:func:`predict_tide_at_points`
    Low-level: predict elevations at arbitrary (lon, lat) points over
    a user-supplied time array. The SFINCS forcing stage uses this
    directly so it can manage its own cadence.
:func:`write_schism_boundary`
    Read a SCHISM ``open_bnds_hgrid.nc``, predict elevations at the
    user-specified cadence over a forecast window, and write the
    canonical 4-D ``elev2D.th.nc``. The cadence is the
    ``time_step_seconds`` parameter, not a hardcoded hourly value —
    SCHISM interpolates between forcing rows at its own integration
    dt, so any cadence the user chooses works.
:func:`extend_schism_boundary`
    Append a tidal-only fill to an existing ``elev2D.th.nc``, reading
    the existing file's ``time_step`` rather than assuming hourly. Used
    for NWM medium/extended-range runs that exceed the 180 h STOFS
    window.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import netCDF4
import numpy as np

from coastal_calibration.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# pyTMD measures ``delta_time`` from this epoch in seconds; the same
# epoch is passed through every compute call so the predictions are
# expressed in UTC and lined up with the SCHISM model clock.
_PYTMD_EPOCH = (2000, 1, 1, 0, 0, 0)
_PYTMD_EPOCH_DT = datetime(*_PYTMD_EPOCH)

# Default tidal model. Matches the TPXO10-atlas-v2 netCDF distribution
# that the project has historically shipped against. Override per-run
# via :attr:`BoundaryConfig.tidal_model` to use FES2014, GOT4.10, etc.
DEFAULT_TIDAL_MODEL = "TPXO10-atlas-v2-nc"


def _stage_atlas_layout(tidal_model: str, atlas_dir: Path, root: Path) -> None:
    """Populate *root* with the directory layout pyTMD expects for *tidal_model*.

    pyTMD's bundled database identifies each atlas file by a relative
    path rooted at a conventional subdirectory (e.g.
    ``TPXO10_atlas_v2/h_m2_tpxo10_atlas_30_v2.nc``). Users in this
    project may supply the atlas files in a directory of any name —
    this helper symlinks the user's files into the expected layout
    under *root* so pyTMD's ``directory + model`` resolution just works,
    without us having to roll our own path resolver against pyTMD's
    internal conventions.

    All atlas groups referenced by the model spec (``z``, ``u``, ``v``)
    are staged. pyTMD's high-level entry points validate every group
    by default, so failing to provide the current files would surface
    as a ``FileNotFoundError`` deep inside the predictor — even though
    we only ever read the elevation group.

    The elevation (``z``) files must exist in *atlas_dir*; missing
    current files (``u``/``v``) are tolerated as a warning since they
    aren't read for tide-elevation predictions.
    """
    import pyTMD  # local import keeps the data package import light

    db = pyTMD.io.load_database()
    known = list(db.keys())
    if tidal_model not in known:
        examples = ", ".join(sorted(k for k in known if "atlas-nc" in k or "nc" in k.lower())[:10])
        raise ValueError(
            f"Unknown tidal_model {tidal_model!r}. Examples of supported "
            f"netcdf models: {examples}, ..."
        )

    spec: dict[str, Any] = db[tidal_model]
    if "z" not in spec:
        raise ValueError(f"Tidal model {tidal_model!r} has no elevation (z) group")

    # Elevation files are required.
    z = spec["z"]
    z_relpaths = [*z["model_file"], z["grid_file"]]
    missing = [p for p in z_relpaths if not (atlas_dir / Path(p).name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} atlas elevation file(s) under {atlas_dir} "
            f"for model {tidal_model!r}: {[Path(p).name for p in missing[:3]]}..."
        )

    # Collect every relpath the spec references (z + optionally u/v).
    # pyTMD's high-level entry points validate every model group, so we
    # symlink current files too when present; absent current files are
    # tolerated since elevation predictions never read them.
    all_relpaths = list(z_relpaths)
    for group in ("u", "v"):
        if group in spec:
            all_relpaths.extend(spec[group]["model_file"])
            all_relpaths.append(spec[group]["grid_file"])

    for relpath in all_relpaths:
        src = atlas_dir / Path(relpath).name
        if not src.exists():
            continue  # current file missing — see docstring
        target = root / relpath
        if target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(src)


def predict_tide_at_points(
    lons: NDArray[np.floating],
    lats: NDArray[np.floating],
    times: Sequence[datetime] | NDArray[np.floating],
    atlas_dir: Path,
    tidal_model: str = DEFAULT_TIDAL_MODEL,
) -> NDArray[np.float64]:
    """Predict tidal elevations at (lon, lat) points over the given times.

    Parameters
    ----------
    lons, lats
        1-D arrays of equal length giving the point coordinates in
        degrees east / degrees north.
    times
        Either a sequence of ``datetime`` objects, or a 1-D array of
        seconds since 2000-01-01 UTC.
    atlas_dir
        Directory containing the atlas constituent and grid files for
        *tidal_model*.
    tidal_model
        pyTMD model identifier (see ``pyTMD.io.load_database``). Defaults
        to TPXO10-atlas-v2 netCDF.

    Returns
    -------
    ndarray of shape ``(n_points, n_times)``
        Elevations in meters. Points outside the model domain return 0
        rather than NaN, matching the historical OTPSnc behavior.
    """
    if len(lons) != len(lats):
        raise ValueError(f"lons/lats length mismatch: {len(lons)} vs {len(lats)}")

    if len(times) and isinstance(times[0], datetime):
        delta_time = np.array(
            [(t - _PYTMD_EPOCH_DT).total_seconds() for t in times], dtype=np.float64
        )
    else:
        delta_time = np.asarray(times, dtype=np.float64)

    # pyTMD's compute submodule is registered via a lazy importer and
    # isn't exposed on the package object pyright sees; the local import
    # also defers the heavy xarray/h5netcdf load until first use.
    from pyTMD import compute as _pytmd_compute

    # Stage user's atlas files under the layout pyTMD's database
    # expects so ``directory + model`` resolution just works. The temp
    # tree is symlinks only and is cleaned up on exit.
    with tempfile.TemporaryDirectory(prefix="pytmd_atlas_") as tmp:
        _stage_atlas_layout(tidal_model, atlas_dir, Path(tmp))
        elev = _pytmd_compute.tide_elevations(
            x=np.asarray(lons, dtype=np.float64),
            y=np.asarray(lats, dtype=np.float64),
            delta_time=delta_time,
            directory=tmp,
            model=tidal_model,
            epoch=_PYTMD_EPOCH,
            standard="UTC",
            type="time series",
            method="linear",
            extrapolate=False,
        )

    arr = np.asarray(elev, dtype=np.float64)
    if hasattr(elev, "mask"):
        arr = np.where(np.ma.getmaskarray(elev), 0.0, arr)
    # pyTMD returns shape (n_points, n_times); SFINCS / SCHISM callers
    # both want that orientation, so don't transpose here.
    return arr


def _read_open_bnd_coords(grid_file: Path) -> NDArray[np.float64]:
    """Return ``(n_bnd, 2)`` lon/lat array from an ``open_bnds_hgrid.nc``."""
    with netCDF4.Dataset(grid_file) as ds:
        coords = ds["nodeCoords"][:]
        idx = ds["openBndNodes"][:]
    return np.asarray(coords[idx], dtype=np.float64)


def write_schism_boundary(
    grid_file: Path,
    output_file: Path,
    start_dt: datetime,
    duration_hours: int,
    atlas_dir: Path,
    tidal_model: str = DEFAULT_TIDAL_MODEL,
    time_step_seconds: int = 3600,
) -> None:
    """Predict tides at SCHISM open-boundary nodes and write ``elev2D.th.nc``.

    Produces a canonical 4-D ``elev2D.th.nc`` (``time``,
    ``nOpenBndNodes``, ``nLevels=1``, ``nComponents=1``) with the
    user-specified cadence. SCHISM reads ``elev2D.th.nc`` and
    interpolates between rows at its own integration dt, so any
    cadence chosen here works — the file is correct so long as it
    covers the simulation window.

    Parameters
    ----------
    grid_file
        SCHISM ``open_bnds_hgrid.nc`` containing ``nodeCoords`` and
        ``openBndNodes``.
    output_file
        Destination path for ``elev2D.th.nc``.
    start_dt
        Simulation start (UTC, naive).
    duration_hours
        Number of forecast hours. The file covers exactly
        ``duration_hours`` from *start_dt*, with the row count
        determined by *time_step_seconds*.
    atlas_dir
        Directory holding the tidal atlas constituent files.
    tidal_model
        pyTMD model identifier (default TPXO10-atlas-v2-nc).
    time_step_seconds
        Cadence between successive boundary rows. Defaults to 3600 s
        (hourly) to match the STOFS / NWM convention. Set this to any
        positive integer when the workflow needs sub-hourly forcing
        (e.g. for short, high-resolution SCHISM runs).
    """
    from coastal_calibration._nc_io import ELEV2D_MISSING, write_elev2d_th

    if time_step_seconds <= 0:
        raise ValueError(f"time_step_seconds must be positive, got {time_step_seconds}")

    duration_hours = abs(int(duration_hours))
    duration_seconds = duration_hours * 3600
    nsteps = duration_seconds // time_step_seconds + 1
    coords = _read_open_bnd_coords(grid_file)
    n_bnd = coords.shape[0]
    logger.info(
        "    pyTMD tide predict: %d nodes x %d steps at %d s cadence from %s (model=%s)",
        n_bnd,
        nsteps,
        time_step_seconds,
        start_dt.isoformat(),
        tidal_model,
    )

    times = [start_dt + timedelta(seconds=i * time_step_seconds) for i in range(nsteps)]
    elev = predict_tide_at_points(
        lons=coords[:, 0],
        lats=coords[:, 1],
        times=times,
        atlas_dir=atlas_dir,
        tidal_model=tidal_model,
    )  # (n_bnd, nsteps)

    # Transpose to (nsteps, n_bnd) matching write_elev2d_th's 2-D form.
    series = elev.T.astype(np.float64)
    time_seconds = np.arange(0, nsteps * time_step_seconds, time_step_seconds, dtype=np.float64)
    base = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    write_elev2d_th(
        output_file,
        n_open_bnd_nodes=n_bnd,
        time_seconds=time_seconds,
        time_step_seconds=time_step_seconds,
        time_series=series,
        time_attrs={
            "long_name": "model time",
            "standard_name": "time",
            "units": f"seconds since {base}        ! NCDASE - BASE_DAT",
            "base_date": f"{base}        ! NCDASE - BASE_DATE",
            "start_time": 0.0,
        },
        missing=ELEV2D_MISSING,
    )


def extend_schism_boundary(
    grid_file: Path,
    output_file: Path,
    start_dt: datetime,
    duration_hours: int,
    atlas_dir: Path,
    tidal_model: str = DEFAULT_TIDAL_MODEL,
    fill_from_hour: int = 181,
) -> None:
    """Append a tidal-only fill to an existing ``elev2D.th.nc``.

    SCHISM medium/extended-range NWM runs need >180 h of open-boundary
    forcing, but STOFS only publishes the first 180 h. This function
    extends the existing file with pyTMD tidal predictions from
    ``fill_from_hour`` through ``duration_hours``.

    The file's existing ``time_step`` variable is read and used as the
    fill cadence — no assumption is made about hourly spacing. The
    fill aligns with the existing time grid so SCHISM sees a single
    consistent series.

    The output file must already exist with the SCHISM 4-D schema (the
    upstream ``regrid_estofs`` stage produces it). This function only
    appends rows; it never reshapes or recreates the file.
    """
    if duration_hours < fill_from_hour + 1:
        logger.debug("Duration %dh < %dh, skipping tidal fill", duration_hours, fill_from_hour + 1)
        return

    if not Path(output_file).exists():
        raise FileNotFoundError(
            f"{output_file} does not exist. extend_schism_boundary is a "
            "tidal-fill step that must run after regrid_estofs has "
            "created the canonical 4-D elev2D.th.nc."
        )

    # Read the existing file's cadence so the fill rows line up with
    # the STOFS rows already written. ``time_step`` is a scalar in the
    # SCHISM schema (dimension ``one``).
    with netCDF4.Dataset(output_file, "r") as ds:
        dt_s = int(float(ds["time_step"][0]))
    if dt_s <= 0:
        raise ValueError(f"{output_file} has non-positive time_step={dt_s}")

    coords = _read_open_bnd_coords(grid_file)
    n_bnd = coords.shape[0]
    # Convert the hour-based fill window into the file's native cadence.
    fill_from_step = (fill_from_hour * 3600) // dt_s
    last_step = (duration_hours * 3600) // dt_s
    n_fill = last_step - fill_from_step
    if n_fill <= 0:
        logger.debug(
            "Nothing to fill at %d s cadence between hour %d and %d",
            dt_s,
            fill_from_hour,
            duration_hours,
        )
        return

    logger.info(
        "    pyTMD tidal fill: %d nodes x %d steps at %d s cadence from hour %d (model=%s)",
        n_bnd,
        n_fill,
        dt_s,
        fill_from_hour,
        tidal_model,
    )

    fill_start = start_dt + timedelta(seconds=fill_from_step * dt_s)
    times = [fill_start + timedelta(seconds=i * dt_s) for i in range(n_fill)]
    elev = predict_tide_at_points(
        lons=coords[:, 0],
        lats=coords[:, 1],
        times=times,
        atlas_dir=atlas_dir,
        tidal_model=tidal_model,
    )  # (n_bnd, n_fill)

    series = elev.T  # (n_fill, n_bnd)
    new_times = np.arange(
        fill_from_step * dt_s,
        (fill_from_step + n_fill) * dt_s,
        dt_s,
    )
    with netCDF4.Dataset(output_file, "a", format="NETCDF4") as ds:
        ds["time"][fill_from_step : fill_from_step + n_fill] = new_times
        ds["time_series"][fill_from_step : fill_from_step + n_fill] = series[
            :, :, np.newaxis, np.newaxis
        ]
