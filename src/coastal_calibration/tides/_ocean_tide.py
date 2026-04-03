"""Ocean tidal level generation for long SCHISM forecasts.

Refactored from ``wrf_hydro_workflow_dev/coastal/Tides/makeOceanTide.py``.
Extends ESTOFS boundary conditions with TPXO tidal predictions beyond
hour 180 for NWM medium/extended-range forecasts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc  # noqa: N813
import numpy as np
from scipy.interpolate import griddata

import coastal_calibration.tides.pytides.constituent as con
from coastal_calibration.logging import logger
from coastal_calibration.tides.pytides.tide import Tide


def _generate_tidal_levels(  # noqa: PLR0915
    consts_path: str,
    grid_file: str,
    output_file: str,
    start_time: datetime,
    total_hours: range,
) -> None:
    """Core tidal prediction logic (internal use)."""
    lon: list[float] = []
    lat: list[float] = []
    bnodes: list[int] = []

    with Path(grid_file).open() as f:
        next(f)
        line = f.readline()
        ne = int(line.split()[0])
        nn = int(line.split()[1])
        for _ in range(nn):
            line = f.readline()
            lon.append(float(line.split()[1]))
            lat.append(float(line.split()[2]))
        for _ in range(ne):
            f.readline()
        num_open_boundaries = int(f.readline().split()[0])
        total_open_boundary_nodes = int(f.readline().split()[0])
        for _ in range(num_open_boundaries):
            num_boundary_nodes = int(f.readline().split()[0])
            bnodes.extend(int(f.readline()) for _ in range(num_boundary_nodes))
    if len(bnodes) != total_open_boundary_nodes:
        msg = f"Parsed {len(bnodes)} boundary nodes but header declares {total_open_boundary_nodes}"
        raise ValueError(msg)

    lo = [lon[b - 1] for b in bnodes]
    la = [lat[b - 1] for b in bnodes]

    tide_constants = ["k1", "k2", "m2", "n2", "o1", "p1", "q1", "s2"]
    consfiles = [Path(consts_path) / f"{x}.nc" for x in tide_constants]
    pha = np.zeros((len(lo), len(consfiles)))
    amp = np.zeros((len(lo), len(consfiles)))

    x = y = i = None  # populated on first constituent file
    for col, file in enumerate(consfiles):
        data = nc.Dataset(file, "r")
        if col == 0:
            x = data.variables["lon"][:]
            y = data.variables["lat"][:]
            x, y = np.meshgrid(x, y)
            i = np.where(x > 180.0)
            x[i] = x[i] - 360.0
            i = np.where(
                (x < max(lo) + 1) & (x > min(lo) - 1) & (y < max(la) + 1) & (y > min(la) - 1)
            )
            x = x[i]
            y = y[i]

        a = data.variables["amplitude"][:]
        a = a[i]
        p = data.variables["phase"][:]
        p = p[i]

        mask = ~a.mask if hasattr(a, "mask") else np.ones(a.shape, dtype=bool)
        if x is None or y is None:
            msg = "Constituent coordinate arrays were not initialized"
            raise RuntimeError(msg)
        xI = x[mask]  # noqa: N806
        yI = y[mask]  # noqa: N806
        p = p[mask]
        a = a[mask]

        amp[:, col] = griddata((xI, yI), a, (lo, la), method="linear")
        pha[:, col] = griddata((xI, yI), p, (lo, la), method="linear")
        data.close()

    pred_times = Tide._times(start_time, total_hours)  # pyright: ignore[reportPrivateUsage]
    if not isinstance(pred_times, list):
        msg = "Expected list of prediction times"
        raise TypeError(msg)

    wl = np.zeros((len(pred_times), amp.shape[0]))
    cons = [c for c in con.noaa if c != con._Z0]  # pyright: ignore[reportPrivateUsage]
    n = [3, 34, 0, 2, 5, 29, 25, 1]
    cons = [cons[c] for c in n]
    model = np.zeros(len(cons), dtype=Tide.dtype)
    model["constituent"] = cons
    for i in range(amp.shape[0]):
        model["amplitude"] = amp[i, :]
        model["phase"] = pha[i, :]
        tide = Tide(model=model, radians=False)
        wl[:, i] = tide.at(pred_times) / 100.0

    mode = "a" if Path(output_file).exists() else "w"
    ncout = nc.Dataset(output_file, mode, format="NETCDF4")

    if mode == "w":
        ncout.createDimension("time", None)
        ncout.createDimension("nOpenBndNodes", amp.shape[0])
        nctime = ncout.createVariable("time", "f8", ("time",))
        ncwl = ncout.createVariable("time_series", "f8", ("time", "nOpenBndNodes"))
        nctime[:] = np.arange(651600.0, 865000, 3600.0)
        ncwl[:] = wl
        start = 0
    else:
        nctime = ncout["time"]
        ncwl = ncout["time_series"]
        start = 181

    t_step = 3600
    t_start = start * t_step
    t_end = (start + total_hours.stop) * t_step
    new_times = np.arange(t_start, t_end, t_step)
    nctime[start:] = new_times
    ncwl[start:] = wl

    ncout.close()


def generate_ocean_tide(
    hgrid_gr3: Path,
    output_file: Path,
    start_dt: datetime,
    duration_hours: int,
    tidal_constants_dir: Path,
) -> None:
    """Generate tidal boundary levels and append to *output_file*.

    In the NWM operational workflow ESTOFS/STOFS data covers hours 0-180.
    For longer forecasts this function fills hours 181+ with tidal
    predictions computed from harmonic constituents (pytides).  The result
    is appended to an existing ``elev2D.th.nc``; if the file does not
    exist yet it is created from scratch.

    Parameters
    ----------
    hgrid_gr3 : Path
        SCHISM ``hgrid.gr3`` text file.
    output_file : Path
        Destination ``elev2D.th.nc``.  Opened in append mode when it
        already exists.
    start_dt : datetime
        Simulation start time (UTC).  Tidal predictions begin at
        ``start_dt + 181 h``.
    duration_hours : int
        Total simulation length.  Must be > 181 for this function to do
        anything.
    tidal_constants_dir : Path
        Directory containing the eight TPXO constituent files
        (``k1.nc``, ``k2.nc``, ``m2.nc``, ``n2.nc``, ``o1.nc``,
        ``p1.nc``, ``q1.nc``, ``s2.nc``).
    """
    if duration_hours < 182:
        logger.debug("Duration %dh < 182h, skipping ocean tide generation", duration_hours)
        return

    logger.info("Generating ocean tide predictions for hours 181-%d", duration_hours)
    tide_start = start_dt + timedelta(hours=181)
    hour_range = range(duration_hours - 181)
    _generate_tidal_levels(
        consts_path=str(tidal_constants_dir),
        grid_file=str(hgrid_gr3),
        output_file=str(output_file),
        start_time=tide_start,
        total_hours=hour_range,
    )
