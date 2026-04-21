"""Multi-run water-level comparison plots.

:func:`plot_station_comparison` overlays one or more simulated runs and
optional NOAA CO-OPS observations at each station, producing 2x2 figures
with one panel per plotable station.

Each run is a ``(times, elevation)`` pair where ``elevation`` is shaped
``(n_times, n_stations)`` and indexed by *station_ids*; run labels become
the legend entries. When ``obs_ds`` is provided, observations are drawn
as a black reference line underneath every run.

Both SFINCS and SCHISM plot stages delegate to this module.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, NDArray

__all__ = ["plot_station_comparison"]

#: Maximum number of stations per figure (2x2 layout).
_STATIONS_PER_FIGURE = 4

#: Marker cycle used for run series (observed always uses a black line, no marker).
_RUN_MARKERS: tuple[str, ...] = ("x", "o", "^", "s", "D", "v", "<", ">", "P", "*")


def _plotable_stations(
    station_ids: list[str],
    runs: Mapping[str, tuple[Any, NDArray[np.float64]]],
    obs_ds: Any | None,
) -> list[tuple[str, int]]:
    """Return ``(station_id, col_idx)`` pairs with at least one finite series.

    A station is plotable when *any* run — or the observations when present —
    has at least one finite value. The list is sorted by numeric station ID
    (falling back to lexicographic order) so figures are deterministic.
    """
    result: list[tuple[str, int]] = []
    for i, sid in enumerate(station_ids):
        any_run = any(bool(np.isfinite(elev[:, i]).any()) for _, elev in runs.values())
        has_obs = False
        if obs_ds is not None and sid in obs_ds.station.values:
            has_obs = bool(np.isfinite(obs_ds.water_level.sel(station=sid)).any())
        if any_run or has_obs:
            result.append((sid, i))
    try:
        result.sort(key=lambda pair: int(pair[0]))
    except ValueError:
        result.sort(key=lambda pair: pair[0])
    return result


def _draw_station_panel(
    ax: Any,
    sid: str,
    col_idx: int,
    runs: Mapping[str, tuple[Any, NDArray[np.float64]]],
    obs_ds: Any | None,
    run_colors: dict[str, Any],
    run_markers: dict[str, str],
) -> None:
    """Draw one station panel with overlaid observed + run series."""
    import matplotlib.dates as mdates

    if obs_ds is not None and sid in obs_ds.station.values:
        obs_wl = obs_ds.water_level.sel(station=sid)
        if bool(np.isfinite(obs_wl).any()):
            ax.plot(
                obs_wl.time.values,
                obs_wl.values,
                label="Observed",
                color="k",
                linewidth=1.0,
            )

    for label, (times, elev) in runs.items():
        sim_ts = elev[:, col_idx]
        if not bool(np.isfinite(sim_ts).any()):
            continue
        color = run_colors[label]
        marker = run_markers[label]
        ax.plot(times, sim_ts, color=color, ls="--", alpha=0.5)
        ax.scatter(times, sim_ts, label=label, color=color, marker=marker, s=25)

    ax.set_title(f"NOAA {sid}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Water Level (m, MSL)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(30)
        tick_label.set_horizontalalignment("right")


def plot_station_comparison(
    runs: Mapping[str, tuple[ArrayLike, NDArray[np.float64]]],
    station_ids: list[str],
    figs_dir: Path | str,
    *,
    obs_ds: Any | None = None,
) -> list[Path]:
    """Create station comparison figures for one or more simulated runs.

    Parameters
    ----------
    runs : Mapping[str, tuple[ArrayLike, NDArray]]
        Maps each run's label to a ``(times, elevation)`` pair. Labels
        become the legend entries. Each run's *elevation* must have the
        same ``n_stations`` columns aligned with *station_ids*; per-run
        time axes may differ.
    station_ids : list[str]
        NOAA station IDs, one per column of every run's elevation array.
    figs_dir : Path or str
        Output directory for figures (created if needed).
    obs_ds : xr.Dataset, optional
        Observed water levels with a ``water_level`` variable indexed by
        ``station`` and ``time``. When *None*, the plots are pure
        model-vs-model comparisons.

    Returns
    -------
    list[pathlib.Path]
        Paths to the saved PNG figures, in pagination order (4 stations
        per 2x2 figure).

    Notes
    -----
    A station is plotable if any run or the observations has at least one
    finite value at that station. Stations with no data anywhere are
    silently skipped.

    Runs are drawn on top of observations. Colours are drawn from the
    ``tab10`` colormap (cycling at 10 runs); markers cycle through a
    10-shape palette independently.
    """
    import sys

    import matplotlib

    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if len(runs) == 0:
        msg = "plot_station_comparison: `runs` is empty."
        raise ValueError(msg)

    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    runs_dict = dict(runs)
    stations = _plotable_stations(station_ids, runs_dict, obs_ds)
    if not stations:
        return []

    palette = plt.colormaps["tab10"]
    labels = list(runs_dict.keys())
    run_colors = {label: palette(i % 10) for i, label in enumerate(labels)}
    run_markers = {label: _RUN_MARKERS[i % len(_RUN_MARKERS)] for i, label in enumerate(labels)}

    n_plotable = len(stations)
    n_figures = math.ceil(n_plotable / _STATIONS_PER_FIGURE)

    saved: list[Path] = []
    for fig_idx in range(n_figures):
        start = fig_idx * _STATIONS_PER_FIGURE
        end = min(start + _STATIONS_PER_FIGURE, n_plotable)
        batch = stations[start:end]
        batch_size = len(batch)

        nrows = 2 if batch_size > 2 else 1
        ncols = 2 if batch_size > 1 else 1

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(16, 5 * nrows),
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for i, (sid, col_idx) in enumerate(batch):
            _draw_station_panel(
                axes_flat[i], sid, col_idx, runs_dict, obs_ds, run_colors, run_markers
            )

        for j in range(batch_size, nrows * ncols):
            axes_flat[j].remove()

        fig.tight_layout()
        fig_path = figs_dir / f"stations_comparison_{fig_idx + 1:03d}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(fig_path)

    return saved
