"""Simulated vs observed water-level comparison plots.

Provides :func:`plot_station_comparison` for creating comparison figures
between model-simulated water levels and NOAA CO-OPS observations.
Both the SFINCS and SCHISM plot stages delegate to this module.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["plot_station_comparison", "plotable_stations"]

#: Maximum number of stations per figure (2x2 layout).
_STATIONS_PER_FIGURE = 4


def plotable_stations(
    station_ids: list[str],
    sim_elevation: NDArray[np.float64],
    obs_ds: Any,
) -> list[tuple[str, int]]:
    """Return ``(station_id, column_index)`` pairs that have data to plot.

    A station is plotable only when *both* its simulated and observed
    time-series contain finite values — a comparison plot with only
    one series is not useful.

    The returned list is sorted by numeric station ID so that figures
    are deterministic across runs.
    """
    result: list[tuple[str, int]] = []
    for i, sid in enumerate(station_ids):
        has_sim = bool(np.isfinite(sim_elevation[:, i]).any())
        has_obs = False
        if sid in obs_ds.station.values:
            has_obs = bool(np.isfinite(obs_ds.water_level.sel(station=sid)).any())
        if has_sim and has_obs:
            result.append((sid, i))
    try:
        result.sort(key=lambda pair: int(pair[0]))
    except ValueError:
        result.sort(key=lambda pair: pair[0])
    return result


def plot_station_comparison(
    sim_times: Any,
    sim_elevation: NDArray[np.float64],
    station_ids: list[str],
    obs_ds: Any,
    figs_dir: Path | str,
) -> list[Path]:
    """Create comparison figures of simulated vs observed water levels.

    Stations that lack *either* valid observations or valid simulated
    data are skipped so that empty panels do not appear.

    Parameters
    ----------
    sim_times : array-like
        Simulation datetimes.
    sim_elevation : ndarray
        Simulated elevation of shape ``(n_times, n_stations)``.
    station_ids : list[str]
        NOAA station IDs (one per column in *sim_elevation*).
    obs_ds : xr.Dataset
        Observed water levels with a ``water_level`` variable indexed
        by ``station`` and ``time``.
    figs_dir : Path or str
        Output directory for figures.

    Returns
    -------
    list[Path]
        Paths to the saved figures.
    """
    import sys

    import matplotlib

    # Force non-interactive backend except inside Jupyter kernels.
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    figs_dir = Path(figs_dir)

    # ── Pre-filter: keep only stations with both obs and sim ──
    stations = plotable_stations(station_ids, sim_elevation, obs_ds)

    if not stations:
        figs_dir.mkdir(parents=True, exist_ok=True)
        return []

    n_plotable = len(stations)
    n_figures = math.ceil(n_plotable / _STATIONS_PER_FIGURE)
    figs_dir.mkdir(parents=True, exist_ok=True)

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
            ax = axes_flat[i]

            # Simulated
            sim_ts = sim_elevation[:, col_idx]
            has_sim = bool(np.isfinite(sim_ts).any())

            # Observed
            has_obs = False
            if sid in obs_ds.station.values:
                obs_wl = obs_ds.water_level.sel(station=sid)
                has_obs = bool(np.isfinite(obs_wl).any())
                if has_obs:
                    ax.plot(
                        obs_wl.time.values,
                        obs_wl.values,
                        label="Observed",
                        color="k",
                        linewidth=1.0,
                    )

            if has_sim:
                ax.plot(
                    sim_times,
                    sim_ts,
                    color="r",
                    ls="--",
                    alpha=0.5,
                )
                ax.scatter(
                    sim_times,
                    sim_ts,
                    label="Simulated",
                    color="r",
                    marker="x",
                    s=25,
                )

            ax.set_title(f"NOAA {sid}", fontsize=14, fontweight="bold")
            ax.set_ylabel("Water Level (m, MSL)", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
            ax.legend(fontsize=11, loc="best")

            # Readable date formatting on x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment("right")

        # Remove unused axes
        for j in range(batch_size, nrows * ncols):
            axes_flat[j].remove()

        fig.tight_layout()
        fig_path = figs_dir / f"stations_comparison_{fig_idx + 1:03d}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(fig_path)

    return saved
