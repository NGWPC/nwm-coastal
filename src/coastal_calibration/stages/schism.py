"""SCHISM model execution stages."""

from __future__ import annotations

import math
import re
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from coastal_calibration.config.schema import SchismModelConfig
from coastal_calibration.stages.base import WorkflowStage

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from coastal_calibration.config.schema import CoastalCalibConfig
    from coastal_calibration.utils.logging import WorkflowMonitor

# Buffer size for reading large hgrid.gr3 files (8 MB).
_HGRID_BUFFER_SIZE = 8 * 1024 * 1024
# Maximum number of stations per figure (2x2 layout).
_STATIONS_PER_FIGURE = 4


# ---------------------------------------------------------------------------
# hgrid.gr3 helpers
# ---------------------------------------------------------------------------


def _read_hgrid_header(hgrid_path: Path) -> tuple[int, int]:
    """Return (n_elements, n_nodes) from hgrid.gr3 header."""
    with hgrid_path.open("r", buffering=_HGRID_BUFFER_SIZE) as f:
        f.readline()  # description
        parts = f.readline().split()
        return int(parts[0]), int(parts[1])


def _read_node_coordinates(hgrid_path: Path, n_nodes: int) -> NDArray[np.float64]:
    """Read all node coordinates from hgrid.gr3 into an (n_nodes, 2) array."""
    coords = np.empty((n_nodes, 2), dtype=np.float64)
    with hgrid_path.open("r", buffering=_HGRID_BUFFER_SIZE) as f:
        f.readline()  # description
        f.readline()  # header
        for i in range(n_nodes):
            parts = f.readline().split()
            coords[i, 0] = float(parts[1])
            coords[i, 1] = float(parts[2])
    return coords


def _read_open_boundary_nodes(hgrid_path: Path, n_nodes: int, n_elements: int) -> list[list[int]]:
    """Read open boundary node lists from hgrid.gr3."""
    with hgrid_path.open("r", buffering=_HGRID_BUFFER_SIZE) as f:
        f.readline()  # description
        f.readline()  # header
        for _ in range(n_nodes):
            f.readline()
        for _ in range(n_elements):
            f.readline()

        # Open boundaries header
        line = f.readline().split("!")[0].split("=")[0].strip()
        n_open_bnds = int(line)
        f.readline()  # total open boundary nodes line

        open_boundaries: list[list[int]] = []
        for _ in range(n_open_bnds):
            bnd_line = f.readline().split("!")[0].split("=")[0].strip()
            n_bnd_nodes = int(bnd_line)
            boundary = [int(f.readline().split()[0]) for _ in range(n_bnd_nodes)]
            open_boundaries.append(boundary)

    return open_boundaries


def _write_station_in(
    base_dir: Path,
    lons: list[float],
    lats: list[float],
) -> Path:
    """Write a station.in file for SCHISM with multiple stations."""
    n = len(lons)
    lines = [
        "1 0 0 0 0 0 0 0 0",  # only elevation output
        str(n),
    ]
    for i, (lon, lat) in enumerate(zip(lons, lats, strict=False), start=1):
        lines.append(f"{i} {lon} {lat} 0.0")
    path = base_dir / "station.in"
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_station_names(base_dir: Path, station_ids: list[str]) -> Path:
    """Write a companion file mapping station indices to NOAA IDs."""
    path = base_dir / "station_noaa_ids.txt"
    path.write_text("\n".join(station_ids) + "\n")
    return path


def _read_station_noaa_ids(base_dir: Path) -> list[str]:
    """Read station NOAA IDs from the companion file."""
    path = base_dir / "station_noaa_ids.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _plotable_stations(
    station_ids: list[str],
    sim_elevation: NDArray[np.float64],
    obs_ds: Any,
) -> list[tuple[str, int]]:
    """Return (station_id, column_index) pairs that have data to plot.

    A station is plotable only when *both* its simulated and observed
    time-series contain finite values — a comparison plot with only
    one series is not useful.
    """
    result: list[tuple[str, int]] = []
    for i, sid in enumerate(station_ids):
        has_sim = bool(np.isfinite(sim_elevation[:, i]).any())
        has_obs = False
        if sid in obs_ds.station.values:
            has_obs = bool(np.isfinite(obs_ds.water_level.sel(station=sid)).any())
        if has_sim and has_obs:
            result.append((sid, i))
    return result


def _read_staout(staout_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read SCHISM station output file (staout_1).

    Returns
    -------
    time_seconds : ndarray
        Time in seconds from simulation start.
    elevation : ndarray
        Water elevation array of shape (n_times, n_stations).
    """
    data = np.loadtxt(staout_path, comments="!")
    time_seconds = data[:, 0]
    elevation = data[:, 1:]
    return time_seconds, elevation


def _patch_param_nml(param_path: Path, nspool_sta: int = 18) -> None:
    """Enable station output in param.nml (SCHOUT namelist).

    Sets ``iout_sta = 1`` and ``nspool_sta`` so that SCHISM writes
    station time-series.  ``nspool_sta`` must divide ``nhot_write``
    evenly or SCHISM will abort with
    ``mod(nhot_write, nspool_sta) /= 0``.

    The default *nspool_sta=18* matches the ``nspool`` value used by
    ``update_param.bash`` and is a divisor of every ``nhot_write``
    value that script produces (18, 72, 162, 2160, …).

    Parameters
    ----------
    param_path : Path
        Path to param.nml.
    nspool_sta : int, default 18
        Station output interval in time-steps.
    """
    text = param_path.read_text()

    # --- iout_sta ---
    new_text, count = re.subn(
        r"(?mi)^(\s*)iout_sta\s*=\s*\d+",
        r"\g<1>iout_sta = 1",
        text,
    )
    if count == 0:
        # Insert after &SCHOUT header
        new_text = re.sub(
            r"(?mi)(^&SCHOUT\s*$)",
            r"\1\n  iout_sta = 1",
            text,
            count=1,
        )
        if new_text == text:
            # Fallback: append a new &SCHOUT block
            lines = text.splitlines(keepends=True)
            lines.append("! Added by coastal_calibration\n&SCHOUT\n  iout_sta = 1\n/\n")
            new_text = "".join(lines)

    # --- nspool_sta ---
    text2, count2 = re.subn(
        r"(?mi)^(\s*)nspool_sta\s*=\s*\d+",
        rf"\g<1>nspool_sta = {nspool_sta}",
        new_text,
    )
    if count2 == 0:
        # Insert after iout_sta (which we just ensured is present)
        text2 = re.sub(
            r"(?mi)(^\s*iout_sta\s*=\s*1)",
            rf"\1\n  nspool_sta = {nspool_sta}",
            new_text,
            count=1,
        )

    param_path.write_text(text2)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class SchismObservationStage(WorkflowStage):
    """Discover NOAA CO-OPS stations and write station.in for SCHISM.

    Uses the open boundaries of the hgrid.gr3 mesh to compute a concave
    hull, then selects all NOAA water-level stations within that polygon.
    A ``station.in`` file is written to the work directory so SCHISM will
    output time series at those locations.

    Gated by ``include_noaa_gages`` on :class:`SchismModelConfig`.
    """

    name = "schism_obs"
    description = "Add NOAA observation stations for SCHISM"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SchismModelConfig)  # noqa: S101
        self.model: SchismModelConfig = config.model_config

    def run(self) -> dict[str, Any]:
        """Discover stations and write station.in."""
        if not self.model.include_noaa_gages:
            self._log("include_noaa_gages is disabled, skipping")
            return {"status": "skipped"}

        import shapely

        from coastal_calibration.coops_api import COOPSAPIClient

        work_dir = self.config.paths.work_dir
        hgrid_path = work_dir / "hgrid.gr3"
        if not hgrid_path.exists():
            raise FileNotFoundError(
                f"hgrid.gr3 not found in {work_dir}. "
                "The update_params stage must run before schism_obs "
                "so that the mesh file is symlinked into the work directory."
            )

        # Read mesh metadata
        self._update_substep("Reading hgrid.gr3 metadata")
        n_elements, n_nodes = _read_hgrid_header(hgrid_path)
        self._log(f"hgrid.gr3: {n_nodes} nodes, {n_elements} elements")

        # Read open boundary nodes
        self._update_substep("Reading open boundaries")
        open_boundaries = _read_open_boundary_nodes(hgrid_path, n_nodes, n_elements)
        if not open_boundaries:
            raise RuntimeError(
                "No open boundaries found in hgrid.gr3. "
                "Cannot discover NOAA stations without boundary geometry."
            )

        total_bnd_nodes = sum(len(b) for b in open_boundaries)
        self._log(
            f"Found {len(open_boundaries)} open boundary segment(s) "
            f"with {total_bnd_nodes} total nodes"
        )

        # Read node coordinates and extract open boundary points
        self._update_substep("Reading node coordinates")
        coords = _read_node_coordinates(hgrid_path, n_nodes)

        # Node IDs in hgrid.gr3 are 1-based
        bnd_node_ids = [nid for bnd in open_boundaries for nid in bnd]
        bnd_pts = coords[np.array(bnd_node_ids) - 1]

        # Compute concave hull
        self._update_substep("Computing domain hull")
        hull = shapely.concave_hull(shapely.MultiPoint(bnd_pts.tolist()), ratio=0.05)

        # Query NOAA stations within the hull
        self._update_substep("Querying NOAA CO-OPS stations")
        client = COOPSAPIClient()
        stations_gdf = client.stations_metadata
        selected = stations_gdf[stations_gdf.within(hull)]

        if selected.empty:
            raise RuntimeError(
                "No NOAA CO-OPS stations found within domain hull. "
                "Set include_noaa_gages to false if station output is not needed."
            )

        candidate_ids = selected["station_id"].tolist()

        # Filter to stations with valid MSL/MLLW datums so that
        # every station written to station.in can later be converted
        # from MLLW to MSL during the plotting stage.
        self._update_substep("Filtering stations by datum availability")
        valid_ids = client.filter_stations_by_datum(candidate_ids)

        dropped = set(candidate_ids) - valid_ids
        if dropped:
            self._log(
                f"Excluded {len(dropped)} station(s) without datum data: "
                f"{', '.join(sorted(dropped))}",
                "warning",
            )

        selected = selected[selected["station_id"].isin(sorted(valid_ids))]
        if selected.empty:
            raise RuntimeError(
                "No NOAA CO-OPS stations with valid datum data found within domain hull. "
                "Set include_noaa_gages to false if station output is not needed."
            )

        station_ids = selected["station_id"].tolist()
        lons = [row.geometry.x for _, row in selected.iterrows()]
        lats = [row.geometry.y for _, row in selected.iterrows()]

        # Write station.in and companion ID file
        self._update_substep("Writing station.in")
        _write_station_in(work_dir, lons, lats)
        _write_station_names(work_dir, station_ids)

        self._log(
            f"station.in written with {len(station_ids)} NOAA station(s): {', '.join(station_ids)}"
        )

        return {
            "status": "completed",
            "noaa_stations": len(station_ids),
            "station_ids": station_ids,
        }


class PreSCHISMStage(WorkflowStage):
    """Prepare input files for SCHISM execution."""

    name = "pre_schism"
    description = "Prepare SCHISM inputs (discharge, partitioning)"
    requires_container = True

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SchismModelConfig)  # noqa: S101
        self.model: SchismModelConfig = config.model_config

    def run(self) -> dict[str, Any]:
        """Execute SCHISM pre-processing."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._log("Running discharge, sink/source, and mesh partitioning")
        self._update_substep("Running pre_schism")
        script_path = self._get_scripts_dir() / "run_sing_coastal_workflow_pre_schism.bash"

        self.run_singularity_command(
            [str(script_path)],
            sif_path=self.config.model_config.singularity_image,
            env=env,
        )

        work_dir = self.config.paths.work_dir

        # Verify critical files produced by earlier stages exist
        required_files = ["source.nc", "param.nml", "hgrid.gr3"]
        missing = [f for f in required_files if not (work_dir / f).exists()]
        if missing:
            raise RuntimeError(
                f"pre_schism: required files missing from {work_dir}: {', '.join(missing)}. "
                "Check logs from earlier stages (initial_discharge, combine_sink_source, "
                "merge_source_sink, update_params) for errors."
            )

        # Patch param.nml to enable station output if station.in exists
        if self.model.include_noaa_gages and (work_dir / "station.in").exists():
            self._update_substep("Patching param.nml for station output")
            _patch_param_nml(work_dir / "param.nml")
            self._log("Set iout_sta = 1, nspool_sta = 18 in param.nml")

        self._log("SCHISM pre-processing complete")
        return {
            "partition_file": str(work_dir / "partition.prop"),
            "outputs_dir": str(work_dir / "outputs"),
            "status": "completed",
        }


class SCHISMRunStage(WorkflowStage):
    """Execute SCHISM model with MPI."""

    name = "schism_run"
    description = "Run SCHISM model (MPI)"
    requires_container = True

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SchismModelConfig)  # noqa: S101
        self.model: SchismModelConfig = config.model_config

    def run(self) -> dict[str, Any]:
        """Execute SCHISM model run."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._update_substep("Setting up MPI environment")
        nfs_mount = str(self.config.paths.nfs_mount)
        env["PATH"] = f"{nfs_mount}/openmpi/bin:{env.get('PATH', '/usr/local/bin:/usr/bin')}"
        env["LD_LIBRARY_PATH"] = f"{nfs_mount}/openmpi/lib:{env.get('LD_LIBRARY_PATH', '')}"
        env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
        env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"

        nscribes = self.model.nscribes
        exec_dir = env.get("EXECnwm", "")
        schism_binary = f"{exec_dir}/{self.model.binary}"

        self._log(
            f"Launching {self.model.binary} with {self.model.total_tasks} MPI tasks "
            f"({self.model.nodes} node(s), {self.model.nscribes} scribe(s))"
        )
        self._update_substep(f"Running pschism with {self.model.total_tasks} MPI tasks")

        self.run_singularity_command(
            ["/bin/bash", "-c", f"{schism_binary} {nscribes}"],
            sif_path=self.model.singularity_image,
            env=env,
            pwd=self.config.paths.work_dir,
            use_mpi=True,
            mpi_tasks=self.model.total_tasks,
        )

        self._log("SCHISM run completed successfully")
        return {
            "outputs_dir": str(self.config.paths.work_dir / "outputs"),
            "status": "completed",
        }


class PostSCHISMStage(WorkflowStage):
    """Post-process SCHISM outputs."""

    name = "post_schism"
    description = "Post-process SCHISM outputs"
    requires_container = True

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SchismModelConfig)  # noqa: S101
        self.model: SchismModelConfig = config.model_config

    def run(self) -> dict[str, Any]:
        """Execute SCHISM post-processing."""
        self._update_substep("Building environment")
        env = self.build_environment()

        self._update_substep("Checking for errors")
        fatal_error = self.config.paths.work_dir / "outputs" / "fatal.error"
        if fatal_error.exists() and fatal_error.stat().st_size > 0:
            error_content = fatal_error.read_text()[-2000:]
            raise RuntimeError(f"SCHISM run failed: {error_content}")

        self._log("Checking outputs and combining hotstarts")
        self._update_substep("Running post_schism")
        script_path = self._get_scripts_dir() / "run_sing_coastal_workflow_post_schism.bash"

        self.run_singularity_command(
            [str(script_path)],
            sif_path=self.model.singularity_image,
            env=env,
        )

        self._log("SCHISM post-processing complete")
        return {
            "outputs_dir": str(self.config.paths.work_dir / "outputs"),
            "status": "completed",
        }


class SchismPlotStage(WorkflowStage):
    """Plot simulated water levels against NOAA CO-OPS observations.

    After the SCHISM run, this stage reads ``staout_1`` (station elevation
    output), identifies stations from ``station_noaa_ids.txt``, fetches
    observed water levels from the NOAA CO-OPS API, and produces
    comparison time-series figures saved to ``<work_dir>/figs/``.

    Each figure contains up to 4 subplots in a 2x2 layout.  For domains
    with many stations, multiple figures are created.

    Observations are fetched in MLLW and converted to MSL using
    per-station datum offsets, matching SCHISM's vertical reference.

    Gated by ``include_noaa_gages`` on :class:`SchismModelConfig`.
    """

    name = "schism_plot"
    description = "Plot simulated vs observed water levels (SCHISM)"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SchismModelConfig)  # noqa: S101
        self.model: SchismModelConfig = config.model_config

    def _fetch_observations_msl(
        self,
        station_ids: list[str],
        begin_date: str,
        end_date: str,
    ) -> Any:
        """Fetch CO-OPS observations in MLLW and convert to MSL.

        Parameters
        ----------
        station_ids : list[str]
            NOAA CO-OPS station IDs.
        begin_date, end_date : str
            Query window formatted as ``%Y%m%d %H:%M``.

        Returns
        -------
        xr.Dataset
            Observed water levels with ``datum`` attribute set to ``MSL``.
        """
        from coastal_calibration.coops_api import COOPSAPIClient, query_coops_byids

        obs_ds = query_coops_byids(
            station_ids,
            begin_date,
            end_date,
            product="water_level",
            datum="MLLW",
            units="metric",
            time_zone="gmt",
        )

        # All stations reaching this point were pre-filtered by
        # SchismObservationStage to have valid MSL/MLLW datums.
        client = COOPSAPIClient()
        datums = client.get_datums(station_ids)

        datum_map = {d.station_id: d for d in datums}
        for sid in station_ids:
            d = datum_map.get(sid)
            if d is None:
                self._log(
                    f"Station {sid}: datum lookup failed unexpectedly, dropping from comparison",
                    "warning",
                )
                obs_ds.water_level.loc[{"station": sid}] = np.nan
                continue
            msl = d.get_datum_value("MSL")
            mllw = d.get_datum_value("MLLW")
            if msl is None or mllw is None:
                self._log(
                    f"Station {sid}: missing MSL/MLLW unexpectedly, dropping from comparison",
                    "warning",
                )
                obs_ds.water_level.loc[{"station": sid}] = np.nan
                continue
            offset = msl - mllw
            if d.units == "feet":
                offset *= 0.3048
            obs_ds.water_level.loc[{"station": sid}] -= offset
            self._log(f"Station {sid}: MLLW->MSL offset = {offset:.4f} m", "debug")

        obs_ds.attrs["datum"] = "MSL"
        return obs_ds

    @staticmethod
    def _plot_figures(
        sim_times: Any,
        sim_elevation: NDArray[np.float64],
        station_ids: list[str],
        obs_ds: Any,
        figs_dir: Path,
    ) -> list[Path]:
        """Create 2x2 comparison figures and save them.

        Stations that have *neither* valid observations nor valid
        simulated data are skipped entirely so that empty panels do
        not appear in the output.

        Parameters
        ----------
        sim_times : array-like
            Simulation datetimes.
        sim_elevation : ndarray
            Simulated elevation of shape (n_times, n_stations).
        station_ids : list[str]
            NOAA station IDs (one per column in ``sim_elevation``).
        obs_ds : xr.Dataset
            Observed water levels.
        figs_dir : Path
            Output directory for figures.

        Returns
        -------
        list[Path]
            Paths to the saved figures.
        """
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        # ── Pre-filter: keep only stations with at least obs or sim ──
        plotable = _plotable_stations(station_ids, sim_elevation, obs_ds)

        if not plotable:
            figs_dir.mkdir(parents=True, exist_ok=True)
            return []

        n_plotable = len(plotable)
        n_figures = math.ceil(n_plotable / _STATIONS_PER_FIGURE)
        figs_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        for fig_idx in range(n_figures):
            start = fig_idx * _STATIONS_PER_FIGURE
            end = min(start + _STATIONS_PER_FIGURE, n_plotable)
            batch = plotable[start:end]
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

    def run(self) -> dict[str, Any]:
        """Read SCHISM output, fetch NOAA observations, and plot comparison."""
        if not self.model.include_noaa_gages:
            self._log("include_noaa_gages is disabled, skipping")
            return {"status": "skipped"}

        work_dir = self.config.paths.work_dir

        # Check required files
        station_ids_file = work_dir / "station_noaa_ids.txt"
        staout_path = work_dir / "outputs" / "staout_1"

        if not station_ids_file.exists():
            self._log("station_noaa_ids.txt not found, skipping plot stage")
            return {"status": "skipped", "reason": "no station IDs file"}

        if not staout_path.exists():
            self._log("outputs/staout_1 not found, skipping plot stage")
            return {"status": "skipped", "reason": "no staout_1"}

        # Read station IDs
        station_ids = _read_station_noaa_ids(work_dir)
        if not station_ids:
            self._log("No station IDs found, skipping plot stage")
            return {"status": "skipped", "reason": "empty station IDs"}

        # Read SCHISM station output
        self._update_substep("Reading SCHISM station output")
        time_seconds, elevation = _read_staout(staout_path)

        if elevation.shape[1] != len(station_ids):
            self._log(
                f"Station count mismatch: staout_1 has {elevation.shape[1]} columns "
                f"but {len(station_ids)} station IDs",
                "warning",
            )
            # Use the minimum to avoid index errors
            n = min(elevation.shape[1], len(station_ids))
            elevation = elevation[:, :n]
            station_ids = station_ids[:n]

        # Convert simulation time to datetimes
        sim = self.config.simulation
        start_dt = sim.start_date
        sim_times = np.array(
            [start_dt + timedelta(seconds=float(t)) for t in time_seconds],
            dtype="datetime64[ns]",
        )

        # Fetch observed water levels (MLLW -> MSL)
        self._update_substep("Fetching NOAA CO-OPS observations")
        begin_date = start_dt.strftime("%Y%m%d %H:%M")
        end_dt = start_dt + timedelta(hours=sim.duration_hours)
        end_date = end_dt.strftime("%Y%m%d %H:%M")

        obs_ds = self._fetch_observations_msl(station_ids, begin_date, end_date)

        # Generate comparison plots (2x2 per figure)
        self._update_substep("Generating comparison plots")
        figs_dir = work_dir / "figs"
        fig_paths = self._plot_figures(sim_times, elevation, station_ids, obs_ds, figs_dir)

        self._log(f"Saved {len(fig_paths)} comparison figure(s) to {figs_dir}")

        return {
            "status": "completed",
            "figures": [str(p) for p in fig_paths],
            "figs_dir": str(figs_dir),
        }
