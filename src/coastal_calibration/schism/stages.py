"""SCHISM model execution stages."""

from __future__ import annotations

import os
import re
import shutil
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from coastal_calibration.base import WorkflowStage

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from coastal_calibration.config.schema import CoastalCalibConfig, SchismModelConfig
    from coastal_calibration.logging import WorkflowMonitor


def _write_station_in(
    base_dir: Path,
    lons: list[float],
    lats: list[float],
) -> Path:
    """Write a station.in file for SCHISM with multiple stations."""
    if len(lons) != len(lats):
        msg = f"Station lons/lats length mismatch: {len(lons)} vs {len(lats)}"
        raise ValueError(msg)
    n = len(lons)
    lines = [
        "1 0 0 0 0 0 0 0 0",  # only elevation output
        str(n),
    ]
    for i, (lon, lat) in enumerate(zip(lons, lats, strict=True), start=1):
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


def _read_station_lonlat(work_dir: Path) -> pd.DataFrame | None:
    """Parse ``station.in`` + ``station_noaa_ids.txt`` into an (id, lon, lat) frame.

    Returns ``None`` if either file is missing.
    """
    station_in = work_dir / "station.in"
    ids_file = work_dir / "station_noaa_ids.txt"
    if not station_in.is_file() or not ids_file.is_file():
        return None
    ids = _read_station_noaa_ids(work_dir)
    # station.in: header line, count line, then one `idx lon lat depth` per row.
    lines = [ln for ln in station_in.read_text().splitlines() if ln.strip()]
    lons: list[float] = []
    lats: list[float] = []
    for raw in lines[2 : 2 + len(ids)]:
        parts = raw.split()
        lons.append(float(parts[1]))
        lats.append(float(parts[2]))
    return pd.DataFrame({"id": ids, "lon": lons, "lat": lats})


def _chain_ring(segments: list[list[int]]) -> list[int]:
    """Chain boundary segments into a single ring of node IDs.

    Handles arbitrary segment order and orientation: each remaining
    segment is appended (or reversed) to extend the current ring at
    either end. Drops shared endpoints to avoid duplicate vertices.

    Subdomain meshes produced by ``extract_mesh`` write original
    boundary segments and new "cut" segments in arbitrary order and
    direction; this routine reconstructs the CCW perimeter ring from
    them without emitting "jump" segments between non-adjacent pieces.

    Kept in sync with ``_chain_outer_ring`` in
    ``qgis_plugin/nwm_coastal/plugin_main.py``.

    Returns the ring's node IDs in traversal order. An empty list
    is returned if no segments were provided.
    """
    remaining = [list(s) for s in segments if s]
    if not remaining:
        return []

    ring = remaining.pop(0)
    while remaining:
        last = ring[-1]
        first = ring[0]
        for i, seg in enumerate(remaining):
            if seg[0] == last:
                ring.extend(seg[1:])
                remaining.pop(i)
                break
            if seg[-1] == last:
                ring.extend(reversed(seg[:-1]))
                remaining.pop(i)
                break
            if seg[-1] == first:
                ring = list(seg) + ring[1:]
                remaining.pop(i)
                break
            if seg[0] == first:
                ring = list(reversed(seg)) + ring[1:]
                remaining.pop(i)
                break
        else:
            # No greedy match — fall back to concatenating the next
            # segment in file order. SCHISM hgrid files normally list
            # boundaries in CCW order, but some real meshes have an
            # off-by-one at the open/land transition (e.g. open ends at
            # node N and land starts at N+1, where the two nodes are
            # spatially adjacent but not identical). The pre-greedy
            # implementation handled this by always concatenating;
            # preserve that behavior so the polygon has a one-node
            # visual gap rather than ending the chain early.
            ring.extend(remaining.pop(0))

    return ring


def _build_domain_polygon(project: Any) -> Any:
    """Build an exact polygon representing the SCHISM mesh domain.

    Chains open + exterior land boundary segments into a single
    outer ring (counterclockwise) and uses island boundaries as
    interior holes.  This is faster and more accurate than a
    concave-hull approximation: ~15x speedup, exact geometry, and
    holes correctly excluded.

    Parameters
    ----------
    project : NWMSCHISMProject
        Loaded SCHISM project.

    Returns
    -------
    shapely.Polygon
        The mesh domain polygon (with holes for islands).
    """
    import shapely

    boundaries = project.read_boundaries()
    coords = project.nodes_coordinates

    ext_segments: list[list[int]] = list(boundaries.open_boundaries)
    ext_segments.extend(list(lb.nodes) for lb in boundaries.land_boundaries if lb.is_exterior)
    islands = [list(lb.nodes) for lb in boundaries.land_boundaries if lb.is_island]

    outer_ids = _chain_ring(ext_segments)
    if not outer_ids:
        raise RuntimeError("Failed to chain mesh boundary segments into a ring")

    outer_pts = coords[np.array(outer_ids) - 1].tolist()
    holes = [coords[np.array(isl) - 1].tolist() for isl in islands]
    return shapely.Polygon(outer_pts, holes=holes)


def _read_staout(staout_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read SCHISM station output file (staout_1).

    Returns
    -------
    time_seconds : ndarray
        Time in seconds from simulation start (empty if file has no data).
    elevation : ndarray
        Water elevation array of shape ``(n_times, n_stations)``
        (empty ``(0, 0)`` if file has no data).
    """
    data = np.loadtxt(staout_path, comments="!")
    if data.ndim < 2 or data.size == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64)
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
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

    def run(self) -> dict[str, Any]:
        """Discover stations and write station.in."""
        if not self.model.include_noaa_gages:
            self._log("include_noaa_gages is disabled, skipping")
            return {"status": "skipped"}

        from coastal_calibration.data.coops_api import COOPSAPIClient
        from coastal_calibration.schism import NWMSCHISMProject

        work_dir = self.config.paths.work_dir
        hgrid_path = work_dir / "hgrid.gr3"
        if not hgrid_path.exists():
            raise FileNotFoundError(
                f"hgrid.gr3 not found in {work_dir}. "
                "The schism_params stage must run before schism_obs "
                "so that the mesh file is symlinked into the work directory."
            )

        # Read mesh using NWMSCHISMProject
        self._update_substep("Reading hgrid.gr3 metadata")
        project = NWMSCHISMProject(work_dir, validate=False)
        self._log(f"hgrid.gr3: {project.n_nodes} nodes, {project.n_elements} elements")

        # Read boundaries
        self._update_substep("Reading boundaries")
        boundaries = project.read_boundaries()
        open_boundaries = boundaries.open_boundaries
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

        # Build an exact polygon from chained boundary segments.
        # This handles extracted subdomains where the land (coastline)
        # is part of the perimeter and islands must be excluded as holes.
        self._update_substep("Building domain polygon")
        hull = _build_domain_polygon(project)

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
        lons = selected.geometry.x.tolist()
        lats = selected.geometry.y.tolist()

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


class SchismDischargeStage(WorkflowStage):
    """Generate river discharge forcing for SCHISM.

    Stages NWM CHRTOUT files, creates initial discharge from
    the configured ``discharge_file`` (``nwmReaches.csv``), runs
    ``combine_sink_source``, and merges river discharge into
    precipitation sources via ``merge_source_sink``.

    Skipped when ``discharge_file`` is not set in the model config
    (matching the SFINCS ``SfincsDischargeStage`` pattern).

    Element areas for source thresholds are computed from the mesh
    via :class:`NWMSCHISMProject` rather than read from a file.
    """

    name = "schism_discharge"
    description = "Generate river discharge forcing"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

    def run(self) -> dict[str, Any]:
        """Stage streamflow data and produce ``source.nc``."""
        discharge_file = self.model.discharge_file
        if discharge_file is None:
            self._log("No discharge_file configured, skipping")
            return {"status": "skipped"}

        from coastal_calibration.schism import NWMSCHISMProject
        from coastal_calibration.schism.prep import (
            make_discharge,
            merge_source_sink,
            run_combine_sink_source,
            stage_chrtout_files,
        )

        work_dir = self.config.paths.work_dir
        sim = self.config.simulation
        paths = self.config.paths
        prebuilt_dir = self.model.coastal_parm

        # 1. Copy discharge file into work_dir as nwmReaches.csv
        self._update_substep("Staging discharge file")
        self._log(f"Copying discharge file: {discharge_file}")
        shutil.copy2(discharge_file, work_dir / "nwmReaches.csv")

        # 2. Stage CHRTOUT files
        end_date = sim.start_date + timedelta(hours=int(sim.duration_hours))

        if sim.meteo_source == "nwm_retro":
            # nwm_retro reads directly from S3 Zarr — no local files needed
            nwm_output_dir = work_dir  # not used but needed for signature
            nwm_ana_dir = None
        else:
            self._update_substep("Staging CHRTOUT files")
            self._log("Symlinking NWM CHRTOUT files")
            nwm_output_dir, nwm_ana_dir = stage_chrtout_files(
                work_dir=work_dir,
                start_date=sim.start_date,
                duration_hours=int(sim.duration_hours),
                coastal_domain=sim.coastal_domain,
                streamflow_dir=paths.streamflow_dir(sim.meteo_source, sim.coastal_domain),
            )

        # 3. Generate discharge files from NWM CHRT output
        self._update_substep("Generating discharge files")
        self._log("Running make_discharge")
        make_discharge(
            work_dir=work_dir,
            nwm_output_dir=nwm_output_dir,
            nwm_ana_dir=nwm_ana_dir,
            is_analysis="analysis" in sim.meteo_source,
            meteo_source=sim.meteo_source,
            domain=sim.coastal_domain,
            start_date=sim.start_date,
            end_date=end_date,
        )

        # 4. Combine sink/source (Fortran binary)
        self._update_substep("Combining sink/source")
        self._log("Running combine_sink_source")
        run_combine_sink_source(work_dir)

        # 5. Merge source/sink into precipitation data
        # Element areas computed from mesh via NWMSCHISMProject
        self._update_substep("Merging source/sink")
        self._log("Running merge_source_sink")
        project = NWMSCHISMProject(prebuilt_dir, validate=False)
        merge_source_sink(
            work_dir=work_dir,
            element_areas=project.element_areas,
            prebuilt_dir=prebuilt_dir,
        )

        # Verify source.nc was produced
        source_nc = work_dir / "source.nc"
        if not source_nc.exists():
            raise RuntimeError(
                f"Discharge stage: source.nc not produced in {work_dir}. "
                "Check logs above for errors."
            )

        self._log("Discharge generation complete")
        return {
            "source_nc": str(source_nc),
            "status": "completed",
        }


class PreSCHISMStage(WorkflowStage):
    """Partition mesh and finalize SCHISM inputs.

    Runs mesh partitioning (``metis_prep`` + ``gpmetis``) and patches
    ``param.nml`` for station output.  All Fortran binaries are expected
    on ``$PATH`` (pixi installs them to ``$CONDA_PREFIX/bin``).
    """

    name = "schism_prep"
    description = "Partition mesh and finalize inputs"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

    def run(self) -> dict[str, Any]:
        """Partition mesh and finalize SCHISM inputs."""
        from coastal_calibration.schism.prep import partition_mesh

        work_dir = self.config.paths.work_dir
        (work_dir / "outputs").mkdir(parents=True, exist_ok=True)

        # 1. Partition mesh
        self._update_substep("Partitioning mesh")
        self._log(
            f"Partitioning for {self.model.total_tasks} tasks ({self.model.nscribes} scribes)"
        )
        partition_mesh(
            work_dir=work_dir,
            total_tasks=self.model.total_tasks,
            nscribes=self.model.nscribes,
        )

        # 2. Verify critical files exist
        required_files = ["param.nml", "hgrid.gr3"]
        if self.model.discharge_file is not None:
            required_files.append("source.nc")
        missing = [f for f in required_files if not (work_dir / f).exists()]
        if missing:
            raise RuntimeError(
                f"pre_schism: required files missing from {work_dir}: "
                f"{', '.join(missing)}. Check logs above for errors."
            )

        # 3. Patch param.nml to enable station output if station.in exists
        if self.model.include_noaa_gages and (work_dir / "station.in").exists():
            self._update_substep("Patching param.nml for station output")
            _patch_param_nml(work_dir / "param.nml")
            self._log("Set iout_sta = 1, nspool_sta = 18 in param.nml")

        # 4. Validate the final param.nml against SCHISM's startup
        #    constraints so misconfigurations surface here with a clear
        #    message instead of a Fortran abort during schism_run.
        from coastal_calibration.schism.prep import validate_param_nml

        param_errors = validate_param_nml(work_dir / "param.nml")
        if param_errors:
            raise RuntimeError("param.nml validation failed:\n  - " + "\n  - ".join(param_errors))

        self._log("SCHISM pre-processing complete")
        return {
            "partition_file": str(work_dir / "partition.prop"),
            "outputs_dir": str(work_dir / "outputs"),
            "status": "completed",
        }


class SCHISMRunStage(WorkflowStage):
    """Execute SCHISM model with MPI.

    The ``pschism`` binary is expected on ``$PATH`` (pixi installs it
    to ``$CONDA_PREFIX/bin``).  MPI is provided by the pixi ``openmpi``
    package.
    """

    name = "schism_run"
    description = "Run SCHISM model (MPI)"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

    @staticmethod
    def _resolve_exe(schism_cfg: SchismModelConfig) -> Path:
        """Return the SCHISM executable or raise with a helpful message."""
        if schism_cfg.schism_exe is not None:
            exe = schism_cfg.schism_exe
            if not exe.is_file():
                msg = f"schism_exe not found: {exe}"
                raise RuntimeError(msg)
            if not os.access(exe, os.X_OK):
                msg = f"schism_exe is not executable: {exe}"
                raise RuntimeError(msg)
            return exe
        found = shutil.which("pschism")
        if found is not None:
            return Path(found)
        raise RuntimeError(
            "SCHISM executable not found.  Either:\n"
            "  1. Activate a pixi environment with the 'schism' feature"
            " (builds automatically on first activation).\n"
            "  2. Set 'schism_exe' in the config to the path of an existing binary."
        )

    def _build_mpi_command(self, exe: Path, env: dict[str, str] | None = None) -> list[str]:
        """Construct the ``mpiexec … pschism N`` command list."""
        from coastal_calibration.utils import build_mpi_cmd

        cmd = build_mpi_cmd(self.model.total_tasks, oversubscribe=self.model.oversubscribe, env=env)
        cmd.extend([str(exe), str(self.model.nscribes)])
        return cmd

    def run(self) -> dict[str, Any]:
        """Execute SCHISM model run."""
        import subprocess

        exe = self._resolve_exe(self.model)

        self._update_substep("Building environment")
        if self.model.schism_exe is not None:
            from coastal_calibration.utils import build_isolated_env

            env = build_isolated_env(
                omp_num_threads=self.model.omp_num_threads,
                runtime_env=self.model.runtime_env,
            )
        else:
            env = self.build_environment()
            if self.model.runtime_env:
                env.update(self.model.runtime_env)

        self._log(
            f"Launching {exe.name} with {self.model.total_tasks} MPI tasks "
            f"({self.model.nodes} node(s), {self.model.nscribes} scribe(s))"
        )
        self._update_substep(f"Running {exe.name} with {self.model.total_tasks} MPI tasks")

        cmd = self._build_mpi_command(exe, env)
        self._log(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=self.config.paths.work_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            stderr = result.stderr or ""
            self._log(f"SCHISM run failed: {stderr[-2000:]}", "error")
            raise RuntimeError(f"SCHISM run failed (exit {result.returncode}): {stderr[-2000:]}")

        self._log("SCHISM run completed successfully")
        return {
            "outputs_dir": str(self.config.paths.work_dir / "outputs"),
            "status": "completed",
        }


class PostSCHISMStage(WorkflowStage):
    """Post-process SCHISM outputs.

    Checks for fatal errors and combines hotstart files when running
    reanalysis or chained runs.  ``combine_hotstart7`` is expected on
    ``$PATH`` (pixi installs it to ``$CONDA_PREFIX/bin``).
    """

    name = "schism_postprocess"
    description = "Post-process SCHISM outputs"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

    def run(self) -> dict[str, Any]:
        """Execute SCHISM post-processing."""
        work_dir = self.config.paths.work_dir
        outputs_dir = work_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        self._update_substep("Checking for errors")
        fatal_error = outputs_dir / "fatal.error"
        if fatal_error.exists() and fatal_error.stat().st_size > 0:
            error_content = fatal_error.read_text()
            # SCHISM writes dry-node diagnostics (QUICKSEARCH) to
            # fatal.error even on successful runs.  Only treat lines
            # that do NOT match known non-fatal patterns as true errors.
            non_fatal_patterns = ("QUICKSEARCH",)
            true_errors = [
                line
                for line in error_content.splitlines()
                if line.strip() and not any(p in line for p in non_fatal_patterns)
            ]
            if true_errors:
                raise RuntimeError(f"SCHISM run failed: {error_content[-2000:]}")
            self._log("fatal.error contains only dry-node warnings (QUICKSEARCH); continuing")

        # Combine hotstarts for reanalysis / chained runs
        sim = self.config.simulation
        is_reanalysis = sim.duration_hours < 0
        if is_reanalysis:
            self._update_substep("Combining hotstarts")
            self._log("Running combine_hotstart7")
            from coastal_calibration.schism.prep import combine_hotstart

            combine_hotstart(outputs_dir)

        self._log("SCHISM post-processing complete")
        return {
            "outputs_dir": str(outputs_dir),
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
        self.model: SchismModelConfig = cast("SchismModelConfig", config.model_config)

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
        from coastal_calibration.data.coops_api import COOPSAPIClient, query_coops_byids

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

    def _per_station_datum_offsets(
        self,
        station_ids: list[str],
        corr_path: Path,
    ) -> NDArray[Any] | None:
        """Compute per-station mesh→MSL offsets, or None if unavailable.

        Reads ``elevation_correction.csv`` and, for each station ID,
        looks up its lon/lat from CO-OPS metadata and uses the nearest
        boundary-node ``conversion_factor`` as the local offset. Stations
        not found in CO-OPS fall back to the boundary mean. Returns
        ``None`` (with a warning) if the CSV columns are malformed.
        """
        try:
            corr_df = pd.read_csv(corr_path)
            corr_lons = corr_df["Field1"].to_numpy()
            corr_lats = corr_df["Field2"].to_numpy()
            conv_factors = corr_df["conversion_factor"].to_numpy()
        except (KeyError, ValueError) as exc:
            self._log(
                f"Could not read elevation correction CSV: {exc}; "
                "comparison may show a datum offset",
                "warning",
            )
            return None
        fallback_conv = float(conv_factors.mean()) if conv_factors.size else 0.0

        from coastal_calibration.data.coops_api import COOPSAPIClient

        client = COOPSAPIClient()
        md = client.stations_metadata
        md_subset = md[md["station_id"].astype(str).isin(station_ids)]
        md_by_id = md_subset.set_index(md_subset["station_id"].astype(str))
        # Vectorised coordinate access via geopandas's .x/.y on the
        # geometry column avoids per-row scalar indexing issues.
        sta_lons = md_by_id.geometry.x.to_dict()
        sta_lats = md_by_id.geometry.y.to_dict()

        per_station_conv = np.full(len(station_ids), fallback_conv)
        for i, sid in enumerate(station_ids):
            if sid in sta_lons:
                slon = float(sta_lons[sid])
                slat = float(sta_lats[sid])
                d2 = (corr_lons - slon) ** 2 + (corr_lats - slat) ** 2
                per_station_conv[i] = float(conv_factors[int(np.argmin(d2))])
            else:
                self._log(
                    f"Station {sid}: no metadata; using boundary mean "
                    f"{fallback_conv:+.4f} m for datum correction",
                    "debug",
                )
        return per_station_conv

    def _load_station_simulation(
        self, work_dir: Path
    ) -> tuple[list[str], NDArray[Any], NDArray[Any]] | None:
        """Read station IDs + ``staout_1`` and return (ids, sim_times, elev_msl).

        Returns ``None`` when prerequisite files are missing — the caller
        should surface that as a skipped substep rather than an error.
        """
        station_ids_file = work_dir / "station_noaa_ids.txt"
        staout_path = work_dir / "outputs" / "staout_1"

        if not station_ids_file.exists():
            self._log("station_noaa_ids.txt not found, skipping station comparison")
            return None
        if not staout_path.exists():
            self._log("outputs/staout_1 not found, skipping station comparison")
            return None

        station_ids = _read_station_noaa_ids(work_dir)
        if not station_ids:
            self._log("No station IDs found, skipping station comparison")
            return None

        self._update_substep("Reading SCHISM station output")
        time_seconds, elevation = _read_staout(staout_path)

        if elevation.size == 0:
            self._log("staout_1 is empty, skipping station comparison")
            return None

        if elevation.shape[1] != len(station_ids):
            msg = (
                f"Station count mismatch: staout_1 has {elevation.shape[1]} "
                f"columns but station_noaa_ids.txt has {len(station_ids)} IDs. "
                "This indicates a corrupted run or a station.in / "
                "station_noaa_ids.txt that is out of sync with the SCHISM run."
            )
            raise RuntimeError(msg)

        # Apply per-station datum correction so simulated values are in MSL.
        corr_path = self.model.elevation_correction_csv
        if corr_path is not None:
            offsets = self._per_station_datum_offsets(station_ids, corr_path)
            if offsets is not None:
                elevation = elevation + offsets[None, :]
                self._log(
                    "Applied per-station datum offset to simulated values "
                    f"(range {offsets.min():+.4f} to {offsets.max():+.4f} m, "
                    f"mean {offsets.mean():+.4f} m)"
                )

        start_ns = np.datetime64(self.config.simulation.start_date, "ns")
        sim_times = start_ns + (time_seconds * 1e9).astype("timedelta64[ns]")
        return station_ids, sim_times, elevation

    def _run_station_comparison(self, figs_dir: Path) -> dict[str, Any]:
        """Produce CO-OPS station comparison figures. No-op if inputs missing.

        This path is gated by ``include_noaa_gages`` on
        :class:`SchismModelConfig` in :meth:`run`; the helper itself assumes
        the caller has already checked the gate.
        """
        loaded = self._load_station_simulation(self.config.paths.work_dir)
        if loaded is None:
            return {"status": "skipped"}
        station_ids, sim_times, elevation = loaded

        sim = self.config.simulation
        self._update_substep("Fetching NOAA CO-OPS observations")
        begin_date = sim.start_date.strftime("%Y%m%d %H:%M")
        end_dt = sim.start_date + timedelta(hours=sim.duration_hours)
        end_date = end_dt.strftime("%Y%m%d %H:%M")

        obs_ds = self._fetch_observations_msl(station_ids, begin_date, end_date)

        self._update_substep("Generating comparison plots")
        from coastal_calibration.plotting import plot_station_comparison

        fig_paths = plot_station_comparison(
            {"Simulated": (sim_times, elevation)},
            station_ids,
            figs_dir,
            obs_ds=obs_ds,
        )
        self._log(f"Saved {len(fig_paths)} comparison figure(s) to {figs_dir}")
        return {
            "status": "completed",
            "figures": [str(p) for p in fig_paths],
        }

    def _run_obs_points(self, work_dir: Path) -> dict[str, Any]:
        """Extract water level at user and NOAA obs points; write parquet."""
        from coastal_calibration.observations import (
            combine_obs_points,
            extract_water_level_series,
            load_obs_points,
            validate_points_in_domain,
        )
        from coastal_calibration.schism.outputs import load_schism_elevation

        outputs_dir = work_dir / "outputs"
        if not outputs_dir.is_dir() or not any(outputs_dir.glob("out2d_*.nc")):
            self._log("No out2d_*.nc blocks, skipping obs-points extraction")
            return {"status": "skipped", "reason": "no out2d_*.nc"}

        user_points: pd.DataFrame | None = None
        if self.model.obs_points_csv is not None:
            user_points = load_obs_points(self.model.obs_points_csv)
            self._log(f"Loaded {len(user_points)} user observation point(s)")

        noaa_points = _read_station_lonlat(work_dir) if self.model.include_noaa_gages else None
        if noaa_points is not None:
            self._log(f"Including {len(noaa_points)} NOAA gauge(s) from station.in")

        all_points = combine_obs_points(user_points, noaa_points)
        if all_points.empty:
            self._log("No obs points to extract (neither user CSV nor NOAA gauges active)")
            return {"status": "skipped", "reason": "no points"}

        self._update_substep("Extracting water level at obs points")
        ds = load_schism_elevation(work_dir, correction_file=self.model.elevation_correction_csv)
        validate_points_in_domain(all_points, ds)
        series = extract_water_level_series(ds, all_points, variable="elevation")

        outfile = work_dir / "obs_water_level.parquet"
        series.to_parquet(outfile)
        self._log(
            f"Wrote {outfile.name} with {len(series.columns)} station(s) "
            f"and {len(series)} timestep(s)"
        )
        return {
            "status": "completed",
            "path": str(outfile),
            "n_points": len(series.columns),
            "n_user": 0 if user_points is None else len(user_points),
            "n_noaa": 0 if noaa_points is None else len(noaa_points),
        }

    def _run_water_level_animation(self, figs_dir: Path) -> dict[str, Any]:
        """Render an MP4/GIF animation from the SCHISM ``out2d_*.nc`` output."""
        work_dir = self.config.paths.work_dir
        outputs_dir = work_dir / "outputs"
        blocks = sorted(outputs_dir.glob("out2d_*.nc")) if outputs_dir.is_dir() else []
        if not blocks:
            self._log("No out2d_*.nc blocks found, skipping water-level animation")
            return {"status": "skipped", "reason": "no out2d_*.nc"}

        self._update_substep("Rendering water-level animation")
        from coastal_calibration.plotting import animate_water_level
        from coastal_calibration.schism.outputs import load_schism_elevation

        ds = load_schism_elevation(work_dir, correction_file=self.model.elevation_correction_csv)
        outfile = figs_dir / "water_level.mp4"
        animate_water_level(
            ds,
            outfile,
            variable="elevation",
            fps=self.model.animation_fps,
            time_stride=self.model.animation_time_stride,
        )
        self._log(f"Saved water-level animation to {outfile}")
        return {"status": "completed", "animation": str(outfile)}

    def run(self) -> dict[str, Any]:
        """Produce SCHISM post-run figures: station comparison + animation.

        Either sub-task can be independently disabled via the
        ``include_noaa_gages`` / ``create_water_level_animation`` fields on
        :class:`SchismModelConfig`. When both are disabled the stage is a
        no-op.
        """
        work_dir = self.config.paths.work_dir
        figs_dir = work_dir / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)

        want_stations = self.model.include_noaa_gages
        want_animation = self.model.create_water_level_animation
        want_obs_parquet = self.model.obs_points_csv is not None or self.model.include_noaa_gages

        if not (want_stations or want_animation or want_obs_parquet):
            self._log(
                "No post-run products requested (set include_noaa_gages, "
                "create_water_level_animation, or obs_points_csv), skipping"
            )
            return {"status": "skipped"}

        result: dict[str, Any] = {"status": "completed", "figs_dir": str(figs_dir)}

        if want_stations:
            result["stations"] = self._run_station_comparison(figs_dir)
        if want_animation:
            result["animation"] = self._run_water_level_animation(figs_dir)
        if want_obs_parquet:
            result["obs_points"] = self._run_obs_points(work_dir)

        return result
