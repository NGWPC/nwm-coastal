"""SFINCS model build stages using HydroMT-SFINCS Python API.

All stages subclass :class:`~coastal_calibration.stages.base.WorkflowStage`
and accept a :class:`~coastal_calibration.config.schema.CoastalCalibConfig`.
SFINCS-specific settings are read from ``config.model_config``
(:class:`~coastal_calibration.config.schema.SfincsModelConfig`).

The HydroMT ``SfincsModel`` instance is shared between stages via a
module-level registry keyed by config ``id``.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from coastal_calibration.config.schema import SfincsModelConfig
from coastal_calibration.stages.base import WorkflowStage
from coastal_calibration.stages.sfincs import create_nc_symlinks, generate_data_catalog

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr
    from hydromt_sfincs import SfincsModel  # pyright: ignore[reportMissingImports]
    from numpy.typing import NDArray

    from coastal_calibration.config.schema import CoastalCalibConfig
    from coastal_calibration.utils.logging import WorkflowMonitor

SFINCS_DOCKER_IMAGE = "deltares/sfincs-cpu"
# Maximum number of stations per figure (2x2 layout).
_STATIONS_PER_FIGURE = 4


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


# ---------------------------------------------------------------------------
# Shared model instance management
# ---------------------------------------------------------------------------
# The SFINCS build stages share a HydroMT ``SfincsModel`` between them.
# We use a module-level dictionary keyed by the ``CoastalCalibConfig`` id to
# store the model instance across stages within a single runner invocation.
#
# When ``--start-from`` skips ``sfincs_init``, the registry is empty.  In
# that case ``_get_model`` lazily re-opens the model from disk (the same
# ``r+`` open + read that ``SfincsInitStage`` performs) so that mid-pipeline
# restarts work transparently.
_MODEL_REGISTRY: dict[int, SfincsModel] = {}


def _set_model(config: CoastalCalibConfig, model: SfincsModel) -> None:
    """Store the SfincsModel instance for the given config."""
    _MODEL_REGISTRY[id(config)] = model


# ---------------------------------------------------------------------------
# Helper — resolve paths from CoastalCalibConfig
# ---------------------------------------------------------------------------


def _sfincs_cfg(config: CoastalCalibConfig) -> SfincsModelConfig:
    """Return the SFINCS model config, raising if not the active model."""
    if not isinstance(config.model_config, SfincsModelConfig):
        raise TypeError("model_config must be SfincsModelConfig when model='sfincs'")
    return config.model_config


def get_model_root(config: CoastalCalibConfig) -> Path:
    """Effective model output directory."""
    sfincs = _sfincs_cfg(config)
    return sfincs.model_root or (config.paths.work_dir / "sfincs_model")


def _data_catalog_path(config: CoastalCalibConfig) -> Path | None:
    """Return catalog path if it exists on disk, else None."""
    candidate = config.paths.work_dir / "data_catalog.yml"
    return candidate if candidate.exists() else None


def _get_model(config: CoastalCalibConfig) -> SfincsModel:
    """Retrieve — or lazily re-open — the SfincsModel for *config*.

    On a fresh run the model is populated by ``SfincsInitStage``.  When
    the runner restarts mid-pipeline (``--start-from``), the registry is
    empty.  In that case we re-open the model from ``model_root`` in the
    same way ``SfincsInitStage.run`` does (``mode="r+"``, then ``read()``).
    """
    try:
        return _MODEL_REGISTRY[id(config)]
    except KeyError:
        pass

    # Lazy re-init from disk
    root = get_model_root(config)
    inp_file = root / "sfincs.inp"
    if not inp_file.exists():
        raise RuntimeError(
            "SFINCS model not initialised and no sfincs.inp found at "
            f"{root}.  Ensure the 'sfincs_init' stage runs first."
        )

    from hydromt_sfincs import SfincsModel as _Sfincs  # pyright: ignore[reportMissingImports]

    data_libs: list[str] = []
    catalog_path = _data_catalog_path(config)
    if catalog_path is not None:
        data_libs.append(str(catalog_path))

    model = _Sfincs(data_libs=data_libs, root=str(root), mode="r+", write_gis=True)
    model.read()
    _MODEL_REGISTRY[id(config)] = model
    return model


def _clear_model(config: CoastalCalibConfig) -> None:
    """Remove the SfincsModel instance for the given config."""
    _MODEL_REGISTRY.pop(id(config), None)


def _meteo_dst_res(config: CoastalCalibConfig, model: SfincsModel) -> float:
    """Return the output resolution (m) for gridded meteo forcing.

    When the user specifies ``meteo_res`` in the configuration it is
    returned directly.  Otherwise the base (coarsest) cell size of the
    SFINCS quadtree grid is used: meteorological fields vary slowly in
    space, so there is no benefit to a meteo grid finer than the
    coarsest computational cell.

    This also avoids the LCC → UTM reprojection inflation problem where
    ``rioxarray.reproject`` produces an output grid spanning the entire
    CONUS extent when no target resolution is specified.
    """
    sfincs = _sfincs_cfg(config)
    if sfincs.meteo_res is not None:
        return sfincs.meteo_res

    # Read the base cell size from the quadtree grid stored in sfincs.nc.
    # The grid structure stores ``dx`` and ``dy`` as attributes of the
    # mesh topology variable, or can be inferred from sfincs.inp.
    dx = float(model.config.get("dx", 0))
    dy = float(model.config.get("dy", 0))
    if dx > 0 and dy > 0:
        return max(dx, dy)

    # Quadtree grids: ``dx`` / ``dy`` are not always in the config but
    # the grid NetCDF records them.  Fall back to the grid dataset.
    try:
        grid_ds = model.components["quadtree_grid"].data
        dx = float(grid_ds.attrs.get("dx", 0))
        dy = float(grid_ds.attrs.get("dy", 0))
        if dx > 0 and dy > 0:
            return max(dx, dy)
    except Exception:  # noqa: S110
        pass

    # Absolute fallback: NWM native resolution (~1 km).
    return 1000.0


def _clip_meteo_to_domain(
    model: SfincsModel,
    component_name: str,
    buffer: float = 10_000.0,
) -> tuple[int, int]:
    """Clip a gridded meteo component to the model domain.

    After ``rioxarray.reproject`` converts NWM data from Lambert
    Conformal Conic to UTM, the output grid can cover the entire CONUS
    extent, producing multi-GB files that cripple the SFINCS simulation.
    This helper uses :meth:`rioxarray.Dataset.rio.clip_box` to trim the
    named component to the model region bounds plus a generous buffer so
    that only the data the SFINCS binary actually reads is kept on disk.

    Parameters
    ----------
    model : SfincsModel
        The HydroMT SFINCS model with meteo data already loaded.
    component_name : str
        Name of the meteo component (``"precipitation"``, ``"wind"``,
        or ``"pressure"``).
    buffer : float
        Extra metres around the model region bounds to include.
        Default 10 km gives SFINCS comfortable headroom for
        interpolation near the boundary.

    Returns
    -------
    (nx_before, nx_after)
        Number of x-cells before and after clipping, for logging.
    """
    component = getattr(model, component_name)
    ds: xr.Dataset = component.data  # type: ignore[assignment]
    nx_before = ds.sizes.get("x", 0)

    # Determine the clip bounds in UTM from the model grid.
    region_bounds = model.region.total_bounds  # (xmin, ymin, xmax, ymax)
    minx = region_bounds[0] - buffer
    miny = region_bounds[1] - buffer
    maxx = region_bounds[2] + buffer
    maxy = region_bounds[3] + buffer

    # rio.clip_box handles CRS-aware clipping and automatically deals
    # with descending y-coordinates, unlike a raw ds.sel() slice.
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activate .rio accessor

    # Ensure the CRS is set — hydromt's reproject → rename → set chain
    # can lose the spatial_ref coordinate.
    if ds.rio.crs is None:
        ds = ds.rio.write_crs(model.crs)

    clipped = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    nx_after = clipped.sizes.get("x", 0)

    if nx_after > 0 and clipped.sizes.get("y", 0) > 0:
        # Replace the in-memory data with the clipped version.
        # We assign ``_data`` directly because ``set()`` checks that
        # spatial coordinates match, which would fail when the new
        # data has different dimensions.
        component._data = clipped

    return nx_before, nx_after


def _waterlevel_geodataset(config: CoastalCalibConfig) -> str | None:
    """Return the geodataset name for water-level forcing, or None."""
    catalog_path = _data_catalog_path(config)
    if catalog_path is None:
        return None
    coastal_source = config.boundary.source
    return f"{coastal_source}_waterlevel" if coastal_source != "tpxo" else "tpxo_tidal"


# ---------------------------------------------------------------------------
# Singularity helpers (used by the run stage)
# ---------------------------------------------------------------------------


def resolve_sif_path(config: CoastalCalibConfig) -> Path:
    """Resolve the Singularity SIF path from configuration.

    When no explicit ``container_image`` is set, the SIF file is stored
    in the download directory so that it can be reused across runs
    without re-downloading.
    """
    sfincs = _sfincs_cfg(config)
    if sfincs.container_image is not None:
        return sfincs.container_image
    return config.paths.download_dir / f"sfincs-cpu_{sfincs.container_tag}.sif"


def _pull_singularity_image(
    config: CoastalCalibConfig,
    sif_path: Path | None = None,
) -> Path:
    """Pull the SFINCS Docker image as a Singularity SIF file."""
    if sif_path is None:
        sif_path = resolve_sif_path(config)

    if sif_path.exists():
        return sif_path

    sif_path.parent.mkdir(parents=True, exist_ok=True)
    sfincs = _sfincs_cfg(config)
    docker_uri = f"docker://{SFINCS_DOCKER_IMAGE}:{sfincs.container_tag}"
    cmd = ["singularity", "pull", str(sif_path), docker_uri]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"singularity pull failed: {result.stderr}")

    return sif_path


def _run_and_log(
    cmd: list[str],
    model_root: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, stream output to ``sfincs_log.txt``, and raise on failure.

    This is the shared subprocess helper used by both the Singularity and
    native-executable code paths.

    Parameters
    ----------
    cmd : list[str]
        Command and arguments to execute.
    model_root : Path
        Working directory for the process *and* location of the log file.
    env : dict, optional
        Environment variables.  ``None`` inherits the current environment.
    """
    log_path = model_root / "sfincs_log.txt"
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    with subprocess.Popen(
        cmd,
        cwd=model_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        with log_path.open("w") as f:
            assert proc.stdout is not None  # noqa: S101
            assert proc.stderr is not None  # noqa: S101
            for line in proc.stdout:
                stdout_lines.append(line)
                f.write(line)
            for line in proc.stderr:
                stderr_lines.append(line)
                f.write(line)
        proc.wait()

    if proc.returncode == 127:
        raise RuntimeError(f"{cmd[0]!r} not found. Make sure it is installed and on PATH.")
    if proc.returncode != 0:
        tail_stdout = "".join(stdout_lines[-20:]).rstrip()
        tail_stderr = "".join(stderr_lines[-20:]).rstrip()
        detail = ""
        if tail_stderr:
            detail += f"\n--- stderr (last 20 lines) ---\n{tail_stderr}"
        if tail_stdout:
            detail += f"\n--- stdout (last 20 lines) ---\n{tail_stdout}"
        if not detail:
            detail = f"\n(no output captured -- check {log_path})"
        raise RuntimeError(f"SFINCS run failed with return code {proc.returncode}{detail}")

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
    )


def _run_singularity(
    config: CoastalCalibConfig,
    sif_path: Path,
    model_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run SFINCS inside a Singularity container."""
    if model_root is None:
        model_root = get_model_root(config)

    cmd = ["singularity", "run", f"-B{model_root}:/data", str(sif_path)]
    return _run_and_log(cmd, model_root)


# ---------------------------------------------------------------------------
# SFINCS workflow stages  (all subclass WorkflowStage)
# ---------------------------------------------------------------------------


class _SfincsStageBase(WorkflowStage):
    """Common base for SFINCS stages that need ``self.sfincs``.

    Avoids repeating the ``isinstance`` assertion and attribute
    assignment in every stage ``__init__``.
    """

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        assert isinstance(config.model_config, SfincsModelConfig)  # noqa: S101
        self.sfincs: SfincsModelConfig = config.model_config


class SfincsInitStage(_SfincsStageBase):
    """Initialise the SFINCS model (pre-built mode).

    Copies pre-built model files to the output directory, opens the model
    in ``r+`` mode, reads it, and clears missing file references.
    """

    name = "sfincs_init"
    description = "Initialise SFINCS model (pre-built)"

    #: Config attributes that reference external files.  When a pre-built
    #: ``sfincs.inp`` lists placeholder names for files that haven't been
    #: generated yet, we clear them so HydroMT won't fail on read/write.
    _FILE_REF_ATTRS: tuple[str, ...] = (
        # Meteorological ASCII forcing files
        "amufile",
        "amvfile",
        "ampfile",
        "amprfile",
        # Other optional file references
        "sbgfile",
        "srcfile",
        "disfile",
        "bzsfile",
        "bzifile",
        "precipfile",
        "prcfile",
        "wndfile",
        "spwfile",
        "inifile",
        "rstfile",
        "ncinifile",
        "weirfile",
        "thdfile",
        "drnfile",
        "scsfile",
        "netspwfile",
        "netsrcdisfile",
        "netbndbzsbzifile",
        # NetCDF forcing files (gridded meteo)
        "netamprfile",
        "netampfile",
        "netamuamvfile",
    )

    #: Generated NetCDF files that downstream stages will recreate.
    #: Stale files from a previous run can crash the HDF5 library when
    #: ``write_netcdf_safely`` (or a lazy ``model.read()``) tries to open
    #: them, so we delete them at the start of every fresh initialisation.
    _GENERATED_NC_FILES: tuple[str, ...] = (
        "sfincs_netbndbzsbzifile.nc",
        "sfincs_netamuv.nc",
        "sfincs_netamp.nc",
        "sfincs_netampr.nc",
        "sfincs_netsrcdisfile.nc",
        # SFINCS output files (not strictly forcing but can interfere)
        "sfincs_map.nc",
        "sfincs_his.nc",
    )

    def _remove_stale_outputs(self, root: Path) -> None:
        """Delete generated netCDF files left over from a previous run.

        The HydroMT ``write_netcdf_safely`` helper opens existing files to
        check whether the data changed.  If a previous run produced files
        with an incompatible schema (e.g. different number of stations)
        the HDF5 library can segfault.  Removing them up front avoids the
        issue entirely; every downstream stage will regenerate its files.
        """
        removed: list[str] = []
        for name in self._GENERATED_NC_FILES:
            p = root / name
            if p.exists():
                p.unlink()
                removed.append(name)
        if removed:
            self._log(f"Removed stale output files: {', '.join(removed)}")

    def _clear_missing_file_refs(self, model: SfincsModel) -> None:
        """Clear config references to files that do not exist on disk."""
        model_root = model.root.path
        cfg_data = model.config.data
        cleared: list[str] = []
        for attr in self._FILE_REF_ATTRS:
            val = getattr(cfg_data, attr, None)
            if val and not (model_root / val).exists():
                setattr(cfg_data, attr, None)
                cleared.append(f"{attr}={val}")
        if cleared:
            self._log(f"Cleared missing file references: {', '.join(cleared)}")

    def run(self) -> dict[str, Any]:
        """Load a pre-built SFINCS model and register it for subsequent stages."""
        from hydromt_sfincs import SfincsModel  # pyright: ignore[reportMissingImports]

        root = get_model_root(self.config)

        data_libs: list[str] = []
        catalog_path = _data_catalog_path(self.config)
        if catalog_path is not None:
            data_libs.append(str(catalog_path))

        self._update_substep("Loading pre-built SFINCS model")

        # Copy pre-built files to model_root if source is different
        source_dir = self.sfincs.prebuilt_dir
        if source_dir.resolve() != root.resolve():
            root.mkdir(parents=True, exist_ok=True)
            for src_file in source_dir.iterdir():
                if src_file.is_file():
                    dst_file = root / src_file.name
                    if not dst_file.exists():
                        shutil.copy2(src_file, dst_file)
            self._log(f"Copied pre-built model from {source_dir} to {root}")

        # Remove generated netCDF files from any previous run so that
        # downstream stages start fresh and never trip over stale data.
        self._remove_stale_outputs(root)

        model = SfincsModel(
            data_libs=data_libs,
            root=str(root),
            mode="r+",
            write_gis=True,
        )

        # Read existing model to detect grid type and load components
        model.read()

        # Clear config references to missing files
        self._clear_missing_file_refs(model)

        _set_model(self.config, model)

        self._log(f"SFINCS model initialised (grid_type={model.grid_type}) at {root}")

        return {
            "model_root": str(root),
            "grid_type": model.grid_type,
            "status": "completed",
        }


class SfincsTimingStage(WorkflowStage):
    """Set simulation timing on the SFINCS model."""

    name = "sfincs_timing"
    description = "Set SFINCS timing"

    def run(self) -> dict[str, Any]:
        """Configure tref, tstart, and tstop on the model."""
        model = _get_model(self.config)
        sim = self.config.simulation
        start = sim.start_date
        stop = start + timedelta(hours=sim.duration_hours)

        self._update_substep("Setting simulation timing")

        model.config.update(
            {
                "tref": start,
                "tstart": start,
                "tstop": stop,
            }
        )

        self._log(f"Timing set: {start} to {stop}")

        return {"status": "completed"}


class SfincsForcingStage(_SfincsStageBase):
    """Add water level boundary forcing.

    When ``boundary.source`` is ``"tpxo"``, tide predictions are generated
    using the OTPS ``predict_tide`` Fortran binary inside the SCHISM
    Singularity container and then injected into the HydroMT model.
    For all other sources the standard HydroMT geodataset path is used.
    """

    name = "sfincs_forcing"
    description = "Add water level forcing"

    # Hourly OTPS predictions, interpolated to this interval (seconds)
    _TPXO_RAW_DT = 3600
    _TPXO_INTERP_DT = 600

    # ------------------------------------------------------------------
    # TPXO tidal forcing via OTPS predict_tide
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_bnd_file(bnd_path: Path) -> list[tuple[float, float, str]]:
        """Parse a SFINCS ``.bnd`` file into (x, y, name) tuples.

        The format is identical to ``.src`` / ``.obs`` files::

            x  y  "name"
        """
        points: list[tuple[float, float, str]] = []
        for raw_line in bnd_path.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split('"')
            coords = parts[0].strip().split()
            name = parts[1].strip() if len(parts) > 1 else f"bnd_{len(points)}"
            x, y = float(coords[0]), float(coords[1])
            points.append((x, y, name))
        return points

    @staticmethod
    def _write_otps_input(
        otps_path: Path,
        lonlats: list[tuple[float, float]],
        tstart: Any,
        tstop: Any,
        dt_seconds: int,
    ) -> None:
        """Write the ``otps_lat_lon_time.txt`` input file for ``predict_tide``.

        Each boundary point gets one line per timestep in the format
        ``lat  lon  YYYY MM DD HH MM SS`` expected by the OTPS binary.
        """
        from datetime import timedelta as _td

        dt = _td(seconds=dt_seconds)
        with otps_path.open("w") as f:
            for lon, lat in lonlats:
                current = tstart
                while current <= tstop:
                    f.write(f"{lat:12.6f}  {lon:12.6f}  {current.strftime('%Y %m %d %H %M %S')}\n")
                    current += dt

    def _prepare_tpxo_files(self, model_root: Path) -> None:
        """Copy OTPS setup files and symlink the TPXO atlas data."""
        import shutil as _shutil

        from coastal_calibration.scripts_path import get_scripts_dir

        scripts_dir = get_scripts_dir()
        for fname in ("setup_tpxo.txt", "Model_tpxo10_atlas"):
            src = scripts_dir / fname
            dst = model_root / fname
            if not dst.exists():
                _shutil.copy2(src, dst)

        tpxo_data_dir = self.config.paths.tpxo_data_dir
        link_target = model_root / "TPXO10_atlas_v2_nc"
        if not link_target.exists():
            link_target.symlink_to(tpxo_data_dir)
            self._log(f"Symlinked {tpxo_data_dir} -> {link_target}")

    def _run_predict_tide(self, model_root: Path) -> Path:
        """Run the OTPS ``predict_tide`` binary inside Singularity.

        Returns the path to the ``otps_out.txt`` output file.
        """
        env = self.build_environment()
        otps_dir = str(self.config.paths.otps_dir)

        bindings = self._get_default_bindings()
        bindings.append(str(model_root))
        tpxo_real = str(self.config.paths.tpxo_data_dir.resolve())
        if tpxo_real not in bindings:
            bindings.append(tpxo_real)

        self.run_singularity_command(
            command=[
                "bash",
                "-c",
                f"cd {model_root} && {otps_dir}/predict_tide < setup_tpxo.txt",
            ],
            bindings=bindings,
            pwd=model_root,
            env=env,
        )

        otps_out = model_root / "otps_out.txt"
        if not otps_out.exists():
            raise FileNotFoundError(
                f"predict_tide did not produce {otps_out}.  "
                "Check that the TPXO atlas data and OTPS binary are present."
            )
        return otps_out

    @staticmethod
    def _parse_otps_output(
        otps_out: Path,
        lonlats: list[tuple[float, float]],
    ) -> Any:
        """Parse ``otps_out.txt`` into a wide DataFrame (time x points).

        Returns a :class:`~pandas.DataFrame` with a ``DatetimeIndex`` and
        one integer column per boundary point.

        The OTPS output has a 3-line header followed by a column header::

            Lat       Lon     mm.dd.yyyy  hh:mm:ss   z(m)

        We let :func:`pandas.read_csv` merge the date and time columns
        into a single ``datetime`` column via ``parse_dates``.
        """
        import pandas as pd

        # Column header names from OTPS: Lat, Lon, mm.dd.yyyy, hh:mm:ss, z(m)
        df_raw = pd.read_csv(
            otps_out,
            sep=r"\s+",
            header=3,
            on_bad_lines="skip",
        )
        df_raw["datetime"] = pd.to_datetime(df_raw["mm.dd.yyyy"] + " " + df_raw["hh:mm:ss"])

        point_dfs: list[pd.Series] = []
        for idx, (lon, lat) in enumerate(lonlats):
            mask = ((df_raw["Lat"] - lat).abs() < 0.01) & ((df_raw["Lon"] - lon).abs() < 0.01)
            subset = df_raw.loc[mask].sort_values("datetime")
            if subset.empty:
                raise ValueError(
                    f"No OTPS output for boundary point {idx} (lat={lat:.4f}, lon={lon:.4f})"
                )
            series = subset.set_index("datetime")["z(m)"].astype(float)
            series.name = idx
            point_dfs.append(series)

        df_ts = pd.concat(point_dfs, axis=1)
        df_ts.index.name = "time"
        return df_ts

    def _inject_water_level(
        self,
        model: SfincsModel,
        df_ts: Any,
        gdf_bnd: Any,
    ) -> None:
        """Inject water-level forcing into the HydroMT model.

        SFINCS needs *both* ``bndfile`` (boundary point locations on the
        grid) and ``netbndbzsbzifile`` (time-varying water levels at those
        points).  We temporarily clear all boundary file refs so that the
        lazy initialisation of the water_level component starts with an
        empty dataset, then restore ``bndfile`` before writing so the
        SFINCS binary can find boundary locations.

        A zero-filled ``bzi`` (infragravity) variable is added because the
        SFINCS binary unconditionally queries ``zi`` in the netCDF file
        (``sfincs_ncinput.F90:118``) and crashes if it is absent.
        """
        import xarray as xr_

        for key in ("bzsfile", "bzifile", "bndfile", "bcafile", "netbndbzsbzifile"):
            model.config.set(key, None)

        _ = model.water_level.data  # force lazy init with empty data

        # Anchor the forcing signal to the mesh datum.  For tidal-only
        # sources (e.g. TPXO) this places the mean water level at the
        # correct geodetic height on the mesh.
        forcing_offset = self.sfincs.forcing_to_mesh_offset_m
        if forcing_offset != 0.0:
            df_ts = df_ts + forcing_offset
            self._log(f"Applied forcing→mesh vdatum offset: {forcing_offset:+.4f} m")

        wl_min = float(df_ts.min().min())
        wl_max = float(df_ts.max().max())
        wl_floor, wl_ceil = -15.0, 15.0
        if wl_min < wl_floor or wl_max > wl_ceil:
            self._log(
                f"Boundary water levels after vdatum adjustment are outside "
                f"[{wl_floor}, {wl_ceil}] m (min={wl_min:.3f}, "
                f"max={wl_max:.3f}).  Check that "
                f"forcing_to_mesh_offset_m={forcing_offset} has the "
                f"correct sign and magnitude.",
                "warning",
            )

        self._update_substep("Setting water level forcing on model")
        model.water_level.set(df=df_ts, gdf=gdf_bnd, merge=False)

        # Add zero-filled bzi so the netCDF writer includes a ``zi``
        # variable (SFINCS crashes without it).
        ds = model.water_level.data
        if "bzi" not in ds.data_vars:
            ds["bzi"] = xr_.zeros_like(ds["bzs"])

        # Restore bndfile so the SFINCS binary can locate boundary cells,
        # and set netCDF output for the water level timeseries.
        model.config.set("bndfile", "sfincs.bnd")
        nc_name = "sfincs_netbndbzsbzifile.nc"
        model.config.set("netbndbzsbzifile", nc_name)
        model.water_level.write()
        self._log(f"Wrote boundary forcing to {nc_name}")

    def _create_tpxo_forcing(self, model: SfincsModel) -> None:
        """Synthesize water-level forcing from TPXO tidal constituents.

        The workflow mirrors the SCHISM ``make_tpxo_ocean.bash`` script:
        read boundary locations, generate OTPS input, run ``predict_tide``
        inside the Singularity container, parse and interpolate the output,
        then inject into the HydroMT model via ``water_level.set()``.
        """
        from datetime import datetime as _dt

        import geopandas as gpd
        from pyproj import Transformer
        from shapely.geometry import Point

        model_root = get_model_root(self.config)
        bnd_path = model_root / "sfincs.bnd"
        if not bnd_path.exists():
            raise FileNotFoundError(
                f"Boundary file {bnd_path} not found.  "
                "Ensure the pre-built model includes sfincs.bnd."
            )

        # 1. Read boundary points
        bnd_points = self._parse_bnd_file(bnd_path)
        if not bnd_points:
            raise ValueError("No boundary points found in sfincs.bnd")
        self._log(f"Read {len(bnd_points)} boundary point(s) from {bnd_path}")

        # 2. Transform to lon/lat
        model_crs = model.crs
        transformer = Transformer.from_crs(model_crs, 4326, always_xy=True)
        lonlats = [transformer.transform(x, y) for x, y, _ in bnd_points]

        # 3. Generate OTPS input file
        tstart = model.config.data.tstart
        tstop = model.config.data.tstop
        if isinstance(tstart, str):
            tstart = _dt.fromisoformat(tstart)
        if isinstance(tstop, str):
            tstop = _dt.fromisoformat(tstop)

        otps_input_path = model_root / "otps_lat_lon_time.txt"
        self._write_otps_input(otps_input_path, lonlats, tstart, tstop, self._TPXO_RAW_DT)
        self._log(f"Wrote OTPS input ({len(lonlats)} points) to {otps_input_path}")

        # 4-5. Copy setup files and symlink atlas data
        self._prepare_tpxo_files(model_root)

        # 6. Run predict_tide
        self._update_substep("Running OTPS predict_tide")
        otps_out = self._run_predict_tide(model_root)
        self._log("predict_tide completed successfully")

        # 7. Parse output
        self._update_substep("Parsing TPXO output")
        df_ts = self._parse_otps_output(otps_out, lonlats)
        n_points = len(lonlats)
        self._log(f"Parsed {len(df_ts)} timesteps x {n_points} points from otps_out.txt")

        # 8. Interpolate to finer resolution
        interp_freq = f"{self._TPXO_INTERP_DT}s"
        df_fine = df_ts.resample(interp_freq).interpolate(method="linear")
        df_fine = df_fine.loc[df_ts.index[0] : df_ts.index[-1]]
        self._log(f"Interpolated to {interp_freq}: {len(df_fine)} timesteps")

        # 9. Build GeoDataFrame for boundary locations
        geom = [Point(x, y) for x, y, _ in bnd_points]
        names = [name for _, _, name in bnd_points]
        gdf_bnd = gpd.GeoDataFrame(
            {"name": names},
            geometry=geom,
            index=range(n_points),
            crs=model_crs,
        )

        # 10. Inject into HydroMT model
        self._inject_water_level(model, df_fine, gdf_bnd)

    @staticmethod
    def _idw_interpolate(
        src_xy: NDArray[np.floating[Any]],
        target_xy: NDArray[np.floating[Any]],
        values: NDArray[np.floating[Any]],
        k: int = 4,
    ) -> NDArray[np.floating[Any]]:
        """Inverse-distance weighted interpolation from source to target points.

        Parameters
        ----------
        src_xy : ndarray, shape (N, 2)
            Source point coordinates.
        target_xy : ndarray, shape (M, 2)
            Target point coordinates.
        values : ndarray, shape (T, N)
            Timeseries values at each source point.
        k : int
            Number of nearest neighbours to use (capped at N).

        Returns
        -------
        ndarray, shape (T, M)
            Interpolated values at each target point.
        """
        from scipy.spatial import cKDTree

        k = min(k, len(src_xy))
        tree = cKDTree(src_xy)
        dists, idxs = tree.query(target_xy, k=k)

        n_times, _ = values.shape
        n_targets = len(target_xy)
        result = np.empty((n_times, n_targets))

        for j in range(n_targets):
            d = np.atleast_1d(dists[j])
            ix = np.atleast_1d(idxs[j])
            exact = d < 1e-10
            if np.any(exact):
                result[:, j] = values[:, ix[exact][0]]
            else:
                weights = 1.0 / d
                weights /= weights.sum()
                result[:, j] = np.nansum(values[:, ix] * weights[np.newaxis, :], axis=1)

        return result

    def _load_geodataset_for_bnd(
        self,
        model: SfincsModel,
        geodataset_name: str,
        bnd_points: list[tuple[float, float, str]],
    ) -> xr.DataArray | xr.Dataset:
        """Load a geodataset clipped around the boundary point locations."""
        import geopandas as gpd
        from pyproj import Transformer
        from shapely.geometry import Point

        self._update_substep(f"Loading {geodataset_name} data")
        tstart, tstop = model.get_model_time()

        transformer = Transformer.from_crs(model.crs, 4326, always_xy=True)
        bnd_lonlats = [transformer.transform(x, y) for x, y, _ in bnd_points]
        bbox_gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lon, lat in bnd_lonlats],
            crs=4326,
        )

        da = model.data_catalog.get_geodataset(
            geodataset_name,
            geom=bbox_gdf,
            buffer=50e3,  # 50 km buffer to ensure enough source nodes
            variables=["waterlevel"],
            time_range=(tstart, tstop),
        )
        self._log(f"Loaded {geodataset_name}: {da.sizes} dimensions")
        return da

    def _interpolate_to_bnd(
        self,
        da: xr.DataArray | xr.Dataset,
        bnd_points: list[tuple[float, float, str]],
        model_crs: Any,
    ) -> Any:
        """IDW-interpolate a geodataset to the boundary point locations."""
        import pandas as pd
        from pyproj import Transformer

        src_gdf = da.vector.to_gdf()
        src_xy = np.column_stack([src_gdf.geometry.x.values, src_gdf.geometry.y.values])

        # Transform boundary points to the same CRS as the source data
        src_crs = da.vector.crs
        if src_crs is not None and src_crs != model_crs:
            tr = Transformer.from_crs(model_crs, src_crs, always_xy=True)
            bnd_in_src = [tr.transform(x, y) for x, y, _ in bnd_points]
        else:
            bnd_in_src = [(x, y) for x, y, _ in bnd_points]
        target_xy = np.array(bnd_in_src)

        wl_data: NDArray[np.floating[Any]] = np.asarray(
            da.transpose(..., da.vector.index_dim).values
        )
        bnd_wl = self._idw_interpolate(src_xy, target_xy, wl_data)

        time_vals = pd.DatetimeIndex(da.time.values)
        df_ts = pd.DataFrame(bnd_wl, index=time_vals, columns=range(len(bnd_points)))
        df_ts.index.name = "time"
        return df_ts

    def _create_geodataset_forcing(self, model: SfincsModel, geodataset_name: str) -> None:
        """Create water-level forcing by interpolating a geodataset to boundary points.

        Unlike ``model.water_level.create(geodataset=...)``, which passes
        *all* source stations within the model region to the netCDF output
        (incompatible with SFINCS when a ``.bnd`` file defines explicit
        boundary points), this method:

        1. Reads boundary locations from ``sfincs.bnd``.
        2. Loads the geodataset (e.g. STOFS water levels).
        3. Spatially interpolates the geodataset to the boundary points
           using inverse-distance weighting from the nearest source nodes.
        4. Injects the result into HydroMT the same way the TPXO path does.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        model_root = get_model_root(self.config)
        bnd_path = model_root / "sfincs.bnd"
        if not bnd_path.exists():
            raise FileNotFoundError(
                f"Boundary file {bnd_path} not found.  "
                "Ensure the pre-built model includes sfincs.bnd."
            )

        # 1. Read boundary points (in model CRS, typically UTM)
        bnd_points = self._parse_bnd_file(bnd_path)
        if not bnd_points:
            raise ValueError("No boundary points found in sfincs.bnd")
        n_bnd = len(bnd_points)
        self._log(f"Read {n_bnd} boundary point(s) from {bnd_path}")

        # 2. Load the geodataset clipped around the boundary points
        da = self._load_geodataset_for_bnd(model, geodataset_name, bnd_points)

        # 3. Spatially interpolate geodataset to boundary point locations
        self._update_substep("Interpolating to boundary points")
        df_ts = self._interpolate_to_bnd(da, bnd_points, model.crs)
        self._log(
            f"Interpolated {geodataset_name} to {n_bnd} boundary points ({len(df_ts)} time steps)"
        )

        # 4. Build GeoDataFrame for boundary locations (model CRS)
        gdf_bnd = gpd.GeoDataFrame(
            {"name": [name for _, _, name in bnd_points]},
            geometry=[Point(x, y) for x, y, _ in bnd_points],
            index=range(n_bnd),
            crs=model.crs,
        )

        # 5. Inject into HydroMT model (same approach as TPXO path)
        self._inject_water_level(model, df_ts, gdf_bnd)

    def run(self) -> dict[str, Any]:
        """Add water level boundary forcing."""
        model = _get_model(self.config)

        if self.config.boundary.source == "tpxo":
            self._update_substep("Creating TPXO tidal forcing")
            self._create_tpxo_forcing(model)
            self._log("Water level forcing added from TPXO predict_tide")
            return {"status": "completed", "source": "tpxo"}

        wl_geodataset = _waterlevel_geodataset(self.config)
        if wl_geodataset is None:
            self._log("No water level geodataset configured, skipping")
            return {"status": "skipped"}

        self._update_substep("Adding water level forcing")
        self._create_geodataset_forcing(model, wl_geodataset)
        self._log(f"Water level forcing added from {wl_geodataset}")

        return {"status": "completed", "source": wl_geodataset}


class SfincsObservationPointsStage(_SfincsStageBase):
    """Add observation points to the model."""

    name = "sfincs_obs"
    description = "Add observation points"

    def _add_noaa_gages(self, model: SfincsModel) -> int:
        """Query NOAA CO-OPS and add water-level stations as observation points.

        Parameters
        ----------
        model : SfincsModel
            Initialised SFINCS model with a region and CRS.

        Returns
        -------
        int
            Number of NOAA stations added.
        """
        from coastal_calibration.coops_api import COOPSAPIClient

        model_crs = model.crs
        if model_crs is None:
            self._log("Model CRS is undefined, cannot add NOAA CO-OPS stations")
            return 0

        # Get domain boundary in EPSG:4326 (lon/lat) for the COOPS query
        region_4326 = model.region.to_crs(4326)
        domain_geom = region_4326.union_all()

        client = COOPSAPIClient()
        stations_gdf = client.stations_metadata
        selected = stations_gdf[stations_gdf.within(domain_geom)]

        if selected.empty:
            self._log("No NOAA CO-OPS stations found within model domain")
            return 0

        # Keep only stations with valid MSL/MLLW datums so that the
        # plotting stage can convert observations from MLLW to MSL.
        candidate_ids = selected["station_id"].tolist()
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
            self._log("No NOAA CO-OPS stations with valid datum data in domain")
            return 0

        # Project selected stations into the model CRS
        selected_projected = selected.to_crs(model_crs)

        for _, row in selected_projected.iterrows():
            model.observation_points.add_point(
                x=row.geometry.x,
                y=row.geometry.y,
                name=f"noaa_{row['station_id']}",
            )

        return len(selected_projected)

    def run(self) -> dict[str, Any]:
        """Add observation points from config, file, and/or NOAA gages."""
        model = _get_model(self.config)

        has_file = self.sfincs.observation_locations_file is not None
        has_points = bool(self.sfincs.observation_points)
        has_noaa = self.sfincs.include_noaa_gages

        if not has_file and not has_points and not has_noaa:
            self._log("No observation points configured, skipping")
            return {"status": "skipped"}

        self._update_substep("Adding observation points")

        # When merge=False, clear existing observation points first
        if not self.sfincs.merge_observations:
            try:
                existing = model.observation_points.nr_points
                if existing > 0:
                    model.observation_points.clear()
                    self._log(f"Cleared {existing} existing observation point(s)")
            except Exception:  # noqa: S110
                pass  # No existing points to clear

        if has_file:
            model.observation_points.create(
                locations=str(self.sfincs.observation_locations_file),
                merge=self.sfincs.merge_observations,
            )
            self._log(f"Observation points added from {self.sfincs.observation_locations_file}")
        elif has_points:
            for pt in self.sfincs.observation_points:
                model.observation_points.add_point(
                    x=pt["x"],
                    y=pt["y"],
                    name=pt.get("name", f"obs_{self.sfincs.observation_points.index(pt)}"),
                )
            self._log(f"Added {len(self.sfincs.observation_points)} observation point(s)")

        noaa_count = 0
        if has_noaa:
            self._update_substep("Querying NOAA CO-OPS stations")
            noaa_count = self._add_noaa_gages(model)
            self._log(f"Added {noaa_count} NOAA CO-OPS observation point(s)")

        return {"status": "completed", "noaa_stations": noaa_count}


class SfincsDischargeStage(_SfincsStageBase):
    """Add discharge source points to the model."""

    name = "sfincs_discharge"
    description = "Add discharge sources"

    @staticmethod
    def _parse_src_file(path: Path) -> list[tuple[float, float, str]]:
        """Parse a SFINCS ``.src`` file into (x, y, name) tuples."""
        points: list[tuple[float, float, str]] = []
        for raw_line in path.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split('"')
            coords = parts[0].strip().split()
            name = parts[1].strip() if len(parts) > 1 else f"src_{len(points)}"
            x, y = float(coords[0]), float(coords[1])
            points.append((x, y, name))
        return points

    @staticmethod
    def _filter_active_cells(
        points: list[tuple[float, float, str]],
        model: Any,
    ) -> tuple[list[tuple[float, float, str]], list[str]]:
        """Keep only source points that fall on active grid cells.

        The SFINCS Fortran binary segfaults when a discharge source point
        maps to an inactive cell (``mask != 1``), because the cell index
        is left at 0 and the code later accesses ``zs(0)`` without any
        bounds check.  This filter prevents the crash by dropping points
        that do not land on an active face of the quadtree mesh.

        Returns
        -------
        kept, dropped
            ``kept`` are the (x, y, name) tuples on active cells;
            ``dropped`` are the names of the removed points.
        """
        import numpy as np
        from scipy.spatial import cKDTree

        grid_ds = model.components["quadtree_grid"].data
        ugrid = grid_ds.ugrid.grid
        face_xy = np.column_stack([ugrid.face_x, ugrid.face_y])
        mask = grid_ds["mask"].values

        tree = cKDTree(face_xy)
        kept: list[tuple[float, float, str]] = []
        dropped: list[str] = []
        for x, y, name in points:
            _, idx = tree.query([x, y])
            if mask[idx] == 1:
                kept.append((x, y, name))
            else:
                dropped.append(name)
        return kept, dropped

    def run(self) -> dict[str, Any]:
        """Add discharge source points from a ``.src`` or GeoJSON file."""
        model = _get_model(self.config)

        if self.sfincs.discharge_locations_file is None:
            self._log("No discharge configuration, skipping")
            return {"status": "skipped"}

        self._update_substep("Adding discharge source points")

        # When merge=False, clear existing discharge points first
        if not self.sfincs.merge_discharge:
            try:
                existing = model.discharge_points.nr_points
                if existing > 0:
                    model.discharge_points.clear()
                    self._log(f"Cleared {existing} existing discharge point(s)")
            except Exception:  # noqa: S110
                pass  # No existing points to clear

        src_path = self.sfincs.discharge_locations_file
        suffix = src_path.suffix.lower()

        if suffix == ".src":
            parsed = self._parse_src_file(src_path)
            # Filter out points on inactive grid cells to prevent
            # a segfault in the SFINCS binary (zs(0) out-of-bounds).
            kept, dropped = self._filter_active_cells(parsed, model)
            if dropped:
                self._log(
                    f"Dropped {len(dropped)} source point(s) on inactive "
                    f"cells: {', '.join(dropped)}"
                )
            for x, y, name in kept:
                model.discharge_points.add_point(x=x, y=y, name=name)
            self._log(f"Added {len(kept)} discharge source point(s) from {src_path}")
        else:
            model.discharge_points.create(
                locations=str(src_path),
                merge=self.sfincs.merge_discharge,
            )
            self._log(f"Discharge source points added from {src_path}")

        return {"status": "completed"}


class SfincsPrecipitationStage(_SfincsStageBase):
    """Add precipitation forcing."""

    name = "sfincs_precip"
    description = "Add precipitation forcing"

    def run(self) -> dict[str, Any]:
        """Add precipitation from a dataset in the data catalog."""
        if not self.sfincs.include_precip:
            self._log("Precipitation forcing disabled, skipping")
            return {"status": "skipped"}

        model = _get_model(self.config)
        meteo_dataset = f"{self.config.simulation.meteo_source}_meteo"

        self._update_substep("Adding precipitation forcing")
        dst_res = _meteo_dst_res(self.config, model)
        model.precipitation.create(precip=meteo_dataset, dst_res=dst_res)

        # Clip the reprojected grid to the model domain.  Without this
        # the LCC → UTM reprojection inflates the grid to cover the
        # entire CONUS extent (~5 000 x 4 200 km).
        nx_before, nx_after = _clip_meteo_to_domain(model, "precipitation")
        self._log(
            f"Precipitation forcing added from {meteo_dataset} "
            f"(res={dst_res:.0f} m, clipped {nx_before}→{nx_after} x-cells)"
        )

        # Write immediately and release memory so subsequent forcing
        # stages don't accumulate large gridded data in RAM.
        model.precipitation.write()
        model.precipitation.clear()

        return {"status": "completed"}


class SfincsWindStage(_SfincsStageBase):
    """Add spatially varying wind forcing (u10 / v10)."""

    name = "sfincs_wind"
    description = "Add wind forcing"

    def run(self) -> dict[str, Any]:
        """Add wind forcing from a dataset in the data catalog."""
        if not self.sfincs.include_wind:
            self._log("Wind forcing disabled, skipping")
            return {"status": "skipped"}

        model = _get_model(self.config)
        meteo_dataset = f"{self.config.simulation.meteo_source}_meteo"

        self._update_substep("Adding wind forcing")
        dst_res = _meteo_dst_res(self.config, model)
        model.wind.create(wind=meteo_dataset, dst_res=dst_res)

        nx_before, nx_after = _clip_meteo_to_domain(model, "wind")
        self._log(
            f"Wind forcing added from {meteo_dataset} "
            f"(res={dst_res:.0f} m, clipped {nx_before}→{nx_after} x-cells)"
        )

        # Write immediately and release memory (see SfincsPrecipitationStage).
        model.wind.write()
        model.wind.clear()

        return {"status": "completed"}


class SfincsPressureStage(_SfincsStageBase):
    """Add spatially varying atmospheric pressure forcing."""

    name = "sfincs_pressure"
    description = "Add atmospheric pressure forcing"

    def run(self) -> dict[str, Any]:
        """Add atmospheric pressure from a dataset in the data catalog."""
        if not self.sfincs.include_pressure:
            self._log("Pressure forcing disabled, skipping")
            return {"status": "skipped"}

        model = _get_model(self.config)
        meteo_dataset = f"{self.config.simulation.meteo_source}_meteo"

        self._update_substep("Adding atmospheric pressure forcing")
        dst_res = _meteo_dst_res(self.config, model)
        model.pressure.create(press=meteo_dataset, dst_res=dst_res)

        nx_before, nx_after = _clip_meteo_to_domain(model, "pressure")

        # Enable barometric pressure correction so SFINCS uses the forcing
        model.config.set("baro", 1)
        self._log(
            f"Atmospheric pressure forcing added from {meteo_dataset} "
            f"(baro=1, res={dst_res:.0f} m, clipped {nx_before}→{nx_after} x-cells)"
        )

        # Write immediately and release memory (see SfincsPrecipitationStage).
        model.pressure.write()
        model.pressure.clear()

        return {"status": "completed"}


class SfincsWriteStage(WorkflowStage):
    """Write the SFINCS model files to disk."""

    name = "sfincs_write"
    description = "Write SFINCS model"

    def run(self) -> dict[str, Any]:
        """Write all model files to the model root directory."""
        model = _get_model(self.config)

        self._update_substep("Writing model to disk")
        model.write()

        root = get_model_root(self.config)
        self._log(f"SFINCS model written to {root}")

        return {
            "model_root": str(root),
            "status": "completed",
        }


class SfincsRunStage(_SfincsStageBase):
    """Run the SFINCS model via a native executable or Singularity container.

    When ``model_config.sfincs_exe`` is set the stage invokes the binary
    directly, which avoids the Singularity dependency (useful on macOS or
    other systems where Singularity is unavailable).  Otherwise it falls
    back to the Singularity container workflow.
    """

    name = "sfincs_run"
    description = "Run SFINCS model"

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        super().__init__(config, monitor)
        # Only require a container when no native exe is configured.
        self.requires_container = self.sfincs.sfincs_exe is None

    # ------------------------------------------------------------------ #
    # Stage entry-point
    # ------------------------------------------------------------------ #

    def run(self) -> dict[str, Any]:
        """Execute SFINCS via native binary or Singularity."""
        model_root = get_model_root(self.config)

        if self.sfincs.sfincs_exe is not None:
            # ── Native executable ──
            self._log(f"Running SFINCS via native executable: {self.sfincs.sfincs_exe}")
            self._update_substep("Running SFINCS (native)")
            env = {**os.environ, "OMP_NUM_THREADS": str(self.sfincs.omp_num_threads)}
            _run_and_log([str(self.sfincs.sfincs_exe)], model_root, env=env)
            mode = "native"
        else:
            # ── Singularity container ──
            self._update_substep("Pulling Singularity image")
            sif_path = _pull_singularity_image(self.config)

            self._update_substep("Running SFINCS (Singularity)")
            _run_singularity(self.config, sif_path)
            mode = "singularity"

        self._log("SFINCS run completed")

        # Clean up model registry
        _clear_model(self.config)

        return {"status": "completed", "mode": mode}


class SfincsPlotStage(_SfincsStageBase):
    """Plot simulated water levels against NOAA CO-OPS observations.

    After the SFINCS run, this stage reads ``point_zs`` (water surface
    elevation) from the model output (``sfincs_his.nc``), identifies
    observation points whose names start with ``noaa_`` (added by
    :class:`SfincsObservationPointsStage`), fetches observed water levels
    from the NOAA CO-OPS API, and produces a comparison time-series
    figure saved to ``<model_root>/figs/``.

    Observations are fetched in MLLW (universally supported by all
    CO-OPS stations) and then converted to MSL using per-station datum
    offsets from the CO-OPS metadata API, matching the STOFS boundary
    condition vertical reference used by SFINCS.

    The stage is a no-op when:

    * The model output file (``sfincs_his.nc``) does not exist.
    * No ``point_zs`` (or ``point_h``) variable is present in the output.
    * No observation points with the ``noaa_`` prefix are found.
    """

    name = "sfincs_plot"
    description = "Plot simulated vs observed water levels"

    @staticmethod
    def _station_dim(point_h: Any) -> str:
        """Detect the station dimension name in the ``point_h`` DataArray."""
        for candidate in ("stations", "station_id"):
            if candidate in point_h.dims:
                return candidate
        for dim_name in point_h.dims:
            if "station" in str(dim_name).lower():
                return str(dim_name)
        raise ValueError("Cannot determine station dimension in point_h")

    @staticmethod
    def _parse_obs_names(obs_file: Path) -> list[str]:
        """Parse observation point names from a ``sfincs.obs`` file.

        The file format has lines like::

            x y "name"

        Returns the list of names in order.
        """
        names: list[str] = []
        for raw_line in obs_file.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split('"')
            if len(parts) >= 2:
                names.append(parts[1].strip())
            else:
                names.append(f"obs_{len(names)}")
        return names

    @staticmethod
    def _plot_figures(
        sim_times: Any,
        sim_elevation: NDArray[np.float64],
        station_ids: list[str],
        obs_ds: Any,
        figs_dir: Path,
    ) -> list[Path]:
        """Create comparison figures and save them.

        Stations that lack *both* valid observations and valid simulated
        data are skipped so that empty panels do not appear.

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

        # ── Pre-filter: keep only stations with both obs and sim ──
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
            self._log(f"Station {sid}: MLLW→MSL offset = {offset:.4f} m", "debug")

        obs_ds.attrs["datum"] = "MSL"
        return obs_ds

    @staticmethod
    def _read_obs_names(mod: Any, model_root: Path) -> list[str]:
        """Read observation point names from HydroMT model or sfincs.obs."""
        if hasattr(mod, "observation_points") and hasattr(mod.observation_points, "geodataframe"):
            gdf = mod.observation_points.geodataframe
            if gdf is not None and not gdf.empty:
                return gdf.index.tolist()

        obs_file = model_root / "sfincs.obs"
        if obs_file.exists():
            return SfincsPlotStage._parse_obs_names(obs_file)
        return []

    def run(self) -> dict[str, Any]:
        """Read SFINCS output, fetch NOAA observations, and plot comparison."""
        from hydromt_sfincs import SfincsModel  # pyright: ignore[reportMissingImports]

        model_root = get_model_root(self.config)
        his_file = model_root / "sfincs_his.nc"

        if not his_file.exists():
            self._log("sfincs_his.nc not found, skipping plot stage")
            return {"status": "skipped", "reason": "no output"}

        self._update_substep("Reading SFINCS output")
        mod = SfincsModel(str(model_root), mode="r")
        mod.read()
        mod.output.read()

        # Prefer point_zs (water surface elevation) over point_h (water depth)
        if "point_zs" in mod.output.data:
            point_zs = mod.output.data["point_zs"]
        elif "point_h" in mod.output.data:
            self._log("point_zs not found, falling back to point_h (water depth)")
            point_zs = mod.output.data["point_h"]
        else:
            self._log("No point_zs or point_h in output, skipping plot stage")
            return {"status": "skipped", "reason": "no point_zs"}

        point_zs = cast("xr.DataArray", point_zs)
        station_dim = self._station_dim(point_zs)
        obs_names = self._read_obs_names(mod, model_root)

        # Identify NOAA stations by the "noaa_" prefix
        noaa_indices: list[int] = []
        noaa_station_ids: list[str] = []
        for i, name in enumerate(obs_names):
            if str(name).startswith("noaa_"):
                noaa_indices.append(i)
                noaa_station_ids.append(str(name).removeprefix("noaa_"))

        if not noaa_station_ids:
            self._log("No NOAA observation points found, skipping plot stage")
            return {"status": "skipped", "reason": "no noaa stations"}

        # Extract numpy arrays from xarray for the selected NOAA stations
        sim_times = point_zs["time"].to_numpy()
        sim_elevation = np.column_stack(
            [point_zs.isel({station_dim: idx}).values for idx in noaa_indices]
        )

        # Apply mesh vdatum → MSL correction.  Model output inherits
        # the mesh vertical datum; this offset converts to MSL for
        # comparison with NOAA CO-OPS observations.
        datum_offset = self.sfincs.vdatum_mesh_to_msl_m
        if datum_offset != 0.0:
            sim_elevation = sim_elevation + datum_offset
            self._log(f"Applied mesh→MSL vdatum offset: {datum_offset:+.4f} m")

        # Fetch observed water levels (MLLW → MSL)
        self._update_substep("Fetching NOAA CO-OPS observations")
        sim = self.config.simulation
        begin_date = sim.start_date.strftime("%Y%m%d %H:%M")
        end_dt = sim.start_date + timedelta(hours=sim.duration_hours)
        end_date = end_dt.strftime("%Y%m%d %H:%M")

        obs_ds = self._fetch_observations_msl(noaa_station_ids, begin_date, end_date)

        # Generate comparison plots
        self._update_substep("Generating comparison plots")
        figs_dir = model_root / "figs"
        fig_paths = self._plot_figures(sim_times, sim_elevation, noaa_station_ids, obs_ds, figs_dir)

        self._log(f"Saved {len(fig_paths)} comparison figure(s) to {figs_dir}")
        return {
            "status": "completed",
            "figures": [str(p) for p in fig_paths],
            "figs_dir": str(figs_dir),
        }


# ---------------------------------------------------------------------------
# Infrastructure stages (symlinks, data catalog)
# ---------------------------------------------------------------------------


class SfincsSymlinksStage(WorkflowStage):
    """Create ``.nc`` symlinks for NWM files in the download directory.

    NWM LDASIN and CHRTOUT files lack a ``.nc`` extension, which confuses
    HydroMT's dataset readers.  This stage creates symlinks with the
    ``.nc`` suffix so the data catalog entries resolve correctly.

    If the download directory does not exist, this stage is a no-op.
    """

    name = "sfincs_symlinks"
    description = "Create .nc symlinks for NWM data"

    def run(self) -> dict[str, Any]:
        """Create ``.nc`` symlinks for NWM LDASIN and CHRTOUT files."""
        download_dir = self.config.paths.download_dir
        if not download_dir.exists():
            self._log(f"Download dir {download_dir} does not exist — skipping symlinks")
            return {"meteo_symlinks": 0, "streamflow_symlinks": 0, "status": "skipped"}

        self._update_substep("Creating .nc symlinks")
        meteo_source = self.config.simulation.meteo_source

        created = create_nc_symlinks(
            download_dir,
            meteo_source=meteo_source,
            include_meteo=True,
            include_streamflow=True,
        )

        n_meteo = len(created["meteo"])
        n_stream = len(created["streamflow"])
        self._log(f"Created {n_meteo} meteo + {n_stream} streamflow symlinks in {download_dir}")

        return {
            "meteo_symlinks": n_meteo,
            "streamflow_symlinks": n_stream,
            "status": "completed",
        }


class SfincsDataCatalogStage(WorkflowStage):
    """Generate HydroMT data catalog for the SFINCS pipeline.

    Delegates to :func:`generate_data_catalog` which already accepts a
    :class:`CoastalCalibConfig`.

    If the download directory does not exist (e.g. download is disabled),
    the catalog is skipped — there are no data files to reference.
    """

    name = "sfincs_data_catalog"
    description = "Generate HydroMT data catalog for SFINCS"

    def run(self) -> dict[str, Any]:
        """Generate a HydroMT data catalog YAML for downloaded data."""
        download_dir = self.config.paths.download_dir
        if not download_dir.exists():
            self._log(f"Download dir {download_dir} does not exist — skipping catalog generation")
            return {"catalog_path": None, "entries": [], "status": "skipped"}

        self._update_substep("Generating data catalog")

        catalog_path = self.config.paths.work_dir / "data_catalog.yml"

        catalog = generate_data_catalog(
            self.config,
            output_path=catalog_path,
            catalog_name=f"sfincs_{self.config.simulation.coastal_domain}",
        )

        self._log(f"Data catalog written to {catalog_path}")

        return {
            "catalog_path": str(catalog_path),
            "entries": [e.name for e in catalog.entries],
            "status": "completed",
        }

    def validate(self) -> list[str]:
        """Check that the download directory exists.

        When ``download.enabled`` is ``True`` the directory will be
        created by the download stage, so this check is skipped.
        """
        errors = super().validate()
        if not self.config.download.enabled:
            download_dir = self.config.paths.download_dir
            if not download_dir.exists():
                errors.append(f"Download directory does not exist: {download_dir}")
        return errors
