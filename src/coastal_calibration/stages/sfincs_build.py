"""SFINCS model build stages using HydroMT-SFINCS Python API.

All stages subclass :class:`~coastal_calibration.stages.base.WorkflowStage`
and accept a :class:`~coastal_calibration.config.schema.CoastalCalibConfig`.
SFINCS-specific settings are read from ``config.model_config``
(:class:`~coastal_calibration.config.schema.SfincsModelConfig`).

The HydroMT ``SfincsModel`` instance is shared between stages via a
module-level registry keyed by config ``id``.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pyproj import Transformer

from coastal_calibration.config.schema import SfincsModelConfig
from coastal_calibration.stages.base import WorkflowStage
from coastal_calibration.stages.sfincs import create_nc_symlinks, generate_data_catalog

if TYPE_CHECKING:
    import xarray as xr
    from hydromt_sfincs import SfincsModel
    from numpy.typing import NDArray

    from coastal_calibration.config.schema import CoastalCalibConfig
    from coastal_calibration.utils.logging import WorkflowMonitor


TransformerCRS = lru_cache(Transformer.from_crs)

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


def _filter_chrtout_files(
    files: list[Path],
    start: datetime,
    end: datetime,
) -> list[Path]:
    """Keep only CHRTOUT files whose filename timestamp falls within [start, end].

    Filenames follow ``YYYYMMDDHHMM.CHRTOUT_DOMAIN1`` convention.
    Files that cannot be parsed are silently skipped.
    """
    kept: list[Path] = []
    for f in files:
        stem = f.name.split(".")[0]
        try:
            ts = datetime.strptime(stem[:12], "%Y%m%d%H%M")
        except (ValueError, IndexError):
            try:
                ts = datetime.strptime(stem[:10], "%Y%m%d%H")
            except (ValueError, IndexError):
                continue
        if start <= ts <= end:
            kept.append(f)
    return kept


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
            "SFINCS model not initialized and no sfincs.inp found at "
            f"{root}.  Ensure the 'sfincs_init' stage runs first."
        )

    from hydromt_sfincs import SfincsModel as _Sfincs

    data_libs: list[str] = []
    catalog_path = _data_catalog_path(config)
    if catalog_path is not None:
        data_libs.append(str(catalog_path))

    model = _Sfincs(data_libs=data_libs, root=str(root), mode="r+", write_gis=True)
    model.read()
    _MODEL_REGISTRY[id(config)] = model
    return model


def _clear_model(config: CoastalCalibConfig) -> None:  # pyright: ignore[reportUnusedFunction]
    """Remove the SfincsModel instance for the given config."""
    _MODEL_REGISTRY.pop(id(config), None)


def _set_forcing_filename(component: Any, filename: str) -> None:
    """Set the output filename on a HydroMT-SFINCS forcing component.

    HydroMT-SFINCS (v2.0.0.dev0) does not expose a public API for
    overriding the forcing output filename — only the private
    ``_filename`` attribute exists.  This helper centralises that
    access so it is easy to find and update if the upstream API
    changes.
    """
    component._filename = filename


def _meteo_dst_res(config: CoastalCalibConfig, model: SfincsModel) -> float:
    """Return the output resolution (m) for gridded meteo forcing.

    When the user specifies ``meteo_res`` in the configuration it is
    returned directly.  Otherwise the base (coarsest) cell size of the
    SFINCS grid is used: meteorological fields vary slowly in space, so
    there is no benefit to a meteo grid finer than the coarsest
    computational cell.

    This also avoids the LCC → UTM reprojection inflation problem where
    ``rioxarray.reproject`` produces an output grid spanning the entire
    CONUS extent when no target resolution is specified.
    """
    sfincs = _sfincs_cfg(config)
    if sfincs.meteo_res is not None:
        return sfincs.meteo_res

    # Read the base cell size from the grid.  For both regular and
    # quadtree grids ``dx`` / ``dy`` are usually recorded in sfincs.inp.
    dx = float(model.config.get("dx", 0))
    dy = float(model.config.get("dy", 0))
    if dx > 0 and dy > 0:
        return max(dx, dy)

    # ``dx`` / ``dy`` are not always in the config; fall back to the
    # grid dataset where the values are stored as attributes.
    try:
        grid_ds = model.quadtree_grid.data if model.grid_type == "quadtree" else model.grid.data
        dx = float(getattr(grid_ds, "attrs", {}).get("dx", 0))
        dy = float(getattr(grid_ds, "attrs", {}).get("dy", 0))
        if dx > 0 and dy > 0:
            return max(dx, dy)
    except Exception:  # noqa: S110
        pass

    # Absolute fallback: NWM native resolution (~1 km).
    return 1000.0


def _create_meteo_forcing(
    model: SfincsModel,
    dataset_name: str,
    variables: list[str],
    dst_res: float,
    *,
    fill_value: float = 0.0,
    domain_buffer: float = 10_000.0,
) -> tuple[xr.Dataset | xr.DataArray, float]:
    """Fetch meteo data, clip in source CRS, and reproject to model CRS.

    This replaces the upstream ``component.create()`` +
    ``_clip_meteo_to_domain()`` pattern.  The heavy lifting (clip in
    source CRS, constrained reproject) is delegated to
    :func:`~coastal_calibration.utils.raster.clip_and_reproject`.

    Parameters
    ----------
    model : SfincsModel
        HydroMT SFINCS model (already initialised with data catalog).
    dataset_name : str
        Catalog key for the meteo dataset.
    variables : list[str]
        Variable names to select (e.g. ``["wind10_u", "wind10_v"]``).
    dst_res : float
        Target resolution in metres (model CRS units).
    fill_value : float
        Value used where no data is available after reprojection.
    domain_buffer : float
        Buffer in metres around the model region for the output grid.
        Default 10 km gives SFINCS comfortable interpolation headroom.

    Returns
    -------
    (data, time_interval_s)
        *data* — reprojected :class:`xarray.Dataset` (or DataArray for
        single-variable requests) with dimensions renamed to ``x``/``y``.
        *time_interval_s* — time step of the source data in seconds.
    """
    import pandas as pd

    from coastal_calibration.utils.raster import clip_and_reproject

    # ------------------------------------------------------------------
    # 1. Fetch data clipped roughly to the model domain
    # ------------------------------------------------------------------
    # ``buffer`` in ``get_rasterdataset`` is expressed as a number of
    # source-resolution cells.  30 cells is generous enough to survive
    # the bbox round-trip through WGS 84 without missing any data near
    # the boundary, while staying far below the upstream default of 5 000
    # (which returns essentially all of CONUS for NWM).
    fetch_buffer_cells = 30

    model_bbox: Any = model.bbox
    data_or_none = model.data_catalog.get_rasterdataset(
        dataset_name,
        bbox=model_bbox,  # WGS 84
        buffer=fetch_buffer_cells,
        time_range=model.get_model_time(),
        variables=variables,
        single_var_as_array=len(variables) == 1,
    )
    if data_or_none is None:
        msg = f"Meteo dataset '{dataset_name}' not found in data catalog."
        raise ValueError(msg)
    data: xr.Dataset | xr.DataArray = data_or_none

    # ------------------------------------------------------------------
    # 2. Validate spatial / temporal extent
    # ------------------------------------------------------------------
    y_dim: str
    x_dim: str
    y_dim, x_dim = data.raster.dims
    if data.coords[x_dim].size < 2 or data.coords[y_dim].size < 2:
        msg = (
            f"Meteo data '{dataset_name}' has fewer than 2 cells on "
            f"the x or y axis after spatial clipping.  Check the input "
            f"data and the model region or increase the fetch buffer."
        )
        raise ValueError(msg)
    if data.coords["time"].size < 2:
        msg = (
            f"Meteo data '{dataset_name}' does not overlap with the "
            f"model time range.  Check input data and model config."
        )
        raise ValueError(msg)

    _time_diff: Any = np.diff(data.time).mean()
    _td: Any = pd.to_timedelta(_time_diff)
    time_interval_s: float = float(_td.total_seconds())

    # ------------------------------------------------------------------
    # 3. Clip in source CRS + constrained reproject
    # ------------------------------------------------------------------
    region = model.region.total_bounds  # (xmin, ymin, xmax, ymax)
    _bounds = (float(region[0]), float(region[1]), float(region[2]), float(region[3]))
    data = clip_and_reproject(
        data,
        dst_bounds=_bounds,
        dst_crs=model.crs,
        dst_res=dst_res,
        fill_value=fill_value,
        buffer=domain_buffer,
    )

    # Rename dims to SFINCS convention (always x / y).
    y_dim, x_dim = data.raster.dims
    data = data.rename({y_dim: "y", x_dim: "x"})

    return data, time_interval_s


def _waterlevel_geodataset(config: CoastalCalibConfig) -> str | None:
    """Return the geodataset name for water-level forcing, or None."""
    catalog_path = _data_catalog_path(config)
    if catalog_path is None:
        return None
    coastal_source = config.boundary.source
    return f"{coastal_source}_waterlevel" if coastal_source != "tpxo" else "tpxo_tidal"


# ---------------------------------------------------------------------------
# Subprocess helpers (used by the run stage)
# ---------------------------------------------------------------------------


def _run_and_log(
    cmd: list[str],
    model_root: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, stream output to ``sfincs_log.txt``, and raise on failure.

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
            if proc.stdout is None or proc.stderr is None:
                msg = "Popen streams not available (stdout/stderr must use PIPE)"
                raise RuntimeError(msg)
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
        if not isinstance(config.model_config, SfincsModelConfig):
            msg = f"Expected SfincsModelConfig, got {type(config.model_config).__name__}"
            raise TypeError(msg)
        self.sfincs: SfincsModelConfig = config.model_config


class SfincsInitStage(_SfincsStageBase):
    """Initialize the SFINCS model (pre-built mode).

    Copies pre-built model files to the output directory, opens the model
    in ``r+`` mode, reads it, and clears missing file references.
    """

    name = "sfincs_init"
    description = "Initialize SFINCS model (pre-built)"

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

    #: Generated files that downstream stages will recreate.
    #: Stale files from a previous run can crash the HDF5 library when
    #: ``write_netcdf_safely`` (or a lazy ``model.read()``) tries to open
    #: them, so we delete them at the start of every fresh initialization.
    _STALE_OUTPUT_FILES: tuple[str, ...] = (
        "sfincs_netbndbzsbzifile.nc",
        "sfincs_netamuv.nc",
        "sfincs_netamp.nc",
        "sfincs_netampr.nc",
        "sfincs_netsrcdisfile.nc",
        # SFINCS output files (not strictly forcing but can interfere)
        "sfincs_map.nc",
        "sfincs_his.nc",
    )

    #: Files that must be restored from the pre-built model on every
    #: re-run.  Core model definition files (``sfincs.inp``,
    #: ``sfincs.nc``, ``sfincs.bnd``) are included so that updates to
    #: the pre-built model are always picked up.  Without overwriting,
    #: a stale file from a previous run would persist and cause silent
    #: errors (e.g. wrong boundary cells, wrong grid, etc.).
    #:
    #: ``sfincs_subgrid.nc`` must also be overwritten because it is
    #: tightly coupled to the quadtree grid topology stored in
    #: ``sfincs.nc``.  A stale subgrid file left by a previous run
    #: with a different grid (e.g. different refinement) has a
    #: different number of cells, causing SFINCS to read mismatched
    #: bed-level data and produce incorrect (flat) water-level output.
    _ALWAYS_OVERWRITE_FILES: tuple[str, ...] = (
        "sfincs.inp",
        "sfincs.nc",
        "sfincs.bnd",
        "sfincs.obs",
        "sfincs_subgrid.nc",
        "sfincs.src",
        "sfincs.dis",
    )

    def _remove_stale_outputs(self, root: Path) -> None:
        """Delete generated files left over from a previous run.

        The HydroMT ``write_netcdf_safely`` helper opens existing files to
        check whether the data changed.  If a previous run produced files
        with an incompatible schema (e.g. different number of stations)
        the HDF5 library can segfault.  Removing them up front avoids the
        issue entirely; every downstream stage will regenerate its files.
        """
        removed: list[str] = []
        for name in self._STALE_OUTPUT_FILES:
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
        from hydromt_sfincs import SfincsModel

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
            overwrite = {f.lower() for f in self._ALWAYS_OVERWRITE_FILES}
            for src_file in source_dir.iterdir():
                if src_file.is_file():
                    dst_file = root / src_file.name
                    if not dst_file.exists() or src_file.name.lower() in overwrite:
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

        self._log(f"SFINCS model initialized (grid_type={model.grid_type}) at {root}")

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

        timing: dict[str, object] = {
            "tref": start,
            "tstart": start,
            "tstop": stop,
        }

        # Add a 1-hour spinup period when the simulation is long enough.
        # During spinup, SFINCS gradually ramps the boundary forcing from
        # zero to the actual value, avoiding shock waves in the domain.
        spinup_seconds = 3600
        total_seconds = int(sim.duration_hours * 3600)
        if total_seconds > spinup_seconds:
            timing["tspinup"] = spinup_seconds
            self._log(f"Spinup: {spinup_seconds} s")

        model.config.update(timing)

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
    def _parse_quadtree_bnd(nc_path: Path) -> list[tuple[float, float, str]]:
        """Extract boundary-cell face centers from a quadtree ``sfincs.nc``.

        For quadtree grids HydroMT-SFINCS stores boundary info in the
        mask array (value 2) inside ``sfincs.nc`` instead of writing a
        separate ``.bnd`` file.  This helper reads the UGRID mesh
        connectivity, computes face centers for boundary cells, and
        returns them in the same ``(x, y, name)`` format as
        :meth:`_parse_bnd_file`.
        """
        import numpy as np
        import xarray as xr

        ds = xr.open_dataset(nc_path)
        try:
            mask = ds["mask"].to_numpy()
            bnd_idx = np.where(mask == 2)[0]
            if len(bnd_idx) == 0:
                return []

            node_x = ds["mesh2d_node_x"].to_numpy()
            node_y = ds["mesh2d_node_y"].to_numpy()
            face_nodes = ds["mesh2d_face_nodes"].to_numpy()  # (nFaces, max_nodes)

            face_x = np.array(
                [
                    node_x[face_nodes[i][~np.isnan(face_nodes[i])].astype(int)].mean()
                    for i in bnd_idx
                ]
            )
            face_y = np.array(
                [
                    node_y[face_nodes[i][~np.isnan(face_nodes[i])].astype(int)].mean()
                    for i in bnd_idx
                ]
            )

            # Sort bottom-left → top-right for a deterministic order
            order = np.lexsort((face_x, face_y))
            face_x = face_x[order]
            face_y = face_y[order]
        finally:
            ds.close()

        return [(float(face_x[i]), float(face_y[i]), f"bnd_{i}") for i in range(len(face_x))]

    @staticmethod
    def _parse_regular_grid_bnd(
        model: SfincsModel,
    ) -> list[tuple[float, float, str]]:
        """Extract boundary-cell centers from a regular-grid mask.

        HydroMT-SFINCS marks boundary cells as ``mask==2`` but does not
        always write a separate ``.bnd`` file for regular grids.  This
        helper reads the mask :class:`~xarray.Dataset` via the HydroMT
        component, finds cells with value 2, and returns their grid
        coordinates in the same ``(x, y, name)`` format as
        :meth:`_parse_bnd_file`.
        """
        import numpy as np

        msk_var: Any = model.mask.data["mask"]
        bnd_mask: NDArray[Any] = np.asarray(msk_var.values == 2)
        y_idx, x_idx = np.where(bnd_mask)
        if len(y_idx) == 0:
            return []

        x_coords: NDArray[Any] = np.asarray(msk_var.coords["x"].to_numpy())
        y_coords: NDArray[Any] = np.asarray(msk_var.coords["y"].to_numpy())
        bnd_x: NDArray[Any] = x_coords[x_idx]
        bnd_y: NDArray[Any] = y_coords[y_idx]

        # Sort bottom-left → top-right for a deterministic order
        order = np.lexsort((bnd_x, bnd_y))
        bnd_x = bnd_x[order]
        bnd_y = bnd_y[order]

        return [(float(bnd_x[i]), float(bnd_y[i]), f"bnd_{i}") for i in range(len(bnd_x))]

    def _get_boundary_points(
        self,
        model: SfincsModel,
    ) -> list[tuple[float, float, str]]:
        """Return boundary point coordinates.

        For regular grids the points are read from ``sfincs.bnd``;
        when the file is absent the boundary cells (``mask==2``) are
        extracted directly from the mask.
        For quadtree grids the points are extracted from the mask
        array (value 2) in ``sfincs.nc`` when no ``.bnd`` file exists.
        """
        model_root = get_model_root(self.config)
        bnd_path = model_root / "sfincs.bnd"

        if bnd_path.exists():
            points = self._parse_bnd_file(bnd_path)
            if points:
                self._log(f"Read {len(points)} boundary point(s) from {bnd_path}")
                return points

        # Fall back to mask-based extraction
        if model.grid_type == "quadtree":
            qtr_name = model.config.get("qtrfile", "sfincs.nc")
            nc_path = model_root / qtr_name
            if nc_path.exists():
                points = self._parse_quadtree_bnd(nc_path)
                if points:
                    self._log(
                        f"Extracted {len(points)} boundary point(s) "
                        f"from quadtree mask in {nc_path.name}"
                    )
                    return points
        else:
            points = self._parse_regular_grid_bnd(model)
            if points:
                self._log(f"Extracted {len(points)} boundary point(s) from regular grid mask")
                return points

        raise FileNotFoundError(
            f"No boundary points found.  For regular grids provide "
            f"sfincs.bnd in {model_root} or ensure boundary cells "
            f"(mask==2) exist in the mask.  For quadtree grids ensure "
            f"boundary cells (mask==2) exist in sfincs.nc."
        )

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

        from coastal_calibration.tides import TIDES_DATA_DIR

        for fname in ("setup_tpxo.txt", "Model_tpxo10_atlas"):
            src = TIDES_DATA_DIR / fname
            dst = model_root / fname
            if not dst.exists():
                _shutil.copy2(src, dst)

        tpxo_data_dir = self.config.paths.tpxo_data_dir
        link_target = model_root / "TPXO10_atlas_v2_nc"
        if not link_target.exists():
            link_target.symlink_to(tpxo_data_dir)
            self._log(f"Symlinked {tpxo_data_dir} -> {link_target}")

    def _run_predict_tide(self, model_root: Path) -> Path:
        """Run the OTPS ``predict_tide`` binary.

        Returns the path to the ``otps_out.txt`` output file.
        The ``predict_tide`` binary is expected on ``$PATH``
        (pixi installs it to ``$CONDA_PREFIX/bin``).
        """
        env = self.build_environment()

        if self.config.paths.otps_dir is not None:
            predict_tide_bin = str(self.config.paths.otps_dir / "predict_tide")
        else:
            predict_tide_bin = "predict_tide"

        result = subprocess.run(
            ["bash", "-c", f"cd {model_root} && {predict_tide_bin} < setup_tpxo.txt"],
            cwd=model_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            self._log(f"predict_tide failed: {result.stderr[-2000:]}", "error")
            raise RuntimeError(
                f"predict_tide failed (exit {result.returncode}): {result.stderr[-2000:]}"
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
            series = subset.set_index("datetime")["z(m)"].astype(np.float64)
            series.name = idx
            point_dfs.append(series)

        df_ts = pd.concat(point_dfs, axis=1)
        df_ts.index.name = "time"
        return df_ts

    @staticmethod
    def _write_bnd_file(bnd_path: Path, gdf_bnd: Any) -> None:
        """Write a ``sfincs.bnd`` file from a boundary-point GeoDataFrame.

        Each line contains the X and Y coordinates of a boundary point
        in the model CRS, matching the column order in the
        ``netbndbzsbzifile`` NetCDF.
        """
        lines: list[str] = []
        for _, row in gdf_bnd.iterrows():
            geom = row.geometry
            lines.append(f"    {geom.x:.1f}   {geom.y:.1f}")
        bnd_path.write_text("\n".join(lines) + "\n")

    def _inject_water_level(
        self,
        model: SfincsModel,
        df_ts: Any,
        gdf_bnd: Any,
    ) -> None:
        """Inject water-level forcing into the HydroMT model.

        For regular grids SFINCS needs *both* ``bndfile`` (boundary point
        locations) and ``netbndbzsbzifile`` (time-varying water levels).
        For quadtree grids the boundary cell locations are already encoded
        in the mask (value 2) inside ``sfincs.nc``, so ``bndfile`` is only
        restored when a ``.bnd`` file is present on disk.

        A zero-filled ``bzi`` (infragravity) variable is added because the
        SFINCS binary unconditionally queries ``zi`` in the netCDF file
        (``sfincs_ncinput.F90:118``) and crashes if it is absent.
        """
        import xarray as xr_

        for key in ("bzsfile", "bzifile", "bndfile", "bcafile", "netbndbzsbzifile"):
            model.config.set(key, None)

        _wl_init: Any = model.water_level.data

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

        model_root = get_model_root(self.config)
        bnd_path = model_root / "sfincs.bnd"

        # Ensure a sfincs.bnd file exists.  For quadtree grids the
        # boundary cells are encoded in the mask, but SFINCS still
        # needs ``bndfile`` to map the netbndbzsbzifile columns to
        # boundary cells.  Write one from the GeoDataFrame if missing.
        if not bnd_path.exists():
            self._write_bnd_file(bnd_path, gdf_bnd)

        model.config.set("bndfile", "sfincs.bnd")
        model.config.set("bndtype", 1)
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

        model_root = get_model_root(self.config)

        # 1. Read boundary points (sfincs.bnd or quadtree mask fallback)
        bnd_points = self._get_boundary_points(model)
        xx, yy, names = zip(*bnd_points, strict=True)

        # 2. Build GeoDataFrame for boundary locations
        gdf_bnd = gpd.GeoDataFrame(  # ty: ignore[no-matching-overload]
            {"name": list(names)},
            geometry=gpd.points_from_xy(list(xx), list(yy), crs=model.crs),
        )

        # 3. Transform to lon/lat
        lonlats = cast(
            "list[tuple[float, float]]", gdf_bnd.to_crs(4326).get_coordinates().values.tolist()
        )

        # 4. Generate OTPS input file
        tstart = model.config.data.tstart
        tstop = model.config.data.tstop
        if isinstance(tstart, str):
            tstart = _dt.fromisoformat(tstart)
        if isinstance(tstop, str):
            tstop = _dt.fromisoformat(tstop)

        otps_input_path = model_root / "otps_lat_lon_time.txt"
        self._write_otps_input(otps_input_path, lonlats, tstart, tstop, self._TPXO_RAW_DT)
        self._log(f"Wrote OTPS input ({len(lonlats)} points) to {otps_input_path}")

        # 5. Copy setup files and symlink atlas data
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
            Number of nearest neighbors to use (capped at N).

        Returns
        -------
        ndarray, shape (T, M)
            Interpolated values at each target point.
        """
        from scipy.spatial import KDTree

        k = min(k, len(src_xy))
        tree = KDTree(src_xy)
        _qresult: Any = tree.query(target_xy, k=k)
        dists = np.asarray(_qresult[0])
        idxs = np.asarray(_qresult[1])

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

        self._update_substep(f"Loading {geodataset_name} data")
        tstart, tstop = model.get_model_time()

        xx, yy, _ = zip(*bnd_points, strict=False)
        bbox_gdf = gpd.GeoDataFrame(  # ty: ignore[no-matching-overload]
            geometry=gpd.points_from_xy(list(xx), list(yy), crs=model.crs),
        ).to_crs(4326)

        da = model.data_catalog.get_geodataset(
            geodataset_name,
            geom=bbox_gdf,
            buffer=50e3,  # 50 km buffer to ensure enough source nodes
            variables=["waterlevel"],
            time_range=(tstart, tstop),
        )
        dims = ", ".join(f"{dim}={size}" for dim, size in da.sizes.items())
        self._log(f"Loaded {geodataset_name}: {dims}")
        return da

    def _interpolate_to_bnd(
        self,
        da: xr.DataArray | xr.Dataset,
        bnd_points: list[tuple[float, float, str]],
        model_crs: Any,
    ) -> Any:
        """IDW-interpolate a geodataset to the boundary point locations."""
        import pandas as pd

        src_gdf = da.vector.to_gdf()
        src_xy = np.column_stack([src_gdf.geometry.x.values, src_gdf.geometry.y.values])

        # Transform boundary points to the same CRS as the source data
        src_crs = da.vector.crs
        if src_crs is not None and src_crs != model_crs:
            tr = TransformerCRS(model_crs, src_crs, always_xy=True)
            xx, yy, _ = zip(*bnd_points, strict=False)
            bnd_in_src = list(zip(*tr.transform(xx, yy), strict=False))
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

        1. Reads boundary locations from ``sfincs.bnd`` (regular grids) or
           from the quadtree mask in ``sfincs.nc`` (quadtree grids).
        2. Loads the geodataset (e.g. STOFS water levels).
        3. Spatially interpolates the geodataset to the boundary points
           using inverse-distance weighting from the nearest source nodes.
        4. Injects the result into HydroMT the same way the TPXO path does.
        """
        import geopandas as gpd

        # 1. Read boundary points (sfincs.bnd or quadtree mask fallback)
        bnd_points = self._get_boundary_points(model)
        n_bnd = len(bnd_points)

        # 2. Load the geodataset clipped around the boundary points
        da = self._load_geodataset_for_bnd(model, geodataset_name, bnd_points)

        # 3. Spatially interpolate geodataset to boundary point locations
        self._update_substep("Interpolating to boundary points")
        df_ts = self._interpolate_to_bnd(da, bnd_points, model.crs)
        self._log(
            f"Interpolated {geodataset_name} to {n_bnd} boundary points ({len(df_ts)} time steps)"
        )

        # 4. Build GeoDataFrame for boundary locations (model CRS)
        xx, yy, names = zip(*bnd_points, strict=True)
        gdf_bnd = gpd.GeoDataFrame(  # ty: ignore[no-matching-overload]
            {"name": list(names)},
            geometry=gpd.points_from_xy(list(xx), list(yy), crs=model.crs),
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
        that do not land on an active cell.

        Supports both quadtree (unstructured) and regular (structured)
        SFINCS grids.  For regular grids the cell index is computed
        arithmetically via ``SfincsGrid.get_indices_at_points`` (O(1)
        per point, handles rotated grids).  For quadtree grids a KDTree
        lookup over face centers is used.

        Returns
        -------
        kept, dropped
            ``kept`` are the (x, y, name) tuples on active cells;
            ``dropped`` are the names of the removed points.
        """
        import numpy as np

        kept: list[tuple[float, float, str]] = []
        dropped: list[str] = []

        if model.grid_type == "quadtree":
            from scipy.spatial import KDTree

            grid_ds = model.quadtree_grid.data
            ugrid = grid_ds.ugrid.grid
            face_xy = np.column_stack([ugrid.face_x, ugrid.face_y])
            mask = grid_ds["mask"].to_numpy()

            tree = KDTree(face_xy)
            for x, y, name in points:
                _qr: Any = tree.query([x, y])
                idx: int = int(_qr[1])
                if mask[idx] == 1:
                    kept.append((x, y, name))
                else:
                    dropped.append(name)
        else:
            # Regular grid: direct index computation (no KDTree needed).
            # ``get_indices_at_points`` returns flat indices into the
            # (mmax, nmax) grid; the mask must be ravelled in Fortran
            # order to match.
            mask_flat = model.grid.mask.to_numpy().ravel(order="F")
            pts_x = np.array([x for x, _, _ in points])
            pts_y = np.array([y for _, y, _ in points])
            inds = model.grid.get_indices_at_points(pts_x, pts_y).ravel()

            for (x, y, name), idx in zip(points, inds, strict=True):
                if idx >= 0 and mask_flat[idx] == 1:
                    kept.append((x, y, name))
                else:
                    dropped.append(name)

        return kept, dropped

    def _assign_discharge_timeseries(self, model: Any) -> None:
        """Assign NWM CHRTOUT discharge timeseries to existing source points.

        For ``nwm_retro`` reads directly from the S3 Zarr store.
        For ``nwm_ana`` reads from downloaded CHRTOUT netCDF files.
        """
        import pandas as pd

        from coastal_calibration.utils.streamflow import read_streamflow

        if model.discharge_points.nr_points == 0:
            return

        gdf = model.discharge_points.gdf
        if "name" not in gdf.columns or gdf.empty:
            self._log("Discharge points have no 'name' column, cannot match feature_ids")
            return

        try:
            fid_by_idx = {idx: int(name) for idx, name in gdf["name"].items()}
        except (ValueError, TypeError):
            self._log(
                "Discharge point names are not integer feature_ids, skipping timeseries assignment"
            )
            return

        self._update_substep("Loading discharge timeseries")
        tstart, tstop = model.get_model_time()
        needed_fids = list(fid_by_idx.values())
        sim = self.config.simulation
        start_dt = pd.Timestamp(tstart).to_pydatetime()
        end_dt = pd.Timestamp(tstop).to_pydatetime()

        if sim.meteo_source == "nwm_retro":
            df_fid = read_streamflow(
                needed_fids,
                start_dt,
                end_dt,
                meteo_source="nwm_retro",
                domain=sim.coastal_domain,
            )
        else:
            streamflow_dir = self.config.paths.streamflow_dir(sim.meteo_source, sim.coastal_domain)
            files = _filter_chrtout_files(
                sorted(streamflow_dir.glob("*.CHRTOUT_DOMAIN1*")),
                start_dt,
                end_dt,
            )
            if not files:
                self._log(
                    f"No CHRTOUT files found in {streamflow_dir} "
                    f"for {start_dt:%Y-%m-%d %H:%M}-{end_dt:%Y-%m-%d %H:%M}, "
                    "discharge points will use default (zero) values"
                )
                return
            df_fid = read_streamflow(
                needed_fids,
                start_dt,
                end_dt,
                meteo_source="nwm_ana",
                chrtout_files=files,
            )

        if df_fid.empty:
            self._log(
                "Could not load streamflow data, discharge points will use default (zero) values"
            )
            return

        available_fids = {int(c) for c in df_fid.columns}
        valid_fids = sorted(set(needed_fids) & available_fids)
        missing_fids = set(needed_fids) - available_fids

        if not valid_fids:
            self._log(
                f"No matching feature_ids — all {len(missing_fids)} point(s) keep zero discharge"
            )
            return

        idxs_by_fid: dict[int, list[int]] = {}
        for idx, fid in fid_by_idx.items():
            idxs_by_fid.setdefault(fid, []).append(idx)

        df_ts = pd.DataFrame(index=df_fid.index)
        n_assigned = 0
        for fid in valid_fids:
            for idx in idxs_by_fid[fid]:
                df_ts[idx] = df_fid[fid].values
                n_assigned += 1
        df_ts.index.name = "time"
        df_ts.columns.name = "index"

        model.discharge_points.set_timeseries(df_ts)

        msg = f"Assigned discharge timeseries to {n_assigned} point(s) ({len(valid_fids)} unique feature_id(s))"
        if missing_fids:
            examples = ", ".join(str(f) for f in sorted(missing_fids)[:5])
            msg += f"; {len(missing_fids)} point(s) not found (zero discharge): {examples}"
        self._log(msg)

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
            # When merging, skip points that already exist (by name)
            # to prevent duplicates on re-runs.
            if self.sfincs.merge_discharge and model.discharge_points.nr_points > 0:
                existing_names = set(model.discharge_points.gdf.get("name", []))
                before = len(kept)
                kept = [(x, y, n) for x, y, n in kept if n not in existing_names]
                n_skipped = before - len(kept)
                if n_skipped:
                    self._log(f"Skipped {n_skipped} duplicate source point(s)")
            for x, y, name in kept:
                model.discharge_points.add_point(x=x, y=y, name=name)
            self._log(f"Added {len(kept)} discharge source point(s) from {src_path}")
        else:
            model.discharge_points.create(
                locations=str(src_path),
                merge=self.sfincs.merge_discharge,
            )
            self._log(f"Discharge source points added from {src_path}")

        # Assign real discharge timeseries from the NWM CHRTOUT data
        # catalog entry.  If no streamflow data is available the method
        # is a no-op and the points keep their default zero discharge.
        self._assign_discharge_timeseries(model)

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
        try:
            precip_data, dt_s = _create_meteo_forcing(
                model,
                meteo_dataset,
                ["precip"],
                dst_res,
            )
            precip = cast("xr.DataArray", precip_data)

            # Convert cumulative precipitation (mm) → rate (mm/hr).
            # NWM LDASIN stores accumulated precip over each time step
            # with the timestamp at the end of the interval ("right"
            # label).  Shift left so the rate applies to the upcoming
            # interval, matching SFINCS convention.
            precip = precip / (dt_s / 3600.0)
            precip = precip.shift(time=-1, fill_value=0)

            # Lower dtwnd if the source data has a finer time step.
            dtwnd = model.config.get("dtwnd", 1800)
            if dtwnd > dt_s:
                model.config.set("dtwnd", dt_s)

            precip = precip.rename("precip_2d")
            model.precipitation.set(precip, name="precip_2d")
            model.config.set("netamprfile", "sfincs_netampr.nc")
            _set_forcing_filename(model.precipitation, "sfincs_netampr.nc")

            self._log(f"Precipitation forcing added from {meteo_dataset} (res={dst_res:.0f} m)")

            model.precipitation.write()
        finally:
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
        try:
            wind, dt_s = _create_meteo_forcing(
                model,
                meteo_dataset,
                ["wind10_u", "wind10_v"],
                dst_res,
            )

            dtwnd = model.config.get("dtwnd", 1800)
            if dtwnd > dt_s:
                model.config.set("dtwnd", dt_s)

            model.wind.set(wind, name="wind_2d")
            model.config.set("netamuamvfile", "sfincs_netamuv.nc")
            _set_forcing_filename(model.wind, "sfincs_netamuv.nc")

            self._log(f"Wind forcing added from {meteo_dataset} (res={dst_res:.0f} m)")

            model.wind.write()
        finally:
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
        try:
            press_data, dt_s = _create_meteo_forcing(
                model,
                meteo_dataset,
                ["press_msl"],
                dst_res,
                fill_value=101325.0,
            )
            press = cast("xr.DataArray", press_data)

            dtwnd = model.config.get("dtwnd", 1800)
            if dtwnd > dt_s:
                model.config.set("dtwnd", dt_s)

            press = press.rename("press_2d")
            model.pressure.set(press, name="press_2d")
            model.config.set("netampfile", "sfincs_netamp.nc")
            _set_forcing_filename(model.pressure, "sfincs_netamp.nc")

            # Enable barometric pressure correction so SFINCS uses the forcing.
            # pavbnd / gapres set the reference atmospheric pressure (Pa) at the
            # open boundaries and the "gap" pressure for inverse-barometer
            # correction.  Standard atmosphere ≈ 101 325 Pa; 101 200 Pa is the
            # conventional SFINCS default.
            model.config.set("baro", 1)
            model.config.set("pavbnd", 101200)
            model.config.set("gapres", 101200)

            self._log(
                f"Atmospheric pressure forcing added from {meteo_dataset} "
                f"(baro=1, res={dst_res:.0f} m)"
            )

            model.pressure.write()
        finally:
            model.pressure.clear()

        return {"status": "completed"}


class SfincsWriteStage(WorkflowStage):
    """Write the SFINCS model files to disk."""

    name = "sfincs_write"
    description = "Write SFINCS model"

    def run(self) -> dict[str, Any]:
        """Write all model files to the model root directory."""
        model = _get_model(self.config)

        # Apply user-specified sfincs.inp overrides (e.g. physics params)
        # right before writing so they take final precedence.
        # Parameter name validation is done at config load time in
        # SfincsModelConfig.__post_init__.
        overrides: dict[str, Any] = getattr(self.config.model_config, "inp_overrides", {})
        if overrides:
            model.config.update(overrides)
            self._log(f"Applied {len(overrides)} sfincs.inp override(s): {overrides}")

        self._update_substep("Writing model to disk")
        model.write()

        root = get_model_root(self.config)
        self._log(f"SFINCS model written to {root}")

        return {
            "model_root": str(root),
            "status": "completed",
        }


class SfincsRunStage(_SfincsStageBase):
    """Run the SFINCS model via a compiled native executable.

    Resolution order for the SFINCS binary:

    1. ``model_config.sfincs_exe`` -- explicit path set in the config.
    2. ``sfincs`` found on ``PATH``.

    If neither is available the stage raises :class:`RuntimeError`.
    """

    name = "sfincs_run"
    description = "Run SFINCS model"

    @staticmethod
    def _resolve_exe(sfincs_cfg: SfincsModelConfig) -> Path:
        """Return the SFINCS executable or raise with a helpful message."""
        if sfincs_cfg.sfincs_exe is not None:
            return sfincs_cfg.sfincs_exe
        found = shutil.which("sfincs")
        if found is not None:
            return Path(found)
        raise RuntimeError(
            "SFINCS executable not found.  Either:\n"
            "  1. Activate a pixi environment with the 'sfincs' feature"
            " (builds automatically on first activation).\n"
            "  2. Set 'sfincs_exe' in the config to the path of an existing binary."
        )

    # ------------------------------------------------------------------ #
    # Stage entry-point
    # ------------------------------------------------------------------ #

    def run(self) -> dict[str, Any]:
        """Execute SFINCS via native binary."""
        model_root = get_model_root(self.config)
        exe = self._resolve_exe(self.sfincs)

        self._log(f"Running SFINCS via native executable: {exe}")
        self._update_substep("Running SFINCS")
        if self.sfincs.sfincs_exe is not None:
            from coastal_calibration.utils.mpi import build_isolated_env

            env = build_isolated_env(
                omp_num_threads=self.sfincs.omp_num_threads,
                runtime_env=self.sfincs.runtime_env,
            )
        else:
            env = self.build_environment()
            if self.sfincs.runtime_env:
                env.update(self.sfincs.runtime_env)
        _run_and_log([str(exe)], model_root, env=env)

        self._log("SFINCS run completed")

        # NOTE: do *not* clear the model registry here.  The plot stage
        # runs immediately after and needs the in-memory SfincsModel to
        # avoid a UGRID re-read error on quadtree grids.  The registry
        # is garbage-collected when the process exits.

        return {"status": "completed", "mode": "native"}


class SfincsFloodMapStage(_SfincsStageBase):
    """Downscale SFINCS water levels to a high-resolution flood depth map.

    Reads ``zsmax`` (maximum water surface elevation) from the SFINCS
    map output, creates an index COG that maps DEM pixels to SFINCS
    grid cells, and calls :func:`hydromt_sfincs.utils.downscale_floodmap`
    to produce a Cloud Optimized GeoTIFF of flood depth.

    The stage is a no-op when:

    * ``floodmap_dem`` is not configured.
    * ``sfincs_map.nc`` does not exist.
    * ``zsmax`` is not present in the map output.
    """

    name = "sfincs_floodmap"
    description = "Downscale flood depth map"

    def run(self) -> dict[str, Any]:
        """Generate a downscaled flood depth COG from SFINCS output."""
        if self.sfincs.floodmap_dem is None:
            self._log("floodmap_dem not configured, skipping flood map stage")
            return {"status": "skipped", "reason": "no DEM configured"}

        if not self.sfincs.floodmap_enabled:
            self._log("Flood map generation disabled, skipping")
            return {"status": "skipped", "reason": "disabled"}

        model_root = get_model_root(self.config)
        map_file = model_root / "sfincs_map.nc"

        if not map_file.exists():
            self._log("sfincs_map.nc not found, skipping flood map stage")
            return {"status": "skipped", "reason": "no map output"}

        dem_path = self.sfincs.floodmap_dem
        if not dem_path.exists():
            self._log(f"DEM not found: {dem_path}, skipping flood map stage", "warning")
            return {"status": "skipped", "reason": "DEM not found"}

        self._update_substep("Reading SFINCS output")
        model = _get_model(self.config)

        from coastal_calibration.utils.floodmap import create_flood_depth_map

        self._update_substep("Downscaling flood depth")
        try:
            output_path = create_flood_depth_map(
                model_root=model_root,
                dem_path=dem_path,
                hmin=self.sfincs.floodmap_hmin,
                model=model,
                log=self._log,
            )
        except (KeyError, FileNotFoundError) as exc:
            self._log(
                f"Flood map generation failed ({exc}); skipping flood map stage",
                "warning",
            )
            return {"status": "skipped", "reason": str(exc)}

        self._log(f"Flood depth map: {output_path}")
        return {"status": "completed", "floodmap": str(output_path)}


class SfincsPlotStage(_SfincsStageBase):
    """Plot simulated water levels against NOAA CO-OPS observations.

    After the SFINCS run, this stage reads ``point_zs`` (water surface
    elevation) from the model output (``sfincs_his.nc``), spatially
    matches each model observation point to the nearest NOAA CO-OPS
    tide-gauge station, fetches observed water levels from the CO-OPS
    API, and produces a comparison time-series figure saved to
    ``<model_root>/figs/``.

    Station matching is purely spatial (KDTree nearest-neighbor in
    WGS 84) so the stage works with **any** SFINCS model — not only
    models created by this package.

    Observations are fetched in MLLW (universally supported by all
    CO-OPS stations) and then converted to MSL using per-station datum
    offsets from the CO-OPS metadata API, matching the STOFS boundary
    condition vertical reference used by SFINCS.

    The stage is a no-op when:

    * The model output file (``sfincs_his.nc``) does not exist.
    * No ``point_zs`` (or ``point_h``) variable is present in the output.
    * No observation points are within range of a NOAA CO-OPS station.
    """

    name = "sfincs_plot"
    description = "Plot simulated vs observed water levels"

    #: Maximum distance (degrees) between an observation point and a NOAA
    #: station to consider them a match.  0.1° ≈ 11 km — generous because
    #: obs points may be snapped to wet cells some distance from the gauge.
    _NOAA_MATCH_RADIUS_DEG: float = 0.1

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

    def _match_noaa_stations(self) -> tuple[list[int], list[str]]:
        """Find the nearest NOAA CO-OPS station for each observation point.

        Uses a spatial lookup so that the run workflow works with **any**
        SFINCS model, not only models created by this package.

        Steps:

        1. Read observation-point coordinates from the loaded model.
        2. Reproject to WGS 84.
        3. Query the full NOAA CO-OPS station catalog and build a KDTree.
        4. For each observation point, find the nearest station within
           :attr:`_NOAA_MATCH_RADIUS_DEG`.
        5. Keep only stations that have valid MSL and MLLW datum values
           (required for the MLLW → MSL conversion in the comparison plot).
        6. Deduplicate — if multiple obs points match the same station,
           keep the closest one.

        Returns
        -------
        noaa_indices : list[int]
            Observation-point indices that correspond to a NOAA CO-OPS station.
        noaa_station_ids : list[str]
            Corresponding CO-OPS station IDs (same order as *noaa_indices*).
        """
        import numpy as np
        from scipy.spatial import KDTree

        from coastal_calibration.coops_api import COOPSAPIClient

        model = _get_model(self.config)

        obs_gdf = model.observation_points.data
        if obs_gdf is None or obs_gdf.empty:  # pyright: ignore[reportUnnecessaryComparison]
            return [], []

        # Reproject observation points to WGS 84 for comparison with the
        # NOAA station catalog (also in EPSG:4326).
        obs_4326 = obs_gdf.to_crs(4326)
        obs_xy = np.column_stack(
            [
                obs_4326.geometry.x.to_numpy(),
                obs_4326.geometry.y.to_numpy(),
            ]
        )

        client = COOPSAPIClient()
        stations_gdf = client.stations_metadata
        if stations_gdf.empty:
            self._log("NOAA station catalog is empty", "warning")
            return [], []

        sta_xy = np.column_stack(
            [
                stations_gdf.geometry.x.to_numpy(),
                stations_gdf.geometry.y.to_numpy(),
            ]
        )
        sta_ids = stations_gdf["station_id"].to_numpy()
        tree = KDTree(sta_xy)

        # For each observation point, find the nearest NOAA station.
        _qresult: Any = tree.query(obs_xy)
        dists_arr: NDArray[np.floating[Any]] = np.asarray(_qresult[0])
        idxs_arr: NDArray[np.intp] = np.asarray(_qresult[1])
        radius = self._NOAA_MATCH_RADIUS_DEG

        # Collect (obs_index, station_id, distance) for matches within radius.
        candidates: dict[str, tuple[int, float]] = {}  # sid → (obs_idx, dist)
        for obs_idx, (dist, sta_idx) in enumerate(zip(dists_arr, idxs_arr, strict=True)):
            if dist > radius:
                continue
            sid = str(sta_ids[sta_idx])
            # Deduplicate: keep the closest obs point per station.
            if sid not in candidates or dist < candidates[sid][1]:
                candidates[sid] = (obs_idx, float(dist))

        if not candidates:
            return [], []

        # Validate datum availability (same filter as the create step).
        valid_ids = client.filter_stations_by_datum(list(candidates.keys()))
        dropped = set(candidates.keys()) - valid_ids
        if dropped:
            self._log(
                f"Excluded {len(dropped)} station(s) without datum data: "
                f"{', '.join(sorted(dropped))}",
                "warning",
            )

        # Build the final lists, sorted by observation-point index.
        pairs = sorted(
            ((obs_idx, sid) for sid, (obs_idx, _) in candidates.items() if sid in valid_ids),
            key=lambda t: t[0],
        )
        noaa_indices = [p[0] for p in pairs]
        noaa_station_ids = [p[1] for p in pairs]

        self._log(
            f"Matched {len(noaa_station_ids)} observation point(s) "
            f"to NOAA station(s): {', '.join(noaa_station_ids)}"
        )
        return noaa_indices, noaa_station_ids

    def _read_point_zs(self, his_file: Path) -> xr.DataArray | None:
        """Read ``point_zs`` (or ``point_h``) from SFINCS output.

        Tries HydroMT's output reader first, then falls back to a
        direct :mod:`xarray` open of *sfincs_his.nc*.
        """
        # Reuse the model already in the registry (read during init)
        # instead of opening a fresh SfincsModel.  Creating a new model
        # and calling read() can fail for quadtree grids when the
        # written sfincs.nc is not fully UGRID-compatible.
        mod = _get_model(self.config)

        # Read output via HydroMT (the compat patch in
        # _hydromt_compat.patch_quadtree_output_read handles the
        # missing UGRID topology in quadtree map files).  Fall back
        # to a direct xarray open when HydroMT still fails.
        point_zs: xr.DataArray | None = None
        try:
            mod.output.read()
            if "point_zs" in mod.output.data:
                point_zs = cast("xr.DataArray", mod.output.data["point_zs"])
            elif "point_h" in mod.output.data:
                self._log("point_zs not found, falling back to point_h (water depth)")
                point_zs = cast("xr.DataArray", mod.output.data["point_h"])
        except Exception as exc:
            self._log(f"HydroMT output.read() failed ({exc}), reading sfincs_his.nc directly")

        if point_zs is None:
            import xarray as xr_mod

            his_ds = xr_mod.open_dataset(his_file)
            for var in ("point_zs", "point_h"):
                if var in his_ds:
                    point_zs = his_ds[var]
                    if var == "point_h":
                        self._log("point_zs not found, falling back to point_h (water depth)")
                    break

        return point_zs

    def run(self) -> dict[str, Any]:
        """Read SFINCS output, fetch NOAA observations, and plot comparison."""
        model_root = get_model_root(self.config)
        his_file = model_root / "sfincs_his.nc"

        if not his_file.exists():
            self._log("sfincs_his.nc not found, skipping plot stage")
            return {"status": "skipped", "reason": "no output"}

        self._update_substep("Reading SFINCS output")
        point_zs = self._read_point_zs(his_file)

        if point_zs is None:
            self._log("No point_zs or point_h in output, skipping plot stage")
            return {"status": "skipped", "reason": "no point_zs"}

        station_dim = self._station_dim(point_zs)

        noaa_indices, noaa_station_ids = self._match_noaa_stations()

        if not noaa_station_ids:
            self._log("No NOAA observation points found, skipping plot stage")
            return {"status": "skipped", "reason": "no noaa stations"}

        # Guard against matched station indices that exceed the number
        # of observation points in the current model output.
        n_stations = point_zs.sizes[station_dim]
        valid = [
            (idx, sid)
            for idx, sid in zip(noaa_indices, noaa_station_ids, strict=False)
            if idx < n_stations
        ]
        if not valid:
            self._log("All NOAA station indices are out of bounds, skipping plot stage")
            return {"status": "skipped", "reason": "noaa indices out of bounds"}
        if len(valid) < len(noaa_indices):
            dropped = len(noaa_indices) - len(valid)
            self._log(f"Dropped {dropped} NOAA station(s) with out-of-bounds indices")
        noaa_indices, noaa_station_ids = [v[0] for v in valid], [v[1] for v in valid]

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
        from coastal_calibration.plotting import plot_station_comparison

        figs_dir = model_root / "figs"
        fig_paths = plot_station_comparison(
            sim_times, sim_elevation, noaa_station_ids, obs_ds, figs_dir
        )

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
            return {
                "meteo_symlinks": 0,
                "streamflow_symlinks": 0,
                "meteo_existing": 0,
                "streamflow_existing": 0,
                "status": "skipped",
            }

        self._update_substep("Creating .nc symlinks")
        meteo_source = self.config.simulation.meteo_source

        created, existing = create_nc_symlinks(
            download_dir,
            meteo_source=meteo_source,
            include_meteo=True,
            include_streamflow=True,
        )

        n_meteo = len(created["meteo"])
        n_stream = len(created["streamflow"])
        n_meteo_existing = existing["meteo"]
        n_stream_existing = existing["streamflow"]

        if n_meteo + n_stream > 0:
            self._log(f"Created {n_meteo} meteo + {n_stream} streamflow symlinks in {download_dir}")
        if n_meteo_existing + n_stream_existing > 0:
            self._log(
                f"Skipped {n_meteo_existing} meteo + {n_stream_existing} streamflow"
                " symlinks (already exist)"
            )

        return {
            "meteo_symlinks": n_meteo,
            "streamflow_symlinks": n_stream,
            "meteo_existing": n_meteo_existing,
            "streamflow_existing": n_stream_existing,
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
