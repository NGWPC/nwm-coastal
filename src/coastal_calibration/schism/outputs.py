"""Readers for SCHISM post-processing output.

Currently supports the SCHISM v5.10+ ``out2d_*.nc`` output format written to
the ``outputs/`` sub-directory of a project. The reader concatenates blocks
along *time*, reconstructs an absolute datetime axis from the ``base_date``
attribute, and packages the 2-D water-level field together with the mesh
geometry (node coords + triangle/quad connectivity), the static bathymetric
depth, the model's per-node dry flag (when present), and a derived water
depth ``h = elevation + depth`` into a single :class:`xarray.Dataset` for
downstream plotting.

The returned dataset uses the attribute ``mesh_type = "ugrid-triangle-or-quad"``
so the shared renderer in :mod:`coastal_calibration.plotting.spatial` can
dispatch correctly. Wet-cell masking is supported in the renderer either via
the ``dryFlagNode`` array (preferred when present) or via the derived ``h``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["load_schism_elevation"]


#: Default glob used to locate SCHISM 2-D output blocks.
_OUT2D_GLOB = "out2d_*.nc"

#: Variable names we expect to find in ``out2d_*.nc``.
_REQUIRED_DATA_VARS: tuple[str, ...] = (
    "elevation",
    "depth",
    "SCHISM_hgrid_node_x",
    "SCHISM_hgrid_node_y",
    "SCHISM_hgrid_face_nodes",
)

#: Optional time-varying variables we pull through when present in the file.
_OPTIONAL_TIME_VARS: tuple[str, ...] = ("dryFlagNode",)


def _resolve_outputs_dir(run_dir: Path) -> Path:
    """Return the directory containing ``out2d_*.nc`` files.

    Accepts either the project root (expects ``outputs/`` underneath) or the
    ``outputs/`` directory itself. Raises if neither matches.
    """
    if not run_dir.is_dir():
        raise NotADirectoryError(f"{run_dir} is not a valid directory")
    if list(run_dir.glob(_OUT2D_GLOB)):
        return run_dir
    outputs = run_dir / "outputs"
    if outputs.is_dir() and list(outputs.glob(_OUT2D_GLOB)):
        return outputs
    raise FileNotFoundError(f"No SCHISM {_OUT2D_GLOB} files found under {run_dir} or {outputs}")


def _parse_base_date(attr: str) -> pd.Timestamp:
    """Parse a SCHISM ``base_date`` attribute into a pandas Timestamp.

    The attribute is a whitespace-separated string of five floats:
    ``"YYYY M D H M"`` (year, month, day, hour, minute). Some models
    additionally write a trailing UTC offset which we ignore.
    """
    parts = attr.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Unexpected SCHISM base_date format: {attr!r}")
    year, month, day, hour = (int(float(p)) for p in parts[:4])
    minute = int(float(parts[4])) if len(parts) >= 5 else 0
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)


def _detect_crs(ds0: xr.Dataset) -> str | None:
    """Best-effort CRS detection from a SCHISM ``out2d_*.nc`` block.

    SCHISM v5.10+ writes a ``crs`` data variable with CF-style attributes.
    For lat/lon meshes the ``grid_mapping_name`` attribute is
    ``"latitude_longitude"`` — we map that to ``EPSG:4326``. Projected
    meshes carry ``transverse_mercator`` (or similar) with parameters that
    don't always correspond to a single EPSG; we leave those as *None* and
    let the caller specify the CRS manually if a basemap is needed.
    """
    if "crs" not in ds0.variables:
        return None
    grid_mapping = str(ds0["crs"].attrs.get("grid_mapping_name", "")).lower()
    if grid_mapping == "latitude_longitude":
        return "EPSG:4326"
    return None


def _normalise_face_nodes(
    face_nodes_raw: NDArray[np.floating] | NDArray[np.integer],
) -> NDArray[np.int64]:
    """Normalise SCHISM face_nodes to 0-based with ``-1`` fill for triangles.

    SCHISM writes 1-based indices with a variety of fill values (commonly 0,
    ``-999999``, or NetCDF ``_FillValue``). When xarray auto-decodes the
    ``_FillValue`` attribute, missing entries arrive as ``NaN`` in a
    floating-point array; we handle that explicitly. The output is a
    contiguous ``int64`` array where:

    - valid node indices are 0-based,
    - padding (triangles in a mixed tri/quad mesh) is ``-1``.

    This matches the convention already used by
    :attr:`NWMSCHISMProject.element_connections`.
    """
    raw = np.asarray(face_nodes_raw)
    if np.issubdtype(raw.dtype, np.floating):
        # NaN from decoded _FillValue → padding. Replace before casting.
        nan_mask = np.isnan(raw)
        arr = np.where(nan_mask, 0.0, raw).astype(np.int64, copy=False)
        arr[nan_mask] = 0  # ensure 0 (treated as invalid below)
    else:
        arr = raw.astype(np.int64, copy=False)
    invalid = arr <= 0
    arr = arr - 1
    arr[invalid] = -1
    return arr


def _load_mesh_geometry(
    first_block: Path,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    pd.Timestamp,
    str | None,
]:
    """Open *first_block* and return the static mesh + base_date + CRS label."""
    with xr.open_dataset(first_block, decode_times=False) as ds0:
        node_x = np.asarray(ds0["SCHISM_hgrid_node_x"].to_numpy(), dtype=np.float64)
        node_y = np.asarray(ds0["SCHISM_hgrid_node_y"].to_numpy(), dtype=np.float64)
        face_nodes = _normalise_face_nodes(ds0["SCHISM_hgrid_face_nodes"].to_numpy())
        depth = np.asarray(ds0["depth"].to_numpy(), dtype=np.float64)
        base_date_attr = ds0["time"].attrs.get("base_date")
        crs_label = _detect_crs(ds0)

    if base_date_attr is None:
        raise KeyError(f"Missing 'base_date' attribute on time variable in {first_block.name}")
    return node_x, node_y, face_nodes, depth, _parse_base_date(str(base_date_attr)), crs_label


def _apply_elevation_correction(
    elevation: NDArray[np.float64],
    node_x: NDArray[np.float64],
    node_y: NDArray[np.float64],
    correction_file: Path,
) -> NDArray[np.float64]:
    """Add the per-node geoid-MSL offset interpolated from *correction_file*.

    Reads the boundary-node CSV (``conversion_factor`` column), maps each
    mesh node to its nearest boundary entry via ``KDTree``, and adds
    that local offset to every elevation timestep. Returns the corrected
    elevation array; the caller should use this in place of the original.
    """
    corr_df = pd.read_csv(correction_file)
    corr_lons = corr_df["Field1"].to_numpy()
    corr_lats = corr_df["Field2"].to_numpy()
    conv_factors = corr_df["conversion_factor"].to_numpy()
    tree = KDTree(np.column_stack([corr_lons, corr_lats]))
    _, nn_idx = tree.query(np.column_stack([node_x, node_y]), k=1)
    node_conv = conv_factors[nn_idx]
    return (elevation + node_conv[None, :]).astype(np.float64, copy=False)


def _set_dataset_attrs(out: xr.Dataset, *, has_dry_flag: bool) -> None:
    """Stamp human-readable attributes onto *out* in place."""
    out["elevation"].attrs.update(
        {"long_name": "free-surface elevation", "units": "m", "datum": "MSL"}
    )
    out["h"].attrs.update(
        {
            "long_name": "water depth",
            "units": "m",
            "description": "elevation + depth; ~0 at dry nodes",
        }
    )
    out["depth"].attrs.update(
        {
            "long_name": "bathymetric depth",
            "units": "m",
            "description": "positive when bed is below the datum",
        }
    )
    out["node_x"].attrs.update({"long_name": "node longitude", "units": "degrees_east"})
    out["node_y"].attrs.update({"long_name": "node latitude", "units": "degrees_north"})
    out["face_nodes"].attrs.update(
        {
            "long_name": "face-node connectivity",
            "start_index": 0,
            "_FillValue": -1,
            "description": "0-based node indices; -1 marks unused vertex of a triangle",
        }
    )
    if has_dry_flag:
        out["dryFlagNode"].attrs.update(
            {"long_name": "dry node flag", "description": "1 = dry, 0 = wet"}
        )


def load_schism_elevation(
    run_dir: str | Path,
    *,
    time_slice: slice | None = None,
    correction_file: str | Path | None = None,
) -> xr.Dataset:
    """Load SCHISM 2-D elevation across all output blocks.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        SCHISM run directory. Either the project root (the reader will look
        under ``outputs/``) or the ``outputs/`` directory itself.
    time_slice : slice, optional
        Slice applied along the concatenated time dimension before the
        dataset is returned. Useful for previews of long runs.
    correction_file : str or pathlib.Path, optional
        Path to ``elevation_correction.csv``.  When provided, the
        ``conversion_factor`` is interpolated from the boundary nodes
        listed in the CSV onto every mesh node (nearest-neighbour) and
        added to the elevations to convert from the model's mesh datum
        back to MSL — the inverse of the boundary correction applied by
        ``correct_elevation``.  This preserves the spatial variation of
        the geoid-MSL offset across the domain instead of applying a
        single boundary mean (which under-corrects nodes whose local
        offset deviates from the average).  Use this when the model was
        forced with STOFS via :func:`make_stofs_boundary` so that
        simulated values are MSL-referenced (matching NOAA CO-OPS
        observations).

    Returns
    -------
    xarray.Dataset
        Dataset with variables ``elevation(time, node)``, ``node_x(node)``,
        ``node_y(node)``, and ``face_nodes(face, face_node)``. The time
        coordinate is an absolute ``datetime64[ns]`` axis reconstructed from
        the SCHISM ``base_date`` attribute. Attributes include
        ``mesh_type = "ugrid-triangle-or-quad"`` and ``source_run``.

    Raises
    ------
    NotADirectoryError
        If *run_dir* is not a directory.
    FileNotFoundError
        If no ``out2d_*.nc`` files are found.
    KeyError
        If any required mesh variable is missing from the first block.

    Notes
    -----
    The reader does not require ``dask``. Blocks are concatenated manually
    via :func:`xarray.concat` after opening each file with the default engine.
    Mesh variables (node coordinates and face connectivity) are read from
    the first block only — SCHISM keeps these constant across a run.
    """
    run_dir = Path(run_dir)
    outputs_dir = _resolve_outputs_dir(run_dir)

    files = sorted(outputs_dir.glob(_OUT2D_GLOB))
    if not files:
        # Unreachable because _resolve_outputs_dir checks, but kept for clarity.
        raise FileNotFoundError(f"No {_OUT2D_GLOB} under {outputs_dir}")

    # Open the first block to discover which optional time-varying variables
    # are actually present (e.g. dryFlagNode), then open every block and pull
    # only what we need.
    with xr.open_dataset(files[0], decode_times=False) as ds0_probe:
        missing = [v for v in _REQUIRED_DATA_VARS if v not in ds0_probe.variables]
        if missing:
            raise KeyError(
                f"SCHISM output file {files[0].name} is missing required "
                f"variables: {', '.join(missing)}"
            )
        present_optional = tuple(v for v in _OPTIONAL_TIME_VARS if v in ds0_probe.variables)

    time_vars = ("elevation", "time", *present_optional)
    per_block: list[xr.Dataset] = []
    for path in files:
        with xr.open_dataset(path, decode_times=False) as ds:
            per_block.append(ds[list(time_vars)].load())

    raw = xr.concat(per_block, dim="time", data_vars="minimal").sortby("time")

    node_x, node_y, face_nodes, depth, base_date, crs_label = _load_mesh_geometry(files[0])

    # SCHISM writes seconds-since-base_date as the raw time values.
    seconds = raw["time"].to_numpy().astype("float64")
    times = base_date + pd.to_timedelta(seconds, unit="s")

    elevation = raw["elevation"].to_numpy()

    # Optionally convert from mesh datum back to MSL by applying the
    # inverse of the boundary forcing correction. See
    # :func:`_apply_elevation_correction` for the per-node lookup.
    if correction_file is not None:
        corr_path = Path(correction_file)
        if corr_path.exists():
            elevation = _apply_elevation_correction(elevation, node_x, node_y, corr_path)

    # Water depth = water surface + bathymetric depth (depth is positive when
    # the bed is below the datum). For dry nodes, SCHISM reports
    # ``elevation = -depth + h_min``, so h ≈ 0 there.
    h = elevation + depth[np.newaxis, :]

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "elevation": (("time", "node"), elevation),
        "h": (("time", "node"), h),
        "depth": (("node",), depth),
        "node_x": (("node",), node_x),
        "node_y": (("node",), node_y),
        "face_nodes": (("face", "face_node"), face_nodes),
    }
    if "dryFlagNode" in present_optional:
        data_vars["dryFlagNode"] = (
            ("time", "node"),
            raw["dryFlagNode"].to_numpy().astype(np.int8, copy=False),
        )

    out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "node": np.arange(node_x.size, dtype=np.int64),
            "face": np.arange(face_nodes.shape[0], dtype=np.int64),
        },
        attrs={
            "mesh_type": "ugrid-triangle-or-quad",
            "source_run": str(run_dir),
        },
    )
    if crs_label is not None:
        out.attrs["crs"] = crs_label

    _set_dataset_attrs(out, has_dry_flag="dryFlagNode" in out.data_vars)

    if time_slice is not None:
        out = out.isel(time=time_slice)
    return out
