"""Readers for SFINCS post-processing output.

Loads the time-dependent water-surface field from ``sfincs_map.nc`` together
with the static bed elevation and (when present) the model mask, and adds a
derived water-depth field ``h = zs - zb`` to the returned dataset. The
canonical layout returned by :func:`load_sfincs_water_level` lets the shared
renderer in :mod:`coastal_calibration.plotting.spatial` dispatch via the
``mesh_type`` attribute and apply wet-cell masking against ``h``.

Two layouts are supported:

- ``mesh_type="regular"`` — fields indexed by ``(time, y, x)`` with 1-D
  ``x`` / ``y`` coord vectors. This matches a non-rotated regular SFINCS grid.
- ``mesh_type="ugrid-quadtree"`` — face-valued fields indexed by
  ``(time, face)`` together with mesh geometry (``node_x(node)``,
  ``node_y(node)``, ``face_nodes(face, 4)``). This matches the UGRID-style
  ``sfincs_map.nc`` that SFINCS emits for quadtree grids.

The ``(n, m)`` structured layout used by SFINCS for *structured-output-of-
quadtree-model* files is not yet handled here — a clear
:class:`NotImplementedError` is raised pointing at the existing regridder in
:mod:`coastal_calibration.sfincs._hydromt_compat`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["load_sfincs_water_level"]


#: Default map-output filename written by SFINCS.
_MAP_FILENAME = "sfincs_map.nc"


def _resolve_map_file(run_dir: Path) -> Path:
    """Return the ``sfincs_map.nc`` path under *run_dir*.

    Accepts either the run directory itself or a direct path to the file.
    """
    if run_dir.is_file() and run_dir.name == _MAP_FILENAME:
        return run_dir
    if not run_dir.is_dir():
        raise NotADirectoryError(f"{run_dir} is not a valid directory")
    candidate = run_dir / _MAP_FILENAME
    if not candidate.is_file():
        raise FileNotFoundError(f"SFINCS map output not found: {candidate}")
    return candidate


def _detect_crs(map_file: Path) -> str | None:
    """Best-effort SFINCS CRS detection.

    SFINCS's ``sfincs_map.nc`` carries a ``crs`` data variable with an
    ``EPSG`` attribute, but in practice that field is often left as the
    placeholder ``"-"``. Two other in-tree sources are reliable:

    1. ``sfincs.inp`` next to the map file has an ``epsg = <code>`` line.
    2. ``sfincs.nc`` (the model grid file) carries the full ``crs.crs_wkt``
       attribute from which we can extract the trailing ``ID["EPSG", N]``.

    Returns a string like ``"EPSG:32614"`` when found, else *None*.
    """
    run_dir = map_file.parent

    # 1. sfincs_map.nc itself (rarely populated but cheap to check).
    try:
        with xr.open_dataset(map_file) as ds:
            if "crs" in ds.variables:
                epsg = str(ds["crs"].attrs.get("EPSG", "")).strip()
                if epsg.isdigit():
                    return f"EPSG:{epsg}"
    except (OSError, ValueError):
        pass

    # 2. sfincs.inp epsg line.
    inp = run_dir / "sfincs.inp"
    if inp.is_file():
        for raw in inp.read_text().splitlines():
            line = raw.strip().lower()
            if line.startswith("epsg"):
                _, _, value = line.partition("=")
                value = value.strip()
                if value.isdigit():
                    return f"EPSG:{value}"

    # 3. sfincs.nc crs_wkt — pull the trailing EPSG ID via a small regex.
    sfincs_nc = run_dir / "sfincs.nc"
    if sfincs_nc.is_file():
        try:
            with xr.open_dataset(sfincs_nc) as ds:
                if "crs" in ds.variables:
                    wkt = str(ds["crs"].attrs.get("crs_wkt", ""))
                    import re

                    matches = re.findall(r'ID\["EPSG",\s*(\d+)\]', wkt)
                    if matches:
                        return f"EPSG:{matches[-1]}"
        except (OSError, ValueError):
            pass

    return None


def _detect_layout(ds: xr.Dataset) -> str:
    """Identify the SFINCS output layout.

    Returns one of ``"regular"``, ``"ugrid-quadtree"``, or ``"structured-nm"``.
    The latter is the raw ``(n, m)`` format that requires the HydroMT compat
    regridder and is not yet handled here.
    """
    dims = set(ds.dims)
    if "n" in dims and "m" in dims:
        return "structured-nm"
    has_ugrid_nodes = "mesh2d_node_x" in ds.variables and "mesh2d_node_y" in ds.variables
    has_ugrid_faces = "mesh2d_face_nodes" in ds.variables
    if has_ugrid_nodes and has_ugrid_faces:
        return "ugrid-quadtree"
    if "x" in dims and "y" in dims:
        return "regular"
    return "unknown"


def _normalise_face_nodes_1based(
    raw: NDArray[np.floating] | NDArray[np.integer],
) -> NDArray[np.int64]:
    """Convert SFINCS ``mesh2d_face_nodes`` to 0-based int64 with ``-1`` fill.

    SFINCS writes face-node connectivity as 1-based integers (sometimes
    stored as ``float64``) with ``0`` as the padding value for sub-quad
    cells. We normalise to the same convention used by
    :func:`coastal_calibration.schism.outputs.load_schism_elevation`:
    0-based indices with ``-1`` for fill. NaN entries (from decoded
    ``_FillValue``) are treated as padding.
    """
    src = np.asarray(raw)
    if np.issubdtype(src.dtype, np.floating):
        nan_mask = np.isnan(src)
        arr = np.where(nan_mask, 0.0, src).astype(np.int64, copy=False)
        arr[nan_mask] = 0
    else:
        arr = src.astype(np.int64, copy=False)
    invalid = arr <= 0
    arr = arr - 1
    arr[invalid] = -1
    return arr


def _build_quadtree(ds: xr.Dataset) -> xr.Dataset:
    """Build a canonical ``mesh_type="ugrid-quadtree"`` dataset.

    Loads ``zs(time, face)``, the static bed elevation ``zb(face)``, and
    (when present) the static cell mask ``msk(face)``; derives water depth
    ``h = zs - zb``.
    """
    if "zs" not in ds.variables:
        msg = "Required variable 'zs' not found in sfincs_map.nc"
        raise KeyError(msg)
    if "zb" not in ds.variables:
        msg = "Required static variable 'zb' not found in sfincs_map.nc"
        raise KeyError(msg)

    zs_da = ds["zs"]
    face_dim_candidates = [
        d for d in zs_da.dims if str(d).startswith("nmesh2d_face") or d == "face"
    ]
    if "time" not in zs_da.dims or not face_dim_candidates:
        msg = f"Expected zs to be face-valued with a 'time' dim; got dims {tuple(zs_da.dims)}"
        raise ValueError(msg)
    face_dim = face_dim_candidates[0]
    zs_da = zs_da.transpose("time", face_dim)

    zs = np.asarray(zs_da.to_numpy(), dtype=np.float64)
    zb = np.asarray(ds["zb"].to_numpy(), dtype=np.float64)
    h = zs - zb[np.newaxis, :]

    node_x = np.asarray(ds["mesh2d_node_x"].to_numpy(), dtype=np.float64)
    node_y = np.asarray(ds["mesh2d_node_y"].to_numpy(), dtype=np.float64)
    face_nodes = _normalise_face_nodes_1based(ds["mesh2d_face_nodes"].to_numpy())
    times = np.asarray(zs_da["time"].to_numpy())

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "zs": (("time", "face"), zs),
        "h": (("time", "face"), h),
        "zb": (("face",), zb),
        "node_x": (("node",), node_x),
        "node_y": (("node",), node_y),
        "face_nodes": (("face", "face_node"), face_nodes),
    }
    if "msk" in ds.variables:
        data_vars["msk"] = (("face",), np.asarray(ds["msk"].to_numpy(), dtype=np.int8))

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "node": np.arange(node_x.size, dtype=np.int64),
            "face": np.arange(face_nodes.shape[0], dtype=np.int64),
        },
        attrs={"mesh_type": "ugrid-quadtree"},
    )


def _build_regular(ds: xr.Dataset) -> xr.Dataset:
    """Build a canonical ``mesh_type="regular"`` dataset.

    Loads ``zs(time, y, x)``, the static bed elevation ``zb(y, x)``, and
    (when present) the static cell mask ``msk(y, x)``; derives water depth
    ``h = zs - zb``. Optional 2-D geographic coordinates ``xc`` / ``yc``
    are preserved.
    """
    if "zs" not in ds.variables:
        msg = "Required variable 'zs' not found in sfincs_map.nc"
        raise KeyError(msg)
    if "zb" not in ds.variables:
        msg = "Required static variable 'zb' not found in sfincs_map.nc"
        raise KeyError(msg)

    zs_da = ds["zs"]
    required_dims = {"time", "y", "x"}
    dims_str = {str(d) for d in zs_da.dims}
    missing = required_dims - dims_str
    if missing:
        msg = (
            f"Expected zs to have dims {required_dims}; got {tuple(zs_da.dims)} (missing {missing})"
        )
        raise ValueError(msg)
    zs_da = zs_da.transpose("time", "y", "x")

    zs = np.asarray(zs_da.to_numpy(), dtype=np.float64)
    zb = np.asarray(ds["zb"].transpose("y", "x").to_numpy(), dtype=np.float64)
    h = zs - zb[np.newaxis, :, :]

    times = np.asarray(zs_da["time"].to_numpy())
    y = np.asarray(ds["y"].to_numpy(), dtype=np.float64)
    x = np.asarray(ds["x"].to_numpy(), dtype=np.float64)

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "zs": (("time", "y", "x"), zs),
        "h": (("time", "y", "x"), h),
        "zb": (("y", "x"), zb),
    }
    if "msk" in ds.variables:
        data_vars["msk"] = (
            ("y", "x"),
            np.asarray(ds["msk"].transpose("y", "x").to_numpy(), dtype=np.int8),
        )
    for name in ("xc", "yc"):
        if name in ds.coords or name in ds.variables:
            da = ds[name]
            if set(da.dims) == {"y", "x"}:
                data_vars[name] = (
                    ("y", "x"),
                    np.asarray(da.transpose("y", "x").to_numpy()),
                )

    return xr.Dataset(
        data_vars=data_vars,
        coords={"time": times, "y": y, "x": x},
        attrs={"mesh_type": "regular"},
    )


def _annotate_attrs(out: xr.Dataset) -> None:
    """Attach standard long_name / units to the canonical variables."""
    out["zs"].attrs.setdefault("long_name", "water-surface elevation")
    out["zs"].attrs.setdefault("units", "m")
    out["h"].attrs.setdefault("long_name", "water depth")
    out["h"].attrs.setdefault("units", "m")
    out["h"].attrs.setdefault("description", "zs - zb; ~0 at dry cells")
    out["zb"].attrs.setdefault("long_name", "bed elevation")
    out["zb"].attrs.setdefault("units", "m")
    if "msk" in out.data_vars:
        out["msk"].attrs.setdefault(
            "description", "SFINCS active-cell mask: 1 = active, 0 = inactive"
        )


def load_sfincs_water_level(
    run_dir: str | Path,
    *,
    time_slice: slice | None = None,
) -> xr.Dataset:
    """Load SFINCS time-dependent water-level + depth from ``sfincs_map.nc``.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        SFINCS run directory containing ``sfincs_map.nc``, or the path to
        the file itself.
    time_slice : slice, optional
        Slice applied along the time dimension before the dataset is
        returned. Useful for previews of long runs.

    Returns
    -------
    xarray.Dataset
        Canonical dataset containing both ``zs(time, ...)`` (water-surface
        elevation) and ``h(time, ...)`` (water depth, derived as
        ``zs - zb``), the static bed elevation ``zb(...)``, and (when
        present) the static cell mask ``msk(...)``. The ``mesh_type``
        attribute is ``"regular"`` or ``"ugrid-quadtree"``.

        Any 2-D geographic coordinates ``xc(y, x)`` / ``yc(y, x)`` found in
        the source file are preserved as auxiliary coordinates.

    Raises
    ------
    NotADirectoryError
        If *run_dir* is neither a directory nor a ``sfincs_map.nc`` file.
    FileNotFoundError
        If the map output file cannot be located.
    KeyError
        If a required variable (``zs`` or ``zb``) is missing.
    NotImplementedError
        If the file uses the structured ``(n, m)`` quadtree layout.
    ValueError
        If ``zs`` does not have the expected dims for the detected layout.

    Notes
    -----
    The reader intentionally opens ``sfincs_map.nc`` directly with *xarray*
    rather than going through :class:`hydromt_sfincs.SfincsModel`. This
    keeps post-processing independent of a full HydroMT model setup and
    works in lightweight environments where only ``xarray`` + ``netcdf4``
    are available. A HydroMT-based path can be added in a follow-up if
    needed for quadtree outputs.
    """
    run_dir = Path(run_dir)
    map_file = _resolve_map_file(run_dir)

    with xr.open_dataset(map_file) as ds:
        layout = _detect_layout(ds)
        if layout == "structured-nm":
            raise NotImplementedError(
                "Structured (n, m) quadtree SFINCS outputs are not yet supported by "
                "load_sfincs_water_level. See "
                "coastal_calibration.sfincs._hydromt_compat.patch_quadtree_output_read "
                "for the regridding logic to plumb in."
            )
        if layout == "ugrid-quadtree":
            out = _build_quadtree(ds)
        elif layout == "regular":
            out = _build_regular(ds)
        else:
            msg = (
                f"Unrecognised SFINCS map layout in {map_file.name}. "
                f"Dims: {sorted(str(d) for d in ds.dims)}"
            )
            raise ValueError(msg)

    out.attrs["source_run"] = str(run_dir)
    crs_label = _detect_crs(map_file)
    if crs_label is not None:
        out.attrs["crs"] = crs_label
    _annotate_attrs(out)

    if time_slice is not None:
        out = out.isel(time=time_slice)
    return out
