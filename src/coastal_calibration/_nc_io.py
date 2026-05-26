"""Internal NetCDF I/O helpers shared across SCHISM/SFINCS writers.

Three narrow helpers, applied opportunistically. There is intentionally
no read helper: every reader in the codebase is a one-liner of the form
``with netCDF4.Dataset(p) as ds: arr = ds["VAR"][:]`` which is already
clearer than any wrapper would be.

Why these three:

* :func:`write_elev2d_th` collapses the two writers (``regrid_estofs``
  and ``tides/_otps``) that produce SCHISM's canonical ``elev2D.th.nc``
  schema. Both spelled out the same dimension/variable layout
  inline, with subtly different fill-value constants and time-attribute
  strings; consolidating them gives a single source of truth tied to
  the SCHISM file format.
* :func:`create_var` is a thin wrapper around
  :meth:`netCDF4.Dataset.createVariable` that takes the variable's
  attribute dict alongside the create call and applies it via
  ``setncatts``. The structured writers in this codebase routinely
  follow each ``createVariable`` with three or four ``var.attr = ...``
  lines; this collapses that idiom.
* :func:`write_var` pairs with :func:`create_var` for whole-array
  writes that shape-checks the data against the variable and casts to
  ``var.dtype`` so silent precision changes don't slip past assignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import netCDF4
import numpy as np

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import NDArray


# Default fill value used by SCHISM's elev2D.th.nc consumer. The
# original writers used both ``-9999.0`` and ``-9999`` interchangeably;
# pin a single typed constant so that round-tripping is bit-exact.
ELEV2D_MISSING: float = -9999.0


def write_var(var: netCDF4.Variable, data: NDArray[Any], *, index: int | None = None) -> None:
    """Write a numpy array into ``var`` with shape and dtype enforcement.

    Two guarantees over a bare slice assignment:

    * **Shape check** — raises a clear Python exception upfront when
      shapes disagree, instead of letting HDF5 surface a cryptic
      broadcasting error from deep inside the C library. Unlimited
      dimensions are skipped in the comparison since they grow on
      assignment from an initial size of 0.
    * **Dtype enforcement** — casts *data* to ``var.dtype`` via
      :func:`numpy.asarray` so the on-disk dtype always matches the
      variable's declared type. netCDF4-python silently converts on
      assignment; doing the cast explicitly here makes that conversion
      visible at the call site and prevents surprise downcasts.

    Without ``index`` this writes the whole variable (``var[:] = data``).
    With ``index`` it writes one slab along the leading dimension
    (``var[index] = data``); ``data`` must then match ``var.shape[1:]``
    exactly — caller is expected to ``[0]``-strip any leading singleton
    dim from upstream sources.

    Use plain slice assignment for compound indexing (``var[:, 0, :]``)
    and scalar broadcasts (``var[:] = 3600``); this helper covers the
    whole-array and per-slab array-to-array cases.

    Parameters
    ----------
    var
        Open netCDF variable in write mode.
    data
        Numpy array (or masked array) whose shape must match
        ``var.shape`` (whole-array write) or ``var.shape[1:]``
        (per-slab write) on every fixed-size dimension.
    index
        If given, write to ``var[index]`` instead of the whole variable.
        Typically the timestep counter inside a per-step loop.
    """
    if index is None:
        if data.ndim != var.ndim:
            raise ValueError(
                f"Rank mismatch writing to {var.name!r}: "
                f"data is {data.ndim}-D, variable is {var.ndim}-D"
            )
        for i, dim in enumerate(var.get_dims()):
            if not dim.isunlimited() and data.shape[i] != var.shape[i]:
                raise ValueError(
                    f"Shape mismatch writing to {var.name!r}: "
                    f"data {data.shape} vs variable {var.shape}"
                )
        var[:] = np.asarray(data, dtype=var.dtype)
        return

    expected = var.shape[1:]
    if data.shape != expected:
        raise ValueError(
            f"Shape mismatch writing slab {index} of {var.name!r}: "
            f"data {data.shape} vs expected {expected}"
        )
    var[index] = np.asarray(data, dtype=var.dtype)


def create_var(
    ds: netCDF4.Dataset,
    name: str,
    dtype: Any,
    dims: tuple[str, ...],
    *,
    attrs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> netCDF4.Variable:
    """Create a netCDF variable and set its attributes; return the variable.

    Replaces the recurring three-line attribute-setting pattern::

        v = ds.createVariable("foo", "f4", ("y", "x"))
        v.long_name = "Foo"
        v.units = "m"

    with one call::

        v = create_var(ds, "foo", "f4", ("y", "x"),
                       attrs={"long_name": "Foo", "units": "m"})

    The function does *only* create-and-attribute; it never writes data.
    Every caller is expected to capture the returned ``Variable`` and
    populate it explicitly on a separate line — either via
    :func:`write_var` (for whole-array writes or per-slab loops with
    ``index=i``), or ``v[:] = scalar`` for scalar broadcasts. This
    keeps every call site uniform (always two lines: create + write),
    and writes are never hidden inside a helper kwarg.

    Parameters
    ----------
    ds
        Open ``netCDF4.Dataset`` in write mode.
    name
        Variable name.
    dtype
        Type code understood by netCDF4 (``"f4"``, ``"f8"``, ``"i4"``,
        etc.) or any numpy dtype.
    dims
        Tuple of dimension names; pass ``()`` for a scalar.
    attrs
        Optional attribute dict; applied via ``setncatts`` when not
        empty.
    **kwargs
        Forwarded to :meth:`netCDF4.Dataset.createVariable`
        (``zlib``, ``fill_value``, ``chunksizes``, ...).

    Returns
    -------
    netCDF4.Variable
        The newly created variable.
    """
    var = ds.createVariable(name, dtype, dims, **kwargs)
    if attrs:
        var.setncatts(attrs)
    return var


def write_elev2d_th(
    path: str | PathLike[str],
    *,
    n_open_bnd_nodes: int,
    time_seconds: NDArray[Any],
    time_step_seconds: float,
    time_series: NDArray[np.floating[Any]],
    time_attrs: dict[str, Any] | None = None,
    missing: float = ELEV2D_MISSING,
) -> None:
    """Write a SCHISM ``elev2D.th.nc`` boundary forcing file.

    SCHISM expects a fixed schema with five dimensions
    (``time``, ``nOpenBndNodes``, ``nLevels``, ``nComponents``, ``one``)
    and three variables (``time_step``, ``time``, ``time_series``).
    This helper enforces that schema in one place so the multiple
    writers in the codebase stay consistent.

    Parameters
    ----------
    path
        Output file path.
    n_open_bnd_nodes
        Number of open boundary nodes (size of the ``nOpenBndNodes``
        dimension).
    time_seconds
        1-D array of time values written into the ``time`` variable
        (typically ``np.arange(0, nt * dt, dt)``).
    time_step_seconds
        Scalar written into the ``time_step`` variable.
    time_series
        Either a 2-D array of shape ``(nt, n_open_bnd_nodes)`` (one
        elevation per node per timestep), or a 4-D array of the
        already-shaped ``(nt, n_open_bnd_nodes, 1, 1)`` form. Values
        less than or equal to *missing* are zeroed before writing.
    time_attrs
        Optional dict of attributes to set on the ``time`` variable
        (e.g. ``{"units": "seconds since ..."}``). When omitted no
        attributes are set.
    missing
        ``_FillValue`` used for the ``time_series`` variable.
    """
    if time_series.ndim == 2:
        if time_series.shape != (len(time_seconds), n_open_bnd_nodes):
            raise ValueError(
                f"time_series 2-D shape {time_series.shape} does not match "
                f"(nt={len(time_seconds)}, n_open_bnd_nodes={n_open_bnd_nodes})"
            )
        time_series = time_series.reshape(len(time_seconds), n_open_bnd_nodes, 1, 1)
    elif time_series.ndim != 4:
        raise ValueError(f"time_series must be 2-D or 4-D, got shape {time_series.shape}")
    elif time_series.shape != (len(time_seconds), n_open_bnd_nodes, 1, 1):
        raise ValueError(
            f"time_series 4-D shape {time_series.shape} does not match "
            f"(nt={len(time_seconds)}, n_open_bnd_nodes={n_open_bnd_nodes}, 1, 1)"
        )

    # Zero out fill-value entries to match the historical writer behavior.
    time_series = np.where(time_series > missing, time_series, 0.0)

    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as nc:
        nc.createDimension("time", None)  # unlimited
        nc.createDimension("nOpenBndNodes", n_open_bnd_nodes)
        nc.createDimension("nLevels", 1)
        nc.createDimension("nComponents", 1)
        nc.createDimension("one", 1)

        ts_var = create_var(nc, "time_step", "f8", ("one",))
        time_var = create_var(nc, "time", "f8", ("time",), attrs=time_attrs)
        series_var = create_var(
            nc,
            "time_series",
            "f8",
            ("time", "nOpenBndNodes", "nLevels", "nComponents"),
            fill_value=missing,
            zlib=True,
        )

        write_var(ts_var, np.array([time_step_seconds]))
        write_var(time_var, time_seconds)
        write_var(series_var, time_series)
