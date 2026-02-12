"""Compatibility patches for hydromt bugs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr


def register_round_coords_preprocessor() -> None:
    """Register a ``round_coords`` preprocessor in hydromt.

    NWM LDASIN files store projected coordinates (LCC, in meters) with
    floating-point rounding errors up to ~0.125 m.  hydromt's raster
    accessor rejects them as "not a regular grid" because its tolerance
    (``atol=5e-4``) is far too tight for metre-scale coordinates.

    This preprocessor rounds x/y coordinates to the nearest integer,
    which makes the grid perfectly regular.
    """
    try:
        from hydromt.data_catalog.drivers.preprocessing import PREPROCESSORS
    except ImportError:
        return

    if "round_coords" in PREPROCESSORS:
        return

    import numpy as np

    def round_coords(ds: xr.Dataset) -> xr.Dataset:
        """Round x and y coordinates to the nearest integer."""
        x_dim = ds.raster.x_dim
        y_dim = ds.raster.y_dim
        ds[x_dim] = np.round(ds[x_dim], decimals=0)
        ds[y_dim] = np.round(ds[y_dim], decimals=0)
        return ds

    PREPROCESSORS["round_coords"] = round_coords


def patch_serialize_crs() -> None:
    """Fix hydromt ``_serialize_crs`` crashing on CRS without an authority.

    hydromt's ``_serialize_crs`` calls ``list(crs.to_authority())`` without
    guarding against ``to_authority()`` returning ``None``, which raises
    ``TypeError: 'NoneType' object is not iterable`` for any CRS that has
    no EPSG code and no recognised authority (e.g. custom proj strings).

    This patch rewrites the function's code object in-place so that even
    references already captured by Pydantic's ``PlainSerializer`` pick up
    the fix.
    """
    try:
        import hydromt.typing.crs as _crs_mod
    except ImportError:
        return

    _original = _crs_mod._serialize_crs

    # Only patch once
    if getattr(_original, "_patched", False):
        return

    def _safe_serialize_crs(crs):  # type: ignore[no-untyped-def]
        epsg = crs.to_epsg()
        if epsg:
            return epsg
        auth = crs.to_authority()
        if auth is not None:
            return list(auth)
        return crs.to_wkt()

    # Replace the *code* of the original function so Pydantic's
    # already-captured reference to the function object sees the fix.
    _original.__code__ = _safe_serialize_crs.__code__
    _original._patched = True  # type: ignore[attr-defined]
