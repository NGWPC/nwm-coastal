"""Temporary compatibility patches for hydromt bugs.

All patches here are stopgaps until the fixes land upstream.
Each one is idempotent (safe to call more than once) and logs
when it is applied so problems are easy to trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coastal_calibration.utils.logging import logger as _log

if TYPE_CHECKING:
    import geopandas as gpd
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
    _log.info("Registered 'round_coords' preprocessor in hydromt.")


def patch_serialize_crs() -> None:
    """Fix hydromt ``_serialize_crs`` crashing on CRS without an authority.

    hydromt's ``_serialize_crs`` calls ``list(crs.to_authority())`` without
    guarding against ``to_authority()`` returning ``None``, which raises
    ``TypeError: 'NoneType' object is not iterable`` for any CRS that has
    no EPSG code and no recognised authority (e.g. custom proj strings).

    Pydantic's ``PlainSerializer`` captures a direct reference to the
    function object at class-definition time, so replacing the function
    on the module would not affect already-imported Pydantic models.
    We therefore swap the function's ``__code__`` in-place so the
    existing reference picks up the fix.
    """
    try:
        import hydromt.typing.crs as _crs_mod
    except ImportError:
        return

    _original = _crs_mod._serialize_crs  # type: ignore[reportPrivateUsage]

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

    _original.__code__ = _safe_serialize_crs.__code__
    _original._patched = True  # type: ignore[attr-defined]
    _log.info("Patched hydromt _serialize_crs to handle CRS without an authority.")


def patch_boundary_conditions_index_dim() -> None:
    """Fix hydromt-sfincs ``_validate_and_prepare_gdf`` not normalising index name.

    ``BoundaryConditionComponent._create_dummy_dataset`` hard-codes
    ``dims=("time", "index")``, but ``GeoDataset.from_gdf`` derives
    ``index_dim`` from ``gdf.index.name``.  When the geodataset's
    spatial dimension is not ``"index"`` (e.g. ``"node"`` for ADCIRC /
    STOFS data), the two names diverge and ``from_gdf`` raises
    ``ValueError: Index dimension node not found in data_vars``.

    This patch wraps ``_validate_and_prepare_gdf`` to rename the GDF
    index to ``"index"`` after validation, keeping everything consistent.
    """
    try:
        from hydromt_sfincs.components.forcing.boundary_conditions import (
            SfincsBoundaryBase,
        )
    except ImportError:
        return

    _original_validate = SfincsBoundaryBase._validate_and_prepare_gdf  # type: ignore[reportPrivateUsage]

    if getattr(_original_validate, "_patched", False):
        return

    def _validate_and_normalise(self: Any, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = _original_validate(self, gdf)
        if gdf.index.name != "index":
            gdf.index.name = "index"
        return gdf

    SfincsBoundaryBase._validate_and_prepare_gdf = _validate_and_normalise  # type: ignore[reportPrivateUsage]
    SfincsBoundaryBase._validate_and_prepare_gdf._patched = True  # type: ignore[reportPrivateUsage, attr-defined]
    _log.info("Patched hydromt-sfincs _validate_and_prepare_gdf to normalise index name.")


def patch_meteo_write_gridded() -> None:
    """Avoid OOM in ``write_gridded`` by keeping dask arrays lazy.

    ``SfincsMeteo.write_gridded`` calls ``self.data.load()`` which
    materialises the entire lazy dask dataset into memory.  For a
    typical NWM forcing setup (precip + wind + pressure) this can
    exceed 90 GB — far more than a login-node's 32 GB.

    This patch replaces ``write_gridded`` with a version that writes
    one time-step at a time via netCDF4 directly, keeping peak memory
    near ~150 MB instead of the full 3-D array.
    """
    try:
        from hydromt_sfincs.components.forcing.meteo import SfincsMeteo
    except ImportError:
        return

    _original = SfincsMeteo.write_gridded
    if getattr(_original, "_patched", False):
        return

    def _write_gridded_lazy(
        self: Any,
        filename: str | None = None,
        rename: dict[str, str] | None = None,
    ) -> None:
        from pathlib import Path

        import netCDF4
        import numpy as np

        tref = self.model.config.get("tref")
        tref_str = tref.strftime("%Y-%m-%d %H:%M:%S")

        ds = self.data

        # Apply variable renaming (swap keys/values to match upstream convention)
        # and select only the renamed variables.  The upstream code does the
        # same: merge only the matching variables, then rename.
        if rename is not None:
            remap = {v: k for k, v in rename.items() if v in ds}
            if remap:
                ds = ds[list(remap)].rename(remap)

        out_path = Path(filename) if filename is not None else Path("output.nc")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        time_vals = ds["time"].values
        var_names = list(ds.data_vars)

        nc = netCDF4.Dataset(str(out_path), "w")
        try:
            # --- dimensions ---
            nc.createDimension("time", None)  # unlimited
            for dim in ds.dims:
                if dim != "time":
                    nc.createDimension(dim, ds.sizes[dim])

            # --- time variable ---
            time_var = nc.createVariable("time", "f8", ("time",))
            time_var.units = f"minutes since {tref_str}"

            # --- spatial coordinate variables ---
            for coord in ds.coords:
                if coord == "time":
                    continue
                arr = ds.coords[coord].values
                nc_coord = nc.createVariable(coord, arr.dtype, ds.coords[coord].dims)
                nc_coord[:] = arr

            # --- data variables ---
            nc_vars = {}
            for vname in var_names:
                da = ds[vname]
                dims = tuple(str(d) for d in da.dims)
                nc_vars[vname] = nc.createVariable(vname, da.dtype, dims)

            # --- write one time-step at a time ---
            t0 = np.datetime64(tref_str)
            for i in range(len(time_vals)):
                time_var[i] = (time_vals[i] - t0) / np.timedelta64(1, "m")
                chunk = ds.isel(time=i).compute()
                for vname in var_names:
                    nc_vars[vname][i, :] = chunk[vname].values
                del chunk
        finally:
            nc.close()

    SfincsMeteo.write_gridded = _write_gridded_lazy
    SfincsMeteo.write_gridded._patched = True  # type: ignore[attr-defined]
    _log.info("Patched hydromt-sfincs write_gridded to stream time-steps lazily.")


def apply_all_patches() -> None:
    """Apply all hydromt/hydromt-sfincs compatibility patches."""
    patch_serialize_crs()
    register_round_coords_preprocessor()
    patch_boundary_conditions_index_dim()
    patch_meteo_write_gridded()
