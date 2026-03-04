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
    (``atol=5e-4``) is far too tight for meter-scale coordinates.

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
    no EPSG code and no recognized authority (e.g. custom proj strings).

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

    _original = _crs_mod._serialize_crs

    if getattr(_original, "_patched", False):
        return

    def _safe_serialize_crs(crs: Any) -> Any:
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

    _original_validate = SfincsBoundaryBase._validate_and_prepare_gdf

    if getattr(_original_validate, "_patched", False):
        return

    def _validate_and_normalize(self: Any, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = _original_validate(self, gdf)
        if gdf.index.name != "index":
            gdf.index.name = "index"
        return gdf

    SfincsBoundaryBase._validate_and_prepare_gdf = _validate_and_normalize  # type: ignore[reportPrivateUsage]
    SfincsBoundaryBase._validate_and_prepare_gdf._patched = True  # type: ignore[reportPrivateUsage, attr-defined]
    _log.info("Patched hydromt-sfincs _validate_and_prepare_gdf to normalize index name.")


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

        time_vals = ds["time"].to_numpy()
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
                arr = ds.coords[coord].to_numpy()
                nc_coord = nc.createVariable(coord, arr.dtype, ds.coords[coord].dims)
                nc_coord[:] = arr

            # --- data variables (float32: SFINCS reads forcing as real*4) ---
            nc_vars = {}
            for vname in var_names:
                da = ds[vname]
                dims = tuple(str(d) for d in da.dims)
                nc_vars[vname] = nc.createVariable(vname, "f4", dims)

            # --- write one time-step at a time ---
            t0 = np.datetime64(tref_str)
            for i in range(len(time_vals)):
                time_var[i] = (time_vals[i] - t0) / np.timedelta64(1, "m")
                chunk = ds.isel(time=i).compute()
                for vname in var_names:
                    nc_vars[vname][i, :] = chunk[vname].to_numpy()
                del chunk
        finally:
            nc.close()

    SfincsMeteo.write_gridded = _write_gridded_lazy  # type: ignore[assignment]
    SfincsMeteo.write_gridded._patched = True  # type: ignore[attr-defined]
    _log.info("Patched hydromt-sfincs write_gridded to stream time-steps lazily.")


def patch_quadtree_subgrid_data_setter() -> None:
    """Add a ``data`` setter to ``SfincsQuadtreeSubgridTable``.

    The upstream ``read`` method does ``self.data = xr.load_dataset(...)``
    but ``data`` is a read-only ``@property`` (no setter), so reading a
    model that contains quadtree subgrid tables crashes with
    ``AttributeError: property 'data' … has no setter``.

    The ``create`` method correctly uses ``self._data = …``, so the fix
    is simply to add a setter that forwards to ``_data``.
    """
    try:
        from hydromt_sfincs.components.quadtree.quadtree_subgrid import (
            SfincsQuadtreeSubgridTable,
        )
    except ImportError:
        return

    prop = SfincsQuadtreeSubgridTable.__dict__.get("data")
    if not isinstance(prop, property) or prop.fset is not None:
        return  # already has a setter or is not a property

    def _set_data(self: Any, value: Any) -> None:
        self._data = value

    SfincsQuadtreeSubgridTable.data = prop.setter(_set_data)
    _log.info("Patched SfincsQuadtreeSubgridTable.data to add missing setter.")


def patch_parse_river_list_geoms() -> None:
    """Fix ``_parse_river_list`` crash when ``geoms`` component is missing.

    ``SfincsModel._parse_river_list`` checks whether a river centerline
    dataset name already exists as a model geometry via
    ``rivers in self.geoms``.  On freshly created models (``mode="w+"``),
    the ``geoms`` component is never registered in hydromt's ``Model``
    component dict, so the property access raises
    ``KeyError: 'geoms'`` via ``Model.__getattr__`` →
    ``Model.get_component``.

    This patch wraps ``_parse_river_list`` to temporarily inject an
    empty dict into the instance ``__dict__`` under the key ``"geoms"``
    when the component is missing.  Python's normal attribute lookup
    checks the instance dict before falling through to ``__getattr__``,
    so ``self.geoms`` resolves to ``{}`` and ``rivers in self.geoms``
    evaluates to ``False`` — falling through to the correct
    ``self.data_catalog.get_geodataframe()`` path.
    """
    try:
        from hydromt_sfincs import SfincsModel
    except ImportError:
        return

    _original = SfincsModel._parse_river_list

    if getattr(_original, "_patched", False):
        return

    def _parse_river_list_safe(self: Any, river_list: Any) -> Any:
        _injected = False
        try:
            self.geoms  # noqa: B018  # test access
        except KeyError:
            # Shadow __getattr__ via the instance dict so the
            # ``rivers in self.geoms`` check evaluates to False.
            self.__dict__["geoms"] = {}
            _injected = True
        try:
            return _original(self, river_list)
        finally:
            if _injected:
                self.__dict__.pop("geoms", None)

    SfincsModel._parse_river_list = _parse_river_list_safe  # type: ignore[assignment]
    SfincsModel._parse_river_list._patched = True  # type: ignore[attr-defined]
    _log.info("Patched SfincsModel._parse_river_list to handle missing 'geoms' component.")


def patch_quadtree_output_read() -> None:
    """Re-grid quadtree ``sfincs_map.nc`` from structured *(n, m)* to UGRID.

    The SFINCS Fortran executable writes map output on a regular *(n, m)*
    grid even for quadtree models, whereas ``hydromt_sfincs`` expects
    UGRID topology and calls ``xugrid.load_dataset`` which fails.

    This patch intercepts ``read_map_file`` for quadtree grids, reads the
    file with plain *xarray*, re-indexes every variable from *(n, m)* to
    the ``mesh2d_nFaces`` dimension using the ``n`` / ``m`` index arrays
    stored in ``sfincs.nc``, wraps the result in a
    ``xugrid.UgridDataset``, and hands it back to the normal output
    pipeline.
    """
    try:
        from hydromt_sfincs.components.output import SfincsOutput
    except ImportError:
        return

    _original = SfincsOutput.read_map_file

    if getattr(_original, "_patched", False):
        return

    def _read_map_file_safe(
        self: Any,
        fn_map: str = "sfincs_map.nc",
        drop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _drop: list[str] = drop if drop is not None else ["crs", "sfincsgrid"]

        if self.model.grid_type != "quadtree":
            return _original(self, fn_map=fn_map, drop=_drop, **kwargs)

        # Try the original first — works if UGRID topology is present.
        try:
            return _original(self, fn_map=fn_map, drop=_drop, **kwargs)
        except Exception:  # noqa: S110
            # Silently fall through to the manual UGRID reconstruction below.
            pass

        import xarray as xr
        import xugrid as xu
        from pyproj import CRS

        grid_ds = self.model.quadtree_grid.data
        ugrid = grid_ds.ugrid.grid
        face_dim = ugrid.face_dimension  # "mesh2d_nFaces"

        # n, m arrays map each face → its (row, col) in the structured
        # output grid (1-indexed).
        n_idx = grid_ds["n"].to_numpy() - 1  # → 0-indexed row
        m_idx = grid_ds["m"].to_numpy() - 1  # → 0-indexed col

        ds_map = xr.open_dataset(fn_map)

        # Build a new dataset with face-indexed variables.
        face_vars: dict[str, xr.DataArray] = {}
        for vname, da in ds_map.data_vars.items():
            if vname in _drop:
                continue
            dims = [str(d) for d in da.dims]

            # Only remap variables that live on the (n, m) grid.
            if "n" not in dims or "m" not in dims:
                continue

            vals = da.values  # e.g. (n, m) or (time, n, m)
            n_pos = dims.index("n")
            m_pos = dims.index("m")

            if n_pos == 0 and m_pos == 1 and len(dims) == 2:
                # (n, m) → (nFaces,)
                face_data = vals[n_idx, m_idx]
                face_vars[vname] = xr.DataArray(
                    face_data,
                    dims=[face_dim],
                )
            elif len(dims) == 3 and n_pos > 0 and m_pos > 0:
                # (time, n, m) → (time, nFaces)
                leading_dim = dims[0]
                face_data = vals[:, n_idx, m_idx]
                face_vars[vname] = xr.DataArray(
                    face_data,
                    dims=[leading_dim, face_dim],
                    coords={leading_dim: da.coords[leading_dim]},
                )

        ds_map.close()

        if not face_vars:
            return None

        ds_face = xr.Dataset(face_vars)
        uds = xu.UgridDataset(ds_face, grids=[ugrid])

        model_crs = self.model.crs
        if model_crs is not None:
            uds.ugrid.grid.set_crs(
                model_crs if isinstance(model_crs, CRS) else CRS.from_user_input(model_crs),
            )

        self.set(uds, split_dataset=True)

    SfincsOutput.read_map_file = _read_map_file_safe  # type: ignore[assignment]
    SfincsOutput.read_map_file._patched = True  # type: ignore[attr-defined]
    _log.info("Patched SfincsOutput.read_map_file to re-grid quadtree map output to UGRID.")


def apply_all_patches() -> None:
    """Apply all hydromt/hydromt-sfincs compatibility patches."""
    patch_serialize_crs()
    register_round_coords_preprocessor()
    patch_boundary_conditions_index_dim()
    patch_meteo_write_gridded()
    patch_quadtree_subgrid_data_setter()
    patch_parse_river_list_geoms()
    patch_quadtree_output_read()
