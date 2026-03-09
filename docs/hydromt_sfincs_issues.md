# `hydromt-sfincs` Bug Reports

Tested against **`hydromt-sfincs`** commit
[`41aac0a`](https://github.com/Deltares/hydromt_sfincs/commit/41aac0a3980fc2714ec28eafb0463d40abfc979a).
All issues were discovered while integrating `hydromt-sfincs` into an automated coastal
calibration pipeline that builds SFINCS models from NWM and STOFS/TPXO forcing data.

______________________________________________________________________

## 1. Gridded meteo reprojection produces CONUS-extent output (critical performance bug)

**Summary**:

`SfincsPrecipitation.create()`, `SfincsWind.create()`, and `SfincsPressure.create()`
produce output grids that can be orders of magnitude larger than the model domain when
the source data uses a different CRS than the model.

**Root cause**:

Each `create()` method follows a two-step pattern (`meteo.py`, precipitation example at
lines 351-413):

```python
# Step 1 — clip in source CRS (correct)
precip = self.data_catalog.get_rasterdataset(
    precip,
    bbox=self.model.bbox,   # model bbox in geographic coords
    buffer=buffer,           # 5 km default
    ...
)

# Step 2 — reproject to model CRS (no post-clip)
precip_out = precip.raster.reproject(
    dst_crs=self.model.crs, dst_res=dst_res, **kwargs
).fillna(0)
```

Step 1 correctly extracts a small rectangle in the source CRS (e.g. Lambert Conformal
Conic for NWM LDASIN). Step 2 calls `rioxarray.reproject()`, which computes the output
extent by transforming the corners of the input rectangle into the target CRS. For
projections with significant non-linearity (LCC → UTM), a small source-CRS rectangle
maps to an enormously inflated bounding box in the target CRS. `reproject()` faithfully
allocates the full output grid, filling unreachable cells with NaN (subsequently
replaced by `.fillna(0)`).

**There is no clip to the model domain after the reprojection.**

**Observed impact**:

For a SFINCS model on the Texas Gulf Coast (UTM zone 14N, domain ~143 km x 92 km) with
NWM LDASIN forcing (LCC, ~1 km):

| File                          | Expected size | Actual size | Grid dimensions |
| ----------------------------- | ------------- | ----------- | --------------- |
| `sfincs_netampr.nc` (precip)  | ~11 MB        | **12 GB**   | 4797 x 4085     |
| `sfincs_netamuv.nc` (wind)    | ~45 MB        | **49 GB**   | 4797 x 4085     |
| `sfincs_netamp.nc` (pressure) | ~23 MB        | **24 GB**   | 4797 x 4085     |

The output grid covered x = -1,793 km to 3,118 km and y = 2,339 km to 6,581 km in UTM
zone 14N — essentially the entire CONUS. The SFINCS simulation, which must read these
files every time step, slowed from ~7 minutes to an estimated **15+ hours**.

**Suggested fix**:

Add a `rio.clip_box` call after `reproject()` in each `create()` method, clipping to
`self.model.region.total_bounds` with a buffer:

```python
precip_out = precip.raster.reproject(dst_crs=self.model.crs, dst_res=dst_res, **kwargs).fillna(0)

# --- add this ---
region = self.model.region.total_bounds  # (xmin, ymin, xmax, ymax)
buf = buffer  # reuse the same buffer parameter
precip_out = precip_out.rio.clip_box(
    minx=region[0] - buf,
    miny=region[1] - buf,
    maxx=region[2] + buf,
    maxy=region[3] + buf,
)
```

This applies identically to `SfincsPrecipitation.create()` (line 411),
`SfincsWind.create()` (line 684), and `SfincsPressure.create()` (line 581).

**Current workaround**:

In our pipeline we bypass the upstream `component.create()` entirely via
`_create_meteo_forcing()` (in `sfincs_build.py`), which clips the source data in its
**native CRS** before reprojecting — so the CONUS-scale grid is never allocated. The
destination grid is also constrained to the model domain bounds. This replaces the
earlier post-hoc `_clip_meteo_to_domain()` approach that still triggered the full CONUS
reprojection in memory. The `meteo_res` config option allows users to override the
automatic resolution.

______________________________________________________________________

## 2. `write_gridded` loads entire dataset into memory (OOM on large grids)

**Summary**:

`SfincsMeteo.write_gridded()` (`meteo.py`, line 180) calls `self.data.load()`, which
materializes the full lazy `dask`-backed dataset into RAM. Combined with issue 1
(CONUS-extent grids), this easily exceeds available memory. Even with properly clipped
grids, a 7-day NWM setup with precip + wind + pressure can require ~10-15 GB of
contiguous memory.

**Root cause**:

```python
def write_gridded(self, filename=None, rename=None):
    ...
    ds = self.data.load()  # <-- full materialization
    ...
    ds.to_netcdf(filename, ...)
```

**Suggested fix**:

Stream one time step at a time via the netCDF4 library, keeping peak memory near ~150 MB
regardless of the total dataset size. See the `_write_gridded_lazy` implementation in
our monkey-patch for a working reference (writes each `ds.isel(time=i).compute()` to a
pre-created netCDF4 file with an unlimited time dimension).

**Current workaround**:

We monkey-patch `SfincsMeteo.write_gridded` at import time (via
`patch_meteo_write_gridded()` in `_hydromt_compat.py`, called from
`apply_all_patches()`) to replace it with a streaming writer that writes one time-step
at a time via `netCDF4.Dataset`, keeping peak memory near ~150 MB instead of
materializing the full 3-D array.

______________________________________________________________________

## 3. `write_netcdf_safely` can crash the HDF5 C library on incompatible files

**Summary**:

`write_netcdf_safely()` (`utils.py`, line 1921) opens an existing netCDF file to check
whether the data changed before overwriting. If the existing file has an incompatible
schema (e.g. different number of stations, different dimension sizes), the HDF5 library
can segfault during the comparison read.

**Root cause**:

```python
def write_netcdf_safely(ds, abs_file_path, encoding=None):
    ds = ds.load()
    if abs_file_path.exists():
        try:
            existing_ds = GeoDataset.from_netcdf(  # <-- crash here
                abs_file_path, crs=ds.raster.crs, chunks="auto"
            )
            changed = not ds.equals(existing_ds)
            existing_ds.close()
        except Exception:
            changed = True  # fail-safe
```

The `except Exception` on line 1952 is intended as a fail-safe, but the crash occurs
inside the HDF5 C library (`H5VL__native_blob_specific` → `H5T__vlen_disk_isnull`),
which raises `SIGTERM`/`SIGSEGV` rather than a Python exception. The process is killed
before the except clause can execute.

**Reproduction**:

1. Run a SFINCS pipeline that writes `sfincs_netbndbzsbzifile.nc` with 119,903 stations
    (e.g. from un-interpolated STOFS data).
1. Fix the pipeline to produce the correct 15 stations.
1. Re-run. `write_netcdf_safely` opens the old 119,903-station file to compare with the
    new 15-station dataset. The HDF5 library segfaults (exit code 139) during
    `GeoDataset.from_netcdf()`.

**Suggested fix**:

The comparison-before-write optimization is fragile when files can have entirely
different schemas across runs. Options:

- **Option A (simple):** Delete the existing file before writing instead of comparing.
    The comparison only saves a write when the data is identical, which is uncommon in a
    re-run scenario.
- **Option B (defensive):** Compare only metadata (dimensions, shapes, dtypes) before
    attempting a full `ds.equals()`. Skip the comparison entirely if schemas differ.

**Current workaround**:

`SfincsInitStage._remove_stale_outputs()` deletes all generated netCDF files (forcing,
output, boundary) at the start of every pipeline run, so `write_netcdf_safely` never
encounters a stale file with an incompatible schema.

______________________________________________________________________

## 4. `_validate_and_prepare_gdf` hard-codes `"index"` dimension name

**Summary**:

`BoundaryConditionComponent._create_dummy_dataset` uses `dims=("time", "index")`, but
`GeoDataset.from_gdf` derives the spatial dimension name from `gdf.index.name`. When the
input GeoDataFrame's index is named something other than `"index"` (e.g. `"node"` from
ADCIRC/STOFS data), the dimension names diverge and `from_gdf` raises:

```python
ValueError: Index dimension node not found in data_vars
```

**Root cause**:

`_validate_and_prepare_gdf` does not normalize the GDF index name to `"index"` before it
is used downstream.

**Suggested fix**:

At the end of `_validate_and_prepare_gdf`, ensure the index name is always `"index"`:

```python
def _validate_and_prepare_gdf(self, gdf):
    ...  # existing validation
    if gdf.index.name != "index":
        gdf.index.name = "index"
    return gdf
```

**Current workaround**:

`patch_boundary_conditions_index_dim()` in `_hydromt_compat.py` (called from
`apply_all_patches()`) monkey-patches `SfincsBoundaryBase._validate_and_prepare_gdf` to
normalize the GDF index name to `"index"` after validation.

______________________________________________________________________

## 5. `_serialize_crs` crashes on CRS without an authority code

**Summary**:

`hydromt.typing.crs._serialize_crs` calls `list(crs.to_authority())` without guarding
against `to_authority()` returning `None`. This raises
`TypeError: 'NoneType' object is not iterable` for any CRS that has no EPSG code and no
recognized authority (e.g. custom `proj` strings).

**Suggested fix**:

```python
def _serialize_crs(crs):
    epsg = crs.to_epsg()
    if epsg:
        return epsg
    auth = crs.to_authority()
    if auth is not None:
        return list(auth)
    return crs.to_wkt()
```

Note: this is in **`hydromt`** core, not `hydromt-sfincs`.

**Current workaround**:

`patch_serialize_crs()` in `_hydromt_compat.py` (called from `apply_all_patches()`)
swaps the function's `__code__` in-place at import time so that existing `Pydantic`
`PlainSerializer` references pick up the fix.

______________________________________________________________________

## 6. NWM LDASIN coordinate rounding errors rejected as irregular grid

**Summary**:

NWM LDASIN files store projected coordinates (LCC, in meters) with floating-point
rounding errors up to ~0.125 m. `hydromt`'s raster accessor rejects them with:

```python
ValueError: not a regular grid
```

because its internal tolerance (`atol=5e-4`) is too tight for meter-scale coordinates.

**Suggested fix**:

Either increase the tolerance or provide a built-in `round_coords` preprocessor that
users can reference in their data catalog YAML:

```yaml
nwm_ana_meteo:
  ...
  driver_kwargs:
    preprocess: round_coords
```

Note: this is in **`hydromt`** core.

**Current workaround**:

`register_round_coords_preprocessor()` in `_hydromt_compat.py` (called from
`apply_all_patches()`) registers a custom `round_coords` preprocessor in `hydromt`'s
`PREPROCESSORS` dictionary that rounds x/y coordinates to the nearest integer before the
regularity check.

______________________________________________________________________

## 7. `water_level.create(geodataset=...)` passes all source stations to netCDF output

**Summary**:

`SfincsWaterLevel.create(geodataset=...)` passes *all* source stations found within the
model region to the boundary netCDF file. When the model has a `.bnd` file that defines
explicit boundary points (the normal SFINCS setup), the station count in the netCDF must
exactly match the boundary point count. The mismatch causes SFINCS to either crash or
silently produce incorrect boundary forcing.

**Root cause**:

The `create()` code path loads the `geodataset`, clips it to the model region, and
writes all matching stations to `sfincs_netbndbzsbzifile.nc` without checking whether a
`.bnd` file exists or how many boundary points it defines. For a STOFS `geodataset` over
the Texas Gulf Coast domain, this can yield ~119,903 source stations, far more than the
15 boundary points in `sfincs.bnd`.

**Suggested fix**:

When a `.bnd` file exists, `water_level.create()` should spatially interpolate the
`geodataset` to the boundary point locations (e.g. via IDW from the nearest source
nodes) and write only those interpolated time-series, matching the `.bnd` point count.

**Current workaround**:

`SfincsForcingStage._create_geodataset_forcing()` in `sfincs_build.py` bypasses
`model.water_level.create(geodataset=...)` entirely. It reads the `.bnd` file, loads the
`geodataset`, spatially interpolates to each boundary point using inverse-distance
weighting (IDW) from the nearest source nodes, then injects the result via
`model.water_level.set(df=..., gdf=...)`.

______________________________________________________________________

## 8. Missing `bzi` (infragravity) variable in boundary netCDF

**Summary**:

SFINCS unconditionally reads a `zi` (infragravity water level) variable from the
boundary netCDF file (`sfincs_ncinput.F90:118`). When `hydromt-sfincs` writes
`sfincs_netbndbzsbzifile.nc` it only writes `bzs` (surface water level), omitting `bzi`.
SFINCS crashes with a netCDF read error on startup.

**Root cause**:

The boundary writer in `hydromt-sfincs` does not generate the `bzi` variable. SFINCS's
Fortran I/O code does not check whether `zi` exists before attempting to read it, so the
missing variable causes a hard crash rather than a graceful fallback to zero.

**Suggested fix**:

`hydromt-sfincs` should always write a `bzi` variable alongside `bzs`. If no
infragravity data is available, it should be zero-filled with the same shape as `bzs`.

**Current workaround**:

`SfincsForcingStage._inject_water_level()` in `sfincs_build.py` manually adds a
zero-filled `bzi` variable after calling `model.water_level.set()`:

```python
ds = model.water_level.data
if "bzi" not in ds.data_vars:
    ds["bzi"] = xr.zeros_like(ds["bzs"])
```

______________________________________________________________________

## 9. Discharge source points on inactive grid cells cause segfault

**Summary**:

When a discharge source point (from the `.src` file) falls on an inactive grid cell
(`mask != 1`), SFINCS segfaults during the simulation. This is a Fortran-side bug but
`hydromt-sfincs` should validate source locations against the grid mask before writing.

**Root cause**:

The SFINCS Fortran binary maps each source point to the nearest grid cell. If that cell
is inactive, the cell index is left at its default value of 0. The code later accesses
`zs(0)` (water level at cell 0) without a bounds check, causing a segfault (exit code
139\) or out-of-bounds array access.

**Suggested fix**:

`hydromt-sfincs` should validate discharge source points against the grid mask when they
are added and either:

- **Option A (strict):** Raise an error listing the offending points.
- **Option B (lenient):** Drop inactive-cell points with a warning, keeping only those
    on active cells.

**Current workaround**:

`SfincsDischargeStage._filter_active_cells()` in `sfincs_build.py` uses
`scipy.spatial.cKDTree` to map each source point to its nearest quadtree face, checks
the face mask value, and drops points on inactive cells. The names of dropped points are
logged as a warning.

______________________________________________________________________

## 10. `_parse_river_list` crashes on freshly created models (`KeyError: 'geoms'`)

**Summary**:

`SfincsModel._parse_river_list()` crashes with `KeyError: 'geoms'` when called on a
model created in write mode (`mode="w+"`). This makes it impossible to use river burning
(`burn_river_rect`) during subgrid table creation on new models — i.e. whenever
`subgrid.create(river_list=...)` is called as part of a model-building workflow rather
than on a model read from disk.

**Root cause**:

`_parse_river_list` (`sfincs.py`, line 744) checks whether a river centerline dataset
name already exists as a loaded model geometry:

```python
if isinstance(rivers, str) and rivers in self.geoms:
    gdf_riv = self.geoms[rivers].copy()
else:
    gdf_riv = self.data_catalog.get_geodataframe(rivers, ...)
```

In `hydromt` v1+, `self.geoms` is resolved via `Model.__getattr__` →
`Model.get_component`, which looks up `"geoms"` in the component registry
(`self.components["geoms"]`). On freshly created models the `geoms` component is never
registered, so the lookup raises:

```python
KeyError: "geoms"
```

Python's `and` operator should short-circuit, but the `KeyError` is raised by the
attribute access itself (`self.geoms`), not by the `in` operator. The exception
propagates before the `in` check can execute, and the `else` branch
(`data_catalog.get_geodataframe`) — which is the correct code path for new models — is
never reached.

**Reproduction**:

```python
from hydromt_sfincs import SfincsModel

sf = SfincsModel(root="./test_model", mode="w+", data_libs=["my_catalog.yml"])

# ... create grid, elevation, mask, roughness ...

sf.subgrid.create(
    elevation_list=[{"elevation": "dem"}],
    roughness_list=[{"lulc": "esa_worldcover"}],
    river_list=[{"centerlines": "my_rivers"}],  # <-- KeyError: 'geoms'
    nr_subgrid_pixels=5,
)
```

**Suggested fix**:

Guard the `self.geoms` access in `_parse_river_list` so that a missing component is
treated the same as an empty one:

```python
try:
    model_geoms = self.geoms
except KeyError:
    model_geoms = {}

if isinstance(rivers, str) and rivers in model_geoms:
    gdf_riv = model_geoms[rivers].copy()
else:
    gdf_riv = self.data_catalog.get_geodataframe(rivers, ...)
```

**Current workaround**:

`patch_parse_river_list_geoms()` in `_hydromt_compat.py` (called from
`apply_all_patches()`) wraps `_parse_river_list` to temporarily inject an empty dict
into the model instance's `__dict__` under the key `"geoms"` when the component is
missing. Python's normal attribute lookup finds the instance dict entry before falling
through to `__getattr__`, so `self.geoms` resolves to `{}` and the `in` check evaluates
to `False`.

______________________________________________________________________

## 11. `read_map_file` fails for quadtree grids (missing UGRID topology)

**Summary**:

`SfincsOutput.read_map_file()` (`output.py`, line 149) calls `xugrid.load_dataset()` on
`sfincs_map.nc` for quadtree grids, expecting UGRID mesh topology. The SFINCS Fortran
executable writes the map output on a regular *(n, m)* structured grid — even for
quadtree models — without any UGRID topology variables. `xugrid` raises:

```python
ValueError: The file or object does not contain UGRID conventions data.
One or more UGRID topologies are required.
```

This aborts the entire `output.read()` call, so history-file data (`point_zs`,
`point_h`) is also lost even though `sfincs_his.nc` is plain NetCDF and would read fine.

**Root cause**:

The SFINCS Fortran executable outputs map data on a flat *(n, m)* grid (dimensions
matching the quadtree's structured index space) using SGRID conventions. The model's
`sfincs.nc` stores `n` and `m` variables (indexed by `mesh2d_nFaces`) that map each
UGRID face to its *(row, col)* position in this structured grid, but the map output
itself contains no mesh topology.

```python
# output.py, line 148-156 (upstream)
elif self.model.grid_type == "quadtree":
    with xu.load_dataset(fn_map) as ds:     # <-- fails here
        ds = ds.set_coords(["mesh2d_node_x", "mesh2d_node_y"])
        crs = ds["crs"].values
        ds.drop_vars("crs")
        ds.grid.set_crs(CRS.from_user_input(crs))
        self.set(ds, split_dataset=True)
```

**Suggested fix**:

When `xugrid.load_dataset` fails for a quadtree map file, read the file with plain
`xarray`, re-index each *(n, m)* variable to the `mesh2d_nFaces` dimension using the
`n`/`m` index arrays from the model grid, and wrap the result in a
`xugrid.UgridDataset`:

```python
elif self.model.grid_type == "quadtree":
    try:
        uds = xu.load_dataset(fn_map)
    except Exception:
        grid_ds = self.model.quadtree_grid.data
        ugrid = grid_ds.ugrid.grid
        n_idx = grid_ds["n"].values - 1   # 0-indexed row
        m_idx = grid_ds["m"].values - 1   # 0-indexed col

        ds_map = xr.open_dataset(fn_map)
        face_vars = {}
        for vname, da in ds_map.data_vars.items():
            if vname in drop:
                continue
            dims = list(da.dims)
            if "n" in dims and "m" in dims:
                vals = da.values
                if len(dims) == 2:          # (n, m) → (nFaces,)
                    face_vars[vname] = xr.DataArray(
                        vals[n_idx, m_idx],
                        dims=[ugrid.face_dimension],
                    )
                elif len(dims) == 3:        # (time, n, m) → (time, nFaces)
                    face_vars[vname] = xr.DataArray(
                        vals[:, n_idx, m_idx],
                        dims=[dims[0], ugrid.face_dimension],
                        coords={dims[0]: da.coords[dims[0]]},
                    )
        ds_map.close()
        uds = xu.UgridDataset(xr.Dataset(face_vars), grids=[ugrid])
        uds.ugrid.grid.set_crs(self.model.crs)

    self.set(uds, split_dataset=True)
```

**Current workaround**:

`patch_quadtree_output_read()` in `_hydromt_compat.py` (called from
`apply_all_patches()`) monkey-patches `SfincsOutput.read_map_file` to perform the
re-indexing described above when the original `xugrid.load_dataset` call fails for
quadtree grids.

______________________________________________________________________

## 12. `SfincsQuadtreeSubgridTable.data` property has no setter (read crashes)

**Summary**:

`SfincsQuadtreeSubgridTable.read()` (`quadtree_subgrid.py`, line 82) assigns
`self.data = xr.load_dataset(...)`, but `data` is a read-only `@property` (getter only,
no setter). This raises
`AttributeError: property 'data' of 'SfincsQuadtreeSubgridTable' object has no setter`
when reading a model that contains quadtree subgrid tables.

**Root cause**:

The `data` property (line 28) is defined as a getter that returns `self._data` with lazy
initialization, but has no corresponding setter:

```python
@property
def data(self):
    if self._data is None:
        self.read()      # recursion if read() also uses self.data
    return self._data

def read(self, filename=None, ...):
    ...
    self.data = xr.load_dataset(filename)  # <-- AttributeError: no setter
```

The `create()` method (line 193) correctly uses `self._data = ...`, so the `read()`
method simply needs to follow the same pattern.

**Suggested fix**:

Add a setter to the `data` property:

```python
@data.setter
def data(self, value):
    self._data = value
```

**Current workaround**:

`patch_quadtree_subgrid_data_setter()` in `_hydromt_compat.py` (called from
`apply_all_patches()`) adds a setter to the `data` property that forwards to
`self._data`.

______________________________________________________________________

## 13. `get_indices_at_points` broken line-continuation (`ifirst` receives a tuple)

**Summary**:

`SfincsQuadtreeGrid.get_indices_at_points()` (`quadtree.py`, line 581) has a broken line
continuation that causes `ifirst[ilev]` to receive the raw `np.where` tuple instead of
the scalar first-index. This eventually raises
`TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'`
when `ifirst` values are used in arithmetic.

**Root cause**:

The upstream code (lines 581–582) reads:

```python
ifirst[ilev] = np.where(self.data["level"].to_numpy()[:] == ilev + 1)
                [0][0]
```

Because the closing `)` on line 581 terminates the expression, `[0][0]` on line 582 is
parsed as a standalone no-op statement (list literal indexing). `ifirst[ilev]` receives
the full `np.where` result tuple `(array([...]),)` instead of the intended scalar
first-index.

**Suggested fix**:

Put `[0][0]` on the same line as `np.where()`:

```python
ifirst[ilev] = np.where(self.data["level"].to_numpy()[:] == ilev + 1)[0][0]
```

**Current workaround**:

`patch_quadtree_get_indices_at_points()` in `_hydromt_compat.py` (called from
`apply_all_patches()`) replaces the entire method with a corrected copy that fixes the
`ifirst` computation and caches the result for reuse.

______________________________________________________________________

## 14. `make_index_cog` — wrong component names, closed-dataset access, missing CRS reprojection

**Summary**:

`make_index_cog()` (`workflows/downscaling.py`) has three bugs that collectively make it
impossible to build a flood-depth index COG for quadtree models when the DEM uses a
different CRS than the model grid.

**Root cause**:

**(a) Wrong component attribute names** (lines 113–116):

```python
if model.grid_type == "quadtree":
    indices = model.quadtree.get_indices_at_points(xx, yy)  # <-- wrong
elif model.grid_type == "regular":
    indices = model.reggrid.get_indices_at_points(xx, yy)  # <-- wrong
```

The actual HydroMT component names are `"quadtree_grid"` and `"grid"`, not `"quadtree"`
and `"reggrid"`. These raise `AttributeError`.

**(b) Closed-dataset access** (lines 107–108):

```python
with rasterio.open(topobathy_fn) as src:
    ...  # src closes here

# Later, inside the block loop:
x_coords = transform[2] + (np.arange(bm0, bm1) + 0.5) * src.transform[0]  # <-- closed
y_coords = transform[5] + (np.arange(bn0, bn1) + 0.5) * src.transform[4]  # <-- closed
```

`src` is accessed after the `with` block exits, reading from a closed rasterio dataset.
`transform` was correctly saved earlier, but the code inconsistently references both
`transform[2]` (saved) and `src.transform[0]` (closed).

**(c) Missing CRS reprojection** (lines 107–114):

The `xx`, `yy` meshgrid coordinates are in the DEM's CRS (e.g. EPSG:4269, NAD83
geographic, degrees), but `get_indices_at_points` expects coordinates in the model's CRS
(e.g. EPSG:32614, UTM, metres). Without reprojection, every point falls outside the grid
and the entire index COG is filled with nodata.

**Suggested fix**:

```python
# (a) Use correct component names:
grid_comp = model.quadtree_grid if model.grid_type == "quadtree" else model.grid

# (b) Use saved transform, not src.transform:
x_coords = transform[2] + (np.arange(bm0, bm1) + 0.5) * transform[0]
y_coords = transform[5] + (np.arange(bn0, bn1) + 0.5) * transform[4]

# (c) Reproject DEM coordinates → model CRS before indexing:
from pyproj import Transformer

proj = Transformer.from_crs(dem_crs, model.crs, always_xy=True)
xx, yy = proj.transform(xx, yy)
indices = grid_comp.get_indices_at_points(xx, yy)
```

**Current workaround**:

`patch_make_index_cog()` in `_hydromt_compat.py` (called from `apply_all_patches()`)
replaces `make_index_cog` with a corrected version that fixes all three bugs: resolves
the correct grid component, uses the saved `transform` variable, and adds a
`pyproj.Transformer` to reproject DEM coordinates to the model CRS before calling
`get_indices_at_points`.
