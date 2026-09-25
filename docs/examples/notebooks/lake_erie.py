# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lake Erie SFINCS Tutorial - bring your own DEM
#
# This notebook builds and runs a [SFINCS](https://sfincs.readthedocs.io)
# model for a spot on the Ohio shore of Lake Erie (Fairport Harbor), forced at
# the open boundary by NOAA's **Lake Erie Operational Forecast System
# (LEOFS)** - the FVCOM model behind GLOFS.
#
# It follows the same three phases as the
# [Lavaca Bay notebook](lavaca.ipynb), create, run, visualize, and differs
# from it in three ways that matter:
#
# | | Lavaca Bay | Lake Erie |
# | --- | --- | --- |
# | Boundary forcing | `stofs` (ocean) | `glofs` + `glofs_model: leofs` (FVCOM) |
# | Elevation | auto-fetched `noaa_3m` + `gebco_15arcs` | **hand-downloaded DEM + your own data catalog** |
# | Vertical datum | NAVD88, tides about MSL | Lake Erie **Low Water Datum** (173.5 m IGLD85) |
#
# None of the built-in elevation sources (`noaa_3m`, `noaa_crm`, `nws_30m`,
# `copdem_30m`, `gebco_15arcs`) carry Great Lakes bathymetry, so **step 1 is
# the part you cannot skip**: download a DEM and register it with HydroMT
# yourself. The rest of the pipeline then behaves exactly as it does on the
# coast.
#
# ## The domain files
#
# This example runs end to end as shipped - a working Fairport Harbor domain is
# included, so you can execute every cell before changing anything. The geometry
# lives in `docs/examples/lake-erie/` next to the configs:
#
# | File | Shipped? | What it is |
# | --- | --- | --- |
# | `sfincs_aoi_lake_erie.geojson` | **yes** | model domain polygon, 425 km² at Fairport Harbor |
# | `discharge_nwm.geojson` | **yes** | 4 NWM flowpaths - the Chagrin and Grand Rivers - keyed by an `ID` column |
# | `refine.geojson` | no | optional sub-area to resolve finer quadtree levels; `grid.refinement` is commented out in `create.yaml` unless you add one |
#
# **To model somewhere else on the lake, replace those two files** and re-run
# from section 2, which measures whatever geometry it finds against the DEM and
# prints the `grid` and `mask` settings to paste into `create.yaml` - so those
# numbers come from your domain rather than from the shipped one. Any CRS is
# fine; everything is reprojected. Section 1 needs no geometry at all, since the
# DEM covers the whole lake.
#
# The two sections below show how the shipped files were made, which is also the
# recipe for making your own.
#
# ### Working in your own copy
#
# Every path in `create.yaml`, `run.yaml` and `dem_catalog.yml` is relative to
# the example folder, and everything the example downloads lands in
# `downloads/` inside it, so the folder can be copied and run anywhere:
#
# ```bash
# cp -r docs/examples/lake-erie ~/my_erie_run
# LAKE_ERIE_DIR=~/my_erie_run jupyter lab docs/examples/notebooks/lake_erie.ipynb
# ```
#
# Three things land under `downloads/`, all disposable: `dem/` (~70 MB, step
# 1b), `grid/` (ESA WorldCover clips written by `create`), and `forcing/`
# (GB-scale GLOFS and NWM data written by `run`). If you already have a shared
# forcing cache, point `paths.raw_download_dir` at it and leave the other two
# alone.
#
# ### Drawing the AOI
#
# The shipped domain was created with the `nwm_coastal` QGIS plugin's polygon
# sketcher - the orange rectangle below, on the Lake Erie shore at Fairport
# Harbor, Ohio. Use the same workflow to draw your own.
#
# ![AOI sketched over Lake Erie with CO-OPS gauges](../images/Lake_Erie_SFINCS_SketcherPoly.jpg)
#
# Two things to note from how it is drawn:
#
# - **It straddles the shoreline.** The lakeward half gives the LEOFS boundary
#   somewhere to attach to; the landward half is the floodplain you
#   actually want results in. An AOI entirely offshore has nothing to inundate,
#   and one entirely onshore has no boundary.
# - **It contains a gauge.** The stars are NOAA CO-OPS stations; this domain
#   captures **9063053 (Fairport Harbor)**, which is what the pipeline validates
#   against later. Cleveland (9063063) sits just west of the box - if you want a
#   particular gauge in the comparison, make sure the polygon covers it.
#
# ### Selecting the flowpaths
#
# The plugin loads the National Hydrofabric, so the flowpaths (blue), their
# nexus points (green) and the catchment divides can be seen while selecting.
# Because this notebook runs a historical case, the flowpaths override option
# was chosen when loading the basemap. The NWM_v3_hydrofabric.gdb was chosen
# as the override, with nwm_reaches_conus as the name. After selecting the
# flowpaths entering the domain, they are exported to `discharge_nwm.geojson`.
#
# ![Merged divides polygon with the selected discharge flowpaths](../images/Lake_Erie_SFINCS_MergedPoly_SelectedFlowpaths.jpg)
#
# Select the reach that **crosses the AOI boundary**, since that crossing point
# is where the pipeline injects the discharge.

# %% [markdown]
# ## Setup
#
# Every path below is relative to the example folder, so the notebook starts by
# moving into it. It looks in three places, in order: `$LAKE_ERIE_DIR`, the
# current directory, then a `lake-erie` sibling. That covers running from
# `docs/examples/notebooks/` as shipped, and running from inside a copy of the
# folder you made somewhere else.

# %%
from __future__ import annotations

import os
from pathlib import Path

candidates = [Path.cwd(), Path.cwd().parent / "lake-erie"]
if os.environ.get("LAKE_ERIE_DIR"):
    candidates.insert(0, Path(os.environ["LAKE_ERIE_DIR"]))

for candidate in candidates:
    if (candidate / "create.yaml").exists():
        example_dir = candidate.resolve()
        break
else:
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find the example folder (no create.yaml in):\n  {searched}\n"
        "Set LAKE_ERIE_DIR to point at your copy."
    )

os.chdir(example_dir)
print(f"working in {example_dir}")

# %% [markdown]
# ## 1. The DEM, step by step
#
# ### 1a. Pick a source
#
# We use NCEI's
# [Bathymetry of Lake Erie and Lake Saint Clair](https://www.ncei.noaa.gov/products/great-lakes-bathymetry)
# - a seamless 3 arc-second (~70 x 90 m) topobathy grid covering the whole
# lake, distributed as a 22 MB tarball. Good properties for this tutorial:
# one small file, real bathymetry, and a documented datum.
#
# Its ~70 m resolution is the ceiling on what the model can resolve on land,
# which is why `create.yaml` uses a uniform 256 m grid: refining below the DEM
# pixel would invent detail the data does not have. Add a `refine.geojson` and
# re-enable `grid.refinement` if you bring a finer DEM. For a production run,
# merge in a finer topographic source - see 1e.
#
# Other candidates, if you need more detail:
#
# - **NOAA OCM Coastal DEM: Lake Erie** - ~3 m lidar + sonar topobathy,
#   NAVD88, via the [Data Access Viewer](https://coast.noaa.gov/dataviewer/).
#   Best resolution, but it is an interactive order rather than a direct
#   download, and it is NAVD88 rather than LWD.
# - **USGS Lake Erie seamless topobathymetric DEM**
#   ([doi:10.5066/P1DA6L6U](https://doi.org/10.5066/P1DA6L6U)) - lidar plus
#   USACE dredge surveys.
# - **USGS 3DEP** 1 m / 10 m - land only, no lake bed.

# %% [markdown]
# ### 1b. Download and unpack
#
# The example catalog provided with the demo expects the GeoTIFF at
# `downloads/dem/erie_lld.tif`, relative to the example folder.
#
# It is deliberately *not* under `create.yaml`'s `download_dir` as
# the `create` stage populates that.

# %%
import tarfile
import urllib.request

DEM_URL = "https://www.ngdc.noaa.gov/mgg/greatlakes/erie/data/geotiff/erie_lld.geotiff.tar.gz"
dem_dir = Path("./downloads/dem")
dem_path = dem_dir / "erie_lld.tif"

if dem_path.exists():
    print(f"already present: {dem_path} ({dem_path.stat().st_size / 1e6:.1f} MB)")
else:
    dem_dir.mkdir(parents=True, exist_ok=True)
    tarball = dem_dir / "erie_lld.geotiff.tar.gz"
    print(f"downloading {DEM_URL}")
    urllib.request.urlretrieve(DEM_URL, tarball)
    with tarfile.open(tarball) as tar:
        for member in tar.getmembers():
            if member.name.endswith((".tif", ".prj", ".tfw")):
                member.name = Path(member.name).name  # flatten erie_lld/ prefix
                tar.extract(member, dem_dir)
    tarball.unlink()
    print(f"extracted: {sorted(p.name for p in dem_dir.iterdir())}")

# %% [markdown]
# ### 1c. Inspect it before trusting it
#
# Three things decide whether a DEM can be used as-is: its CRS, its NoData
# value, and its **vertical datum**. The first two are in the file; the third
# is not, so it has to be inferred and cross-checked against something you
# know. This reads the whole lake - no AOI needed yet.

# %%
import numpy as np
import rasterio

with rasterio.open(dem_path) as src:
    print(f"crs        : {src.crs.to_string()}")
    print(f"size       : {src.width} x {src.height}")
    print(f"pixel size : {src.res[0]:.8f} deg  (~{src.res[0] * 111e3 * 0.74:.0f} m in x at 42N)")
    print(f"dtype      : {src.dtypes[0]}")
    print(f"nodata     : {src.nodata}")
    print(f"bounds     : {src.bounds}")
    band = src.read(1, out_shape=(src.height // 4, src.width // 4))  # decimated overview

values = band[band > -9000]
print(f"\nelevation  : {values.min():+.1f} .. {values.max():+.1f} m")
print(f"median     : {np.median(values):+.1f} m")
print(f"below zero : {(values < 0).mean():.0%} of the lake-wide grid")

# %% [markdown]
# The datum is the interesting one. The lake bed reads **negative**, bottoming
# out near -62 m, which matches Lake Erie's true ~64 m maximum depth. The lake
# surface sits at ~174 m IGLD85, so these cannot be absolute elevations - they
# are heights relative to the lake's **Low Water Datum (173.5 m IGLD85)**,
# which is what the NCEI product description says and, conveniently, the same
# datum GLOFS publishes water levels in. That is why `run.yaml` can leave
# `forcing_to_mesh_offset_m` at `0.0`.

# %% [markdown]
# Rendered, the seamless topobathy looks like this - lake bed in blues, land in
# greens and tans, with this example's AOI in red.
#
# ![Lake Erie topobathy under the AOI](../images/Lake_Erie_Topobathy.jpg)

# %% [markdown]
# ### 1d. Write the HydroMT data catalog
#
# A dataset with `source: null` in `create.yaml` is looked up by name in the
# HydroMT data catalogs listed under `data_catalog.data_libs`. `dem_catalog.yml`
# is that catalog, and it ships with this example:
#
# ```yaml
# meta:
#   version: v1.0.0
#   name: lake_erie_dem
#   hydromt_version: '>1.0a,<2'
#
# erie_lld:
#   data_type: RasterDataset
#   uri: downloads/dem/erie_lld.tif
#   driver:
#     name: rasterio
#   metadata:
#     category: topography
#     crs: 4269
#   data_adapter:
#     rename:
#       erie_lld: elevtn
# ```
#
# Four things to get right when you adapt this to your own DEM:
#
# - **`uri`** is resolved relative to the catalog file, so keep the catalog
#   and the raster in a fixed relationship to each other.
# - **`data_type: RasterDataset`** with **`driver: {name: rasterio}`** for any
#   GeoTIFF; use `driver: {name: raster_xarray}` for NetCDF.
# - **`metadata.crs`** is only needed when the file itself lacks a CRS, but
#   stating it is good practice.
# - **`data_adapter.rename`** must map the variable HydroMT reads to
#   **`elevtn`**, which is the name HydroMT-SFINCS looks for. For a
#   single-band GeoTIFF, HydroMT names the variable after the *catalog entry
#   key* - `erie_lld` here - so the mapping is `erie_lld: elevtn`. Get this
#   wrong and the elevation stage will fail with a missing-variable error.
#
# The cell below is the check that saves you a failed `create` run: it reads
# the dataset through the catalog exactly as the pipeline will.

# %%
from hydromt import DataCatalog

catalog = DataCatalog(data_libs=["./dem_catalog.yml"])
da = catalog.get_rasterdataset("erie_lld", bbox=[-83.6, 41.55, -82.9, 41.85])
assert da.name == "elevtn", f"expected variable 'elevtn', got {da.name!r} - fix data_adapter.rename"
print(da)

# %% [markdown]
# ## 2. Measure your AOI against the DEM
#
# This reads the AOI in the example folder and reports what the DEM says
# about it. Nothing here is a pass/fail check - it is the measurement that
# tells you what `grid.resolution`, `grid.refinement` and the `mask`
# thresholds should be, in the LWD datum the DEM turned out to use.

# %%
import geopandas as gpd

aoi_candidates = sorted(Path().glob("*aoi*.geojson"))
aoi_path = aoi_candidates[0] if aoi_candidates else Path("./aoi.geojson")
if not aoi_path.exists():
    raise FileNotFoundError(
        f"No {aoi_path.resolve()}. Digitize your model domain and save it there "
        '(see "The domain files" at the top of this notebook), then re-run from here.'
    )

aoi = gpd.read_file(aoi_path).to_crs(4326)
refine = (
    gpd.read_file("./refine.geojson").to_crs(4326) if Path("./refine.geojson").exists() else None
)
flow_path = Path("./discharge_nwm.geojson")
flowlines = gpd.read_file(flow_path) if flow_path.exists() else None

utm = aoi.estimate_utm_crs()
area_km2 = aoi.to_crs(utm).area.sum() / 1e6
print(f"aoi       : {len(aoi)} feature(s), {area_km2:,.0f} km^2")
print(f"bounds    : {np.round(aoi.total_bounds, 4)}")
print(f"utm zone  : {utm.to_string()}  <- what grid.crs 'utm' will resolve to")
if refine is not None:
    overlap_m2 = refine.to_crs(utm).intersection(aoi.to_crs(utm).union_all()).area.sum()
    print(
        f"refine    : {len(refine)} feature(s), {overlap_m2 / 1e6:,.0f} km^2 of it inside the AOI"
    )
else:
    print("refine    : none - drop grid.refinement from create.yaml")
if flowlines is not None:
    cols = [c for c in flowlines.columns if c != "geometry"]
    print(f"flowpaths : {len(flowlines)} feature(s); columns {cols}")
else:
    print("flowpaths : none - drop river_discharge from create.yaml")

# %% [markdown]
# ### What the DEM says inside your AOI

# %%
clipped = catalog.get_rasterdataset("erie_lld", bbox=list(aoi.total_bounds)).load()
clipped = clipped.where(clipped > -9000)  # drop NoData before any statistic
inside = clipped.values[np.isfinite(clipped.values)]

print(f"elevation : {inside.min():+.2f} .. {inside.max():+.2f} m (LWD)")
print(f"below LWD : {(inside < 0).mean():.0%} of the AOI bounding box")
wet = inside[inside < 0]
if wet.size:
    print("\nsubmerged part, m relative to LWD:")
    for pct in (5, 25, 50, 75, 95):
        print(f"  {pct:>2}% of submerged cells are deeper than {np.percentile(wet, pct):+.2f}")
else:
    print("\nno submerged cells - this AOI is entirely above low water, which cannot be")
    print("right for a lake model; extend it lakeward before going on")

# %% [markdown]
# ### Settings derived from the above
#
# `mask.zmin` is the floor for active cells, so it goes below the deepest bed
# in the domain. `mask.boundary_zmax` selects which cells can carry the LEOFS
# water-level boundary: deep enough to pick open water rather than shoreline,
# shallow enough that a usable band of cells qualifies. Taking the depth that
# a quarter of the submerged cells are deeper than is a reasonable first cut.
#
# For resolution, the DEM is the ceiling - refining below its ~70 m pixel
# invents detail. This targets a finest cell near the pixel size and backs out
# the base resolution from the refinement level.

# %%
deepest = float(np.floor(inside.min()))
boundary_zmax = float(np.round(np.percentile(wet, 25), 1)) if wet.size else None
dem_res_m = float(clipped.raster.res[0]) * 111e3 * 0.74
level = 2
finest = 2 ** int(np.round(np.log2(dem_res_m)))
base = finest * 2**level

print("# paste into create.yaml")
print("grid:")
print(f"  resolution: {base}")
print("  refinement:")
print("  - polygon: ./refine.geojson")
print(
    f"    level: {level}                # {base} -> {finest} m, vs a ~{dem_res_m:.0f} m DEM pixel"
)
print("mask:")
print(f"  zmin: {deepest - 5}")
print(f"  boundary_zmax: {boundary_zmax}")
print("  reset_bounds: true")
print("  keep_largest_only: true")
print(
    f"\n# cells that would qualify as boundary: {(inside < boundary_zmax).mean():.1%} of the bbox"
)

# %% [markdown]
# Sanity-check that against the map below: the boundary cells should form a
# band along the open-water edge of the AOI, not a patch in the middle. Raise
# `boundary_zmax` toward zero if too few cells qualify, lower it if the band
# reaches inshore.

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
clipped.plot(ax=axes[0], cmap="terrain", cbar_kwargs={"label": "elevation (m, LWD)"})
axes[0].set_title("DEM over the AOI bounding box")
(clipped < boundary_zmax).plot(ax=axes[1], cmap="Blues", add_colorbar=False)
axes[1].set_title(f"cells below boundary_zmax = {boundary_zmax} m")
for ax in axes:
    aoi.boundary.plot(ax=ax, color="red", linewidth=1.5, label="AOI")
    if refine is not None:
        refine.boundary.plot(ax=ax, color="magenta", linewidth=1.2, label="refine")
    if flowlines is not None and len(flowlines):
        flowlines.to_crs(4326).plot(ax=ax, color="black", linewidth=0.8, label="NWM flowpaths")
    ax.legend(loc="upper right", fontsize=8)

# %% [markdown]
# ## 3. Create the SFINCS model
#
# From here on this is the Lavaca workflow. The only differences in the create
# config are the `data_catalog` block, the `source: null` elevation entry, and
# mask thresholds expressed in LWD instead of NAVD88.
#
# The `grid` and `mask` values below, and the same ones in `create.yaml`, are
# what section 2 derived for a 425 km^2 Fairport Harbor domain. Replace
# them with what section 2 printed for your AOI.

# %%
from coastal_calibration import SfincsCreateConfig, SfincsCreator, configure_logger

configure_logger(level="INFO")

create_config = SfincsCreateConfig.from_dict(
    {
        "aoi": "./sfincs_aoi_lake_erie.geojson",
        "output_dir": "./sfincs_fh_lake_erie",
        "download_dir": "./downloads/grid",
        "grid": {
            "resolution": 256,
            "crs": "utm",
            "rotated": False,
            # add a refine.geojson and re-enable to resolve the river mouths finer
            "refinement": [],
        },
        "data_catalog": {"data_libs": ["./dem_catalog.yml"]},
        "elevation": {
            "datasets": [
                {"name": "erie_lld", "zmin": -20000, "source": None},
            ],
            "buffer_cells": 1,
        },
        "mask": {
            "zmin": -27.0,
            "boundary_zmax": -19.1,
            "reset_bounds": True,
            "keep_largest_only": True,
        },
        "subgrid": {
            "nr_subgrid_pixels": 4,
            "lulc_dataset": "esa_worldcover",
            "manning_land": 0.04,
            "manning_sea": 0.02,
        },
        # Drop this block if you have no flowpaths yet.
        "river_discharge": {
            "flowlines": "./discharge_nwm.geojson",
            "nwm_id_column": "ID",
        },
        "add_noaa_gages": True,
    }
)

# %%
creator = SfincsCreator(create_config)
result = creator.run()
if not result.success:
    raise RuntimeError(f"Model creation failed at stage '{result.stages_failed}': {result.errors}")
print(result)

# %% [markdown]
# ### Inspect the created model

# %%
output = Path("./sfincs_fh_lake_erie")
assert output.exists(), (
    f"Output directory not found: {output.resolve()} - run the create step first."
)

for f in sorted(output.iterdir()):
    if f.name.startswith(".") or f.suffix == ".log":
        continue
    size = f.stat().st_size
    label = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.1f} KB"
    print(f"  {f.name:<30s} {label}")

# %% [markdown]
# ## 4. Run the simulation pipeline
#
# ### LEOFS boundary forcing
#
# `boundary.source: glofs` with `glofs_model: leofs` replaces STOFS. Two
# constraints come with it:
#
# - `simulation.coastal_domain` must be `greatlakes`, and that domain accepts
#   no other boundary source - the tidal atlases and STOFS do not cover the
#   lakes. Either setting without the other is rejected at config load.
# - Only **nowcast** files are read, from NCEI's archive, so the run window
#   must be in the past (2016 onward, depending on the lake).
#
# The download stage reads only `time`, `lon`, `lat` and `zeta` out of each
# hourly FVCOM file - those files run up to ~180 MB an hour - and caches the
# slices under `downloads/forcing/coastal/glofs/leofs/`. If LEOFS changed its
# unstructured grid inside your window, the stage stops and tells you where;
# split the run at that time.
#
# Meteorology still comes from NWM (`meteo_source: nwm_ana`), whose CONUS
# domain covers the lakes.

# %% [markdown]
# ### Why these overrides
#
# A few `run_param_overrides` are specific to running SFINCS on a lake surface
# 174 m above sea level:
#
# | Override | Default | Why it is wrong here |
# | --- | --- | --- |
# | `zsini` = `0.89` | `0` | Initial water level, in mesh datum. On LWD, 0 means "exactly at low water" - usable, but Erie typically sits a few decimetres above LWD. Set it from the CO-OPS Fairport Harbor (9063053) record at your start time. On an *absolute* IGLD85 mesh the default would start the model dry. |
# | `latitude` = `41.75` | `0` | A projected grid carries no latitude, so Coriolis would be computed at the equator. |
#
# `forcing_to_mesh_offset_m: 0.0` is correct because the mesh and GLOFS
# share the LWD datum. If you rebuild the mesh from an absolute-elevation DEM
# (the NOAA OCM 3 m product, say), set it to `173.5` and raise `zsini`
# accordingly.

# %%
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

run_config = CoastalCalibConfig.from_dict(
    {
        "model": "sfincs",
        "simulation": {
            "start_date": "2026-09-21 12:00:00",
            "duration_hours": 24,
            "coastal_domain": "greatlakes",
            "meteo_source": "nwm_ana",
        },
        "boundary": {"source": "glofs", "glofs_model": "leofs"},
        "paths": {
            "work_dir": "./run",
            "raw_download_dir": "./downloads/forcing",
        },
        "download": {"enabled": True},
        "model_config": {
            "prebuilt_dir": "./sfincs_fh_lake_erie",
            "discharge_locations_file": "./sfincs_fh_lake_erie/sfincs_nwm.src",
            "merge_discharge": True,
            "forcing_to_mesh_offset_m": 0.0,  # GLOFS and the mesh are both on LWD
            "vdatum_mesh_to_msl_m": 0.0,  # unused on the lakes
            "include_precip": True,
            "include_wind": True,
            "include_pressure": True,
            "floodmap_dem": "./downloads/dem/erie_lld.tif",
            "run_param_overrides": {
                "tspinup": 10800,
                "advection": 0,
                "viscosity": 0,
                "nuvisc": 0.01,
                "cdnrb": 3,
                "cdwnd": [0.0, 28.0, 50.0],
                "cdval": [0.001, 0.0025, 0.0025],
                "zsini": 0.89,
                "latitude": 41.75,
            },
        },
    }
)

# %%
runner = CoastalCalibRunner(run_config)
result = runner.run()
if not result.success:
    raise RuntimeError(f"Model run failed at stage '{result.stages_failed}': {result.errors}")
print(result)

# %% [markdown]
# ## 5. Gauge comparison
#
# With `add_noaa_gages: true` the pipeline compares against NOAA CO-OPS lake
# gauges - Fairport Harbor (9063053) and its neighbours for this AOI. Lake gauges publish observations but no tide
# predictions and no MSL or MLLW, so observations are fetched in the lake's
# low-water datum and shifted by `forcing_to_mesh_offset_m`: the comparison
# is made in the **mesh datum**, not MSL as on the coast.

# %%
from IPython.display import Image, display

figs_dir = Path("run/sfincs_model/figs")
assert figs_dir.exists(), f"Results not found: {figs_dir.resolve()} - run the pipeline first."

for png in sorted(figs_dir.glob("stations_comparison_*.png")):
    display(Image(filename=str(png), width=800))

# %% [markdown]
# ## 6. Mesh and flood depth map

# %%
from coastal_calibration.plotting import SfincsGridInfo, plot_floodmap, plot_mesh

info = SfincsGridInfo.from_model_root("run/sfincs_model")
print(info)

# %%
fig, ax = plot_mesh(info, title="Lake Erie (Fairport Harbor) SFINCS mesh")

# %%
fig, ax = plot_floodmap(
    "run/sfincs_model/floodmap_hmax.tif",
    title="Max water depth, Fairport Harbor, Lake Erie",
)
fig.savefig("../images/lake_erie_thumb.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 7. Water-level field
#
# Identical to Lavaca from here - the post-processing API does not care which
# boundary source drove the run. One reading difference: `zs` is in the mesh
# datum, so on this model the numbers are metres **above Lake Erie LWD**, not
# above MSL.

# %%
from coastal_calibration.sfincs.outputs import load_sfincs_water_level

run_dir = Path("run/sfincs_model")
ds = load_sfincs_water_level(run_dir)
print(f"mesh_type     : {ds.attrs['mesh_type']}")
print(f"crs           : {ds.attrs.get('crs', '(not detected)')}")
print(f"dims          : {dict(ds.sizes)}")
print(f"zs range (m)  : {float(ds['zs'].min()):+.3f} .. {float(ds['zs'].max()):+.3f}")

# %%
from coastal_calibration.plotting import animate_water_level, plot_water_level

DRY_THRESHOLD = 0.05  # m - same default as plot_water_level
wet = ds["h"] > DRY_THRESHOLD
vmin, vmax = (float(v) for v in ds["zs"].where(wet).quantile([0.02, 0.98]).values)

fig, ax = plt.subplots(figsize=(11, 8))
plot_water_level(
    ds,
    time=ds.sizes["time"] // 2,
    variable="zs",
    ax=ax,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    colorbar=True,
    basemap=True,
    title="Fairport Harbor water level (m above LWD)",
)
snapshot_png = figs_dir / "water_level_snapshot.png"
fig.savefig(snapshot_png, dpi=150, bbox_inches="tight")
plt.close(fig)
display(Image(filename=str(snapshot_png), width=800))

# %% [markdown]
# ### Anomaly from the time-mean
#
# The wind-driven seiche is the signal worth looking at on Lake Erie: a
# strong southwesterly piles water into the eastern basin and draws down
# Fairport Harbor, and the basin then rocks back. Subtracting each cell's
# time-mean isolates that from the static bed elevation.

# %%
zs_anom = ds["zs"] - ds["zs"].mean("time")
ds_anom = ds.assign(zs_anom=zs_anom)
ds_anom["zs_anom"].attrs.update({"long_name": "water-level anomaly from time-mean", "units": "m"})
amp = float(abs(zs_anom.where(wet)).quantile(0.98).values)

fig, ax = plt.subplots(figsize=(11, 8))
plot_water_level(
    ds_anom,
    time=ds.sizes["time"] // 2,
    variable="zs_anom",
    ax=ax,
    cmap="RdBu_r",
    vmin=-amp,
    vmax=+amp,
    colorbar=True,
    title="Fairport Harbor water-level anomaly",
)
anomaly_png = figs_dir / "water_level_anomaly.png"
fig.savefig(anomaly_png, dpi=150, bbox_inches="tight")
plt.close(fig)
display(Image(filename=str(anomaly_png), width=800))

# %%
from IPython.display import Video

anim_path = animate_water_level(
    ds,
    figs_dir / "water_level_animation.mp4",
    variable="zs",
    fps=10,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    title_prefix="Fairport Harbor",
    mask_dry=True,
    dry_threshold=DRY_THRESHOLD,
)
Video(str(anim_path), embed=True, width=800)

# %% [markdown]
# ## Summary
#
# 1. Downloaded a Great Lakes topobathy DEM by hand, checked its CRS, NoData
#    and vertical datum, and registered it in `dem_catalog.yml` - the step
#    the coastal examples get for free from the built-in fetchers.
# 2. Validated AOI, refinement polygon and flowpaths against that DEM before
#    building anything.
# 3. Built the quadtree mesh with `SfincsCreator`, elevations on Lake Erie
#    Low Water Datum.
# 4. Ran the pipeline with LEOFS (FVCOM) boundary forcing and the four
#    elevated-domain `run_param_overrides`, then compared against CO-OPS lake
#    gauges in the mesh datum.
# 5. Plotted the mesh, flood depth map, water-level snapshot, seiche anomaly
#    and animation.
#
# ### Adapting this to another lake
#
# Swap `glofs_model` (`lmhofs` for Michigan-Huron, `loofs` for Ontario,
# `lsofs` for Superior), download that lake's grid from the same NCEI
# product, and change the LWD constants: Michigan-Huron 176.0 m, Ontario
# 74.2 m, Superior 183.2 m (IGLD85). Everything else carries over.
