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
# # Cook Inlet SFINCS - building a model in Alaska
#
# Builds and runs a SFINCS model for Cook Inlet, Alaska. The build inputs - an
# AOI polygon, the flowpaths entering it, and the quadtree refinement zones -
# were all drawn in QGIS with the `nwm_coastal` plugin, and sections 1 to 3
# show how.
#
# **These are suggested steps.** Any method of producing a polygon GeoJSON is
# fine; nothing below depends on having used the plugin.
#
# Alaska differs from the Lavaca and Lake Erie examples in one way that matters:
# no built-in elevation source covers it, so the DEM is downloaded by hand and
# registered in a data catalog, as in
# [Lake Erie SFINCS](lake_erie_sfincs.ipynb).
#
# ## Setup

# %%
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

notebook_dir = Path.cwd()  # assumes run from docs/examples/notebooks/
example_dir = (notebook_dir.parent / "alaska_sfincs").resolve()
os.chdir(example_dir)


def cli(*args: str) -> None:
    """Run a coastal-calibration CLI command and stream its output."""
    print(f"$ coastal-calibration {' '.join(args)}\n")
    subprocess.run([sys.executable, "-m", "coastal_calibration.cli", *args], check=True)


print(f"Working directory: {example_dir}")
print("Inputs:", sorted(p.name for p in example_dir.glob("*.geojson")))

# %% [markdown]
# ## 1. The AOI polygon
#
# Sketch a polygon around the area of interest, using the NHF catchment divides
# as a guide.
#
# Union it with the divides and look at the result.
#
# <div style="display: flex; gap: 1em; align-items: flex-start; flex-wrap: wrap;">
#   <figure style="flex: 1.89; margin: 0;">
#     <img src="../images/Alaska_SFINCS_initial_sketcher_poly.jpg" alt="Initial sketched polygon over Cook Inlet" style="width: 100%;">
#     <figcaption>Initial sketched polygon over Cook Inlet</figcaption>
#   </figure>
#   <figure style="flex: 1.84; margin: 0;">
#     <img src="../images/Alaska_SFINCS_intial_sketcher_merge.jpg" alt="Polygon merged with divides" style="width: 100%;">
#     <figcaption>Polygon merged with divides</figcaption>
#   </figure>
# </div>
#
# Edit the polygon to drop divides you do not want, re-merging as you go.
#
# <div style="display: flex; gap: 1em; align-items: flex-start; flex-wrap: wrap;">
#   <figure style="flex: 1.76; margin: 0;">
#     <img src="../images/Alaska_SFINCS_edit_sketcher.jpg" alt="Editing the sketched polygon" style="width: 100%;">
#     <figcaption>Editing the sketched polygon</figcaption>
#   </figure>
#   <figure style="flex: 1.74; margin: 0;">
#     <img src="../images/Alaska_SFINCS_edit_merged.jpg" alt="Re-merged after editing" style="width: 100%;">
#     <figcaption>Re-merged after editing</figcaption>
#   </figure>
# </div>
#
# Some nearshore areas have no divides at all; trace those by hand so the
# polygon follows the coast.
#
# <div style="display: flex; gap: 1em; align-items: flex-start; flex-wrap: wrap;">
#   <figure style="flex: 1.81; margin: 0;">
#     <img src="../images/Alaksa_SFINCS_missing_divide.jpg" alt="Nearshore area with no divide" style="width: 100%;">
#     <figcaption>Nearshore area with no divide</figcaption>
#   </figure>
#   <figure style="flex: 1.57; margin: 0;">
#     <img src="../images/Alaska_SFINCS_include_missing_divide.jpg" alt="Polygon extended over the missing divide" style="width: 100%;">
#     <figcaption>Polygon extended over the missing divide</figcaption>
#   </figure>
# </div>
#
# Merging can leave invalid geometry - self-intersections and the like. Check
# validity and fix it with QGIS's own geometry tools before saving. The result
# is `sfincs_alaska_cookinlet.geojson`.
#
# ![Final Cook Inlet AOI](../images/Alaska_SFINCS_final.jpg)

# %% [markdown]
# ## 2. Flowpaths entering the domain
#
# Load the NWM v3 geodatabase or an NHF GeoPackage through the plugin, select
# the flowpaths that cross into the AOI, and export them.
#
# In Alaska the NWM carries far more flowpaths than the NHF hydrofabric does,
# which makes the NWM set hard to choose from on its own. Here the NHF set was
# used to guide which NWM reaches to keep.
#
# <div style="display: flex; gap: 1em; align-items: flex-start; flex-wrap: wrap;">
#   <figure style="flex: 1.75; margin: 0;">
#     <img src="../images/SFINCS_NWM_vs_NGEN_flowpaths.jpg" alt="NWM and NGEN flowpaths compared" style="width: 100%;">
#     <figcaption>NWM and NGEN flowpaths compared</figcaption>
#   </figure>
#   <figure style="flex: 0.86; margin: 0;">
#     <img src="../images/SFINCS_example_selecting_flowpaths.jpg" alt="Selecting the flowpaths entering the domain" style="width: 100%;">
#     <figcaption>Selecting the flowpaths entering the domain</figcaption>
#   </figure>
# </div>
#
# Both sets are shipped. `create.yaml` uses the NWM one, keyed on its `ID`
# column; switch `river_discharge.flowlines` to the `_ngen` file and
# `nwm_id_column` to `fp_id` for a NextGen-keyed run.

# %%
import geopandas as gpd

for name in ("sfincs_alaska_flowpaths_nwm.geojson", "sfincs_alaska_flowpaths_ngen.geojson"):
    g = gpd.read_file(name)
    print(f"{name}: {len(g)} flowpaths, {g.crs}")

# %% [markdown]
# ## 3. Refinement zones
#
# The quadtree is refined where the detail is worth paying for. Open Alaska
# geodata helped decide - a roads layer and a structures layer, to find where
# people and assets actually are.
#
# ![Using additional data to place refinement zones](../images/SFINCS_use_adtnl_data_refine.jpg)
#
# QGIS's `buffer` around the flowpaths is another way to raise resolution along
# the rivers. Four steps, starting from the flowpaths layer: clip them to the
# AOI, buffer the result, split the buffer into single parts, and save it as a
# refinement polygon.
#
# <div style="display: flex; gap: 1em; align-items: flex-start; flex-wrap: wrap;">
#   <figure style="flex: 2.01; margin: 0;">
#     <img src="../images/SFINCS_make_buffered_rivers_clip_step.jpg" alt="1. Clip the flowpaths to the AOI" style="width: 100%;">
#     <figcaption>1. Clip the flowpaths to the AOI</figcaption>
#   </figure>
#   <figure style="flex: 1.98; margin: 0;">
#     <img src="../images/SFINCS_make_buffered_rivers_buffer_step.jpg" alt="2. Buffer the clipped flowpaths" style="width: 100%;">
#     <figcaption>2. Buffer the clipped flowpaths</figcaption>
#   </figure>
# </div>
#
# <div style="display: flex; gap: 1em; align-items: flex-start; flex-wrap: wrap;">
#   <figure style="flex: 1.99; margin: 0;">
#     <img src="../images/SFINCS_make_buffered_rivers_multi_to_single_step.jpg" alt="3. Multipart to single part" style="width: 100%;">
#     <figcaption>3. Multipart to single part</figcaption>
#   </figure>
#   <figure style="flex: 0.89; margin: 0;">
#     <img src="../images/SFINCS_make_buffered_rivers_save_step.jpg" alt="4. Save as a refinement polygon" style="width: 100%;">
#     <figcaption>4. Save as a refinement polygon</figcaption>
#   </figure>
# </div>
#
# The seven `sfincs_refine_*.geojson` polygons each refine two levels, taking
# 1024 m cells down to 256 m.

# %%
for p in sorted(example_dir.glob("sfincs_refine_*.geojson")):
    g = gpd.read_file(p)
    print(f"{p.name}: {g.to_crs(3338).area.sum() / 1e6:.0f} km2")

# %% [markdown]
# ## 4. Elevation
#
# Nothing auto-fetched reaches Alaska: the NOAA 3 m topobathy index holds only
# CONUS, Hawaii, PR and USVI, and the NCEI Coastal Relief Model stops at 49 N.
# NCEI's regional DEMs do cover Cook Inlet, over plain HTTPS.
#
# Two are used here, finest first:
#
# | file | resolution | extent |
# | --- | --- | --- |
# | `cook_inlet_ak_3_mhhw_2020.nc` | 3" (~90 m) | 60.20-61.70 N, upper inlet only |
# | `cook_inlet_ak_8_mhhw_2020.nc` | 8" (~240 m) | 58.40-61.75 N, spans the AOI |
#
# `gebco_15arcs` fills what is left past their footprints.
#
# With AWS credentials, `coastal-calibration prepare-topobathy <aoi> --domain
# alaska` fetches the NWS 30 m topobathy instead, which is finer than either.

# %%
NCEI = "https://www.ngdc.noaa.gov/thredds/fileServer/regional"
dem_dir = Path("./downloads/dem")
dem_dir.mkdir(parents=True, exist_ok=True)

for name in ("cook_inlet_ak_3_mhhw_2020.nc", "cook_inlet_ak_8_mhhw_2020.nc"):
    dest = dem_dir / name
    if dest.exists():
        print(f"already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"downloading {NCEI}/{name}")
        urllib.request.urlretrieve(f"{NCEI}/{name}", dest)
        print(f"  -> {dest.stat().st_size / 1e6:.1f} MB")

# %% [markdown]
# ### The vertical datum
#
# Both DEMs are on **MHHW**, not MSL. Cook Inlet has one of the largest tide
# ranges anywhere, so that is not a small difference: at Anchorage (9455920)
# MHHW sits 3.87 m above MSL, at Seldovia (9455500) 2.59 m.
#
# `create.yaml` adds `offset: 3.87` to both DEMs, which puts them on MSL so
# they merge cleanly with GEBCO and match the STOFS forcing datum. That single
# number is measured at Anchorage and the separation varies by more than a
# metre across the inlet, so it is exact near the city and approximate at the
# mouth. For a domain this wide the better answer is to pre-transform the
# rasters; the offset is the quick version.

# %%
import xarray as xr

for name in ("cook_inlet_ak_3_mhhw_2020.nc", "cook_inlet_ak_8_mhhw_2020.nc"):
    ds = xr.open_dataset(dem_dir / name)
    z = ds["z"]
    print(
        f"{name}: {dict(ds.sizes)} | lat {float(ds.lat.min()):.2f}..{float(ds.lat.max()):.2f}"
        f" | lon {float(ds.lon.min()):.2f}..{float(ds.lon.max()):.2f}"
        f" | z {float(z.min()):.1f}..{float(z.max()):.1f} m MHHW"
    )
    ds.close()

# %% [markdown]
# ### Land cover
#
# Roughness comes from ESA WorldCover, which ships at 10 m. Over this AOI that
# is a 73,500 x 36,830 raster - 2.7 billion pixels - and building the subgrid
# tables from it needs about 30 GB of memory.
#
# Resampling to 100 m cuts the raster 100x,
# to 27 million pixels, and the class distribution is unchanged. `mode` is the
# resampling here because the classes are categorical.

# %%
import subprocess as _sp

from coastal_calibration.data.esa_worldcover import fetch_esa_worldcover

lulc_dir = Path("./downloads/lulc")
lulc_tif = lulc_dir / "esa_worldcover.tif"

if lulc_tif.exists():
    print(f"already present: {lulc_tif} ({lulc_tif.stat().st_size / 1e6:.1f} MB)")
else:
    lulc_dir.mkdir(parents=True, exist_ok=True)
    native, _, _ = fetch_esa_worldcover(
        aoi=Path("./sfincs_alaska_cookinlet.geojson"),
        output_dir=lulc_dir,
        catalog_name="esa_worldcover_10m",
    )
    _sp.run(
        [
            "gdalwarp",
            "-tr",
            "0.000833",
            "0.000833",
            "-r",
            "mode",
            "-ot",
            "Byte",
            "-dstnodata",
            "0",
            "-co",
            "COMPRESS=DEFLATE",
            str(native),
            str(lulc_tif),
        ],
        check=True,
        capture_output=True,
    )
    native.unlink()
    print(f"resampled to 100 m: {lulc_tif} ({lulc_tif.stat().st_size / 1e6:.1f} MB)")

# %%
import rasterio

with rasterio.open(lulc_tif) as r:
    print(f"{r.width} x {r.height} = {r.width * r.height / 1e6:.1f} Mpx, dtype={r.dtypes[0]}")

# %% [markdown]
# ## 5. Create the model
#
# 1024 m base cells, refined to 256 m inside the seven zones, with 4 subgrid
# pixels per cell.

# %%
print(Path("create.yaml").read_text())

# %%
cli("create", "create.yaml")

# %% [markdown]
# ### What was built

# %%
model_dir = Path("sfincs_cookinlet")
print(sorted(p.name for p in model_dir.iterdir()))

# %% [markdown]
# ## 6. Run it
#
# 12 hours, the same window as the Alaska SCHISM example, forced by STOFS.
# This is the slow part - everything above is the model build.

# %%
cli("run", "run.yaml", "--dry-run")

# %%
cli("run", "run.yaml")

# %% [markdown]
# ## Summary
#
# The build inputs were drawn in QGIS, the elevation came from NCEI over plain
# HTTPS through a data catalog, and `create` plus `run` did the rest.
