# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lavaca Bay SFINCS Tutorial — CLI Workflow
#
# This notebook demonstrates how to build and run a
# [SFINCS](https://sfincs.readthedocs.io) coastal flood model for
# Lavaca Bay, Texas using the `coastal-calibration` command-line interface.
#
# The workflow has three phases:
#
# 1. **Create** — build a SFINCS model from an Area of Interest (AOI)
#    polygon using HydroMT-SFINCS.  This produces the grid, elevation,
#    subgrid tables, and boundary conditions.
# 2. **Run** — execute the full simulation pipeline: download forcing
#    data, write SFINCS input files, run the model, produce a
#    downscaled flood depth map, and compare results against NOAA
#    tide-gauge observations.
# 3. **Visualize** — plot the flood depth map and station comparisons.

# %% [markdown]
# ## Setup
#
# Change to the `texas-lavaca/` directory so that relative paths in
# the YAML configs resolve correctly.

# %%
import os
from pathlib import Path

notebook_dir = Path.cwd()  # assumes notebook is run from docs/examples/notebooks/
os.chdir(notebook_dir.parent / "texas-lavaca")
print("Working directory:", Path.cwd())

# %% [markdown]
# ## 1. Create the SFINCS model
#
# ### Explore the create configuration
#
# The `create.yaml` file specifies the AOI, grid resolution, elevation
# datasets, mask thresholds, and subgrid parameters.

# %%
print(Path("create.yaml").read_text())

# %% [markdown]
# ### Run the create workflow
#
# This downloads elevation data (NOAA 3 m topobathy + GEBCO bathymetry)
# and ESA WorldCover land-use, then builds the quadtree grid, elevation,
# mask, boundary conditions, and subgrid tables.

# %%
!coastal-calibration create create.yaml

# %% [markdown]
# ### Inspect the created model
#
# The output directory now contains all SFINCS model files.

# %%
output = Path("output")
assert output.exists(), f"Output directory not found: {output.resolve()} — run the create command first."

for f in sorted(output.iterdir()):
    if f.name.startswith(".") or f.suffix == ".log":
        continue
    size = f.stat().st_size
    label = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.1f} KB"
    print(f"  {f.name:<30s} {label}")

# %% [markdown]
# ## 2. Run the simulation pipeline
#
# ### Explore the run configuration
#
# The `run.yaml` file configures the simulation period, boundary source,
# download paths, and SFINCS runtime parameters.

# %%
print(Path("run.yaml").read_text())

# %% [markdown]
# ### Note on the SFINCS executable
#
# The `sfincs_exe` field in `run.yaml` points to a compiled SFINCS binary.
# You have two options:
#
# 1. **Compile SFINCS** yourself and update the path in the config if it
#    differs from `~/.local/bin/sfincs`.
# 2. **Use Singularity** — comment out the `sfincs_exe` line entirely.
#    The pipeline will then use the `ngen-coastal.sif` Singularity image.
#
# If neither is available, the pipeline will complete all stages up to
# `sfincs_run` (downloading data, writing inputs) and then fail at the
# model execution step.  You can re-start from that point later with
# `--start-from sfincs_run`.

# %% [markdown]
# ### Run the pipeline

# %%
!coastal-calibration run run.yaml

# %% [markdown]
# ## 3. View results
#
# The pipeline generates station comparison plots (modelled vs. observed
# water levels at NOAA CO-OPS tide gauges).

# %%
from IPython.display import Image, display

figs_dir = Path("run/sfincs_model/figs")
assert figs_dir.exists(), f"Results not found: {figs_dir.resolve()} — run the pipeline first."

for png in sorted(figs_dir.glob("stations_comparison_*.png")):
    display(Image(filename=str(png), width=800))

# %% [markdown]
# ## 4. Quadtree mesh
#
# The SFINCS model uses a quadtree grid with local refinement.  Coarser
# cells (512 m) cover the offshore domain while regions near the coastline
# and inside the bay are refined to smaller cell sizes (down to 64 m).
# The map output file (`sfincs_map.nc`) stores the mesh as standard UGRID
# quadrilaterals — opening it in QGIS with *native mesh rendering*
# immediately shows the refinement structure.

# %%
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
import xugrid as xu
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from pyproj import CRS, Transformer

map_file = Path("run/sfincs_model/sfincs_map.nc")
assert map_file.exists(), (
    f"Map output not found: {map_file.resolve()} — run the pipeline first."
)

ds = xu.open_dataset(map_file)
grid = ds.ugrid.grid
grid_crs = CRS.from_epsg(int(ds["inp"].attrs["epsg"]))

# Derive refinement level from cell size (the grid is in UTM metres).
fnc = grid.face_node_connectivity  # (n_face, 4) — all quads
node_x, node_y = grid.node_x, grid.node_y
cell_width = node_x[fnc].max(axis=1) - node_x[fnc].min(axis=1)
base_res = 512
level = np.round(np.log2(base_res / cell_width) + 1).astype(int)

# Print grid summary
levels, counts = np.unique(level, return_counts=True)
print(f"  Grid CRS:  EPSG:{grid_crs.to_epsg()}")
print(f"  Faces:     {grid.n_face:,}")
print(f"  Edges:     {grid.n_edge:,}")
for lv, cnt in zip(levels, counts, strict=True):
    print(f"  Level {lv}:   {cnt:>6,} cells ({base_res / 2**(lv-1):.0f} m)")

# %%
# Transform node coordinates to geographic CRS for the cartopy overlay.
transformer = Transformer.from_crs(grid_crs, "EPSG:4326", always_xy=True)
node_lon, node_lat = transformer.transform(node_x, node_y)

# Build vertex arrays for PolyCollection (all faces are quads).
n_verts = fnc.shape[1]
verts = np.zeros((grid.n_face, n_verts, 2))
for j in range(n_verts):
    verts[:, j, 0] = node_lon[fnc[:, j]]
    verts[:, j, 1] = node_lat[fnc[:, j]]

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent(
    [node_lon.min(), node_lon.max(), node_lat.min(), node_lat.max()],
    crs=ccrs.PlateCarree(),
)

# Satellite background
tiles = cimgt.QuadtreeTiles()
ax.add_image(tiles, 11)

# Overlay mesh cells colored by refinement level
colors = ["#4575b4", "#91bfdb", "#fee090", "#d73027"]
cmap = ListedColormap(colors)
norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], ncolors=4)

pc = PolyCollection(verts, edgecolors="black", linewidths=0.1, alpha=0.4)
pc.set_array(level.astype(float))
pc.set_cmap(cmap)
pc.set_norm(norm)
ax.add_collection(pc)

# Legend
legend_handles = [
    Patch(
        facecolor=cmap(norm(lv)), edgecolor="black", linewidth=0.5,
        alpha=0.5, label=f"Level {lv} ({base_res / 2**(lv-1):.0f} m)",
    )
    for lv in levels
]
ax.legend(handles=legend_handles, loc="lower right", title="Refinement level")
ax.set_title("Lavaca Bay SFINCS quadtree mesh")
plt.show()

# %% [markdown]
# ## 5. Flood depth map
#
# When `floodmap_dem` is configured in `run.yaml`, the pipeline
# automatically produces a downscaled flood depth map.  The
# `sfincs_floodmap` stage reads the maximum water surface elevation
# (`zsmax`) from the SFINCS map output, builds an index COG mapping DEM
# pixels to SFINCS grid cells, and writes a Cloud Optimized GeoTIFF of
# flood depth at the DEM resolution.

# %%
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
import rasterio

floodmap = Path("run/sfincs_model/floodmap_hmax.tif")
assert floodmap.exists(), (
    f"Flood map not found: {floodmap.resolve()} — "
    "ensure floodmap_dem is set in run.yaml and sfincs_map.nc contains zsmax."
)

# Print metadata at full resolution, then read at a coarser overview
# for display — the full raster can be too large for cartopy to render.
with rasterio.open(floodmap) as src:
    bounds = src.bounds
    raster_crs = src.crs
    print(f"  CRS:          {raster_crs}")
    print(f"  Size:         {src.width} x {src.height}")
    res_unit = raster_crs.linear_units if raster_crs.is_projected else "deg"
    print(f"  Resolution:   {abs(src.res[0]):.6g} x {abs(src.res[1]):.6g} {res_unit}")
    print(f"  File size:    {floodmap.stat().st_size / 1e6:.1f} MB")
    overviews = src.overviews(1)
    # Pick an overview that gives roughly 2000 px on the longest axis.
    ovr_idx = next(
        (i for i, f in enumerate(overviews) if max(src.height, src.width) / f <= 2000),
        len(overviews) - 1,
    )

with rasterio.open(floodmap, overview_level=ovr_idx) as src:
    hmax = src.read(1)
    print(f"  Display size: {src.width} x {src.height} (overview {overviews[ovr_idx]}x)")

# Mask dry / NaN pixels.  The floodmap already contains NaN outside the
# SFINCS domain and where depth < hmin, so no extra DEM masking is needed.
hmax_masked = np.where(np.isfinite(hmax) & (hmax > 0), hmax, np.nan)
valid = np.isfinite(hmax_masked)
print(f"  Valid pixels: {valid.sum():,} / {hmax.size:,} ({valid.sum() / hmax.size:.1%})")
if valid.any():
    print(f"  Depth range:  {np.nanmin(hmax_masked):.2f} - {np.nanmax(hmax_masked):.2f} m")

# %%
# Build a cartopy projection that matches the raster CRS.
if raster_crs.is_projected:
    proj = ccrs.epsg(raster_crs.to_epsg())
    data_crs = proj
else:
    proj = ccrs.PlateCarree()
    data_crs = ccrs.PlateCarree()

extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(extent, crs=data_crs)

# Satellite background tiles
tiles = cimgt.QuadtreeTiles()
ax.add_image(tiles, 12)

# Overlay flood depth — use a masked array so cartopy renders
# invalid pixels as fully transparent over the satellite tiles.
cmap = plt.cm.viridis.copy()
cmap.set_bad(alpha=0)
hmax_plot = np.ma.masked_invalid(hmax_masked)

im = ax.imshow(
    hmax_plot,
    extent=extent,
    origin="upper",
    transform=data_crs,
    cmap=cmap,
    vmin=0,
    vmax=np.nanpercentile(hmax_masked, 98),
    interpolation="nearest",
    zorder=2,
)
fig.colorbar(im, ax=ax, label="Flood depth (m)", shrink=0.6, pad=0.02, extend="both")
ax.set_title("Lavaca Bay flood depth (hmax) from SFINCS simulation")
plt.show()

# %% [markdown]
# The flood depth COG can also be opened directly in QGIS or any GIS
# viewer.  For standalone usage outside the pipeline see the API
# notebook or use:
#
# ```python
# from coastal_calibration.utils.floodmap import create_flood_depth_map
#
# create_flood_depth_map(
#     model_root="run/sfincs_model",
#     dem_path="../downloads/lavaca_grid/noaa_3m.tif",
# )
# ```

# %% [markdown]
# ## Summary
#
# This notebook demonstrated the full Lavaca Bay SFINCS workflow via the CLI:
#
# 1. `coastal-calibration create create.yaml` — built the model from an AOI
# 2. `coastal-calibration run run.yaml` — downloaded data, ran SFINCS, and
#    compared results against NOAA observations
# 3. Inspected the quadtree mesh and its refinement levels
# 4. Visualized the downscaled flood depth map (`floodmap_hmax.tif`)
