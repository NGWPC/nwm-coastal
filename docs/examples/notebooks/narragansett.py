# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Narragansett Bay: SFINCS Demo
#
# This notebook demonstrates the full SFINCS workflow for
# Narragansett Bay, Rhode Island:
#
# 1. **QGIS Plugin**: define the model domain and select discharge points
# 2. **Create** the model from the AOI polygon
# 3. **Run** the simulation with compound forcing
#    (ocean boundary + river discharge + precipitation + wind + pressure)
# 4. **Visualize** water level comparisons and the flood depth map

# %% [markdown]
# ## QGIS Plugin: Defining the Model Domain
#
# Before building a SFINCS model, we need two inputs:
#
# 1. An **AOI polygon** that defines the model boundary
# 2. A **discharge points file** listing NWM flowlines that enter the domain
#
# The NWM Coastal QGIS plugin provides an interactive workflow for
# creating both. Here is a step-by-step walkthrough.
#
# ### Step 1: Install the Plugin
#
# Install the NWM Coastal plugin from the QGIS Plugin Manager.
#
# ![QGIS Plugin Manager](../images/qgis_plugin_window.png)
#
# ### Step 2: Load the Basemap
#
# Click "Add Basemap" in the toolbar to load National HydroFabric data
# (watershed divides, NWM flowlines, gages) and NOAA CO-OPS station
# locations. Set the minimum stream order to filter small tributaries.
#
# ![Add Basemap Dialog](../images/qgis_plugin_basemap.png)
#
# ### Step 3: Explore the Data
#
# The plugin adds several layers: OSM basemap, watershed divides,
# NWM flowpaths, USGS gages, NHF nexus points, and CO-OPS tide
# stations. Use the layers panel to toggle visibility and inspect
# the coastal area.
#
# ![QGIS Toolbar and Layers](../images/qgis_plugin_menu.png)
#
# ### Step 4: Draw the AOI Polygon
#
# Click "Draw Polygon" in the toolbar, then sketch the model domain
# by clicking vertices on the map. Right-click to finish. The polygon
# should cover the coastal area and extend offshore to capture the
# tidal boundary.
#
# ![Sketched AOI Polygon](../images/qgis_plugin_poly.png)
#
# ### Step 5: Merge with Watershed Boundaries
#
# Click "Union with NHF Divides" to snap the polygon boundary to
# watershed divide lines. This ensures the model domain aligns
# with hydrologic boundaries.
#
# ![Merged Polygon](../images/qgis_plugin_merge.png)
#
# ### Step 6: Identify River Discharge Points
#
# With the merged polygon in place, zoom in to find NWM flowlines
# that connect to the domain boundary. These flowlines will become
# river discharge sources in the model. The red arrows below
# highlight flowlines entering the merged polygon.
#
# ![NWM Flowlines](../images/qgis_plugin_flowlines.png)
#
# ### Step 7: Save and Export
#
# Click "Save Polygon" to export the merged polygon as `aoi.geojson`
# and the selected discharge points as `discharge_nwm.geojson`.
# These two files are the inputs for the SFINCS model creation step below.
#
# ### Tip: Drawing Refinement Regions
#
# The same "Draw Polygon" tool can be used to define refinement
# regions. A typical workflow is to first run the model without
# refinement, inspect the results, identify areas that need higher
# resolution (e.g., narrow channels or areas with steep gradients),
# then draw a polygon around those areas and export it as a GeoJSON.
# The exported polygon can then be passed to the `grid.refinement`
# config to increase mesh resolution locally.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import os
from pathlib import Path

notebook_dir = Path.cwd()  # assumes notebook is run from docs/examples/notebooks/
os.chdir(notebook_dir.parent / "narragansett-ri")

# %% [markdown]
# ## 1. Create the SFINCS model
#
# `SfincsCreateConfig.from_dict` builds a configuration from a plain
# dictionary (same structure as the YAML file). The key settings are:
#
# - **grid**: base resolution of 512 m with 3 levels of refinement near the coast
# - **elevation**: merged from NWS 30 m coastal DEM and GEBCO bathymetry
# - **subgrid**: 4x subgrid pixels with land-use-based Manning coefficients
# - **river_discharge**: NWM flowlines exported by the QGIS plugin
# - **add_noaa_gages**: automatically discover NOAA tide gauges in the domain

# %%
from coastal_calibration import SfincsCreateConfig, SfincsCreator, configure_logger

configure_logger(level="INFO")

create_config = SfincsCreateConfig.from_dict(
    {
        "aoi": "./aoi.geojson",
        "output_dir": "./output",
        "download_dir": "../downloads/narragansett_grid",
        "grid": {
            "resolution": 512,  # base cell size in meters
            "crs": "utm",
            "rotated": False,
            "refinement": [
                {"polygon": "./aoi.geojson", "level": 3, "buffer_m": -200},
            ],
        },
        "elevation": {
            "datasets": [
                {
                    "name": "nws_30m",
                    "zmin": -20000,
                    "source": "nws_30m",
                    "coastal_domain": "atlgulf",
                },
                {"name": "gebco_15arcs", "zmin": -20000, "source": "gebco_15arcs"},
            ],
            "buffer_cells": 1,
        },
        "mask": {"zmin": -50.0, "boundary_zmax": -1.0, "reset_bounds": True},
        "subgrid": {
            "nr_subgrid_pixels": 4,
            "lulc_dataset": "esa_worldcover",
            "manning_land": 0.04,
            "manning_sea": 0.02,
        },
        "river_discharge": {
            "flowlines": "./discharge_nwm.geojson",  # from QGIS plugin
            "nwm_id_column": "flowpath_id",
        },
        "add_noaa_gages": True,
    }
)

# %% [markdown]
# ### Run the create workflow

# %%
creator = SfincsCreator(create_config)
result = creator.run()
if not result.success:
    raise RuntimeError(f"Model creation failed at stage '{result.stages_failed}': {result.errors}")
print(result)

# %% [markdown]
# ### Inspect the created model

# %%
output = Path("output")
assert output.exists(), (
    f"Output directory not found: {output.resolve()} — run the create step first."
)

for f in sorted(output.iterdir()):
    if f.name.startswith(".") or f.suffix == ".log":
        continue
    size = f.stat().st_size
    label = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.1f} KB"
    print(f"  {f.name:<30s} {label}")

# %% [markdown]
# ## 2. Run the simulation pipeline
#
# The run configuration specifies:
#
# - **Simulation period**: 60 hours starting 2024-01-09
# - **Forcing sources**: NWM analysis for meteorology, STOFS for ocean boundary
# - **Compound forcing**: ocean boundary + river discharge + precipitation +
#   wind + barometric pressure
# - **Flood depth map**: downscaled to the 30 m NWS DEM

# %%
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

run_config = CoastalCalibConfig.from_dict(
    {
        "model": "sfincs",
        "simulation": {
            "start_date": "2024-01-09",
            "duration_hours": 60,
            "coastal_domain": "atlgulf",
            "meteo_source": "nwm_ana",
        },
        "boundary": {"source": "stofs"},
        "paths": {
            "work_dir": "./run",
            "raw_download_dir": "../downloads",
        },
        "download": {"enabled": True},
        "model_config": {
            "prebuilt_dir": "./output",
            "discharge_locations_file": "./output/sfincs_nwm.src",
            "merge_discharge": True,
            "forcing_to_mesh_offset_m": 0.0,  # STOFS already in NAVD88
            "vdatum_mesh_to_msl_m": 0.1,  # NAVD88 mesh -> MSL
            "include_precip": True,
            "include_wind": True,
            "include_pressure": True,
            "floodmap_dem": "../downloads/narragansett_grid/nws_30m.tif",
            "inp_overrides": {
                "tspinup": 10800,  # 3-hour spinup
            },
        },
    }
)

# %% [markdown]
# ### Run the pipeline
#
# The pipeline executes 14 stages: download forcing data, build the
# SFINCS input files (timing, forcing, discharge, precipitation, wind,
# pressure), run the model, generate the flood depth map, and produce
# comparison plots against NOAA observations.

# %%
runner = CoastalCalibRunner(run_config)
result = runner.run()
if not result.success:
    raise RuntimeError(f"Model run failed at stage '{result.stages_failed}': {result.errors}")
print(result)

# %% [markdown]
# ## 3. View results
#
# The pipeline compares modeled water levels against NOAA CO-OPS
# tide gauge observations at stations within the domain.

# %%
from IPython.display import Image, display

figs_dir = Path("run/sfincs_model/figs")
assert figs_dir.exists(), f"Results not found: {figs_dir.resolve()} — run the pipeline first."

for png in sorted(figs_dir.glob("stations_comparison_*.png")):
    display(Image(filename=str(png), width=800))

# %% [markdown]
# ## 4. SFINCS mesh
#
# The SFINCS model uses a quadtree grid. Coarser cells (512 m) cover
# the offshore domain while regions near the coastline and inside the
# bay are refined to higher resolution.

# %%
from coastal_calibration.plotting import SfincsGridInfo, plot_mesh

info = SfincsGridInfo.from_model_root("run/sfincs_model")
print(info)

# %%
fig, ax = plot_mesh(info, title="Narragansett Bay SFINCS mesh")

# %% [markdown]
# ## 5. Flood depth map
#
# The pipeline produces a downscaled flood depth map when
# `floodmap_dem` is configured. The process has three steps:
#
# 1. **Read `zsmax`**: extract the maximum water surface elevation
#    over the simulation period from `sfincs_map.nc`
# 2. **Build an index COG**: for each pixel in the high-resolution DEM,
#    find the SFINCS grid cell it falls in and store the mapping as a
#    GeoTIFF (`floodmap_index.tif`). This index is reusable across runs
#    with the same grid.
# 3. **Compute depth**: for each DEM pixel, look up the `zsmax` value
#    via the index and subtract the DEM elevation. Pixels with depth
#    below 5 cm are masked out. The result is written as a
#    Cloud Optimized GeoTIFF at the DEM resolution (30 m in this case).

# %%
from coastal_calibration.plotting import plot_floodmap

fig, ax = plot_floodmap(
    "run/sfincs_model/floodmap_hmax.tif",
    title="Max water depth, Narragansett Bay, RI",
)
fig.savefig("../images/narragansett_thumb.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated the full SFINCS workflow:
#
# 1. **QGIS Plugin**: drew the AOI polygon and identified NWM discharge points
# 2. **Model Creation**: built a quadtree mesh with elevation, subgrid,
#    and river discharge sources using `SfincsCreator`
# 3. **Simulation**: ran the pipeline with compound forcing (ocean + river +
#    meteo) and validated against NOAA observations using `CoastalCalibRunner`
# 4. **Visualization**: inspected the quadtree mesh and flood depth map
