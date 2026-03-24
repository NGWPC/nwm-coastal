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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lavaca Bay SFINCS Tutorial
#
# This notebook demonstrates how to build and run a
# [SFINCS](https://sfincs.readthedocs.io) coastal flood model for
# Lavaca Bay, Texas using the `coastal_calibration` Python API.
#
# The workflow has three phases:
#
# 1. **Create** — build a SFINCS model from an Area of Interest (AOI)
#    polygon using HydroMT-SFINCS.
# 2. **Run** — execute the full simulation pipeline: download forcing
#    data, write SFINCS input files, run the model, produce a
#    downscaled flood depth map, and compare results against NOAA
#    tide-gauge observations.
# 3. **Visualize** — plot the flood depth map and station comparisons.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import os
from pathlib import Path

notebook_dir = Path.cwd()  # assumes notebook is run from docs/examples/notebooks/
os.chdir(notebook_dir.parent / "lavaca-tx")

# %% [markdown]
# ## 1. Create the SFINCS model
#
# ### Build the create configuration
#
# `SfincsCreateConfig.from_dict` accepts a plain dictionary with the same
# structure as the YAML file.

# %%
from coastal_calibration import SfincsCreateConfig, SfincsCreator, configure_logger

configure_logger(level="INFO")

create_config = SfincsCreateConfig.from_dict(
    {
        "aoi": "./aoi.geojson",
        "output_dir": "./output",
        "download_dir": "../downloads/lavaca_grid",
        "grid": {
            "resolution": 512,
            "crs": "utm",
            "rotated": False,
            "refinement": [
                {"polygon": "./refine.geojson", "level": 3},
            ],
        },
        "elevation": {
            "datasets": [
                {"name": "noaa_3m", "zmin": -20000, "source": "noaa_3m"},
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
            "flowlines": "./discharge_nwm.geojson",
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
# ### Build the run configuration
#
# `CoastalCalibConfig.from_dict` accepts the same dictionary structure as
# the run YAML file.

# %%
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

run_config = CoastalCalibConfig.from_dict(
    {
        "model": "sfincs",
        "simulation": {
            "start_date": "2025-06-01",
            "duration_hours": 100,
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
            "vdatum_mesh_to_msl_m": 0.17,  # NAVD88 mesh -> MSL
            "include_precip": True,
            "include_wind": True,
            "include_pressure": True,
            "inp_overrides": {
                "tspinup": 10800,
                "advection": 0,
                "viscosity": 0,
                "nuvisc": 0.01,
                "cdnrb": 3,
                "cdwnd": [0.0, 28.0, 50.0],
                "cdval": [0.001, 0.0025, 0.0025],
            },
            # Flood depth map — path to a high-resolution DEM.
            # Here we reuse the NOAA 3m DEM fetched during model creation.
            "floodmap_dem": "../downloads/lavaca_grid/noaa_3m.tif",
        },
    }
)

# %% [markdown]
# ### Note on the SFINCS executable
#
# The `sfincs_exe` field overrides the default PATH lookup for the SFINCS binary.
# When running inside a pixi environment with the `sfincs` feature, the binary
# is compiled automatically and available on PATH — no `sfincs_exe` needed.
#
# If you compiled SFINCS manually, set `sfincs_exe` to the path of the binary.
# If neither is available, the pipeline will complete all stages up to
# `sfincs_run` and then fail at model execution.

# %% [markdown]
# ### Run the pipeline

# %%
runner = CoastalCalibRunner(run_config)
result = runner.run()
if not result.success:
    raise RuntimeError(f"Model run failed at stage '{result.stages_failed}': {result.errors}")
print(result)

# %% [markdown]
# ## 3. View results
#
# The pipeline generates station comparison plots (modeled vs. observed
# water levels at NOAA CO-OPS tide gauges).

# %%
from IPython.display import Image, display

figs_dir = Path("run/sfincs_model/figs")
assert figs_dir.exists(), f"Results not found: {figs_dir.resolve()} — run the pipeline first."

for png in sorted(figs_dir.glob("stations_comparison_*.png")):
    display(Image(filename=str(png), width=800))

# %% [markdown]
# ## 4. SFINCS mesh
#
# The SFINCS model uses a quadtree grid with local refinement.  Coarser
# cells (512 m) cover the offshore domain while regions near the coastline
# and inside the bay are refined to smaller cell sizes (down to 64 m).

# %%
from coastal_calibration.plotting import SfincsGridInfo, plot_floodmap, plot_mesh

info = SfincsGridInfo.from_model_root("run/sfincs_model")
print(info)

# %%
fig, ax = plot_mesh(info, title="Lavaca Bay SFINCS mesh")

# %% [markdown]
# ## 5. Flood depth map
#
# The pipeline automatically produces a downscaled flood depth map when
# `floodmap_dem` is configured.  The `sfincs_floodmap` stage reads the
# maximum water surface elevation (`zsmax`) from the SFINCS map output,
# builds an index COG mapping DEM pixels to SFINCS grid cells, and
# writes a Cloud Optimized GeoTIFF of flood depth at the DEM resolution.

# %%
fig, ax = plot_floodmap(
    "run/sfincs_model/floodmap_hmax.tif",
    title="Lavaca Bay flood depth (hmax) from SFINCS simulation",
)
fig.savefig("../images/lavaca_thumb.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# The flood depth COG can be opened in QGIS or any GIS viewer.
# You can also generate a flood depth map outside the pipeline
# using the standalone function:
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
# This notebook demonstrated the full Lavaca Bay SFINCS workflow via the
# Python API:
#
# 1. `SfincsCreateConfig.from_dict({...})` + `SfincsCreator(config).run()`
#    — built the model from an AOI
# 2. `CoastalCalibConfig.from_dict({...})` + `CoastalCalibRunner(config).run()`
#    — downloaded data, ran SFINCS, and compared results against NOAA observations
# 3. Inspected the quadtree mesh and its refinement levels
# 4. Visualized the downscaled flood depth map (`floodmap_hmax.tif`)
