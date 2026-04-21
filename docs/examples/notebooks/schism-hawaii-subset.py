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
# # Hawaii: SCHISM Demo
#
# This notebook demonstrates the SCHISM ocean model workflow using
# the same `coastal-calibration` API shown in the SFINCS demo.
#
# SCHISM differs from SFINCS in a few key ways:
#
# - **Prebuilt mesh**: SCHISM uses an unstructured triangular mesh
#   that is prepared ahead of time (hgrid.gr3, vgrid.in, etc.)
# - **MPI execution**: runs across multiple nodes using `mpiexec`
#   and `pschism`
# - **Atmospheric regridding**: NWM forcing is regridded onto the
#   SCHISM mesh using ESMF, which requires a geogrid file
#
# Despite these differences, the Python API is identical:
# `CoastalCalibConfig` for configuration and `CoastalCalibRunner`
# for execution. Only the `model_config` section changes.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import os
from pathlib import Path

notebook_dir = Path.cwd()  # assumes notebook is run from docs/examples/notebooks/
os.chdir(notebook_dir.parent / "hawaii")

# %%
import shapely

import coastal_calibration.schism.subsetter as ss

project = "model"
nwm = ss.NWMSCHISMProject(project)

line = shapely.LineString([[-155.778814, 20.624375], [-156.563293, 20.071111]])
res = ss.divide_mesh(project, line, ".", side_a_name="model_a", side_b_name="model_b")

# %%
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner, configure_logger

configure_logger(level="INFO")

config_a = CoastalCalibConfig.from_dict(
    {
        "model": "schism",
        "simulation": {
            "start_date": "2025-11-26",
            "duration_hours": 50,
            "coastal_domain": "hawaii",
            "meteo_source": "nwm_ana",
            "timestep_seconds": 300,  # 5-minute timestep
        },
        "boundary": {"source": "stofs"},
        "paths": {
            "work_dir": "./run_a",
            "raw_download_dir": "../downloads",
        },
        "download": {"enabled": True},
        "model_config": {
            "prebuilt_dir": "./model_a",  # pre-built mesh and config files
            "geogrid_file": "./geo_em_HI.nc",  # for ESMF atmospheric regridding
            "discharge_file": "./model_a/nwmReaches.csv",  # NWM reach → element mapping
            "nodes": 1,  # number of compute nodes
            "ntasks_per_node": 8,  # MPI tasks per node
            "nscribes": 2,  # I/O server tasks
            "oversubscribe": True,
            "include_noaa_gages": True,
        },
    }
)

print(f"Work directory: {config_a.paths.work_dir}")
print(f"Domain:         {config_a.simulation.coastal_domain}")
print(f"Duration:       {config_a.simulation.duration_hours}h")

# %% [markdown]
# ## 2. Run the pipeline
#
# The pipeline executes 11 stages grouped into four phases:
#
# 1. **Download**: fetch NWM meteorological data and STOFS boundary data
# 2. **Forcing**: regrid atmospheric forcing onto the SCHISM mesh (ESMF),
#    generate sflux files, process river discharge, and set up boundary
#    conditions
# 3. **Model Prep**: update parameters, discover NOAA stations, partition
#    the mesh for MPI
# 4. **Run & Validate**: execute `pschism` via `mpiexec`, post-process
#    outputs, and generate comparison plots against NOAA observations

# %%
runner_a = CoastalCalibRunner(config_a)
result_a = runner_a.run()
if not result_a.success:
    raise RuntimeError(f"Pipeline failed at stage '{result_a.stages_failed}': {result_a.errors}")
print(result_a)

# %% [markdown]
# ## 3. View results
#
# The pipeline compares modeled water levels against NOAA CO-OPS
# tide gauge observations at stations within the domain.

# %%
from IPython.display import Image

figs = sorted(Path("run_a/figs").glob("stations_comparison_*.png"))
if not figs:
    print("No station comparison figures were generated; skipping thumbnail creation and display.")
else:
    for png in figs:
        display(Image(filename=str(png), width=800))

# %%
config_b = CoastalCalibConfig.from_dict(
    {
        "model": "schism",
        "simulation": {
            "start_date": "2025-11-26",
            "duration_hours": 50,
            "coastal_domain": "hawaii",
            "meteo_source": "nwm_ana",
            "timestep_seconds": 300,  # 5-minute timestep
        },
        "boundary": {"source": "stofs"},
        "paths": {
            "work_dir": "./run_b",
            "raw_download_dir": "../downloads",
        },
        "download": {"enabled": True},
        "model_config": {
            "prebuilt_dir": "./model_b",  # pre-built mesh and config files
            "geogrid_file": "./geo_em_HI.nc",  # for ESMF atmospheric regridding
            "discharge_file": "./model_b/nwmReaches.csv",  # NWM reach → element mapping
            "nodes": 1,  # number of compute nodes
            "ntasks_per_node": 8,  # MPI tasks per node
            "nscribes": 2,  # I/O server tasks
            "oversubscribe": True,
            "include_noaa_gages": True,
        },
    }
)

print(f"Work directory: {config_b.paths.work_dir}")
print(f"Domain:         {config_b.simulation.coastal_domain}")
print(f"Duration:       {config_b.simulation.duration_hours}h")

runner_b = CoastalCalibRunner(config_b)
result_b = runner_b.run()
if not result_b.success:
    raise RuntimeError(f"Pipeline failed at stage '{result_b.stages_failed}': {result_b.errors}")
print(result_b)

figs = sorted(Path("run_b/figs").glob("stations_comparison_*.png"))
if not figs:
    print("No station comparison figures were generated; skipping thumbnail creation and display.")
else:
    for png in figs:
        display(Image(filename=str(png), width=800))

# %% [markdown]
# ## Summary
#
# This notebook ran the full SCHISM pipeline for Hawaii using the
# same API as SFINCS:
#
# 1. Configured `CoastalCalibConfig` with SCHISM-specific settings
#    (MPI layout, geogrid for ESMF regridding)
# 2. Executed the 11-stage pipeline with `CoastalCalibRunner`
#    (download, forcing, model prep, run, validate)
# 3. Compared modeled water levels against NOAA observations
# 4. Showed how the same config can be submitted to an HPC cluster
#    via `sbatch` or any other job scheduler
#
# The interface is identical to SFINCS. Only the `model_config`
# section differs (MPI layout vs. OpenMP, geogrid for atmospheric
# regridding, prebuilt mesh vs. automated creation).
