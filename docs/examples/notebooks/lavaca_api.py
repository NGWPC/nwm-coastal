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
# # Lavaca Bay SFINCS Tutorial — Python API
#
# This notebook demonstrates how to build and run a
# [SFINCS](https://sfincs.readthedocs.io) coastal flood model for
# Lavaca Bay, Texas using the `coastal_calibration` Python API.
#
# The workflow has two phases:
#
# 1. **Create** — build a SFINCS model from an Area of Interest (AOI)
#    polygon using HydroMT-SFINCS.
# 2. **Run** — execute the full simulation pipeline: download forcing
#    data, write SFINCS input files, run the model, and compare results
#    against NOAA tide-gauge observations.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import os
from pathlib import Path

notebook_dir = Path.cwd()  # assumes notebook is run from docs/examples/notebooks/
os.chdir(notebook_dir.parent / "texas-lavaca")

# %% [markdown]
# ## 1. Create the SFINCS model
#
# ### Build the create configuration
#
# `SfincsCreateConfig.from_dict` accepts a plain dictionary with the same
# structure as the YAML file.

# %%
from coastal_calibration import SfincsCreateConfig, SfincsCreator

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
        "add_noaa_gages": True,
    }
)

# %% [markdown]
# ### Run the create workflow

# %%
creator = SfincsCreator(create_config)
result = creator.run()
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
            "duration_hours": 10,
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
            "merge_discharge": True,
            "forcing_to_mesh_offset_m": 0.0,
            "vdatum_mesh_to_msl_m": 0.30,
            "include_precip": True,
            "include_wind": True,
            "include_pressure": True,
            # SFINCS executable path. Either:
            #   1. Compile SFINCS and set the path here, or
            #   2. Remove this key to use Singularity (requires ngen-coastal.sif)
            "sfincs_exe": "~/.local/bin/sfincs",
            "inp_overrides": {
                "tspinup": 10800,
                "advection": 0,
                "viscosity": 0,
                "nuvisc": 0.01,
                "cdnrb": 3,
                "cdwnd": [0.0, 28.0, 50.0],
                "cdval": [0.001, 0.0025, 0.0025],
            },
        },
    }
)

# %% [markdown]
# ### Note on the SFINCS executable
#
# The `sfincs_exe` field points to a compiled SFINCS binary.
# You have two options:
#
# 1. **Compile SFINCS** yourself and update the path if it differs
#    from `~/.local/bin/sfincs`.
# 2. **Use Singularity** — remove the `sfincs_exe` key from the dict.
#    The pipeline will then use the `ngen-coastal.sif` Singularity image.
#
# If neither is available, the pipeline will complete all stages up to
# `sfincs_run` and then fail at model execution.

# %% [markdown]
# ### Run the pipeline

# %%
runner = CoastalCalibRunner(run_config)
result = runner.run()
print(result)

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
# ## Summary
#
# This notebook demonstrated the full Lavaca Bay SFINCS workflow via the
# Python API:
#
# 1. `SfincsCreateConfig.from_dict({...})` + `SfincsCreator(config).run()`
#    — built the model from an AOI
# 2. `CoastalCalibConfig.from_dict({...})` + `CoastalCalibRunner(config).run()`
#    — downloaded data, ran SFINCS, and compared results against NOAA observations
