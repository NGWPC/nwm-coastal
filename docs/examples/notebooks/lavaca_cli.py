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
# The workflow has two phases:
#
# 1. **Create** — build a SFINCS model from an Area of Interest (AOI)
#    polygon using HydroMT-SFINCS.  This produces the grid, elevation,
#    subgrid tables, and boundary conditions.
# 2. **Run** — execute the full simulation pipeline: download forcing
#    data, write SFINCS input files, run the model, and compare results
#    against NOAA tide-gauge observations.

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
# ## Summary
#
# This notebook demonstrated the full Lavaca Bay SFINCS workflow via the CLI:
#
# 1. `coastal-calibration create create.yaml` — built the model from an AOI
# 2. `coastal-calibration run run.yaml` — downloaded data, ran SFINCS, and
#    compared results against NOAA observations
