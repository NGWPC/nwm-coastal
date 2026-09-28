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
# # Lake Erie SCHISM - preparing a mesh for the pipeline
#
# A SCHISM mesh usually arrives with the grid and the physics settings but
# without the derived files the pipeline needs. This notebook generates those
# with the three `prepare-schism-*` commands, then runs a 12-hour simulation
# forced at the open boundary by NOAA's Lake Erie OFS (`leofs`).
#
# The SFINCS counterpart on the same lake and the same window is
# [Lake Erie SFINCS](lake_erie_sfincs.ipynb).
#
# ## What ships, and what is generated
#
# | Shipped with the mesh | Generated here |
# | --- | --- |
# | `hgrid.gr3`, `hgrid.ll`, `hgrid.cpp` | `hgrid.nc`, `open_bnds_hgrid.nc` |
# | `vgrid.in`, `param.nml`, `bctides.in` | `manning.gr3` |
# | `elev.ic`, `windrot_geo2proj.gr3`, `sflux/` | `nwmReaches.csv`, `ngenReaches.csv` |
#
# `param.nml` asks for all three: `nchi = -1` reads `manning.gr3`, and
# `if_source = -1` reads the reach crosswalk. Without them SCHISM starts and
# runs with no bottom friction file and no river inflow.
#
# ## Setup
#
# The mesh, geogrid and hydrofabric are not in the repo. Create the symlinks
# once, pointing at wherever yours live:
#
# ```bash
# cd docs/examples/lake-erie_schism
# ln -s /path/to/schism_models/lake_erie        model
# ln -s /path/to/geo_em_CONUS.nc                geo_em_CONUS.nc
# ln -s /path/to/NWM_v3_hydrofabric.gdb         hydrofabric.gdb
# ln -s /path/to/nhf_1.2.2.gpkg                 hydrofabric.gpkg
# ```
#
# The hydrofabric links point at the files themselves, so it does not matter how
# your copies are organised.

# %%
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

notebook_dir = Path.cwd()  # assumes run from docs/examples/notebooks/
example_dir = (notebook_dir.parent / "lake-erie_schism").resolve()
os.chdir(example_dir)

required = (
    "model",
    "geo_em_CONUS.nc",
    "hydrofabric.gdb",
    "hydrofabric.gpkg",
    "run.yaml",
)
missing = [n for n in required if not Path(n).exists()]
if missing:
    raise FileNotFoundError(f"Missing in {example_dir}: {missing}. See ../README.md.")


def cli(*args: str) -> None:
    """Run a coastal-calibration CLI command and stream its output."""
    print(f"$ coastal-calibration {' '.join(args)}\n")
    subprocess.run([sys.executable, "-m", "coastal_calibration.cli", *args], check=True)


print(f"Working directory: {example_dir}")
print(f"Mesh: {Path('model').resolve()}")
print("Mesh contents:", sorted(p.name for p in Path("model").iterdir()))

# %% [markdown]
# The mesh boundary can be pulled out in QGIS with the `nwm_coastal` plugin,
# which is how the domain outline below was made. It is only for looking at -
# the prepare commands read `hgrid.gr3` directly.
#
# ![Lake Erie SCHISM mesh boundary extracted in QGIS](../images/Lake_Erie_SCHISM_extract_mesh_boundary.jpg)

# %% [markdown]
# ## 1. ESMF mesh files
#
# Converts `hgrid.gr3` into the NetCDF form the atmospheric regridder and the
# boundary stages read.

# %%
cli("prepare-schism-mesh", "./model")

# %% [markdown]
# ## 2. Manning roughness
#
# Samples ESA WorldCover land cover at every node and maps each class to a
# Manning's n. Add `--force` to overwrite an existing file.

# %%
if Path("model/manning.gr3").exists():
    print("manning.gr3 already present; pass --force to regenerate")
else:
    cli("prepare-schism-manning", "./model")

# %% [markdown]
# ## 3. Reach crosswalks
#
# Intersects the mesh boundary with a hydrofabric and writes one row per
# crossing: a flowline entering the mesh becomes a source, one leaving it
# becomes a sink. `nwmReaches.csv` keys on NWM COMIDs and is what `nwm_ana`
# and `nwm_retro` runs use; `ngenReaches.csv` keys on NextGen `fp_id` for
# `ngen_forecast` runs.

# %%
cli(
    "prepare-schism-reaches",
    "./model",
    "--nwm-gdb",
    "./hydrofabric.gdb",
    "--ngen-gpkg",
    "./hydrofabric.gpkg",
    "--force",
)

# %% [markdown]
# ### What landed
#
# The source block should be dominated by the Detroit River, and the sink block
# by the Niagara - Erie's inflow and outflow. If the largest reach shows up as a
# source rather than a sink, the crossing direction was read wrong.

# %%
from coastal_calibration.schism.ngen_reaches import read_reaches_blocks

for name in ("nwmReaches.csv", "ngenReaches.csv"):
    sources, sinks = read_reaches_blocks(Path("model") / name)
    print(f"{name}: {len(sources)} sources, {len(sinks)} sinks")

for name in ("hgrid.nc", "open_bnds_hgrid.nc", "manning.gr3"):
    p = Path("model") / name
    print(f"{name}: {p.stat().st_size / 1e6:.1f} MB")

# %% [markdown]
# ## 4. The run configuration
#
# Everything above is preparation and only has to happen once per mesh. The
# cell after this one is the simulation, which is the slow part - stop here if
# you only wanted the model built.

# %%
print(Path("run.yaml").read_text())

# %% [markdown]
# Two settings are specific to a Great Lakes mesh:
#
# - `forcing_to_mesh_offset_m: 173.5` - GLOFS reports water levels on the
#   lake's low water datum while this mesh carries absolute IGLD85 elevations
#   (`elev.ic` starts at 174.73). The offset is added to the boundary forcing
#   and to the gauge observations, so both land in the mesh datum.
# - `coastal_domain: greatlakes` and `boundary.source: glofs` are required
#   together; either alone fails validation.
#
# `ntasks_per_node: 16` suits a workstation. Each rank holds roughly 1.5 GB on
# this 1.9M-element mesh, so memory runs out before cores do.

# %%
cli("run", "run.yaml", "--dry-run")

# %% [markdown]
# ## 5. Run it
#
# 12 hours of simulation. The solver is the bulk of the wall time; everything
# before it - forcing, sflux, boundary, discharge, partitioning - takes a
# couple of minutes.

# %%
cli("run", "run.yaml")

# %% [markdown]
# ## Results
#
# `run/figs/` holds the water-level comparison against the NOAA CO-OPS gauges
# inside the domain, and `run/outputs/` the SCHISM NetCDF output.

# %%
figs = sorted(Path("run/figs").glob("*.png")) if Path("run/figs").exists() else []
print(f"figures: {[p.name for p in figs]}")

outputs = sorted(Path("run/outputs").glob("out2d_*.nc")) if Path("run/outputs").exists() else []
print(f"outputs: {[p.name for p in outputs]}")

# %% [markdown]
# ## Summary
#
# Three commands turned a shipped mesh into a runnable model:
# `prepare-schism-mesh`, `prepare-schism-manning`, `prepare-schism-reaches`.
# They are per-mesh, not per-run, so a second simulation on this domain only
# needs a new `run.yaml`.
#
# To model a different lake, point `model` at that mesh, set `glofs_model` to
# its OFS (`lmhofs`, `loofs`, `lsofs`) and check the mesh datum before trusting
# `forcing_to_mesh_offset_m`.
