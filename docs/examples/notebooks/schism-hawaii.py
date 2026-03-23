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
# # Hawaii SCHISM Tutorial
#
# This notebook demonstrates how to run the SCHISM ocean model for the
# Hawaii coast using the `coastal_calibration` Python API.
#
# The workflow mirrors the SFINCS pipeline: given a **prebuilt model
# directory** and a **geogrid file** the runner downloads forcing data,
# regrids atmospheric forcing, generates boundary conditions, runs
# SCHISM, and produces comparison plots against NOAA CO-OPS tide gauge
# observations.
#
# ## Prerequisites
#
# - The **pixi `schism` environment** must be active so that `pschism`,
#   `metis_prep`, `gpmetis`, and other SCHISM binaries are on `$PATH`.
# - **Pre-built model files** (hgrid.gr3, vgrid.in, param.nml,
#   bctides.in, nwmReaches.csv, etc.).
# - A **geogrid file** (e.g. `geo_em_HI.nc`) for atmospheric forcing
#   regridding.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import os
from pathlib import Path

notebook_dir = Path.cwd()  # assumes notebook is run from docs/examples/notebooks/
os.chdir(notebook_dir.parent / "hawaii")

# %% [markdown]
# ## 1. Build the run configuration
#
# `CoastalCalibConfig.from_dict` accepts the same dictionary structure as
# the run YAML file.  The interface is identical to the SFINCS workflow:
# `prebuilt_dir` and `geogrid_file` under `model_config` point to the
# model inputs, `work_dir` is the run directory, and `raw_download_dir`
# is where NWM data is cached.

# %%
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner, configure_logger

configure_logger(level="INFO")

MODEL_DIR = "/Volumes/data/schism_models/hawaii"
GEOGRID = "/Volumes/data/schism_models/geo_em_HI.nc"

run_config = CoastalCalibConfig.from_dict(
    {
        "model": "schism",
        "simulation": {
            "start_date": "2025-11-26",
            "duration_hours": 50,
            "coastal_domain": "hawaii",
            "meteo_source": "nwm_ana",
            "timestep_seconds": 300,
        },
        "boundary": {"source": "stofs"},
        "paths": {
            "work_dir": "./run",
            "raw_download_dir": "../downloads",
        },
        "download": {"enabled": True},
        "model_config": {
            "prebuilt_dir": MODEL_DIR,
            "geogrid_file": GEOGRID,
            "nodes": 1,
            "ntasks_per_node": 4,
            "nscribes": 2,
            "oversubscribe": True,
            "include_noaa_gages": True,
        },
    }
)

print(f"Work directory: {run_config.paths.work_dir}")
print(f"Prebuilt dir:   {run_config.model_config.prebuilt_dir}")
print(f"Domain:         {run_config.simulation.coastal_domain}")
print(f"Duration:       {run_config.simulation.duration_hours}h")

# %% [markdown]
# ## 2. Run the pipeline
#
# `CoastalCalibRunner` executes all stages in order:
#
# 1. `download` — fetch NWM CHRTOUT, LDASIN, and STOFS data
# 2. `schism_forcing_prep` — stage LDASIN files
# 3. `schism_forcing` — ESMF regridding of atmospheric forcing (MPI)
# 4. `schism_sflux` — generate sflux files
# 5. `schism_params` — create param.nml, symlink mesh files
# 6. `schism_obs` — discover NOAA tide gauge stations
# 7. `schism_boundary` — boundary forcing (TPXO or STOFS → elev2D.th.nc)
# 8. `schism_prep` — discharge generation, mesh partitioning
# 9. `schism_run` — run `pschism` via `mpiexec`
# 10. `schism_postprocess` — check outputs, combine hotstarts
# 11. `schism_plot` — sim vs obs comparison plots

# %%
runner = CoastalCalibRunner(run_config)
result = runner.run()
if not result.success:
    raise RuntimeError(f"Pipeline failed at stage '{result.stages_failed}': {result.errors}")
print(result)

# %% [markdown]
# ## 3. View results
#
# The pipeline generates station comparison plots (modeled vs. observed
# water levels at NOAA CO-OPS tide gauges).

# %%
from IPython.display import Image, display

figs_dir = Path("run/figs")
for png in sorted(figs_dir.glob("stations_comparison_*.png")):
    display(Image(filename=str(png), width=800))

# %% [markdown]
# ## 4. Inspect outputs

# %%
outputs_dir = Path("run/outputs")
if outputs_dir.exists():
    all_outputs = sorted(outputs_dir.iterdir())
    for f in all_outputs[:20]:
        sz = f.stat().st_size
        label = f"{sz / 1e6:.1f} MB" if sz > 1e6 else f"{sz / 1e3:.1f} KB"
        print(f"  {f.name:<40s} {label}")
    if len(all_outputs) > 20:
        print(f"  ... and {len(all_outputs) - 20} more files")
else:
    print("No outputs directory found")

# %% [markdown]
# ## Summary
#
# This notebook ran the full SCHISM pipeline for Hawaii using the same
# `CoastalCalibConfig` + `CoastalCalibRunner` API as SFINCS:
#
# 1. Pointed `prebuilt_dir` at the pre-built model files and
#    `geogrid_file` at the WRF geogrid for regridding
# 2. Downloaded NWM and STOFS coastal data
# 3. Regridded atmospheric forcing onto the SCHISM mesh
# 4. Generated boundary conditions (elev2D.th.nc)
# 5. Prepared discharge inputs and partitioned the mesh
# 6. Ran `pschism` via `mpiexec`
# 7. Post-processed outputs and generated comparison plots
#
# The interface is identical to the SFINCS workflow — only the
# `model_config` fields differ (MPI layout vs. OpenMP threads).
