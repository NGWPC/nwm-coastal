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
# # Alaska SCHISM - full domain and a Cook Inlet subset
#
# Prepares the full Alaska SCHISM mesh, cuts a Cook Inlet subdomain out of it,
# and runs both against STOFS boundary forcing.
#
# Everything up to section 5 is preparation and takes a few minutes. The two
# simulations are last, so you can build both models and stop there.
#
# Subsetting **after** preparing is deliberate: `extract_mesh` re-keys and
# carries over `manning.gr3`, `windrot_geo2proj.gr3`, `nwmReaches.csv`,
# `ngenReaches.csv` and the NetCDF mesh, so no prepare command runs twice.
#
# ## Setup
#
# The mesh, geogrid and hydrofabric are not in the repo. Create the symlinks
# once, pointing at wherever yours live:
#
# ```bash
# cd docs/examples/alaska_schism
# ln -s /path/to/schism_models/alaska           model
# ln -s /path/to/geo_em_Alaska.nc               geo_em_Alaska.nc
# ln -s /path/to/NWM_v3_hydrofabric.gdb         hydrofabric.gdb
# ln -s /path/to/ak_nhf_1.2.2.gpkg              hydrofabric.gpkg
# ```
#
# Note the Alaska NextGen hydrofabric is the `ak_` prefixed GeoPackage. The
# links point at the files themselves, so it does not matter how your copies
# are organised.

# %%
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

notebook_dir = Path.cwd()  # assumes run from docs/examples/notebooks/
example_dir = (notebook_dir.parent / "alaska_schism").resolve()
os.chdir(example_dir)

required = (
    "model",
    "geo_em_Alaska.nc",
    "hydrofabric.gdb",
    "hydrofabric.gpkg",
    "run_full.yaml",
    "run_subset.yaml",
)
missing = [n for n in required if not Path(n).exists()]
if missing:
    raise FileNotFoundError(f"Missing in {example_dir}: {missing}. See ../README.md.")


def cli(*args: str) -> None:
    """Run a coastal-calibration CLI command and stream its output."""
    print(f"$ coastal-calibration {' '.join(args)}\n")
    subprocess.run([sys.executable, "-m", "coastal_calibration.cli", *args], check=True)


print(f"Working directory: {example_dir}")
print("Mesh contents:", sorted(p.name for p in Path("model").iterdir()))

# %% [markdown]
# ## 1. Looking at the mesh
#
# Load the mesh with the `nwm_coastal` QGIS plugin. Switching the project to an
# Alaska projection makes it easier to orient.
#
# Zoomed in, the extent of the modelled nearshore is visible.
#
# <div style="display: flex; gap: 1em; align-items: flex-start; flex-wrap: wrap;">
#   <figure style="flex: 1.43; margin: 0;">
#     <img src="../images/Alaska_SCHISM.jpg" alt="Full Alaska SCHISM mesh in QGIS" style="width: 100%;">
#     <figcaption>Full Alaska SCHISM mesh in QGIS</figcaption>
#   </figure>
#   <figure style="flex: 2.11; margin: 0;">
#     <img src="../images/Alaska_SCHISM_zoom.jpg" alt="Alaska SCHISM mesh, zoomed" style="width: 100%;">
#     <figcaption>Alaska SCHISM mesh, zoomed</figcaption>
#   </figure>
# </div>
#
# Switch the project back to EPSG:4326 before drawing anything, so polygons are
# saved in that CRS. Most tools reproject, but it is the expected one.
#
# ![Back to EPSG:4326](../images/Alaska_SCHISM_zoom_EPSG4326.jpg)
#
# The plugin can also write the mesh outline to a GeoJSON, which is handy for
# drawing against. `schism_alaska_full_boundary.geojson` in this folder came
# from these two steps.
#
# <div style="display: flex; gap: 1em; align-items: flex-start; flex-wrap: wrap;">
#   <figure style="flex: 2.36; margin: 0;">
#     <img src="../images/Alaska_SCHISM_extract_mesh_boundary.jpg" alt="Extract mesh boundary" style="width: 100%;">
#     <figcaption>Extract mesh boundary</figcaption>
#   </figure>
#   <figure style="flex: 0.89; margin: 0;">
#     <img src="../images/Alaska_SCHISM_save_mesh_boundary.jpg" alt="Save mesh boundary" style="width: 100%;">
#     <figcaption>Save mesh boundary</figcaption>
#   </figure>
# </div>

# %% [markdown]
# ## 2. Prepare the full domain
#
# ESMF mesh files first.

# %%
cli("prepare-schism-mesh", "./model", "--force")

# %% [markdown]
# `manning.gr3` ships with this mesh. To rebuild it from ESA WorldCover:
#
# ```bash
# coastal-calibration prepare-schism-manning ./model --force
# ```

# %%
print("manning.gr3 present:", Path("model/manning.gr3").exists())

# %% [markdown]
# Then the reach crosswalks. Alaska has its own NWM reach layer and its own
# NextGen hydrofabric, both picked automatically from the mesh bounds.

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
# ## 3. Draw the subset polygon
#
# Pick an area that makes sense on its own - Cook Inlet here. Roughly
# surrounding the area of interest is fine, but where the polygon crosses the
# mesh it should cut cleanly, because that edge becomes the subdomain's open
# boundary.
#
# Zoomed in on the cut, so the resulting edge is a sensible line rather than a
# ragged one.
#
# <div style="display: flex; gap: 1em; align-items: flex-start; flex-wrap: wrap;">
#   <figure style="flex: 1.76; margin: 0;">
#     <img src="../images/Alaska_SCHISM_subsetpoly.jpg" alt="Cook Inlet subset polygon" style="width: 100%;">
#     <figcaption>Cook Inlet subset polygon</figcaption>
#   </figure>
#   <figure style="flex: 2.37; margin: 0;">
#     <img src="../images/Alaska_SCHISM_subsetpoly_zoomwest.jpg" alt="Zoomed on the western cut" style="width: 100%;">
#     <figcaption>Zoomed on the western cut</figcaption>
#   </figure>
# </div>
#
# Saved here as `schism_alaska_subset_cookinlet.geojson`.

# %% [markdown]
# ## 4. Extract the subset
#
# Keeps the part of the mesh inside the polygon and rebuilds open boundaries
# along the cut.

# %%
import shapely

import coastal_calibration.schism.subsetter as ss

poly = shapely.get_geometry(
    shapely.from_geojson(Path("schism_alaska_subset_cookinlet.geojson").read_text()), 0
)
print(f"Polygon: {poly.geom_type}, bounds={[round(b, 3) for b in poly.bounds]}")

res = ss.extract_mesh("model", poly, "extracted", output_name="cookinlet")
print(
    f"Extracted: {res.classification.n_side_a:,} nodes, {res.subset.side_a.n_elements:,} elements"
)

# %% [markdown]
# The subset carries the prepared files across, re-keyed to its own element
# numbering - nothing needs regenerating.

# %%
from coastal_calibration.schism.ngen_reaches import read_reaches_blocks

subset_dir = Path("extracted/cookinlet")
print("subset contents:", sorted(p.name for p in subset_dir.iterdir()))

for name in ("nwmReaches.csv", "ngenReaches.csv"):
    if (subset_dir / name).exists():
        sources, sinks = read_reaches_blocks(subset_dir / name)
        full_s, full_k = read_reaches_blocks(Path("model") / name)
        print(f"{name}: subset {len(sources)}/{len(sinks)} vs full {len(full_s)}/{len(full_k)}")

# %% [markdown]
# ### hgrid.cpp
#
# This mesh did not ship an `hgrid.cpp`. SCHISM's own solver never reads one, but the
# `combine_sink_source` utility it runs during the discharge stage opens it
# unconditionally, and aborts when it is absent.
#
# In the mode that utility is called with it uses the element connectivity and the
# boundary block, both of which `hgrid.gr3` already carries, so a link to that file
# satisfies it. Note this writes into the linked mesh directory, not into the example.

# %%
for d in (Path("model"), Path("extracted/cookinlet")):
    cpp = d / "hgrid.cpp"
    if not cpp.is_symlink() and not cpp.exists():
        cpp.symlink_to("hgrid.gr3")
    print(f"{cpp}: -> {cpp.readlink() if cpp.is_symlink() else 'existing file'}")

# %% [markdown]
# ## 5. Run
#
# Both configs are validated first. The subset is much smaller, so run it
# first.

# %%
cli("run", "run_subset.yaml", "--dry-run")
cli("run", "run_full.yaml", "--dry-run")

# %% [markdown]
# `nodes` and `ntasks_per_node` in each yaml size the MPI job, so the same
# command works on a workstation and on the cluster.

# %%
print(Path("run_subset.yaml").read_text())

# %% [markdown]
# ### Cook Inlet subset

# %%
cli("run", "run_subset.yaml")

# %% [markdown]
# ### Full domain
#
# The full Alaska mesh is a much bigger job - `run_full.yaml` asks for 4 nodes.

# %%
cli("run", "run_full.yaml")

# %% [markdown]
# ## Summary
#
# Prepare once per mesh, subset from the prepared mesh, then run. The same
# three `prepare-schism-*` commands apply to any SCHISM domain; only the
# hydrofabric paths and `coastal_domain` change.
