# `docs/examples/` — what's where

This directory contains every notebook and input file needed to reproduce the published
tutorials. The rendered landing page (`index.md`) is for end-users; this README is for
maintainers and the next dev.

## Layout

```
docs/examples/
├── README.md              ← this file
├── index.md               ← rendered docs landing
├── .gitignore             ← single source of truth for what's transient
├── images/                ← screenshots referenced by notebooks / README
├── notebooks/             ← .py + .ipynb pairs (jupytext-paired)
│
├── lavaca-tx/             ← SFINCS Lavaca Bay tutorial inputs
└── walkthrough/           ← SCHISM + SFINCS Mendocino comparison inputs
```

Every per-domain directory contains **only the inputs** (GeoJSONs and YAML configs).
Runtime products (the SCHISM/SFINCS work directories, downloaded forcings, derived
meshes, the proprietary mesh symlinks, etc.) are gitignored.

## Per-domain inputs and notebook map

| Domain         | Notebook            | Tracked input files                                                                   |
| -------------- | ------------------- | ------------------------------------------------------------------------------------- |
| `lavaca-tx/`   | `lavaca.ipynb`      | `aoi.geojson`, `refine.geojson`, `discharge_nwm.geojson`, `create.yaml`, `run.yaml`   |
| `walkthrough/` | `walkthrough.ipynb` | `extract_poly.geojson`, `aoi.geojson`, `refine_poly.geojson`, `discharge_nwm.geojson` |

Earlier per-domain demos (Narragansett, Hawaii, Pacific extract, Hawaii subset, the
single-domain post-run plotting notebooks, and the cluster-side `1_schism_subset.py` /
`2_sfincs_run.py` scripts) are archived under
`/Volumes/data/nwm-coastal-leftover/notebooks_archive/`. They were either redundant with
the walkthrough or have been merged into `lavaca.py`'s post-run section. Pull them back
from the archive if the underlying functionality (e.g. `divide_mesh`, single-domain
SCHISM Hawaii run) needs to be demoed again.

## Proprietary inputs (set up post-clone)

The full Pacific SCHISM mesh and the WRF geogrid are not redistributable. The
`walkthrough/` directory expects them as **gitignored symlinks** that you create once
after cloning, pointing at wherever the data lives on your machine.

```bash
ln -s /path/to/schism_models/pacific          docs/examples/walkthrough/model
ln -s /path/to/schism_models/geo_em_CONUS.nc  docs/examples/walkthrough/geo_em_CONUS.nc
```

The walkthrough fails with a clear error if the symlinks are missing. The `model`
symlink is matched by the `**/model` pattern in `.gitignore`; the geogrid by
`**/geo_em_*.nc`.

## Runtime products (gitignored, fine to delete)

After running a notebook each per-domain directory grows additional subdirectories that
the `.gitignore` keeps untracked:

| Subdir          | Source                                                                  | Safe to delete?                    |
| --------------- | ----------------------------------------------------------------------- | ---------------------------------- |
| `cache/`        | per-notebook lookup cache (CO-OPS metadata, station lists, …)           | yes                                |
| `run/`          | SCHISM work directory (param.nml, partitioned mesh, model outputs, log) | yes (re-runs the pipeline)         |
| `output/`       | SFINCS work directory (sfincs.inp, sfincs\_\*.nc, …)                    | yes                                |
| `extracted/`    | output of `extract_mesh` — the small subset SCHISM project              | yes (re-runs the extract notebook) |
| `figs/`         | per-run plotting output                                                 | yes                                |
| `sfincs_model/` | HydroMT-SFINCS model root                                               | yes                                |
| `downloads/`    | shared NWM/STOFS forcing cache (top-level, ~83 GB)                      | only if you're OK re-downloading   |

## Notebooks

Both notebooks are paired with a `.py` source via `jupytext`. Edit the `.py` and run
`pixi r -e dev jupytext --sync <name>.py` to regenerate the `.ipynb`. Both files must be
staged together — `pre-commit` enforces this.

Each notebook starts with `os.chdir(notebook_dir.parent / "<domain>")`, then references
inputs as plain `./<file>.geojson` relative to that working directory.
