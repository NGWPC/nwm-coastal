# Concepts and Workflows

NWM Coastal is a toolset for configuring, calibrating, and validating SCHISM and SFINCS
coastal models for the Next Generation National Water Model, driving each simulation
from a single configuration file and command-line interface. Routed NWM streamflow
enters at the models' river inflow points, tides and storm surge at the ocean boundary,
and meteorological data over the domain surface. From these inputs, the models simulate
total water level and coastal flooding. NOAA CO-OPS water level observations can be
downloaded automatically and compared against the simulated results.

Each NWM v3.0 coastal domain has an existing SCHISM mesh, and this package can subset
one to a smaller region of interest. Generating new SCHISM meshes programmatically is
not yet feasible for implementation into the package. SFINCS was included for the
complementary capability: HydroMT-SFINCS can build a complete model — grid, elevation,
roughness, and boundaries — programmatically from an area-of-interest polygon, making it
practical to stand up coverage or improve resolution wherever it may be needed.

This page describes the system as a whole: the directory structure it expects, the
topobathy and land cover available for building models, the forcing data that drives
simulations, and the additional components required to run a forecast through the
NextGen framework.

For per-stage details, see [Workflow Stages](user-guide/workflow-stages.md).

## Directory Layout

`nwm-coastal` expects several sibling directories. Everything except the repository
itself is populated by
[`scripts/setup_data_coastal.sh`](getting-started/installation.md#download-model-data).

```
ngwpc/
├── nwm-coastal/      this repository — the coastal-calibration package and CLI
├── nwm-rte/          NextGen runtime; needed only for forecasts
├── run_ngen/         forcing engine and t-route outputs, ESMF mesh/domain files
├── run_coastal/      coastal models and simulation configs
└── coastal_data/     TPXO tidal atlas, hydrofabric copies
```

`run_coastal/` is where the modeling actually happens:

| Directory | Contents |
| --------- | -------- |
| `schism_models/` | Prebuilt SCHISM meshes, one per domain, plus subsets |
| `sfincs_models/` | SFINCS models you have created |
| `schism_sims/` | SCHISM run configurations and per-cycle outputs |
| `sfincs_sims/` | SFINCS run configurations and per-cycle outputs |

Paths in a run configuration are resolved relative to the configuration file, which is
why the example configs use relative paths like `../schism_models/atlgulf`.

## SCHISM: Prebuilt Meshes

SCHISM meshes already exist for every supported domain, so there is no creation step.
A run points at one with `model_config.prebuilt_dir`:

```yaml
model_config:
  prebuilt_dir: ../schism_models/atlgulf
  geogrid_file: ../../run_ngen/data/esmf_mesh/NWM/domain/geo_em_CONUS.nc
```

A mesh directory holds the model definition only — `hgrid.gr3`, `vgrid.in`,
`param.nml`, `bctides.in`, and `manning.gr3` are required; forcing and output files are
generated per run and should not be copied between machines.

### Running the full NWMv3 domain

Running a whole domain needs no special mode — you simply skip the subsetting step and
point `prebuilt_dir` at the full mesh instead of a subset. A worked retrospective
configuration is at
[`schism_retro_full_domain.yaml`](examples/schism_retro_full_domain.yaml).

Full domains are large. The Atlantic/Gulf mesh is roughly 10.5 million nodes and 2.7
million elements, so it needs a genuine multi-node MPI allocation rather than a
workstation — see the [sbatch pattern](user-guide/cli.md#using-run-inside-a-slurm-job-heredoc-recommended).
Subsetting exists precisely so that regional studies do not pay that cost.

## QGIS Plugin

The [QGIS plugin](user-guide/qgis-plugin.md) produces the GeoJSON inputs for both
models. It does two jobs.

![The nwm_coastal toolbar in QGIS](examples/images/plugin_window.png)

**Subsetting a SCHISM mesh.** Load a full mesh, draw a polygon around the region you
care about, and save it. `extract_mesh` then clips the mesh to that polygon and rebuilds
the open boundaries where the polygon cuts across it, producing a small, self-contained
model directory. A regional subset runs in minutes on a workstation where the full
domain needs a cluster.

![Drawing a subset polygon over the Pacific SCHISM mesh](examples/images/plugin_extract_schism.png)

**Defining a SFINCS domain.** The same drawing tools produce the AOI polygon that
`create` consumes, an optional refinement polygon for finer quadtree resolution, and the
NWM flowpaths that become river discharge points. Aligning the AOI to watershed
boundaries keeps the hydrology coherent.

![Model domain aligned to watershed boundaries](examples/images/plugin_divide_union.png)

## SFINCS: Models Are Created

There is no prebuilt SFINCS library — coverage does not exist everywhere yet, so you
build a model for your area of interest:

```bash
pixi r -e dev coastal-calibration create create_config.yaml
```

`create` uses [HydroMT-SFINCS](https://deltares.github.io/hydromt_sfincs/) under the
hood. It builds a quadtree grid from the AOI, fetches and applies elevation and land
cover, masks active cells, sets boundary cells, adds river discharge points, builds
subgrid tables, and writes the model. The output directory becomes the `prebuilt_dir`
for simulation runs. See the [create stages](user-guide/workflow-stages.md#sfincs-creation-stages)
for what each step does.

## Topobathy and Land Cover

Several datasets are wired up for automatic download. Set `source` on an elevation
dataset and `create_fetch_data` retrieves it, clipped to your AOI.

| `source` | Dataset | Resolution | Best suited to | Retrieved from |
| -------- | ------- | ---------- | -------------- | -------------- |
| `nws_30m` | NWS topo-bathymetric DEM | 30 m | NWM coastal domains; requires `coastal_domain` | `s3://ngwpc-dev/nwm-tools-data/nws-topobathy/` (credentials required) |
| `noaa_3m` | NOAA NCEI coastal topobathy | ~3 m | US coastal areas needing fine detail | [`noaa-nos-coastal-lidar-pds`](https://registry.opendata.aws/noaa-coastal-lidar/) (public) |
| `noaa_crm` | NOAA Coastal Relief Model | ~90 m | Broad US coastal coverage | [NCEI ArcGIS image service](https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/CRM_mosaic/ImageServer) |
| `copdem_30m` | Copernicus DEM | 30 m | Global land elevation (default) | [`copernicus-dem-30m`](https://registry.opendata.aws/copernicus-dem/) (public) |
| `gebco_15arcs` | GEBCO 2025 | ~450 m | Global bathymetry (default) | [GEBCO via CEDA](https://www.gebco.net/data_and_products/gridded_bathymetry_data/) |
| `esa_worldcover` | ESA WorldCover 2020 | 10 m | Land cover, used for roughness | [`esa-worldcover`](https://registry.opendata.aws/esa-worldcover-vito/) (public) |

The defaults are `copdem_30m` for land and `gebco_15arcs` for bathymetry. Multiple
datasets can be combined; finer data is used where it is available.

These are conveniences, not a closed list. If a fetch fails — an external service is
down, or credentials are unavailable — download the data yourself or choose a different
source, then apply it as described below. The fetch happens once during model setup, so
a failure costs you a retry, not a simulation.

### Using your own DEM

An elevation dataset with **no `source`** is looked up by `name` in a
[HydroMT data catalog](https://deltares.github.io/hydromt/)
listed under `data_catalog.data_libs`. This is how you use any DEM the auto-fetch
sources do not cover — including **Great Lakes bathymetry**, which none of the sources
above provide.

Write a catalog next to your GeoTIFF:

```yaml
# my_dem/data_catalog.yml
meta:
  version: v1.0.0
  name: my_dem
  hydromt_version: '>1.0a,<2'

my_dem:
  data_type: RasterDataset
  uri: my_dem.tif
  driver:
    name: rasterio
  metadata:
    category: topography
    crs: 4326
  data_adapter:
    rename:
      elevation: elevtn
```

Then reference it from the create configuration by name, with no `source`:

```yaml
elevation:
  datasets:
    - name: my_dem
      zmin: -20000

data_catalog:
  data_libs:
    - ./my_dem/data_catalog.yml
```

The `rename` entry matters: HydroMT-SFINCS expects the elevation variable to be called
`elevtn`. If your raster already uses that name, omit `data_adapter`.

For a working reference, run `coastal-calibration prepare-topobathy`, which downloads a
DEM and writes exactly this catalog structure alongside it.

## Running Models

Both models use the same command and the same configuration schema; only
`model_config` differs:

```bash
pixi r -e dev coastal-calibration run config.yaml
```

Stages run in sequence and are resumable, so a failure part-way through can be picked up
with `--start-from` rather than restarted. `--dry-run` validates without executing.

Forcing comes from `simulation.meteo_source`: `nwm_retro` for historical periods and
`nwm_ana` for 2018 onward. Both are public and need no credentials. See
[Supported Data Sources](user-guide/configuration.md#domains) for coverage by domain.

## Forecasts

Forecast runs need more than this repository. The coastal models consume two products
generated upstream by [`nwm-rte`](https://github.com/NGWPC/nwm-rte):

1. **Gridded meteorological forcing** from the NextGen forcing engine, passed as
   `paths.forecast_meteo_file`.
2. **River discharge** from a t-route regionalization run for the VPU, passed as
   `paths.troute_file`.

With those in hand the coastal configuration sets `meteo_source: ngen_forecast` and runs
exactly like any other simulation. Each cycle warm-starts from the previous cycle's
hotstart or restart file, so the sequence is continuous rather than a series of cold
starts.

The [forecast demo](https://github.com/NGWPC/nwm-coastal/tree/development/forecast_demo)
wires this together with ecFlow, running SCHISM and SFINCS hourly for analysis and
short-range cycles.

## Proprietary Data and Symlinks

Some inputs are OWP's own data and are not redistributable, so they are not in the
repository and not in the download script. The example walkthrough expects two
gitignored symlinks that you create once, pointing at wherever that data lives on your
machine:

```bash
ln -s /path/to/schism_models/pacific          docs/examples/walkthrough/model
ln -s /path/to/schism_models/geo_em_CONUS.nc  docs/examples/walkthrough/geo_em_CONUS.nc
```

The walkthrough fails with a clear error if they are missing. See the
[examples README](https://github.com/NGWPC/nwm-coastal/blob/development/docs/examples/README.md)
for the full list of per-domain inputs.

## SCHISM and SFINCS Differences

The two models handle river discharge differently, which matters when comparing results:

- **SCHISM** uses paired **sources and sinks**. Where an NWM flowpath crosses the mesh
  boundary it is flagged as a source if flow enters and a sink if it leaves. Because the
  prebuilt meshes were not built with flowpaths as a constraint, a meandering boundary
  can flag the same river repeatedly.
- **SFINCS** uses **source points only**. `create_discharge` locates where selected
  flowpaths enter the AOI and snaps those inflows to active cells; water leaves through
  the open coastal boundary rather than through a sink.

The practical recommendation for SFINCS is to define the domain so rivers enter across
the boundary and drain to the open coast, giving sources with no sinks. See
[Spurious source/sink points](dev/schism_sink_source_issue.md) for the full analysis.
