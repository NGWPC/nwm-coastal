# Examples

Seven notebooks cover the client-facing surface of the library:

- **Mendocino Walkthrough** is the headliner — extract a SCHISM subdomain from the
    Pacific mesh, drive it through the 12-stage SCHISM pipeline, derive the SFINCS AOI
    from the SCHISM mesh boundary using the `nwm_coastal` QGIS plugin, run SFINCS on the
    matching domain, then compare both models against a NOAA tide gauge and animate the
    two water-level fields side-by-side.
- **Lavaca Bay** is a SFINCS-only "build from an AOI" demo on a different domain. It
    covers the create + run flow with `SfincsCreator` and `CoastalCalibRunner`, then
    drives the post-processing plotting API directly (mesh inspection, flood depth map,
    water-surface/depth/anomaly snapshots, satellite basemap overlay, animation, and
    time series at user-specified observation points).
- **Lake Erie** is the Great Lakes counterpart to Lavaca: boundary forcing from NOAA's
    Lake Erie OFS (`leofs`, the FVCOM model behind GLOFS) instead of STOFS, and a DEM
    you download and register in your own HydroMT data catalog instead of one of the
    built-in auto-fetched sources, none of which carry Great Lakes bathymetry. It also
    documents the elevated-domain settings a lake 174 m above sea level needs.
- **Lake Erie SCHISM** is the mesh-preparation counterpart: a SCHISM mesh arrives without
    the derived files the pipeline needs, and the three `prepare-schism-*` commands
    generate them before a 12-hour LEOFS-forced run on the same window as the SFINCS
    Lake Erie example.
- **Alaska SCHISM** prepares the full Alaska mesh, cuts a Cook Inlet subdomain from the
    prepared mesh — so no preparation step repeats — and runs both against STOFS.
- **Cook Inlet SFINCS** builds a SFINCS model in a region no auto-fetched elevation source
    covers, using NCEI's regional topobathy through a data catalog, and shows the QGIS
    steps behind the AOI, the flowpath selection and the quadtree refinement zones.
- **Forecast Walkthrough** is the operational pipeline rather than a single model run —
    one spinup, one analysis-and-assimilation hour, and one short-range cycle, driving
    the nwm-rte forcing engine and t-route regionalization alongside SCHISM and SFINCS.
    Each expensive stage is its own cell so a failure can be retried in place.

!!! note "Prerequisites"

    The Mendocino, Lavaca, Lake Erie SFINCS and Cook Inlet notebooks need a compiled
    SFINCS executable; Mendocino, Lake Erie SCHISM and Alaska SCHISM need SCHISM. The
    three SCHISM notebooks also need a `model` symlink pointing at a mesh — see
    [`docs/examples/README.md`](https://github.com/NGWPC/nwm-coastal/blob/development/docs/examples/README.md). Both binaries are built automatically when activating a
    pixi environment with the corresponding feature (`schism` or `sfincs`), so no manual
    build is needed in the standard workflow. See
    [Compiling SFINCS](../dev/sfincs_compilation.md) for build instructions when not using
    pixi.

!!! warning "Forecast Walkthrough is not self-contained"

    Unlike the other three, it drives external infrastructure: a built `nwm-rte` Docker
    image, `sudo docker` access, staged forcing data, and the four `NWM_COASTAL_ROOT` /
    `NWM_RTE_ROOT` / `RUN_NGEN_ROOT` / `RUN_COASTAL_ROOT` environment variables. Several
    cells may take a long time to run, so start Jupyter under `tmux`/`screen`. Setup is in
    [`forecast_demo/FORECAST_DEMO_README.md`](https://github.com/NGWPC/nwm-coastal/blob/development/forecast_demo/FORECAST_DEMO_README.md).

## Notebooks

<div class="grid cards" markdown>

- [![Mendocino walkthrough (SCHISM + SFINCS)](images/walkthrough_thumb.png){ loading=lazy }](notebooks/walkthrough.ipynb "Mendocino walkthrough — SCHISM + SFINCS comparison")
    **Mendocino Walkthrough (SCHISM + SFINCS)**

    End-to-end side-by-side demo on a single Pacific subdomain. Extract a sub-mesh from
    the full Pacific SCHISM domain with `extract_mesh`, run a SFINCS quadtree model on
    the same boundary (level-4 refinement along the SCHISM mesh edge,
    `mask.keep_largest_only`, tide-stable `run_param_overrides`), then compare the two
    against a shared NOAA gauge in a 3-line plot and render the water-level fields
    side-by-side with a shared colorbar via `animate_water_level_comparison`.

- [![Lavaca Bay (SFINCS)](images/lavaca_thumb.png){ loading=lazy }](notebooks/lavaca.ipynb "Lavaca Bay, TX")
    **Lavaca Bay (SFINCS)**

    SFINCS-only build-from-AOI workflow: `SfincsCreator` produces a quadtree mesh with
    elevation, subgrid, and discharge sources from a single AOI polygon, then
    `CoastalCalibRunner` runs the simulation and validates against NOAA observations.
    Drives the post-processing plotting API directly to produce mesh and flood-map
    inspections, water-surface/depth/anomaly snapshots, a satellite-basemap overlay, an
    animation, and time series at three user-specified observation points.

- [**Lake Erie (SFINCS + LEOFS)**](notebooks/lake_erie_sfincs.ipynb)

    Great Lakes build-from-AOI workflow on the Ohio shore of Lake Erie (Fairport Harbor).
    Walks through downloading a Great Lakes topobathy DEM, checking its CRS, NoData and
    vertical datum, and registering it in a HydroMT data catalog.
    Then runs SFINCS with `boundary.source: glofs` /
    `glofs_model: leofs` FVCOM forcing, validated against CO-OPS Fairport Harbor
    (9063053), on a mesh referenced to Lake Erie Low Water Datum, with the `zsini`
    and `latitude` settings an elevated domain needs.

- [**Lake Erie (SCHISM + LEOFS)**](notebooks/lake_erie_schism.ipynb)

    Turning a shipped SCHISM mesh into a runnable model. `prepare-schism-mesh` writes the
    ESMF NetCDF grid, `prepare-schism-manning` derives roughness from ESA WorldCover, and
    `prepare-schism-reaches` builds the source/sink crosswalk by intersecting the mesh
    boundary with a hydrofabric — the Detroit River in, the Niagara out. Then a 12-hour
    GLOFS-forced run with the absolute-IGLD85 datum offset an elevated mesh needs.

- [**Alaska (SCHISM, full + subset)**](notebooks/alaska_schism.ipynb)

    Full Alaska domain and a Cook Inlet subdomain in one notebook. The full mesh is
    prepared first and the subset cut from it with `extract_mesh`, which re-keys and
    carries over Manning, wind rotation and both reach crosswalks, so nothing is prepared
    twice. QGIS screenshots cover orienting the mesh and cutting a clean subset edge.

- [**Cook Inlet (SFINCS, Alaska)**](notebooks/alaska_sfincs.ipynb)

    Building a SFINCS model where nothing auto-fetched reaches: NOAA's 3 m topobathy stops
    at CONUS/HI/PR, the CRM mosaic at 49 N. Uses NCEI's Cook Inlet regional DEMs through a
    HydroMT catalog, with the MHHW-to-MSL shift a 9 m tide range makes unavoidable. 1024 m
    quadtree refined to 256 m over seven zones, plus the QGIS workflow for the AOI,
    flowpaths and refinement polygons.

- [**Forecast Walkthrough (hourly cycle)**](notebooks/forecast_walkthrough.ipynb)

    One hourly coastal forecast cycle run by hand, with no ecflow: SCHISM/SFINCS spinup
    and the t-route AnA-A bootstrap, then an analysis-and-assimilation hour (t-route
    AnA-A/AnA-B plus met forcing from the nwm-rte forcing engine), then a short-range
    cycle, with observed-vs-SCHISM-vs-SFINCS comparison plots after each. Paired with
    `forecast_demo/forecast_walkthrough.py`, which runs the same steps end to end.

</div>
