# API Reference

This page provides detailed documentation for the NWM Coastal Python API.

## Configuration Classes

### CoastalCalibConfig

::: coastal_calibration.config.schema.CoastalCalibConfig
    options:
      show_source: true
      members:
        - from_yaml
        - from_dict
        - to_yaml
        - to_dict
        - validate
        - model

### SimulationConfig

::: coastal_calibration.config.schema.SimulationConfig

### BoundaryConfig

::: coastal_calibration.config.schema.BoundaryConfig

### PathConfig

::: coastal_calibration.config.schema.PathConfig

### ModelConfig

::: coastal_calibration.config.schema.ModelConfig

### SchismModelConfig

::: coastal_calibration.config.schema.SchismModelConfig

### SfincsModelConfig

::: coastal_calibration.config.schema.SfincsModelConfig

### MonitoringConfig

::: coastal_calibration.config.schema.MonitoringConfig

### DownloadConfig

::: coastal_calibration.config.schema.DownloadConfig

## SFINCS Creation Configuration

### SfincsCreateConfig

::: coastal_calibration.config.create_schema.SfincsCreateConfig
    options:
      show_source: true
      members:
        - from_yaml
        - from_dict
        - to_yaml
        - to_dict
        - validate
        - stage_order

### GridConfig

::: coastal_calibration.config.create_schema.GridConfig

### ElevationConfig

::: coastal_calibration.config.create_schema.ElevationConfig

### MaskConfig

::: coastal_calibration.config.create_schema.MaskConfig

### SubgridConfig

::: coastal_calibration.config.create_schema.SubgridConfig

### RiverDischargeConfig

::: coastal_calibration.config.create_schema.RiverDischargeConfig

## Workflow Runners

### CoastalCalibRunner

::: coastal_calibration.runner.CoastalCalibRunner
    options:
      show_source: true
      members:
        - validate
        - run

### SfincsCreator

::: coastal_calibration.sfincs.create.SfincsCreator
    options:
      show_source: true
      members:
        - run

### WorkflowResult

::: coastal_calibration.runner.WorkflowResult

## Plotting

### SfincsGridInfo

::: coastal_calibration.sfincs.plotting.SfincsGridInfo
    options:
      show_source: true
      members:
        - from_model_root

### plot_mesh

::: coastal_calibration.sfincs.plotting.plot_mesh

### plot_floodmap

::: coastal_calibration.sfincs.plotting.plot_floodmap

### plot_station_comparison

::: coastal_calibration.plotting.stations.plot_station_comparison

### plot_water_level

::: coastal_calibration.plotting.spatial.plot_water_level

### animate_water_level

::: coastal_calibration.plotting.animate.animate_water_level

## SCHISM Mesh Subsetting

Cut a regional subdomain out of a larger SCHISM mesh, or split one along a dividing
line. Used by the [Mendocino](../examples/notebooks/walkthrough.ipynb),
[Alaska](../examples/notebooks/alaska_schism.ipynb) and
[Lake Erie](../examples/notebooks/lake_erie_schism.ipynb) examples, and by the QGIS
plugin.

### extract_mesh

::: coastal_calibration.schism.subsetter.extract_mesh

### split_mesh

::: coastal_calibration.schism.subsetter.split_mesh

### MeshSubsetter

::: coastal_calibration.schism.subsetter.MeshSubsetter

## Tidal Prediction

Tidal boundary conditions via [pyTMD](https://pytmd.readthedocs.io/), against the TPXO10
atlas by default and any netCDF model in `pyTMD.io.load_database()` with an elevation
group.

### predict_tide_at_points

::: coastal_calibration.data.tides.predict_tide_at_points

### write_schism_boundary

::: coastal_calibration.data.tides.write_schism_boundary

### extend_schism_boundary

::: coastal_calibration.data.tides.extend_schism_boundary

## Output Readers

### load_schism_elevation

::: coastal_calibration.schism.outputs.load_schism_elevation

### load_sfincs_water_level

::: coastal_calibration.sfincs.outputs.load_sfincs_water_level

## Observation Points

### load_obs_points

::: coastal_calibration.observations.load_obs_points

### validate_points_in_domain

::: coastal_calibration.observations.validate_points_in_domain

### extract_water_level_series

::: coastal_calibration.observations.extract_water_level_series

## Flood Depth Map

### create_flood_depth_map

::: coastal_calibration.sfincs.floodmap.create_flood_depth_map

## Downloader

### validate_date_ranges

::: coastal_calibration.data.downloader.validate_date_ranges

## NOAA CO-OPS API

### COOPSAPIClient

::: coastal_calibration.data.coops_api.COOPSAPIClient
    options:
      show_source: true
      members:
        - stations_metadata
        - validate_parameters
        - build_url
        - fetch_data
        - get_datums

### query_coops_byids

::: coastal_calibration.data.coops_api.query_coops_byids

### query_coops_bygeometry

::: coastal_calibration.data.coops_api.query_coops_bygeometry

## Type Aliases

```python
# Model type
ModelType = Literal["schism", "sfincs"]

# Meteorological data source
MeteoSource = Literal["nwm_retro", "nwm_ana"]

# Coastal domain identifier
CoastalDomain = Literal["prvi", "hawaii", "atlgulf", "pacific"]

# Boundary condition source
BoundarySource = Literal["tpxo", "stofs"]

# Logging level
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
```

## Constants

### Default Paths

```python
DEFAULT_NFS_MOUNT = Path("/ngen-test")
```

### Default Path Templates

```python
DEFAULT_WORK_DIR_TEMPLATE = (
    "/ngen-test/coastal/${user}/"
    "${model}_${simulation.coastal_domain}_${boundary.source}_${simulation.meteo_source}/"
    "${model}_${simulation.start_date}"
)

DEFAULT_RAW_DOWNLOAD_DIR_TEMPLATE = (
    "/ngen-test/coastal/${user}/"
    "${model}_${simulation.coastal_domain}_${boundary.source}_${simulation.meteo_source}/"
    "raw_data"
)
```

### Model Registry

```python
MODEL_REGISTRY: dict[str, type[ModelConfig]] = {
    "schism": SchismModelConfig,
    "sfincs": SfincsModelConfig,
}
```
