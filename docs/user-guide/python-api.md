# Python API

The Python API provides programmatic access to the coastal calibration workflow,
enabling integration with other tools and custom automation.

## Basic Usage

```python
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

# Load configuration from YAML
config = CoastalCalibConfig.from_yaml("config.yaml")

# Create a runner
runner = CoastalCalibRunner(config)

# Validate configuration
errors = runner.validate()
if errors:
    for error in errors:
        print(f"Error: {error}")
else:
    # Run the workflow
    result = runner.run()

    if result.success:
        print(f"Completed in {result.duration_seconds:.1f}s")
    else:
        print(f"Failed: {result.errors}")
```

## Configuration

### Loading Configuration

```python
from coastal_calibration import CoastalCalibConfig

# From YAML file
config = CoastalCalibConfig.from_yaml("config.yaml")

# From a plain dictionary (useful in notebooks and scripts)
config = CoastalCalibConfig.from_dict(
    {
        "simulation": {
            "start_date": "2021-06-11",
            "duration_hours": 24,
            "coastal_domain": "hawaii",
            "meteo_source": "nwm_ana",
        },
        "boundary": {"source": "stofs"},
    }
)

# Access configuration values
print(config.simulation.coastal_domain)
print(config.paths.work_dir)
print(config.model)  # "schism" or "sfincs"
```

### Creating SCHISM Configuration Programmatically

```python
from datetime import datetime
from pathlib import Path
from coastal_calibration import (
    CoastalCalibConfig,
    SimulationConfig,
    BoundaryConfig,
    PathConfig,
    SchismModelConfig,
    MonitoringConfig,
    DownloadConfig,
)

config = CoastalCalibConfig(
    simulation=SimulationConfig(
        start_date=datetime(2021, 6, 11),
        duration_hours=24,
        coastal_domain="hawaii",
        meteo_source="nwm_ana",
    ),
    boundary=BoundaryConfig(source="stofs"),
    paths=PathConfig(
        work_dir=Path("/ngen-test/coastal/your_username/my_run"),
        raw_download_dir=Path("/ngen-test/coastal/your_username/downloads"),
    ),
    model_config=SchismModelConfig(
        nodes=2,
        ntasks_per_node=18,
    ),
)

# Save to YAML
config.to_yaml("generated_config.yaml")
```

### Enabling NOAA Observation Station Comparison

To automatically discover NOAA CO-OPS water level stations within the model domain and
generate comparison plots after the SCHISM run, set `include_noaa_gages=True` in the
model configuration:

```python
config = CoastalCalibConfig(
    simulation=SimulationConfig(
        start_date=datetime(2021, 6, 11),
        duration_hours=24,
        coastal_domain="hawaii",
        meteo_source="nwm_ana",
    ),
    boundary=BoundaryConfig(source="stofs"),
    paths=PathConfig(
        work_dir=Path("/ngen-test/coastal/your_username/hawaii_obs"),
        raw_download_dir=Path("/ngen-test/coastal/your_username/downloads"),
    ),
    model_config=SchismModelConfig(
        nodes=2,
        ntasks_per_node=18,
        include_noaa_gages=True,  # Enable station discovery & comparison plots
    ),
)
```

This activates two additional stages in the pipeline:

- **`schism_obs`** discovers NOAA CO-OPS stations via the concave hull of the open
    boundary nodes and writes `station.in` and `station_noaa_ids.txt`.
- **`schism_plot`** reads SCHISM station output (`staout_1`), fetches observations from
    the CO-OPS API (with MLLW→MSL datum conversion), and saves comparison plots to
    `figs/`.

### Creating SFINCS Configuration Programmatically

```python
from datetime import datetime
from pathlib import Path
from coastal_calibration import (
    CoastalCalibConfig,
    SimulationConfig,
    BoundaryConfig,
    PathConfig,
    SfincsModelConfig,
)

TEXAS_DIR = Path("/path/to/texas/model")

config = CoastalCalibConfig(
    simulation=SimulationConfig(
        start_date=datetime(2025, 6, 1),
        duration_hours=168,
        coastal_domain="atlgulf",
        meteo_source="nwm_ana",
    ),
    boundary=BoundaryConfig(source="stofs"),
    paths=PathConfig(
        work_dir=Path("/tmp/sfincs_run"),
        raw_download_dir=Path("/tmp/sfincs_downloads"),
    ),
    model_config=SfincsModelConfig(
        prebuilt_dir=TEXAS_DIR,
        discharge_locations_file=TEXAS_DIR / "sfincs_nwm.src",
        observation_points=[
            {"x": 830344.95, "y": 3187383.41, "name": "Sargent"},
        ],
        merge_observations=False,
        merge_discharge=False,
        include_noaa_gages=True,
        include_precip=True,
        include_wind=True,
        include_pressure=True,
        forcing_to_mesh_offset_m=0.0,  # STOFS already in mesh datum
        vdatum_mesh_to_msl_m=0.171,  # mesh datum (NAVD88) → MSL for obs comparison
        meteo_res=2000,  # meteo output resolution in meters (auto-derived if None)
        floodmap_dem=TEXAS_DIR / "dem.tif",  # high-res DEM for flood depth map
        floodmap_hmin=0.05,  # minimum flood depth threshold (m)
        floodmap_enabled=True,  # enable flood depth map generation
    ),
)
```

### Configuration Validation

```python
config = CoastalCalibConfig.from_yaml("config.yaml")

# Validate and get list of errors
errors = config.validate()

if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid")
```

## Workflow Execution

### Run the Workflow

```python
# Run complete workflow
result = runner.run()

# Run partial workflow
result = runner.run(start_from="schism_forcing_prep", stop_after="schism_sflux")

# Run from a specific stage to the end
result = runner.run(start_from="schism_prep")
```

### Workflow Result

The `WorkflowResult` object contains information about the execution:

```python
result = runner.run()

print(f"Success: {result.success}")
print(f"Duration: {result.duration_seconds}s")

if not result.success:
    for error in result.errors:
        print(f"Error: {error}")

# Stage timing (if enable_timing is True)
for stage, duration in result.stage_durations.items():
    print(f"  {stage}: {duration:.1f}s")
```

All result objects (`WorkflowResult`, `DownloadResult`, `DownloadResults`, and
`StageProgress`) have human-readable `__str__` methods, so `print(result)` produces
clean, indented output:

```python
result = runner.run()
print(result)
# WorkflowResult: SUCCESS
#   Start:     2025-06-01 00:00:00
#   End:       2025-06-01 00:12:34
#   Duration:  12m 34s
#   Completed: download, sfincs_init, sfincs_timing, ...
```

## SFINCS Model Creation

The `SfincsCreator` runner builds a new SFINCS quadtree model from an AOI polygon. It
uses a separate `SfincsCreateConfig` configuration schema.

```python
from coastal_calibration import SfincsCreateConfig
from coastal_calibration.creator import SfincsCreator

# Load from YAML
config = SfincsCreateConfig.from_yaml("create_config.yaml")

# Or from a dictionary
config = SfincsCreateConfig.from_dict(
    {
        "aoi": "./texas_aoi.geojson",
        "output_dir": "./my_sfincs_model",
        "elevation": {
            "datasets": [{"name": "nws_30m", "zmin": -20000}],
        },
        "data_catalog": {"data_libs": ["./dem/data_catalog.yml"]},
    }
)

# Run the creation workflow
creator = SfincsCreator(config)
creator.run()

# Resume from a specific stage (uses .create_status.json for tracking)
creator.run(start_from="create_elevation")
```

## Plotting Utilities

The `coastal_calibration.plotting` module provides reusable functions for visualizing
SFINCS grids, flood depth maps, and simulated vs observed water-level comparisons.

### Grid Inspection

```python
from coastal_calibration import SfincsGridInfo, plot_mesh

# Load grid metadata from a SFINCS model directory
info = SfincsGridInfo.from_model_root("run/sfincs_model")
print(info)
# SfincsGridInfo(quadtree, EPSG:32619)
#   Faces:     293,850
#   Edges:     596,123
#   Level 1:    7,090 cells (512 m)
#   Level 2:   14,180 cells (256 m)
#   ...

# Plot the mesh colored by refinement level with satellite basemap
fig, ax = plot_mesh(info)
```

### Flood Depth Map

```python
from coastal_calibration import plot_floodmap

# Plot the flood-depth COG with automatic overview selection
fig, ax = plot_floodmap("run/sfincs_model/floodmap_hmax.tif")
```

### Station Comparison Plots

```python
from coastal_calibration import plot_station_comparison, plotable_stations

# Filter stations that have both simulated and observed data
pairs = plotable_stations(station_ids, sim_elevation, obs_ds)

# Generate 2×2 comparison figures and save to disk
fig_paths = plot_station_comparison(
    sim_times, sim_elevation, station_ids, obs_ds, "run/sfincs_model/figs"
)
```

## Data Sources

### Check Available Date Ranges

```python
from coastal_calibration.downloader import validate_date_ranges

# Validate dates for your configuration
errors = validate_date_ranges(
    start_time=datetime(2021, 6, 11),
    end_time=datetime(2021, 6, 12),
    meteo_source="nwm_ana",
    boundary_source="stofs",
    coastal_domain="hawaii",
)

if errors:
    print("Date range errors:", errors)
```

### Supported Data Sources

| Source      | Date Range               | Description           |
| ----------- | ------------------------ | --------------------- |
| `nwm_retro` | 1979-02-01 to 2023-01-31 | NWM Retrospective 3.0 |
| `nwm_ana`   | 2018-09-17 to present    | NWM Analysis          |
| `stofs`     | 2020-12-30 to present    | STOFS water levels    |
| `glofs`     | 2005-09-30 to present    | Great Lakes OFS       |

## Logging

Configure logging for the workflow:

```python
import logging
from coastal_calibration.logging import setup_logger

# Set up logging
logger = setup_logger(log_level="DEBUG", log_file="workflow.log")

# Now run your workflow
config = CoastalCalibConfig.from_yaml("config.yaml")
runner = CoastalCalibRunner(config)
result = runner.run()
```

## Example: Batch Processing

Run multiple simulations with different parameters:

```python
from datetime import datetime, timedelta
from coastal_calibration import CoastalCalibConfig, CoastalCalibRunner

# Load base configuration
base_config = CoastalCalibConfig.from_yaml("base_config.yaml")

# Run simulations for multiple dates
start_dates = [
    datetime(2021, 6, 1),
    datetime(2021, 6, 15),
    datetime(2021, 7, 1),
]

results = []
for start_date in start_dates:
    # Modify configuration for this run
    config = CoastalCalibConfig.from_yaml("base_config.yaml")
    config.simulation.start_date = start_date

    # Update work directory for this run
    date_str = start_date.strftime("%Y%m%d")
    config.paths.work_dir = config.paths.work_dir.parent / f"run_{date_str}"

    # Run
    runner = CoastalCalibRunner(config)
    result = runner.run()
    results.append((start_date, result))

# Report results
for start_date, result in results:
    status = "Success" if result.success else "Failed"
    print(f"{start_date}: {status}")
```

## Example: Domain Comparison

Run the same simulation across multiple domains:

```python
domains = ["hawaii", "prvi", "atlgulf", "pacific"]

for domain in domains:
    config = CoastalCalibConfig.from_yaml("base_config.yaml")
    config.simulation.coastal_domain = domain

    runner = CoastalCalibRunner(config)
    errors = runner.validate()

    if errors:
        print(f"{domain}: Validation failed - {errors}")
        continue

    result = runner.run()
    print(f"{domain}: {'Success' if result.success else 'Failed'}")
```
