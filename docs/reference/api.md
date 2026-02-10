# API Reference

This page provides detailed documentation for the NWM Coastal Python API.

## Configuration Classes

### CoastalCalibConfig

::: coastal_calibration.config.schema.CoastalCalibConfig
    options:
      show_source: true
      members:
        - from_yaml
        - to_yaml
        - to_dict
        - validate
        - model

### SlurmConfig

::: coastal_calibration.config.schema.SlurmConfig

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

## Workflow Runner

### CoastalCalibRunner

::: coastal_calibration.runner.CoastalCalibRunner
    options:
      show_source: true
      members:
        - validate
        - submit
        - run

### WorkflowResult

::: coastal_calibration.runner.WorkflowResult

## Downloader

### validate_date_ranges

::: coastal_calibration.downloader.validate_date_ranges

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
DEFAULT_SING_IMAGE_PATH = Path("/ngencerf-app/singularity/ngen-coastal.sif")
DEFAULT_PARM_DIR = Path("/ngen-test/coastal/ngwpc-coastal")
DEFAULT_NGEN_APP_DIR = Path("/ngen-app")
DEFAULT_NFS_MOUNT = Path("/ngen-test")
DEFAULT_CONDA_ENV_NAME = "ngen_forcing_coastal"
DEFAULT_NWM_VERSION = "v3.0.6"
DEFAULT_OTPS_DIR = Path("/ngen-app/OTPSnc")
DEFAULT_SLURM_PARTITION = "c5n-18xlarge"
```

### Default Path Templates

```python
DEFAULT_WORK_DIR_TEMPLATE = (
    "/ngen-test/coastal/${slurm.user}/"
    "${model}_${simulation.coastal_domain}_${boundary.source}_${simulation.meteo_source}/"
    "${model}_${simulation.start_date}"
)

DEFAULT_RAW_DOWNLOAD_DIR_TEMPLATE = (
    "/ngen-test/coastal/${slurm.user}/"
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
