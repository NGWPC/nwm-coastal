# NWM Coastal

A Python package for running SCHISM and SFINCS coastal model calibration workflows on
HPC clusters with Singularity containers and SLURM job scheduling.

## Features

- **Multi-Model Support**: SCHISM (multi-node MPI) and SFINCS (single-node OpenMP) via a
    polymorphic `ModelConfig` architecture
- **YAML Configuration**: Simple, human-readable configuration files with variable
    interpolation
- **SLURM Integration**: Automatic job script generation and submission
- **Data Download**: Automated download of NWM and STOFS boundary data
- **Multiple Domains**: Support for Hawaii, Puerto Rico/Virgin Islands, Atlantic/Gulf,
    and Pacific
- **Boundary Conditions**: TPXO tidal model and STOFS water level support
- **NOAA Observation Stations**: Automatic discovery of CO-OPS water level stations
    within the model domain, with post-run comparison plots (simulated vs observed)
- **Workflow Control**: Unified `run` and `submit` pipelines with `--start-from` /
    `--stop-after` support for partial workflows
- **Configuration Inheritance**: Share common settings across multiple runs

## Quick Example

```yaml
# config.yaml - minimal SCHISM configuration
slurm:
  job_name: my_schism_run
  user: your_username

simulation:
  start_date: 2021-06-11
  duration_hours: 24
  coastal_domain: hawaii
  meteo_source: nwm_ana

boundary:
  source: stofs
```

```bash
# Submit and monitor the job
coastal-calibration submit config.yaml --interactive
```

## Supported Models

| Model  | Status    | Description                                                    |
| ------ | --------- | -------------------------------------------------------------- |
| SCHISM | Supported | Semi-implicit Cross-scale Hydroscience Integrated System Model |
| SFINCS | Supported | Super-Fast INundation of CoastS                                |

## Installation

```bash
pip install coastal-calibration
```

See the [Installation Guide](getting-started/installation.md) for detailed instructions.

## License

This project is licensed under the BSD-2-Clause License. See
[LICENSE](https://github.com/NGWPC/nwm-coastal/blob/main/LICENSE) for details.
