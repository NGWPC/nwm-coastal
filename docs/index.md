# NWM Coastal: Coastal Model Workflow

`nwm-coastal` is a collection of tools designed to implement skill assessments for
coastal model runs and execute National Water Model (NWM) hindcast/forecast runs for
standalone coastal models utilized in NWMv4 operations. This includes the Semi-implicit
Cross-scale Hydroscience Integrated System Model (SCHISM) and Super-Fast INundation of
CoastS (SFINCS) as the intended coastal models for implementation in NWMv4 operations.
This repository stores symlinks to both coastal modeling development groups, which are
used to compile and run the NWMv4 coastal models. The coastal modeling output for NWMv4
operations is the Total Water Level (TWL) fields. These fields are the focus of the
coastal tools in this repository for skill assessments across NWM domains, as well as
the end result to retrieve from tools in this repository that set up and execute NWMv4
hindcast and operational forecast runs.

## Features

- **Multi-Model Support**: SCHISM (multi-node MPI) and SFINCS (single-node OpenMP) via a
    polymorphic `ModelConfig` architecture
- **YAML Configuration**: Simple, human-readable configuration files with variable
    interpolation
- **Data Download**: Automated download of NWM and STOFS boundary data
- **Multiple Domains**: Support for Hawaii, Puerto Rico/Virgin Islands, Atlantic/Gulf,
    and Pacific
- **Boundary Conditions**: TPXO tidal model and STOFS water level support
- **NOAA Observation Stations**: Automatic discovery of CO-OPS water level stations
    within the model domain, with post-run comparison plots (simulated vs observed)
- **SFINCS Model Creation**: Build SFINCS quadtree models from an AOI polygon with
    automatic NOAA DEM discovery, elevation/bathymetry, boundary cells, and subgrid
    tables
- **Workflow Control**: `run` pipeline with `--start-from` / `--stop-after` support for
    partial workflows
- **Configuration Inheritance**: Share common settings across multiple runs

## Quick Example

On clusters, the recommended approach is to use a heredoc `sbatch` script:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=coastal_schism
#SBATCH --partition=c5n-18xlarge
#SBATCH -N 2
#SBATCH --ntasks-per-node=18
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out

CONFIG_FILE="/tmp/coastal_config_${SLURM_JOB_ID}.yaml"

cat > "${CONFIG_FILE}" <<'EOF'
simulation:
  start_date: 2021-06-11
  duration_hours: 24
  coastal_domain: hawaii
  meteo_source: nwm_ana

boundary:
  source: stofs
EOF

coastal-calibration run "${CONFIG_FILE}"
rm -f "${CONFIG_FILE}"
```

Submit with `sbatch my_run.sh`. The full NFS path ensures the command is found on
compute nodes. See the [Quick Start](getting-started/quickstart.md) for more options.

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
