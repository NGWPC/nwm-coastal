# Installing `coastal-calibration` on a Shared Cluster

This guide sets up `coastal-calibration` as a globally available CLI tool on a shared
cluster using [pixi](https://pixi.sh). All dependencies (including system libraries like
PROJ, GDAL, HDF5, and NetCDF) are fully isolated and managed by pixi — nothing is
installed into the system Python or shared libraries.

## Prerequisites

Install pixi on the cluster if it's not already available:

```bash
curl -fsSL https://pixi.sh/install.sh | sudo PIXI_BIN_DIR=/usr/local/bin PIXI_NO_PATH_UPDATE=1 bash
```

This assume that `/usr/local/bin` is in the system `PATH` for all users. If not, adjust
`PIXI_BIN_DIR` accordingly and ensure that the wrapper script created later is symlinked
into a directory that is in the `PATH`.

## Setup (one-time, by admin)

### 1. Create the project directory

```bash
sudo mkdir -p /opt/coastal-calibration
sudo chown $(whoami) /opt/coastal-calibration
cd /opt/coastal-calibration
```

### 2. Create `pixi.toml`

```bash
cat > pixi.toml <<'EOF'
[workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = "~=3.14.0"
proj = "*"
libgdal-core = "*"
hdf5 = "*"
libnetcdf = "*"
ffmpeg = "*"

[pypi-dependencies]
coastal-calibration = { git = "https://github.com/cheginit/nwm-coastal.git", extras = ["sfincs", "plot"] }
EOF
```

### 3. Install

By default, `uv` hardlinks files from its cache into `site-packages`. Singularity/Apptainer
cannot resolve these hardlinks when bind-mounting the bundled scripts directory into the
container. Setting `UV_LINK_MODE=copy` forces `uv` to copy the files so they exist as real
files in `site-packages`.

```bash
UV_LINK_MODE=copy pixi install
```

This creates a fully isolated environment under `/opt/coastal-calibration/.pixi/` with
all conda and PyPI dependencies resolved together.

### 5. Create a wrapper script

```bash
cat > /opt/coastal-calibration/coastal-calibration <<'WRAPPER'
#!/bin/sh
exec /opt/coastal-calibration/.pixi/envs/default/bin/coastal-calibration "$@"
WRAPPER
chmod +x /opt/coastal-calibration/coastal-calibration
```

### 6. Make it available to all users

Symlink into a shared bin directory:

```bash
sudo ln -sf /opt/coastal-calibration/coastal-calibration /usr/local/bin/coastal-calibration
```

## Updating (when a new version is pushed)

```bash
cd /opt/coastal-calibration
UV_LINK_MODE=copy pixi update
```

That's it — one command. All conda and PyPI dependencies are re-resolved and the
`coastal-calibration` CLI is updated in place.

## Verifying the installation

```bash
coastal-calibration --help
```

## Uninstalling

```bash
sudo rm -rf /opt/coastal-calibration
sudo rm -f /usr/local/bin/coastal-calibration
```

## How it works

- **pixi** manages an isolated environment in `/opt/coastal-calibration/.pixi/`
- **conda-forge** provides system libraries (`proj`, `gdal`, `hdf5`, `netcdf`) that
    would otherwise require `module load` or system package managers
- **PyPI** provides the Python package (`coastal-calibration`) and its Python
    dependencies, installed from the Git repository
- The wrapper script calls the binary directly from the isolated environment, so users
    don't need pixi installed or any knowledge of the environment
- Nothing is installed into the system Python — the cluster's existing software is
    completely unaffected
