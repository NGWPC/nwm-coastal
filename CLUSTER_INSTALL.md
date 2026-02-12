# Installing `coastal-calibration` on a Shared Cluster

This guide sets up `coastal-calibration` as a globally available CLI tool on a shared
cluster using [pixi](https://pixi.sh). All dependencies (including system libraries like
PROJ, GDAL, HDF5, and NetCDF) are fully isolated and managed by pixi — nothing is
installed into the system Python or shared libraries.

**Important:** The install directory must be on the **shared filesystem** (e.g., NFS) so
that compute nodes can access it when jobs are submitted via Slurm.

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

The directory **must** be on the shared filesystem visible to all compute nodes:

```bash
mkdir -p /ngen-test/coastal-calibration
cd /ngen-test/coastal-calibration
```

### 2. Create `pixi.toml`

```bash
cat > pixi.toml <<'EOF'
[workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = "~=3.14.0"
uv = "*"
proj = "*"
libgdal-core = "*"
hdf5 = "*"
libnetcdf = "*"
ffmpeg = "*"

[pypi-dependencies]
coastal-calibration = { git = "https://github.com/cheginit/nwm-coastal.git", extras = ["sfincs", "plot"] }
hydromt-sfincs = { git = "https://github.com/Deltares/hydromt_sfincs", rev = "41aac0a3980fc2714ec28eafb0463d40abfc979a" }
EOF
```

### 3. Install

```bash
UV_LINK_MODE=copy pixi install
```

This creates a fully isolated environment under `/ngen-test/coastal-calibration/.pixi/`
with all conda and PyPI dependencies resolved together.

### 4. Create a wrapper script

```bash
cat > /ngen-test/coastal-calibration/coastal-calibration <<'WRAPPER'
#!/bin/sh
exec /ngen-test/coastal-calibration/.pixi/envs/default/bin/coastal-calibration "$@"
WRAPPER
chmod +x /ngen-test/coastal-calibration/coastal-calibration
```

### 5. Make it available to all users

Symlink into a shared bin directory:

```bash
sudo ln -sf /ngen-test/coastal-calibration/coastal-calibration /usr/local/bin/coastal-calibration
```

## Updating (when a new version is pushed)

```bash
cd /ngen-test/coastal-calibration
UV_LINK_MODE=copy pixi update
UV_LINK_MODE=copy pixi run uv pip install --reinstall-package coastal-calibration \
  "coastal-calibration[sfincs,plot] @ git+https://github.com/cheginit/nwm-coastal.git"
```

`UV_LINK_MODE=copy` is required on NFS shared filesystems where hardlinks (the default)
don't work across filesystem boundaries. The `--reinstall-package` flag forces `uv` to
re-fetch and rebuild the package from the latest commit on the remote repository.

## Verifying the installation

```bash
coastal-calibration --help
```

## Uninstalling

```bash
rm -rf /ngen-test/coastal-calibration
sudo rm -f /usr/local/bin/coastal-calibration
```

## How it works

- **pixi** manages an isolated environment in `/ngen-test/coastal-calibration/.pixi/`
- **conda-forge** provides system libraries (`proj`, `gdal`, `hdf5`, `netcdf`) that
    would otherwise require `module load` or system package managers
- **PyPI** provides the Python package (`coastal-calibration`) and its Python
    dependencies, installed from the Git repository
- The wrapper script calls the binary directly from the isolated environment, so users
    don't need pixi installed or any knowledge of the environment
- The install lives on the shared filesystem (`/ngen-test`) so all compute nodes can
    access it when running Slurm jobs
- Nothing is installed into the system Python — the cluster's existing software is
    completely unaffected
