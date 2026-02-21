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
coastal-calibration = { git = "https://github.com/NGWPC/nwm-coastal.git", tag = "v0.2.0", extras = ["sfincs", "plot"] }
hydromt-sfincs = { git = "https://github.com/Deltares/hydromt_sfincs", rev = "41aac0a3980fc2714ec28eafb0463d40abfc979a" }
EOF
```

### 3. Install

```bash
UV_CACHE_DIR=/var/tmp/uv-cache UV_LINK_MODE=copy pixi install
```

This creates a fully isolated environment under `/ngen-test/coastal-calibration/.pixi/`
with all conda and PyPI dependencies resolved together.

!!! note "NFS compatibility"

    Both `UV_LINK_MODE=copy` and `UV_CACHE_DIR` are needed on NFS shared filesystems.
    `UV_LINK_MODE=copy` prevents hardlink failures across filesystem boundaries, and
    `UV_CACHE_DIR` redirects uv's cache to a node-local directory to avoid lock-file and
    performance issues on NFS. We use `/var/tmp` rather than `$HOME` because the admin's
    home directory is not accessible to other users, and on some clusters compute nodes may
    not mount user home directories at all. `/var/tmp` is node-local, writable by all users,
    and persists across reboots (unlike `/tmp`).

### 4. Create a wrapper script

```bash
cat > /ngen-test/coastal-calibration/coastal-calibration <<'WRAPPER'
#!/bin/sh
export UV_CACHE_DIR="${UV_CACHE_DIR:-/var/tmp/uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
exec /ngen-test/coastal-calibration/.pixi/envs/default/bin/coastal-calibration "$@"
WRAPPER
chmod +x /ngen-test/coastal-calibration/coastal-calibration
```

The wrapper exports `UV_CACHE_DIR` and `UV_LINK_MODE` so that any internal `uv`
invocations at runtime also work correctly on NFS. The `${VAR:-default}` syntax
preserves any user-set values.

### 5. Make it available to all users

The wrapper script lives on the shared NFS filesystem, so it is already accessible from
every compute node. To make it available as a bare `coastal-calibration` command, add
the install directory to the system `PATH` on **all nodes** (login and compute) via a
profile drop-in:

```bash
sudo tee /etc/profile.d/coastal-calibration.sh > /dev/null <<'PROFILE'
export PATH="/ngen-test/coastal-calibration:$PATH"
PROFILE
```

On most clusters `/etc/profile.d/` is on a shared filesystem or provisioned identically
across nodes, so this single file makes the command available everywhere.

!!! warning "Node-local symlinks don't work"

    Do **not** symlink into `/usr/local/bin/` — that directory is node-local and will only
    exist on the node where the admin ran the command. Compute nodes launched by SLURM will
    not have the symlink and jobs will fail with `command not found`.

Alternatively, use the full NFS path directly in sbatch scripts (this always works
regardless of PATH setup):

```bash
/ngen-test/coastal-calibration/coastal-calibration run "${CONFIG_FILE}"
```

## Updating (when a new version is released)

Update the pinned tag in `pixi.toml` to the desired release, then reinstall:

```bash
cd /ngen-test/coastal-calibration
# Edit pixi.toml: update the `tag` for coastal-calibration to the new version
UV_CACHE_DIR=/var/tmp/uv-cache UV_LINK_MODE=copy pixi install
```

`UV_LINK_MODE=copy` is required on NFS shared filesystems where hardlinks (the default)
don't work across filesystem boundaries. `UV_CACHE_DIR` redirects the uv cache to a
local directory to avoid lock-file and performance issues on NFS.

## Verifying the installation

```bash
coastal-calibration --help
```

## Uninstalling

```bash
rm -rf /ngen-test/coastal-calibration
sudo rm -f /etc/profile.d/coastal-calibration.sh
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
