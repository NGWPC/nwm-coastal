# Cluster Installation

This guide sets up `coastal-calibration` with compiled SFINCS and SCHISM model binaries
on a shared HPC cluster using [pixi](https://pixi.sh). All dependencies (including
system libraries like PROJ, GDAL, HDF5, NetCDF, MPI, and the Fortran compilers needed to
build the models) are fully isolated and managed by pixi. Nothing is installed into the
system Python or shared libraries.

Pixi is only needed at **install time** (compilation and dependency resolution). At
**runtime**, the wrapper script activates the pre-built environment directly -- pixi
does not need to be installed on compute nodes.

!!! important

    The install directory must be on the **shared filesystem** (e.g., NFS, Lustre) so that
    compute nodes can access it when jobs are submitted via Slurm.

## Prerequisites

Install pixi (v0.66+) on the **login node** (not needed on compute nodes):

```bash
curl -fsSL https://pixi.sh/install.sh | sudo PIXI_BIN_DIR=/usr/local/bin PIXI_NO_PATH_UPDATE=1 bash
```

Adjust `PIXI_BIN_DIR` if `/usr/local/bin` is not on the system `PATH`.

## Setup (one-time, by admin)

### 1. Clone the repository

Choose a directory on the shared filesystem visible to all compute nodes.
`--recurse-submodules` fetches the SFINCS and SCHISM source code needed for compilation:

```bash
cd <SHARED_DIR>
git clone --recurse-submodules https://github.com/NGWPC/nwm-coastal.git coastal-calibration
cd coastal-calibration
```

Replace `<SHARED_DIR>` with the appropriate path on your cluster's shared filesystem.

### 2. Install the pixi environment

The `dev` environment includes everything: Python CLI, SFINCS, SCHISM, ESMF/MPI, and
plotting dependencies:

```bash
pixi install -e dev
```

On first install, pixi-build compiles SFINCS and SCHISM from the submodules under
`coastal_models/` using `rattler-build` recipes. The compiled packages are cached as
`.conda` archives. Subsequent installs reuse the cache and complete in under a second.

**Binaries installed:**

| Binary                | Description                              |
| --------------------- | ---------------------------------------- |
| `sfincs`              | SFINCS coastal flooding model            |
| `pschism`             | SCHISM parallel ocean model              |
| `combine_hotstart7`   | Merges rank-specific hotstart files      |
| `combine_sink_source` | Combines NWM sinks with adjacent sources |
| `metis_prep`          | Converts hgrid.gr3 to METIS graph format |
| `gpmetis`             | METIS graph partitioner                  |

### 3. Create a wrapper script

The wrapper activates the pixi environment (setting `PATH`, `LD_LIBRARY_PATH`, and
sourcing conda activation scripts for MPI, ESMF, etc.) then runs `coastal-calibration`.
This means pixi itself is **not needed at runtime** -- the wrapper is a self-contained
entry point that works on any node that can see the shared filesystem.

```bash
INSTALL_DIR="$(pwd)"

cat > "${INSTALL_DIR}/nwm-coastal" <<WRAPPER
#!/bin/bash
set -eu

# Activate the pre-built pixi environment
_ENV="${INSTALL_DIR}/.pixi/envs/dev"

export PATH="\${_ENV}/bin:\${PATH:-}"
export LD_LIBRARY_PATH="\${_ENV}/lib:\${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="\${_ENV}"
export HDF5_USE_FILE_LOCKING=FALSE

# Source conda activation scripts (MPI, ESMF, GDAL, etc.)
for _script in "\${_ENV}"/etc/conda/activate.d/*.sh; do
    [ -f "\$_script" ] && . "\$_script"
done

exec coastal-calibration "\$@"
WRAPPER
chmod +x "${INSTALL_DIR}/nwm-coastal"
```

### 4. Make it available to all users

Add the install directory to the system `PATH` on **all nodes** (login and compute) via
a profile drop-in:

```bash
sudo tee /etc/profile.d/coastal-calibration.sh > /dev/null <<PROFILE
export PATH="${INSTALL_DIR}:\$PATH"
PROFILE
```

On most clusters `/etc/profile.d/` is on a shared filesystem or provisioned identically
across nodes, so this single file makes the command available everywhere.

!!! warning "Node-local symlinks don't work"

    Do **not** symlink into `/usr/local/bin/`. That directory is node-local and will only
    exist on the node where the admin ran the command. Compute nodes launched by Slurm will
    not have the symlink and jobs will fail with `command not found`.

Alternatively, skip the profile drop-in and use the full path to the wrapper directly in
`sbatch` scripts:

```bash
<SHARED_DIR>/coastal-calibration/nwm-coastal run "${CONFIG_FILE}"
```

______________________________________________________________________

## Running

No pixi needed. The wrapper script handles environment activation:

```bash
nwm-coastal run config.yaml
```

In Slurm job scripts:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

nwm-coastal run "${CONFIG_FILE}"
```

The wrapper ensures that `sfincs`, `pschism`, `mpiexec`, and all shared libraries are on
`PATH` / `LD_LIBRARY_PATH`, so `coastal-calibration` can spawn model subprocesses
directly.

______________________________________________________________________

## Updating

On the login node (where pixi is installed):

```bash
cd <SHARED_DIR>/coastal-calibration
git pull --recurse-submodules
pixi clean
pixi install -e dev
```

`pixi clean` removes all cached environments and compiled packages (`.pixi/envs/` and
`.pixi/build/`), forcing a clean rebuild. This avoids stale cache issues (e.g., model
binaries linked against a previous MPI version, corrupted solver state, or recipe
changes not being picked up). The full rebuild including SFINCS and SCHISM compilation
takes only a few minutes.

The wrapper script does not need updating -- it always points to the same environment
directory.

## Verifying the installation

```bash
nwm-coastal --help
nwm-coastal --version
```

Check that model binaries exist in the environment:

```bash
ls <SHARED_DIR>/coastal-calibration/.pixi/envs/dev/bin/{sfincs,pschism,mpiexec,gpmetis}
```

## Uninstalling

```bash
rm -rf <SHARED_DIR>/coastal-calibration
sudo rm -f /etc/profile.d/coastal-calibration.sh
```

______________________________________________________________________

## Using system-compiled model binaries

On clusters where SCHISM or SFINCS must be compiled against system MPI (e.g., WCOSS2
with Cray MPICH), the pixi environment provides only the Python runtime and libraries.
The model binaries are compiled separately using the system toolchain and referenced via
config:

```yaml
model_config:
  schism_exe: /path/to/system/pschism    # system-compiled SCHISM binary
  # or for SFINCS:
  sfincs_exe: /path/to/system/sfincs     # system-compiled SFINCS binary
  runtime_env:                            # optional: extra env vars for model run
    MPICH_ENV_DISPLAY: '1'
```

When `schism_exe` or `sfincs_exe` is set, the run stage automatically:

1. **Strips conda library paths** (`$CONDA_PREFIX/lib`) from `PATH` and
    `LD_LIBRARY_PATH` so the system binary finds system MPI/HDF5/NetCDF instead of
    conda's versions.
1. **Detects the MPI implementation** (`mpiexec --version`) and sets the correct tuning
    variables (MPICH `MPICH_OFI_STARTUP_CONNECT`, etc. for Cray MPICH; OpenMPI
    `OMPI_MCA_*` for OpenMPI).
1. **Applies `runtime_env`** overrides last, so any auto-detected value can be
    overridden.

Python MPI stages (ESMF regridding via `mpi4py`) continue using conda's OpenMPI -- they
are not affected by this isolation.

______________________________________________________________________

## How it works

- **pixi** (login node only) manages a fully isolated environment in `.pixi/` with all
    dependencies resolved together (conda + PyPI)
- **pixi-build** compiles SFINCS and SCHISM from source via `rattler-build` recipes in
    `coastal_models/`. Build dependencies (compilers, cmake, autotools) are resolved
    automatically and only present during compilation. The compiled packages are cached
    as `.conda` archives and reused across environments
- **conda-forge** provides system libraries (`proj`, `gdal`, `hdf5`, `netcdf`,
    `openmpi`) that would otherwise require `module load` or system package managers
- **MPI consistency**: both SFINCS and SCHISM link against MPI-enabled `hdf5` and
    `netcdf-fortran` (`mpi_openmpi_*` build variants), matching ESMF/esmpy's runtime
    expectations. On clusters with system MPI (e.g., WCOSS2 with Cray MPICH), use
    `schism_exe` / `sfincs_exe` config options for automatic environment isolation
- **MPI runtime detection**: at launch, `mpiexec --version` is parsed to identify the
    active MPI implementation (OpenMPI or MPICH/Cray MPICH). The correct tuning
    variables are set automatically. On AWS EFA instances, libfabric transport settings
    are added when `/sys/class/infiniband/efa*` devices are detected. On plain
    NFS/Lustre clusters without EFA, only general settings are applied (shared-memory on
    local `/tmp`, fork-warning suppression)
- **The wrapper script** activates the pre-built environment (`PATH`, `LD_LIBRARY_PATH`,
    conda activation scripts) and runs `coastal-calibration`. Pixi is not needed on
    compute nodes -- the wrapper is self-contained
- The install lives on the shared filesystem so all compute nodes can access it when
    running Slurm jobs
- Nothing is installed into the system Python, so the cluster's existing software is
    completely unaffected
