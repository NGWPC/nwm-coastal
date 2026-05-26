# Compiling SFINCS from Source

SFINCS is included as a Git submodule under `coastal_models/sfincs/repo/`. The
`sfincs_run` workflow stage requires the `sfincs` binary to be on `PATH` (or pointed to
via the `sfincs_exe` config option).

______________________________________________________________________

## Using Pixi (Recommended)

SFINCS is built as a **pixi-build package** using the `rattler-build` backend. The
recipe lives at `coastal_models/sfincs/` with three files:

| File          | Purpose                                                        |
| ------------- | -------------------------------------------------------------- |
| `pixi.toml`   | Package metadata and build backend declaration                 |
| `recipe.yaml` | Conda recipe (source path, build/host/run dependencies, tests) |
| `build.sh`    | Build script (autotools configure + make, macOS SDK probe)     |

Any pixi environment that includes the `sfincs` feature (**`sfincs`**, **`dev`**,
**`test311`**, **`test313`**) will **automatically build and install the `sfincs` conda
package** on first `pixi install`. Subsequent runs use the cached package and complete
instantly (~0.3 s) unless the submodule source or recipe changes.

The binary is installed into the active pixi environment (`$CONDA_PREFIX/bin/sfincs`),
so it is immediately available on `PATH` when running inside that environment.

### Key recipe details

- **Source**: `./repo/source` (the SFINCS autotools tree inside the submodule)
- **MPI variants**: `hdf5` and `libnetcdf` are pinned to `mpi_openmpi_*` builds so that
    SFINCS, SCHISM, and ESMF share the same MPI-linked libraries at runtime
- **macOS SDK probe**: `scripts/find_compatible_sdk.sh` dynamically tests each installed
    SDK and picks the newest one compatible with conda-forge's linker
- **Parallel make disabled**: SFINCS's `Makefile.am` has incomplete Fortran module
    dependency declarations, so the build uses `make -j1`

______________________________________________________________________

## Manual Build

If you prefer to build outside pixi, follow the platform-specific instructions below.
The resulting binary has no runtime library dependencies -- just copy it wherever you
need it and run.

### Ubuntu

#### Install Dependencies

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  gfortran \
  autoconf \
  automake \
  libtool \
  m4 \
  pkg-config \
  libnetcdf-dev
```

#### Build

```bash
cd coastal_models/sfincs/repo/source

# Generate the configure script (only needed when building from the git repo)
autoreconf -vif

# -fallow-argument-mismatch is needed for netcdf-fortran with GCC >= 10
# see https://github.com/Unidata/netcdf-fortran/issues/212
export FCFLAGS="-fopenmp -O3 -fallow-argument-mismatch -w"
export FFLAGS="-fopenmp -O3 -fallow-argument-mismatch -w"

# Only for older SFINCS versions bundling netcdf-fortran 4.4.4:
# GCC >= 14 errors on implicit function declarations in its configure tests.
# Also need explicit netCDF paths since 4.4.4 doesn't pick them up from pkg-config,
# and parallel make must be disabled (use make -j1) due to missing Fortran module
# dependency declarations in the Makefile.
# export CFLAGS="-Wno-error=implicit-function-declaration"
# export CPPFLAGS="-I$(nc-config --includedir)"
# export LDFLAGS="-L$(nc-config --libdir)"

CC=gcc FC=gfortran ./configure --prefix=$HOME/.local --disable-shared --disable-openacc
make -j1
make install
```

The statically linked executable is at `~/.local/bin/sfincs`. Verify with:

```bash
file ~/.local/bin/sfincs   # should show "statically linked"
ldd ~/.local/bin/sfincs    # should print "not a dynamic executable"
```

______________________________________________________________________

### macOS

> **Note:** macOS does not support fully static executables (Apple does not ship static
> versions of system libraries). The instructions below statically link all non-system
> libraries (netCDF, HDF5, gfortran runtime, etc.) so the binary is as self-contained as
> possible.

#### Install Dependencies

Install [Homebrew](https://brew.sh) if not already present:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install packages:

```bash
brew install gcc autoconf automake libtool pkg-config netcdf
```

#### Build

Homebrew installs GCC as versioned commands (`gcc-14`, `gfortran-14`, etc.). The snippet
below detects the version automatically:

```bash
cd coastal_models/sfincs/repo/source
autoreconf -vif

# Detect Homebrew GCC version
GCC_VERSION=$(brew list --versions gcc | grep -oE '[0-9]+' | head -1)

export FCFLAGS="-fopenmp -O3 -fallow-argument-mismatch -w"
export FFLAGS="-fopenmp -O3 -fallow-argument-mismatch -w"

# Enable zstd support in netcdf-fortran (Homebrew netCDF is built with zstd)
export HDF5_PLUGIN_PATH="$(nc-config --plugindir)"

# Only for older SFINCS versions bundling netcdf-fortran 4.4.4:
# GCC >= 14 errors on implicit function declarations in its configure tests.
# Also need explicit netCDF paths since 4.4.4 doesn't pick them up from pkg-config,
# and parallel make must be disabled (use make -j1) due to missing Fortran module
# dependency declarations in the Makefile.
# export CFLAGS="-Wno-error=implicit-function-declaration"
# export CPPFLAGS="-I$(nc-config --includedir)"
# export LDFLAGS="-L$(nc-config --libdir)"

CC=gcc-${GCC_VERSION} FC=gfortran-${GCC_VERSION} \
  ./configure --prefix=$HOME/.local --disable-shared --disable-openacc
make -j1
make install
```

The executable is at `~/.local/bin/sfincs`.

> **Apple Silicon (M1/M2/M3/M4):** Homebrew installs to `/opt/homebrew`. If `pkg-config`
> cannot find `netcdf`, export the path before running `configure`:
>
> ```bash
> export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
> ```

______________________________________________________________________

## Verifying the Build

A minimal smoke test is included in the repository:

```bash
cd coastal_models/sfincs/repo/source
make test   # runs test/01_noadv
```

______________________________________________________________________

## Quick Reference

| Item             | Ubuntu                         | macOS (Homebrew)                    |
| ---------------- | ------------------------------ | ----------------------------------- |
| Fortran compiler | `gfortran` (via `apt`)         | `gfortran` (via `brew install gcc`) |
| C compiler       | `gcc` (via `build-essential`)  | `gcc-N` (via `brew install gcc`)    |
| Autotools        | `autoconf automake libtool m4` | `autoconf automake libtool`         |
| NetCDF C library | `libnetcdf-dev`                | `netcdf`                            |
| pkg-config       | `pkg-config`                   | `pkg-config`                        |
