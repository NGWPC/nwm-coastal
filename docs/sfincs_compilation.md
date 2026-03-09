# Compiling SFINCS from Source

SFINCS is included as a Git submodule under `SFINCS/`. The `sfincs_run`
workflow stage requires the `sfincs` binary to be on `PATH` (or pointed
to via the `sfincs_exe` config option).

---

## Using Pixi (Recommended)

All build dependencies (compilers, autotools, NetCDF) are declared in the
`sfincs` pixi feature. Any environment that includes this feature
(**`sfincs`**, **`dev`**, **`test311`**, **`test314`**) will
**automatically compile SFINCS** on first activation if the binary is
not already present. Subsequent activations skip the build instantly
(just a file-existence check).

The binary is installed into the active pixi environment
(`$CONDA_PREFIX/bin/sfincs`), so it is immediately available on `PATH`
when running inside that environment. The build is **incremental** --
if you have already configured once, subsequent runs skip `autoreconf`
and `configure` and only recompile changed source files.

---

## Manual Build

If you prefer to build outside pixi, follow the platform-specific
instructions below. The resulting binary has no runtime library
dependencies -- just copy it wherever you need it and run.

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
cd SFINCS/source

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

---

### macOS

> **Note:** macOS does not support fully static executables (Apple does not ship
> static versions of system libraries). The instructions below statically link
> all non-system libraries (netCDF, HDF5, gfortran runtime, etc.) so the binary
> is as self-contained as possible.

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

Homebrew installs GCC as versioned commands (`gcc-14`, `gfortran-14`, etc.).
The snippet below detects the version automatically:

```bash
cd SFINCS/source
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

> **Apple Silicon (M1/M2/M3/M4):** Homebrew installs to `/opt/homebrew`. If
> `pkg-config` cannot find `netcdf`, export the path before running `configure`:
>
> ```bash
> export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
> ```

---

## Verifying the Build

A minimal smoke test is included in the repository:

```bash
cd SFINCS/source
make test   # runs test/01_noadv
```

---

## Quick Reference

| Item | Ubuntu | macOS (Homebrew) |
| ---- | ------ | ---------------- |
| Fortran compiler | `gfortran` (via `apt`) | `gfortran` (via `brew install gcc`) |
| C compiler | `gcc` (via `build-essential`) | `gcc-N` (via `brew install gcc`) |
| Autotools | `autoconf automake libtool m4` | `autoconf automake libtool` |
| NetCDF C library | `libnetcdf-dev` | `netcdf` |
| pkg-config | `pkg-config` | `pkg-config` |
