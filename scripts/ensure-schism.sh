#!/bin/bash
# Activation script: build SCHISM and supporting utilities if not yet
# installed in this environment.  Runs automatically on pixi env
# activation.  Subsequent activations skip instantly (file-existence check).
#
# Binaries installed to $CONDA_PREFIX/bin/:
#   pschism             — main SCHISM parallel executable
#   combine_hotstart7   — merges rank-specific hotstart files
#   combine_sink_source — combines NWM sinks with adjacent sources
#   metis_prep          — converts hgrid.gr3 to graphinfo for METIS
#   gpmetis             — METIS graph partitioner

# Fingerprint of the key compiled-against libraries: if any of these
# packages are updated, the binaries must be rebuilt against the new versions.
_dep_fingerprint() {
    local files
    files=$(ls "$CONDA_PREFIX/conda-meta/"libnetcdf-*.json \
               "$CONDA_PREFIX/conda-meta/"hdf5-*.json \
               "$CONDA_PREFIX/conda-meta/"openmpi-*.json \
               "$CONDA_PREFIX/conda-meta/"netcdf-fortran-*.json 2>/dev/null | sort)
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s' "$files" | sha256sum | cut -d' ' -f1
    else
        printf '%s' "$files" | shasum -a 256 | cut -d' ' -f1
    fi
}

FINGERPRINT_FILE="$CONDA_PREFIX/.schism_build_fingerprint"
CURRENT_FP=$(_dep_fingerprint)

if [ -x "$CONDA_PREFIX/bin/pschism" ] \
    && [ -x "$CONDA_PREFIX/bin/combine_hotstart7" ] \
    && [ -x "$CONDA_PREFIX/bin/combine_sink_source" ] \
    && [ -x "$CONDA_PREFIX/bin/metis_prep" ] \
    && [ -x "$CONDA_PREFIX/bin/gpmetis" ] \
    && [ "$(cat "$FINGERPRINT_FILE" 2>/dev/null)" = "$CURRENT_FP" ]; then
    return 0 2>/dev/null || exit 0
fi

echo "SCHISM binaries not found or dependencies changed — building …"
(
    set -e
    cd "$PIXI_PROJECT_ROOT"
    git submodule update --init schism

    SCHISM_SRC="$PIXI_PROJECT_ROOT/schism/src"
    BUILD_DIR="$PIXI_PROJECT_ROOT/schism/build"
    NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

    # ── 1. Build main SCHISM (pschism) + combine_hotstart7 via CMake ──
    _needs_configure=0
    if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
        _needs_configure=1
    elif ! grep -q "CMAKE_INSTALL_PREFIX:PATH=$CONDA_PREFIX" "$BUILD_DIR/CMakeCache.txt"; then
        echo "Stale CMake cache (wrong prefix) — reconfiguring …"
        rm -rf "$BUILD_DIR"
        _needs_configure=1
    fi

    if [ "$_needs_configure" -eq 1 ]; then
        cmake -S "$SCHISM_SRC" -B "$BUILD_DIR" \
            -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
            -DCMAKE_Fortran_COMPILER=mpifort \
            -DCMAKE_C_COMPILER=mpicc \
            -DCMAKE_CXX_COMPILER=mpicxx \
            -DCMAKE_BUILD_TYPE=Release \
            -DNO_PARMETIS=ON \
            -DTVD_LIM=VL \
            -DBLD_STANDALONE=ON \
            -DBUILD_TOOLS=ON \
            -DOLDIO=OFF \
            -DNetCDF_FORTRAN_DIR="$CONDA_PREFIX" \
            -DNetCDF_C_DIR="$CONDA_PREFIX"

        # On macOS, mpicc wraps Clang while mpifort wraps gfortran.
        # SCHISM's SCHISMCompile.cmake detects Clang and sets the
        # Fortran preprocessor flag to --preprocess, but gfortran
        # treats that as "preprocess only" (like -E) and does not
        # produce .mod files.  Fix: replace --preprocess with -cpp
        # in the generated build flags.
        if [ "$(uname -s)" = "Darwin" ]; then
            find "$BUILD_DIR" -name "flags.make" \
                -exec sed -i '' 's/--preprocess/-cpp/g' {} +
        fi
    fi

    echo "Building pschism …"
    make -C "$BUILD_DIR" -j"$NPROC" pschism

    echo "Building combine_hotstart7 …"
    make -C "$BUILD_DIR" -j"$NPROC" combine_hotstart7

    # CMake puts binaries in build/bin/ — copy to CONDA_PREFIX/bin/
    # pschism output name includes tags (e.g., pschism_NO_PARMETIS_TVD-VL)
    PSCHISM_BIN=$(find "$BUILD_DIR/bin" -name 'pschism*' -type f | head -1)
    if [ -z "$PSCHISM_BIN" ]; then
        echo "ERROR: pschism binary not found in $BUILD_DIR/bin/"
        exit 1
    fi
    cp "$PSCHISM_BIN" "$CONDA_PREFIX/bin/pschism"
    chmod +x "$CONDA_PREFIX/bin/pschism"

    if [ -f "$BUILD_DIR/bin/combine_hotstart7" ]; then
        cp "$BUILD_DIR/bin/combine_hotstart7" "$CONDA_PREFIX/bin/"
        chmod +x "$CONDA_PREFIX/bin/combine_hotstart7"
    else
        echo "ERROR: combine_hotstart7 not found in $BUILD_DIR/bin/"
        exit 1
    fi

    # ── 2. Build combine_sink_source (standalone Fortran, no deps) ──
    echo "Building combine_sink_source …"
    COMBINE_SRC="$SCHISM_SRC/Utility/Pre-Processing/NWM/NWM_coupling/combine_sink_source.F90"
    # gfortran 14+ rejects stop('...') — needs stop '...'
    COMBINE_PATCHED="$BUILD_DIR/combine_sink_source_patched.F90"
    sed "s/stop('\(.*\)')/stop '\1'/g" "$COMBINE_SRC" > "$COMBINE_PATCHED"
    gfortran -O2 -cpp -o "$CONDA_PREFIX/bin/combine_sink_source" "$COMBINE_PATCHED"
    chmod +x "$CONDA_PREFIX/bin/combine_sink_source"

    # ── 3. Build metis_prep (standalone Fortran, no deps) ──
    echo "Building metis_prep …"
    METIS_PREP_SRC="$SCHISM_SRC/Utility/Grid_Scripts/metis_prep.f90"
    # -mcmodel=medium is only supported on x86_64
    MCMODEL_FLAG=""
    if [ "$(uname -m)" = "x86_64" ]; then
        MCMODEL_FLAG="-mcmodel=medium"
    fi
    gfortran -O2 $MCMODEL_FLAG -o "$CONDA_PREFIX/bin/metis_prep" "$METIS_PREP_SRC"
    chmod +x "$CONDA_PREFIX/bin/metis_prep"

    # ── 4. Build gpmetis from metis-5.1.0 ──
    echo "Building gpmetis (METIS 5.1.0) …"
    METIS_DIR="$SCHISM_SRC/metis-5.1.0"
    cd "$METIS_DIR"

    # Check if already configured for the right prefix
    METIS_BUILD="$METIS_DIR/build/$(uname -s)-$(uname -m)"
    if [ -f "$METIS_BUILD/CMakeCache.txt" ] \
        && ! grep -q "CMAKE_INSTALL_PREFIX:PATH=$CONDA_PREFIX" "$METIS_BUILD/CMakeCache.txt"; then
        make distclean 2>/dev/null || true
    fi

    if [ ! -f "$METIS_BUILD/Makefile" ]; then
        make config prefix="$CONDA_PREFIX"
    fi
    make -j"$NPROC"
    make install
)

if [ $? -ne 0 ]; then
    echo "ERROR: SCHISM build failed — check the output above."
    return 1 2>/dev/null || exit 1
fi
echo "$CURRENT_FP" > "$FINGERPRINT_FILE"
echo "SCHISM build complete — binaries installed to $CONDA_PREFIX/bin/"
