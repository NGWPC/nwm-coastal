#!/bin/bash
set -euo pipefail

NPROC="${CPU_COUNT:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)}"

# ── macOS SDK compatibility ───────────────────────────────────────────
# Probe installed SDKs for one whose TBD stubs work with conda's linker.
source "$RECIPE_DIR/../../scripts/find_compatible_sdk.sh" || true

# ── 1. Build main SCHISM (pschism) + combine_hotstart7 via CMake ─────
BUILD_DIR="$SRC_DIR/_build"

cmake -S "$SRC_DIR/src" -B "$BUILD_DIR" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_Fortran_COMPILER=mpifort \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_BUILD_TYPE=Release \
    -DNO_PARMETIS=ON \
    -DTVD_LIM=VL \
    -DBLD_STANDALONE=ON \
    -DBUILD_TOOLS=ON \
    -DOLDIO=OFF \
    -DNetCDF_FORTRAN_DIR="$PREFIX" \
    -DNetCDF_C_DIR="$PREFIX"

# On macOS (conda-forge), mpicc wraps Clang while mpifort wraps gfortran.
# SCHISM's SCHISMCompile.cmake detects Clang and sets the Fortran
# preprocessor flag to --preprocess, but gfortran treats that as
# "preprocess only" (like -E) and does not produce .mod files.
# Fix: replace --preprocess with -cpp in the generated build flags.
if [ "$(uname -s)" = "Darwin" ]; then
    find "$BUILD_DIR" -name "flags.make" \
        -exec sed -i '' 's/--preprocess/-cpp/g' {} +
fi

echo "Building pschism ..."
make -C "$BUILD_DIR" -j"$NPROC" pschism

echo "Building combine_hotstart7 ..."
make -C "$BUILD_DIR" -j"$NPROC" combine_hotstart7

# CMake puts binaries in build/bin/ — copy to $PREFIX/bin/
# pschism output name includes tags (e.g., pschism_NO_PARMETIS_TVD-VL)
PSCHISM_BIN=$(find "$BUILD_DIR/bin" -name 'pschism*' -type f | head -1)
if [ -z "$PSCHISM_BIN" ]; then
    echo "ERROR: pschism binary not found in $BUILD_DIR/bin/"
    exit 1
fi
cp "$PSCHISM_BIN" "$PREFIX/bin/pschism"
chmod +x "$PREFIX/bin/pschism"

cp "$BUILD_DIR/bin/combine_hotstart7" "$PREFIX/bin/"
chmod +x "$PREFIX/bin/combine_hotstart7"

# ── 2. Build combine_sink_source (standalone Fortran, no deps) ───────
echo "Building combine_sink_source ..."
COMBINE_SRC="$SRC_DIR/src/Utility/Pre-Processing/NWM/NWM_coupling/combine_sink_source.F90"
# gfortran 14+ rejects stop('...') — needs stop '...'
COMBINE_PATCHED="$BUILD_DIR/combine_sink_source_patched.F90"
sed "s/stop('\(.*\)')/stop '\1'/g" "$COMBINE_SRC" > "$COMBINE_PATCHED"
"${FC:-gfortran}" -O2 -cpp -o "$PREFIX/bin/combine_sink_source" "$COMBINE_PATCHED"
chmod +x "$PREFIX/bin/combine_sink_source"

# ── 3. Build metis_prep (standalone Fortran, no deps) ────────────────
echo "Building metis_prep ..."
METIS_PREP_SRC="$SRC_DIR/src/Utility/Grid_Scripts/metis_prep.f90"
# -mcmodel=medium is only supported on x86_64
MCMODEL_FLAG=""
if [ "$(uname -m)" = "x86_64" ]; then
    MCMODEL_FLAG="-mcmodel=medium"
fi
"${FC:-gfortran}" -O2 $MCMODEL_FLAG -o "$PREFIX/bin/metis_prep" "$METIS_PREP_SRC"
chmod +x "$PREFIX/bin/metis_prep"

# ── 4. Build gpmetis from metis-5.1.0 ───────────────────────────────
echo "Building gpmetis (METIS 5.1.0) ..."
METIS_DIR="$SRC_DIR/src/metis-5.1.0"
cd "$METIS_DIR"
make config prefix="$PREFIX"
make -j"$NPROC"
make install
