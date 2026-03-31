#!/bin/bash
set -euo pipefail

# ── macOS SDK compatibility ───────────────────────────────────────────
# Probe installed SDKs for one whose TBD stubs work with conda's linker.
# The script is in the repo root but $RECIPE_DIR points to packages/sfincs.
source "$RECIPE_DIR/../../scripts/find_compatible_sdk.sh" || true

# ── Clean stale build artifacts ──────────────────────────────────────
make distclean 2>/dev/null || true

# ── Configure ─────────────────────────────────────────────────────────
autoreconf -vif

# Conda-forge compilers target an older darwin SDK (e.g. darwin20)
# while the running system may report a newer one (e.g. darwin25).
# Pass --build/--host so configure doesn't falsely detect cross-compilation.
_cc_triplet=$("${CC:-cc}" -dumpmachine 2>/dev/null || true)

FCFLAGS='-fopenmp -O3 -fallow-argument-mismatch -w' \
FFLAGS='-fopenmp -O3 -fallow-argument-mismatch -w' \
./configure --prefix="$PREFIX" --disable-shared --disable-openacc \
    ${_cc_triplet:+--build="$_cc_triplet" --host="$_cc_triplet"}

# ── Build & install ──────────────────────────────────────────────────
# SFINCS's Fortran module dependencies aren't fully declared in Makefile.am,
# so parallel make can fail with "Cannot open module file" errors.
make -j1
make install
