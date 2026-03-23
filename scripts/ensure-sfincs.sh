#!/bin/bash
# Activation script: build SFINCS if the binary is not yet installed
# in this environment. Runs automatically on pixi env activation.
# Subsequent activations skip instantly (just a file-existence check).

# Fingerprint of the key compiled-against libraries: if any of these
# packages are updated, the binary must be rebuilt against the new versions.
_dep_fingerprint() {
    local files
    files=$(ls "$CONDA_PREFIX/conda-meta/"libnetcdf-*.json \
               "$CONDA_PREFIX/conda-meta/"hdf5-*.json \
               "$CONDA_PREFIX/conda-meta/"libgfortran-*.json 2>/dev/null | sort)
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s' "$files" | sha256sum | cut -d' ' -f1
    else
        printf '%s' "$files" | shasum -a 256 | cut -d' ' -f1
    fi
}

FINGERPRINT_FILE="$CONDA_PREFIX/.sfincs_build_fingerprint"
CURRENT_FP=$(_dep_fingerprint)

if [ -x "$CONDA_PREFIX/bin/sfincs" ] \
    && [ "$(cat "$FINGERPRINT_FILE" 2>/dev/null)" = "$CURRENT_FP" ]; then
    return 0 2>/dev/null || exit 0
fi

echo "SFINCS binary not found or dependencies changed — building …"
(
    set -e
    cd "$PIXI_PROJECT_ROOT"
    git submodule update --init SFINCS
    cd SFINCS/source

    # Reconfigure if Makefile is missing or was configured with a different prefix
    _needs_configure=0
    if [ ! -f Makefile ]; then
        _needs_configure=1
    elif ! grep -Fxq "prefix = $CONDA_PREFIX" Makefile; then
        echo "Stale Makefile (wrong prefix) — reconfiguring …"
        make distclean 2>/dev/null || true
        _needs_configure=1
    fi

    if [ "$_needs_configure" -eq 1 ]; then
        autoreconf -vif
        FCFLAGS='-fopenmp -O3 -fallow-argument-mismatch -w' \
        FFLAGS='-fopenmp -O3 -fallow-argument-mismatch -w' \
        ./configure --prefix="$CONDA_PREFIX" --disable-shared --disable-openacc
    fi

    make -j1
    make install
)

if [ $? -ne 0 ]; then
    echo "ERROR: SFINCS build failed — check the output above."
    return 1 2>/dev/null || exit 1
fi
echo "$CURRENT_FP" > "$FINGERPRINT_FILE"
echo "SFINCS build complete — binary installed to $CONDA_PREFIX/bin/sfincs"
