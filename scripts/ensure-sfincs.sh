#!/bin/bash
# Activation script: build SFINCS if the binary is not yet installed
# in this environment. Runs automatically on pixi env activation.
# Subsequent activations skip instantly (just a file-existence check).

if [ -x "$CONDA_PREFIX/bin/sfincs" ]; then
    return 0 2>/dev/null || exit 0
fi

echo "SFINCS binary not found in this environment — building …"
(
    set -e
    cd "$PIXI_PROJECT_ROOT"
    git submodule update --init SFINCS
    cd SFINCS/source
    if [ ! -f Makefile ]; then
        autoreconf -vif
        FCFLAGS='-fopenmp -O3 -fallow-argument-mismatch -w' \
        FFLAGS='-fopenmp -O3 -fallow-argument-mismatch -w' \
        ./configure --prefix="$CONDA_PREFIX" --disable-shared --disable-openacc
    fi
    make -j1
    make install
)
echo "SFINCS build complete — binary installed to $CONDA_PREFIX/bin/sfincs"
