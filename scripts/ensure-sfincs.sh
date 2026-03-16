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

    # Reconfigure if Makefile is missing or was configured with a different prefix
    _needs_configure=0
    if [ ! -f Makefile ]; then
        _needs_configure=1
    elif ! grep -q "^prefix = $CONDA_PREFIX\$" Makefile; then
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
echo "SFINCS build complete — binary installed to $CONDA_PREFIX/bin/sfincs"
