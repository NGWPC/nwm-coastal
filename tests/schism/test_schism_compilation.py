"""Tests for pixi-compiled SCHISM binaries.

These tests require the ``schism`` pixi environment to be active
(``pixi run -e schism ...``) so that pschism, metis_prep, gpmetis,
combine_hotstart7, and combine_sink_source are on ``$CONDA_PREFIX/bin``.

Run with::

    pixi run -e schism test-schism
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.schism.schism_testkit import generate_test_case, run_schism

pytestmark = pytest.mark.schism

CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")
EXEC_DIR = Path(CONDA_PREFIX) / "bin" if CONDA_PREFIX else None

REQUIRED_BINARIES = ["pschism", "metis_prep", "gpmetis", "combine_hotstart7", "combine_sink_source"]


def _binaries_available() -> bool:
    if EXEC_DIR is None or not EXEC_DIR.exists():
        return False
    return all((EXEC_DIR / name).exists() for name in REQUIRED_BINARIES)


skip_no_binaries = pytest.mark.skipif(
    not _binaries_available(),
    reason="SCHISM binaries not found in $CONDA_PREFIX/bin (run in pixi schism env)",
)


@pytest.fixture
def schism_test_case(tmp_path):
    """Generate a minimal SCHISM test case in a temp directory."""
    project_dir = tmp_path / "schism_test"
    generate_test_case(
        grid_size=(11, 11),
        resolution=(100.0, 100.0),
        boundary_type="shore",
        base_dir=project_dir,
        depth=1.0,
        run_days=1.0,
        dt=100.0,
        tidal_amplitude=1.0,
        tidal_period_hours=12.42,
        station_output=True,
    )
    return project_dir


@skip_no_binaries
class TestSchismBinaries:
    """Verify that the pixi-compiled SCHISM binaries exist and are executable."""

    @pytest.mark.parametrize("binary", REQUIRED_BINARIES)
    def test_binary_exists_and_executable(self, binary):
        path = EXEC_DIR / binary
        assert path.exists(), f"{binary} not found at {path}"
        assert os.access(path, os.X_OK), f"{binary} is not executable"


@skip_no_binaries
class TestSchismRun:
    """Run SCHISM on a minimal test case and validate output."""

    def test_tidal_amplitude(self, schism_test_case):
        """Generate a tidal test case, run pschism, and verify ~1m amplitude."""
        run_schism(
            project_dir=schism_test_case,
            exec_dir=EXEC_DIR,
            num_procs=4,
            num_scribes=2,
            mpi_args=["--oversubscribe"],
            timeout=120,
        )

        staout = schism_test_case / "outputs" / "staout_1"
        assert staout.exists(), "staout_1 not produced"
        data = np.loadtxt(staout)
        assert data.shape[0] > 0, "staout_1 has no data rows"

        # Column 0 is time, column 1 is elevation at the station
        elevations = data[:, 1]
        max_amplitude = np.max(np.abs(elevations))
        assert max_amplitude > 0.5, f"Max amplitude {max_amplitude:.3f}m is too low (expected ~1m)"
        assert max_amplitude < 2.0, f"Max amplitude {max_amplitude:.3f}m is too high (expected ~1m)"

    def test_run_fails_with_missing_inputs(self, tmp_path):
        """run_schism raises FileNotFoundError for missing input files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Missing input files"):
            run_schism(project_dir=empty_dir, exec_dir=EXEC_DIR)
