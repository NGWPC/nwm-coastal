"""Tests for the SCHISM GLOFS boundary (make_glofs_boundary and stage routing)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import netCDF4
import numpy as np
import pytest

from coastal_calibration.config.schema import (
    BoundaryConfig,
    CoastalCalibConfig,
    DownloadConfig,
    PathConfig,
    SchismModelConfig,
    SimulationConfig,
)
from coastal_calibration.data.glofs import _write_geodataset
from coastal_calibration.schism.boundary import BoundaryConditionStage, GLOFSBoundaryStage
from coastal_calibration.schism.prep import make_glofs_boundary
from coastal_calibration.schism.project_reader import NWMSCHISMProject
from tests.schism.schism_testkit import generate_test_case

if TYPE_CHECKING:
    from pathlib import Path

    from coastal_calibration.config.schema import BoundarySource

START = datetime(2025, 6, 10)
DURATION = 3


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Build a 9x7 geographic mesh with one open boundary."""
    d = tmp_path / "mesh"
    generate_test_case(
        grid_size=(9, 7),
        resolution=(1.0, 1.0),
        boundary_type="shore",
        base_dir=d,
        station_output=False,
    )
    return d


def _waterlevel(tmp_path: Path, lon_shift: float = 0.0, hours: int = DURATION + 1) -> Path:
    """GLOFS-style merged file: a dense point cloud with a uniform level per hour."""
    gx, gy = np.meshgrid(np.linspace(-1, 9, 41), np.linspace(-1, 7, 33))
    lon, lat = gx.ravel() + lon_shift, gy.ravel()
    times = [START + timedelta(hours=k) for k in range(hours)]
    zeta = np.array([np.full(lon.size, 0.5 + 0.1 * k) for k in range(hours)], dtype=np.float32)
    path = tmp_path / "glofs_merged.nc"
    _write_geodataset(path, times, lon, lat, zeta, "leofs")
    return path


def _open_nodes(project: Path) -> int:
    return NWMSCHISMProject(project, validate=False).read_boundaries().total_open_nodes


class TestMakeGlofsBoundary:
    def test_writes_offset_levels_for_every_open_node(self, project, tmp_path):
        out = make_glofs_boundary(
            work_dir=tmp_path,
            prebuilt_dir=project,
            waterlevel_file=_waterlevel(tmp_path),
            start_date=START,
            duration_hours=DURATION,
            offset_m=173.5,
        )
        with netCDF4.Dataset(out) as ds:
            ts = ds["time_series"][:]
            assert ts.shape == (DURATION + 1, _open_nodes(project), 1, 1)
            np.testing.assert_array_equal(ds["time"][:], [0.0, 3600.0, 7200.0, 10800.0])
            assert float(ds["time_step"][0]) == 3600.0
        expected = 173.5 + 0.5 + 0.1 * np.arange(DURATION + 1)
        want = np.tile(expected[:, None], ts.shape[1])
        np.testing.assert_allclose(ts[:, :, 0, 0], want, atol=1e-5)

    def test_rejects_glofs_from_another_lake(self, project, tmp_path):
        with pytest.raises(ValueError, match="glofs_model matches"):
            make_glofs_boundary(
                work_dir=tmp_path,
                prebuilt_dir=project,
                waterlevel_file=_waterlevel(tmp_path, lon_shift=20.0),
                start_date=START,
                duration_hours=DURATION,
            )

    def test_rejects_file_for_a_different_window(self, project, tmp_path):
        with pytest.raises(ValueError, match="expected"):
            make_glofs_boundary(
                work_dir=tmp_path,
                prebuilt_dir=project,
                waterlevel_file=_waterlevel(tmp_path, hours=DURATION),
                start_date=START,
                duration_hours=DURATION,
            )


def _config(tmp_path: Path, source: BoundarySource, project: Path) -> CoastalCalibConfig:
    return CoastalCalibConfig(
        simulation=SimulationConfig(
            start_date=START,
            duration_hours=DURATION,
            coastal_domain="greatlakes",
            meteo_source="nwm_retro",
        ),
        boundary=BoundaryConfig(source=source, glofs_model="leofs"),
        paths=PathConfig(work_dir=tmp_path / "work", raw_download_dir=tmp_path / "dl"),
        model_config=SchismModelConfig(prebuilt_dir=project),
        download=DownloadConfig(enabled=False),
    )


class TestBoundaryRouting:
    def test_glofs_routes_to_glofs_stage(self, project, tmp_path, monkeypatch):
        monkeypatch.setattr(GLOFSBoundaryStage, "run", lambda self: {"status": "glofs"})
        stage = BoundaryConditionStage(_config(tmp_path, "glofs", project))
        assert stage.run() == {"status": "glofs"}

    def test_unknown_source_raises_instead_of_using_stofs(self, project, tmp_path):
        cfg = _config(tmp_path, "glofs", project)
        cfg.boundary.source = "fvcom"  # ty: ignore[invalid-assignment]
        with pytest.raises(ValueError, match=r"Unsupported boundary\.source"):
            BoundaryConditionStage(cfg).run()

    def test_glofs_stage_requires_lake(self, project, tmp_path):
        cfg = _config(tmp_path, "glofs", project)
        cfg.boundary.glofs_model = None
        assert GLOFSBoundaryStage(cfg).validate() == [
            "boundary.glofs_model is required when boundary.source is 'glofs'"
        ]
