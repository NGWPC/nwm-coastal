"""Integration tests for the ``create_water_level_animation`` workflow toggle.

Drives :class:`~coastal_calibration.schism.stages.SchismPlotStage` end-to-end
with a synthetic SCHISM ``outputs/`` directory and verifies that the
animation GIF is produced when the toggle is on — and is skipped cleanly
when it's off.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from coastal_calibration.config.schema import (
    BoundaryConfig,
    CoastalCalibConfig,
    DownloadConfig,
    MonitoringConfig,
    PathConfig,
    SchismModelConfig,
    SimulationConfig,
)
from coastal_calibration.schism.stages import SchismPlotStage


def _have_ffmpeg() -> bool:
    from matplotlib.animation import writers

    return writers.is_available("ffmpeg")


_skip_if_no_ffmpeg = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available on PATH")

# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------


_NODE_X = np.array([0.0, 1.0, 2.0, 0.5, 1.5], dtype=np.float64)
_NODE_Y = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
_DEPTH = np.array([5.0, 4.5, 4.0, 3.5, 3.0], dtype=np.float64)
_FACE_NODES = np.array([[1, 2, 4, 0], [2, 5, 4, 0], [2, 3, 5, 0]], dtype=np.int32)


def _write_out2d_block(path: Path, *, seconds: np.ndarray, elev: np.ndarray) -> None:
    ds = xr.Dataset(
        data_vars={
            "elevation": (("time", "nSCHISM_hgrid_node"), elev.astype(np.float32)),
            "depth": (("nSCHISM_hgrid_node",), _DEPTH),
            "SCHISM_hgrid_node_x": (("nSCHISM_hgrid_node",), _NODE_X),
            "SCHISM_hgrid_node_y": (("nSCHISM_hgrid_node",), _NODE_Y),
            "SCHISM_hgrid_face_nodes": (
                ("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"),
                _FACE_NODES,
            ),
        },
        coords={"time": seconds.astype(np.float64)},
    )
    ds["time"].attrs["base_date"] = "2020 1 1 0 0"
    ds.to_netcdf(path)


@pytest.fixture
def schism_run_dir(tmp_path: Path) -> Path:
    """Populate ``<tmp>/work/outputs/`` with two tiny out2d blocks."""
    work = tmp_path / "work"
    outputs = work / "outputs"
    outputs.mkdir(parents=True)

    _write_out2d_block(
        outputs / "out2d_1.nc",
        seconds=np.array([0.0, 3600.0]),
        elev=np.outer(np.arange(1.0, 3.0), np.arange(1.0, 6.0)),
    )
    _write_out2d_block(
        outputs / "out2d_2.nc",
        seconds=np.array([7200.0, 10800.0]),
        elev=np.outer(np.arange(3.0, 5.0), np.arange(1.0, 6.0)),
    )
    return work


def _make_config(work_dir: Path, tmp_path: Path, **schism_kwargs) -> CoastalCalibConfig:
    return CoastalCalibConfig(
        simulation=SimulationConfig(
            start_date=datetime(2021, 6, 11),
            duration_hours=3,
            coastal_domain="pacific",
            meteo_source="nwm_retro",
        ),
        boundary=BoundaryConfig(source="tpxo"),
        paths=PathConfig(
            work_dir=work_dir,
            raw_download_dir=tmp_path / "downloads",
        ),
        model_config=SchismModelConfig(**schism_kwargs),
        monitoring=MonitoringConfig(),
        download=DownloadConfig(enabled=False),
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestSchismPlotStageAnimation:
    def test_animation_disabled_by_default(self, schism_run_dir: Path, tmp_path: Path):
        """Neither NOAA gauges nor animation → the stage is a no-op."""
        cfg = _make_config(schism_run_dir, tmp_path)
        stage = SchismPlotStage(cfg)
        result = stage.run()

        assert result["status"] == "skipped"
        assert not (schism_run_dir / "figs" / "water_level.mp4").exists()

    @_skip_if_no_ffmpeg
    def test_animation_mp4_produced_when_enabled(self, schism_run_dir: Path, tmp_path: Path):
        cfg = _make_config(
            schism_run_dir,
            tmp_path,
            create_water_level_animation=True,
            animation_fps=5,
        )
        stage = SchismPlotStage(cfg)
        result = stage.run()

        assert result["status"] == "completed"
        assert result["animation"]["status"] == "completed"
        out = Path(result["animation"]["animation"])
        assert out.is_file()
        assert out.name == "water_level.mp4"
        assert out.stat().st_size > 0

    def test_animation_skipped_when_outputs_dir_missing(self, tmp_path: Path):
        """No ``outputs/`` dir at all → animation step returns skipped."""
        work = tmp_path / "work"
        work.mkdir()
        cfg = _make_config(
            work,
            tmp_path,
            create_water_level_animation=True,
        )
        stage = SchismPlotStage(cfg)
        result = stage.run()

        assert result["status"] == "completed"
        assert result["animation"]["status"] == "skipped"
        assert result["animation"]["reason"] == "no out2d_*.nc"

    @_skip_if_no_ffmpeg
    def test_stations_and_animation_are_independent(self, schism_run_dir: Path, tmp_path: Path):
        """With NOAA gauges off and animation on, the station path doesn't run."""
        cfg = _make_config(
            schism_run_dir,
            tmp_path,
            include_noaa_gages=False,
            create_water_level_animation=True,
        )
        stage = SchismPlotStage(cfg)
        result = stage.run()

        # Only the animation sub-result should be present.
        assert "animation" in result
        assert "stations" not in result


class TestSchismPlotStageObsPoints:
    """Tests for the user-obs-points path (obs_points_csv config field)."""

    def test_obs_parquet_written_for_user_csv(self, schism_run_dir: Path, tmp_path: Path):
        """A user CSV with valid points triggers the parquet output."""
        import pandas as pd

        # Mesh extent in the schism_run_dir fixture: node_x = 0..2, node_y = 0..1.
        csv = tmp_path / "user_obs.csv"
        pd.DataFrame({"id": ["p1", "p2"], "lon": [0.5, 1.5], "lat": [0.2, 0.8]}).to_csv(
            csv, index=False
        )

        cfg = _make_config(schism_run_dir, tmp_path, obs_points_csv=csv)
        stage = SchismPlotStage(cfg)
        result = stage.run()

        assert result["status"] == "completed"
        assert result["obs_points"]["status"] == "completed"
        outfile = Path(result["obs_points"]["path"])
        assert outfile.exists()
        assert outfile.name == "obs_water_level.parquet"

        df = pd.read_parquet(outfile)
        assert list(df.columns) == ["p1", "p2"]
        # Fixture writes two blocks with 2 timesteps each → 4 total.
        assert len(df) == 4

    def test_obs_csv_outside_domain_raises(self, schism_run_dir: Path, tmp_path: Path):
        """Points outside the mesh bbox should raise before parquet is written."""
        import pandas as pd

        csv = tmp_path / "bad_obs.csv"
        # Mesh lon goes up to 2.0; 99.0 is well outside.
        pd.DataFrame({"id": ["way_off"], "lon": [99.0], "lat": [0.5]}).to_csv(csv, index=False)

        cfg = _make_config(schism_run_dir, tmp_path, obs_points_csv=csv)
        stage = SchismPlotStage(cfg)
        with pytest.raises(ValueError, match="outside the model"):
            stage.run()
