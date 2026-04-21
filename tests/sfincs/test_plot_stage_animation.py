"""Integration tests for SFINCS ``create_water_level_animation`` workflow toggle.

Drives :class:`~coastal_calibration.sfincs.stages.SfincsPlotStage` end-to-end
with a synthetic ``sfincs_map.nc`` and verifies that the animation GIF is
produced when the toggle is on — and is skipped cleanly when it's off.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from coastal_calibration.config.schema import (
    BoundaryConfig,
    CoastalCalibConfig,
    DownloadConfig,
    MonitoringConfig,
    PathConfig,
    SfincsModelConfig,
    SimulationConfig,
)
from coastal_calibration.sfincs.stages import SfincsPlotStage


def _have_ffmpeg() -> bool:
    from matplotlib.animation import writers

    return writers.is_available("ffmpeg")


_skip_if_no_ffmpeg = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available on PATH")

# ---------------------------------------------------------------------------
# Synthetic fixture builder — UGRID quadtree layout matches real SFINCS output.
# ---------------------------------------------------------------------------


def _write_ugrid_quadtree_map(path: Path, *, n_time: int = 3) -> None:
    """Write a minimal UGRID-quadtree ``sfincs_map.nc`` (4 quads on a 3x3 node grid)."""
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)
    face_nodes = np.array(
        [[1, 2, 5, 4], [2, 3, 6, 5], [4, 5, 8, 7], [5, 6, 9, 8]], dtype=np.float64
    )
    times = pd.date_range("2024-01-01", periods=n_time, freq="1h")
    zs = np.outer(np.arange(1.0, n_time + 1.0), np.arange(1.0, 5.0)).astype(np.float32)
    zb = np.array([-2.0, -1.5, -1.0, -0.5], dtype=np.float32)  # bed below datum

    ds = xr.Dataset(
        data_vars={
            "zs": (("time", "nmesh2d_face"), zs),
            "zb": (("nmesh2d_face",), zb),
            "mesh2d_node_x": (("nmesh2d_node",), node_x),
            "mesh2d_node_y": (("nmesh2d_node",), node_y),
            "mesh2d_face_nodes": (
                ("nmesh2d_face", "max_nmesh2d_face_nodes"),
                face_nodes,
            ),
        },
        coords={"time": times},
    )
    ds["mesh2d_face_nodes"].attrs["start_index"] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


@pytest.fixture
def sfincs_model_with_map(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out a minimal SFINCS ``model_root`` containing a synthetic map file.

    Returns ``(work_dir, model_root)``.
    """
    work = tmp_path / "work"
    work.mkdir()
    model_root = work / "sfincs_model"
    _write_ugrid_quadtree_map(model_root / "sfincs_map.nc")
    # SfincsModelConfig requires ``sfincs.inp`` to pass validation; we don't
    # run the full config validator in these tests, so the file doesn't need
    # content — just existence for defensive callers.
    (model_root / "sfincs.inp").write_text("")
    return work, model_root


def _make_config(work_dir: Path, tmp_path: Path, **sfincs_kwargs) -> CoastalCalibConfig:
    # prebuilt_dir is required — point it at the work dir; no files are read
    # from it in the plot stage.
    cfg = CoastalCalibConfig(
        simulation=SimulationConfig(
            start_date=datetime(2024, 1, 1),
            duration_hours=3,
            coastal_domain="atlgulf",
            meteo_source="nwm_retro",
        ),
        boundary=BoundaryConfig(source="tpxo"),
        paths=PathConfig(
            work_dir=work_dir,
            raw_download_dir=tmp_path / "downloads",
        ),
        model_config=SfincsModelConfig(prebuilt_dir=work_dir, **sfincs_kwargs),
        monitoring=MonitoringConfig(),
        download=DownloadConfig(enabled=False),
    )
    return cfg


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestSfincsPlotStageAnimation:
    def test_animation_disabled_by_default(
        self, sfincs_model_with_map: tuple[Path, Path], tmp_path: Path
    ):
        """Default config → animation absent, station compare skipped (no his)."""
        work, model_root = sfincs_model_with_map
        cfg = _make_config(work, tmp_path)
        stage = SfincsPlotStage(cfg)
        result = stage.run()

        assert result["status"] == "completed"
        assert result["stations"]["status"] == "skipped"
        assert "animation" not in result
        assert not (model_root / "figs" / "water_level.mp4").exists()

    @_skip_if_no_ffmpeg
    def test_animation_mp4_produced_when_enabled(
        self, sfincs_model_with_map: tuple[Path, Path], tmp_path: Path
    ):
        work, _model_root = sfincs_model_with_map
        cfg = _make_config(
            work,
            tmp_path,
            create_water_level_animation=True,
            animation_fps=5,
        )
        stage = SfincsPlotStage(cfg)
        result = stage.run()

        assert result["status"] == "completed"
        assert result["animation"]["status"] == "completed"
        out = Path(result["animation"]["animation"])
        assert out.is_file()
        assert out.name == "water_level.mp4"
        assert out.stat().st_size > 0

    def test_animation_skipped_when_map_missing(self, tmp_path: Path):
        """No ``sfincs_map.nc`` → animation step returns skipped."""
        work = tmp_path / "work"
        work.mkdir()
        (work / "sfincs_model").mkdir()
        cfg = _make_config(
            work,
            tmp_path,
            create_water_level_animation=True,
        )
        stage = SfincsPlotStage(cfg)
        result = stage.run()

        assert result["animation"]["status"] == "skipped"
        assert result["animation"]["reason"] == "no sfincs_map.nc"

    @_skip_if_no_ffmpeg
    def test_time_stride_honored(self, sfincs_model_with_map: tuple[Path, Path], tmp_path: Path):
        """A stride of 2 on a 3-frame series still produces a valid animation."""
        work, _model_root = sfincs_model_with_map
        cfg = _make_config(
            work,
            tmp_path,
            create_water_level_animation=True,
            animation_time_stride=2,
        )
        stage = SfincsPlotStage(cfg)
        result = stage.run()

        out = Path(result["animation"]["animation"])
        assert out.is_file()
        assert out.stat().st_size > 0


class TestSfincsPlotStageObsPoints:
    """Tests for the user-obs-points path (obs_points_csv config field)."""

    def test_obs_parquet_written_for_user_csv(
        self, sfincs_model_with_map: tuple[Path, Path], tmp_path: Path
    ):
        import pandas as pd

        work, _model_root = sfincs_model_with_map
        # Synthetic mesh is in a raw 3x3 grid spanning (0, 0) .. (2, 2);
        # without a CRS attr the observations helper treats those as WGS84.
        csv = tmp_path / "user_obs.csv"
        pd.DataFrame({"id": ["p1", "p2"], "lon": [0.5, 1.5], "lat": [0.5, 1.5]}).to_csv(
            csv, index=False
        )

        cfg = _make_config(work, tmp_path, obs_points_csv=csv)
        stage = SfincsPlotStage(cfg)
        result = stage.run()

        assert result["status"] == "completed"
        assert result["obs_points"]["status"] == "completed"
        outfile = Path(result["obs_points"]["path"])
        assert outfile.exists()
        assert outfile.name == "obs_water_level.parquet"

        df = pd.read_parquet(outfile)
        assert set(df.columns) == {"p1", "p2"}
        assert len(df) == 3  # three timesteps in the synthetic fixture

    def test_obs_csv_outside_domain_raises(
        self, sfincs_model_with_map: tuple[Path, Path], tmp_path: Path
    ):
        import pandas as pd

        work, _ = sfincs_model_with_map
        csv = tmp_path / "bad_obs.csv"
        pd.DataFrame({"id": ["way_off"], "lon": [99.0], "lat": [99.0]}).to_csv(csv, index=False)

        cfg = _make_config(work, tmp_path, obs_points_csv=csv)
        stage = SfincsPlotStage(cfg)
        with pytest.raises(ValueError, match="outside the model"):
            stage.run()
